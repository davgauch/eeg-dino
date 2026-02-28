"""
EEG-DINO Evaluation: Linear Probing on Sleep-EDF Test Set
==========================================================

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from eeg_dino_model import StudentModel
from train import SleepEDFDataset

SLEEP_EDF_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/'


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor (frozen backbone)
# ─────────────────────────────────────────────────────────────────────────────

class FrozenBackbone(nn.Module):
    """
    Loads the pre-trained student, freezes all parameters.
    Returns the CLS token embedding for each input.
    """
    def __init__(self, checkpoint_path, n_channels=19, sampling_rate=200,
                 embed_dim=200, n_layers=12, n_heads=8, mlp_dim=512):
        super().__init__()

        self.model = StudentModel(
            n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim
        )

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['student_state_dict'])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(loss: {ckpt['metrics']['loss']:.4f})")

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, x):
        """
        x: [batch, n_channels, n_samples]
        Returns: [batch, 256]  (signal-level features from signal_head)
        """
        # Use all channels (no channel_indices needed for full input)
        return self.model(x, channel_indices=None, return_patch=False)


# ─────────────────────────────────────────────────────────────────────────────
# Extract features for entire dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone, dataloader, device):
    """
    Run the frozen backbone over the dataset and collect features + labels.
    Returns numpy arrays: features [N, 256], labels [N]
    """
    backbone.eval()
    all_feats  = []
    all_labels = []

    for x, y in tqdm(dataloader, desc="Extracting features"):
        x = x.to(device)
        feats = backbone(x)                # [batch, 256]
        all_feats.append(feats.cpu())
        all_labels.append(y)

    return (
        torch.cat(all_feats,  dim=0).numpy(),
        torch.cat(all_labels, dim=0).numpy()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, feat_dim=256, n_classes=5):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(X_train, y_train, X_val, y_val,
                        feat_dim=256, n_classes=5,
                        n_epochs=50, lr=1e-3, batch_size=512, device='cuda'):
    """
    Trains a linear classifier on top of pre-extracted features.
    Uses X_val for early stopping.
    """
    probe = LinearProbe(feat_dim, n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.LongTensor(y_val).to(device)

    train_ds     = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_acc = 0
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Val accuracy for early stopping
        probe.eval()
        with torch.no_grad():
            preds = probe(X_v).argmax(dim=1).cpu().numpy()
        val_ba = balanced_accuracy_score(y_v.cpu().numpy(), preds)

        if val_ba > best_val_acc:
            best_val_acc = val_ba
            best_state   = {k: v.clone() for k, v in probe.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Linear probe epoch {epoch:3d}/{n_epochs} — "
                  f"val balanced acc: {val_ba:.4f}")

    probe.load_state_dict(best_state)
    print(f"  Best val balanced accuracy: {best_val_acc:.4f}")
    return probe


def evaluate_probe(probe, X_test, y_test, device='cuda'):
    probe.eval()
    X = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        preds = probe(X).argmax(dim=1).cpu().numpy()

    ba    = balanced_accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    wf1   = f1_score(y_test, preds, average='weighted')

    print("\n" + "="*50)
    print("Test Set Results (linear probing)")
    print("="*50)
    print(f"  Balanced Accuracy : {ba:.4f}  ({ba*100:.2f}%)")
    print(f"  Cohen's Kappa     : {kappa:.4f}")
    print(f"  Weighted F1       : {wf1:.4f}")
    print("="*50)
    print("\nReference (EEG-DINO-S, paper Table 2, TUEV):")
    print("  Balanced Accuracy : 0.5482 (54.82%)")
    print("  Cohen's Kappa     : 0.5673")
    print("  Weighted F1       : 0.7861")
    print("(Paper used TUEG pre-training data, 1.1M samples vs your ~92K)")

    return {'balanced_accuracy': ba, 'cohens_kappa': kappa, 'weighted_f1': wf1}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--probe_lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load backbone ─────────────────────────────────────────────────────────
    backbone = FrozenBackbone(args.checkpoint).to(device)

    # ── Load Sleep-EDF splits (server already provides Train/Valid/Test) ──────
    print("\nLoading Sleep-EDF splits...")
    ds_train = SleepEDFDataset(SLEEP_EDF_PATH, fold='TrainFold')
    ds_val   = SleepEDFDataset(SLEEP_EDF_PATH, fold='ValidFold')
    ds_test  = SleepEDFDataset(SLEEP_EDF_PATH, fold='TestFold')

    kw = dict(batch_size=args.batch_size, num_workers=4,
               pin_memory=True, shuffle=False)
    train_loader = DataLoader(ds_train, **kw)
    val_loader   = DataLoader(ds_val,   **kw)
    test_loader  = DataLoader(ds_test,  **kw)

    # ── Extract features (one forward pass, no grad) ──────────────────────────
    print("\nExtracting features...")
    X_train, y_train = extract_features(backbone, train_loader, device)
    X_val,   y_val   = extract_features(backbone, val_loader,   device)
    X_test,  y_test  = extract_features(backbone, test_loader,  device)

    print(f"\nFeature shapes: train={X_train.shape}, val={X_val.shape}, "
          f"test={X_test.shape}")
    print(f"Labels: {np.unique(y_train)} (sleep stages 0-4)")

    # ── Train linear probe ────────────────────────────────────────────────────
    print("\nTraining linear probe...")
    probe = train_linear_probe(
        X_train, y_train, X_val, y_val,
        feat_dim=256, n_classes=5,
        n_epochs=args.probe_epochs,
        lr=args.probe_lr,
        device=device
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    results = evaluate_probe(probe, X_test, y_test, device=device)

    return results


if __name__ == '__main__':
    main()
