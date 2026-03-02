"""
EEG-DINO Linear Probing Evaluation
===================================
Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pth --preset tiny
  python evaluate.py --checkpoint checkpoints/best_model.pth --preset small --max_samples 5000
"""

import argparse, torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

from eeg_dino_model import EEGTransformer
from tfe_module import TimeFrequencyEmbedding
from dpe_module import DecoupledPositionalEmbedding
from train import SleepEDFDataset, SLEEP_EDF_PATH, PRESETS


class FrozenBackbone(nn.Module):
    """Student backbone (TFE→DPE→Transformer) with frozen weights."""

    def __init__(self, n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens = self.tfe(x)
        tokens = self.dpe(tokens, channel_indices)
        tokens = self.transformer(tokens)
        return tokens[:, 0]  # CLS token


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.head(x)


def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting", leave=False):
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(backbone, train_loader, val_loader, n_classes, embed_dim,
                        device, epochs=50, lr=1e-3):
    print("Extracting features...")
    tr_f, tr_y = extract_features(backbone, train_loader, device)
    va_f, va_y = extract_features(backbone, val_loader, device)

    probe = LinearProbe(embed_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0, None
    for ep in range(1, epochs + 1):
        probe.train()
        idx = torch.randperm(len(tr_f))
        total_loss = 0
        for i in range(0, len(tr_f), 256):
            b = idx[i:i+256]
            logits = probe(tr_f[b].to(device))
            loss = criterion(logits, tr_y[b].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        probe.eval()
        with torch.no_grad():
            logits = probe(va_f.to(device))
            preds = logits.argmax(1).cpu().numpy()
            gt = va_y.numpy()
            acc = accuracy_score(gt, preds)
            f1 = f1_score(gt, preds, average='macro')

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            n_batches = max(1, len(tr_f) // 256)
            print(f"  Probe ep{ep:3d} — loss:{total_loss/n_batches:.4f} "
                  f"acc:{acc:.4f} f1:{f1:.4f}")

    probe.load_state_dict(best_state)
    return probe, best_acc


def evaluate_probe(probe, backbone, loader, device, class_names=None):
    feats, labels = extract_features(backbone, loader, device)
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device))
        preds = logits.argmax(1).cpu().numpy()
    gt = labels.numpy()

    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average='macro')
    kappa = cohen_kappa_score(gt, preds)

    print(f"\n{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    print(f"  Kappa    : {kappa:.4f}")
    print(f"{'='*40}")

    if class_names:
        from sklearn.metrics import classification_report
        print(classification_report(gt, preds, target_names=class_names, digits=4))

    return {'accuracy': acc, 'f1_macro': f1, 'kappa': kappa}


def main():
    p = argparse.ArgumentParser(description='EEG-DINO Linear Probing')
    p.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    p.add_argument('--preset', default='tiny', choices=PRESETS)
    p.add_argument('--n_channels', type=int, default=2)
    p.add_argument('--n_classes', type=int, default=5)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--probe_epochs', type=int, default=50)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pr = PRESETS[args.preset]

    # Load backbone from checkpoint
    backbone = FrozenBackbone(
        args.n_channels, 200,
        pr['embed_dim'], pr['n_layers'], pr['n_heads'], pr['mlp_dim']
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    student_sd = ckpt['student']

    # Map keys: strip student prefix, keep backbone layers
    bb_sd = {}
    for k, v in student_sd.items():
        for prefix in ('tfe.', 'dpe.', 'transformer.'):
            if k.startswith(prefix):
                bb_sd[k] = v
                break
    backbone.load_state_dict(bb_sd, strict=True)
    print(f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Load Sleep-EDF splits
    train_ds = SleepEDFDataset(SLEEP_EDF_PATH, 'TrainFold',
                                args.n_channels, 200, args.max_samples)
    val_ds = SleepEDFDataset(SLEEP_EDF_PATH, 'ValidFold',
                              args.n_channels, 200, args.max_samples)
    test_ds = SleepEDFDataset(SLEEP_EDF_PATH, 'TestFold',
                               args.n_channels, 200, args.max_samples)

    train_loader = DataLoader(train_ds, 256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, 256, num_workers=4)
    test_loader = DataLoader(test_ds, 256, num_workers=4)

    # Train linear probe
    probe, val_acc = train_linear_probe(
        backbone, train_loader, val_loader, args.n_classes,
        pr['embed_dim'], device, args.probe_epochs, args.probe_lr)

    # Evaluate on test
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    results = evaluate_probe(probe, backbone, test_loader, device, class_names)


if __name__ == '__main__':
    main()
