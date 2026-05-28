"""EEG-DINO Linear Probing Evaluation.

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse, torch, numpy as np
import random
import torch.nn as nn
import logging
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

from model.eeg_dino_model import EEGTransformer
from model.tfe_module import TimeFrequencyEmbedding
from model.dpe_module import DecoupledPositionalEmbedding
from configs import PRESETS
from datasets import SleepEDFDataset, get_dataset_root


class FrozenBackbone(nn.Module):
    """Backbone (TFE → DPE → Transformer) with frozen weights."""

    def __init__(self, n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens, _ = self.tfe(x)  # discard raw_features in eval
        tokens = self.dpe(tokens, channel_indices)
        cls, _ = self.transformer(tokens)
        return cls


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
    logging.getLogger(__name__).info("Extracting features...")
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
            logging.getLogger(__name__).info(
                f"Probe ep{ep:3d} — loss:{total_loss/n_batches:.4f} "
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

    logging.getLogger(__name__).info(f"\n{'='*40}")
    logging.getLogger(__name__).info(f"  Accuracy : {acc:.4f}")
    logging.getLogger(__name__).info(f"  F1 Macro : {f1:.4f}")
    logging.getLogger(__name__).info(f"  Kappa    : {kappa:.4f}")
    logging.getLogger(__name__).info(f"{'='*40}")

    if class_names:
        from sklearn.metrics import classification_report
        logging.getLogger(__name__).info(
            classification_report(gt, preds, target_names=class_names, digits=4))

    return {'accuracy': acc, 'f1_macro': f1, 'kappa': kappa}


def main():
    p = argparse.ArgumentParser(description='EEG-DINO Linear Probing')
    p.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    p.add_argument('--n_classes', type=int, default=5)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--probe_epochs', type=int, default=50)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--preset', default='tiny', choices=list(PRESETS.keys()))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = PRESETS[args.preset]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # configure logging
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    backbone = FrozenBackbone(
        cfg['n_channels'], cfg['sampling_rate'],
        cfg['embed_dim'], cfg['n_layers'], cfg['n_heads'], cfg['mlp_dim']
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    bb_sd = {}
    for k, v in ckpt['student'].items():
        for prefix in ('tfe.', 'dpe.', 'transformer.'):
            if k.startswith(prefix):
                bb_sd[k] = v
                break
    backbone.load_state_dict(bb_sd, strict=True)
    logging.getLogger(__name__).info(
        f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Load Sleep-EDF splits
    n_ch, sr = cfg['n_channels'], cfg['sampling_rate']
    sleep_root = get_dataset_root('sleep_edf')
    train_ds = SleepEDFDataset(sleep_root, 'TrainFold', n_ch, sr)
    val_ds = SleepEDFDataset(sleep_root, 'ValidFold', n_ch, sr)
    test_ds = SleepEDFDataset(sleep_root, 'TestFold', n_ch, sr)

    train_loader = DataLoader(train_ds, 256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, 256, num_workers=4)
    test_loader = DataLoader(test_ds, 256, num_workers=4)

    # Train linear probe
    probe, val_acc = train_linear_probe(
        backbone, train_loader, val_loader, args.n_classes,
        cfg['embed_dim'], device, args.probe_epochs, args.probe_lr)

    # Evaluate on test
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    results = evaluate_probe(probe, backbone, test_loader, device, class_names)


if __name__ == '__main__':
    main()
