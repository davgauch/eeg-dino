"""Evaluate pairwise downstream classification accuracy and save predictions.

Usage:
    python experiments/downstream_pairwise_accuracy.py \
        --checkpoint_root checkpoints/myrun \
        --strategies none random theta delta alpha beta \
        --preset tiny
"""

import os
import sys
import argparse
import json
import glob
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.eeg_dino_model import EEGTransformer
from model.dpe_module import DecoupledPositionalEmbedding
from model.tfe_module import TimeFrequencyEmbedding
from configs import PRESETS
from datasets import SleepEDFDataset, get_dataset_root


STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


class FrozenBackbone(nn.Module):
    def __init__(self, n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe         = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe         = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens, _ = self.tfe(x)
        tokens    = self.dpe(tokens, channel_indices)
        cls, _    = self.transformer(tokens)
        return cls


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.head(x)


class DownstreamClassifier(nn.Module):
    def __init__(self, backbone, probe):
        super().__init__()
        self.backbone = backbone
        self.probe    = probe

    def forward(self, x, channel_indices=None):
        return self.probe(self.backbone(x, channel_indices))


def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting", leave=False):
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(backbone, train_loader, val_loader, embed_dim, device,
                       n_classes=5, epochs=50, lr=1e-3):
    tr_f, tr_y = extract_features(backbone, train_loader, device)
    va_f, va_y = extract_features(backbone, val_loader,   device)

    probe     = LinearProbe(embed_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for _ in range(1, epochs + 1):
        probe.train()
        idx = torch.randperm(len(tr_f))
        for i in range(0, len(tr_f), 256):
            b      = idx[i:i + 256]
            logits = probe(tr_f[b].to(device))
            loss   = criterion(logits, tr_y[b].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(va_f.to(device)).argmax(1).cpu().numpy()
            acc   = accuracy_score(va_y.numpy(), preds)
        if acc > best_acc:
            best_acc  = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_acc


def get_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting", leave=False):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.concatenate(preds), np.concatenate(labels)


def compute_pairwise_accuracy(predictions, labels, stage_pairs, stage_names):
    results = {}
    for (i, j) in stage_pairs:
        mask     = (labels == i) | (labels == j)
        pair_key = f"{stage_names[i]}_vs_{stage_names[j]}"
        results[pair_key] = float((predictions[mask] == labels[mask]).mean()) \
                            if mask.sum() > 0 else float('nan')
    return results


def load_backbone(checkpoint_path, cfg, device):
    backbone = FrozenBackbone(
        n_channels=cfg["n_channels"], sampling_rate=cfg["sampling_rate"],
        embed_dim=cfg["embed_dim"],   n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],       mlp_dim=cfg["mlp_dim"],
    ).to(device)

    ssl_ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone_state = {k: v for k, v in ssl_ckpt["student"].items()
                      if k.startswith(("tfe.", "dpe.", "transformer."))}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    return backbone


def iter_checkpoint_runs(checkpoint_root, strategies, seeds=None):
    root     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', checkpoint_root)) \
               if not os.path.isabs(checkpoint_root) else checkpoint_root
    seed_set = set(seeds) if seeds else None

    for strategy in strategies:
        for run_dir in sorted(glob.glob(os.path.join(root, f"{strategy}_seed*"))):
            if not os.path.isdir(run_dir): continue
            base = os.path.basename(run_dir)
            try:
                seed = int(base.split('_seed')[-1])
            except ValueError:
                continue
            if seed_set and seed not in seed_set: continue
            ckpt = os.path.join(run_dir, 'best_model.pth')
            if os.path.exists(ckpt):
                yield strategy, seed, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      help="Single checkpoint path")
    parser.add_argument("--strategy",        help="Strategy name (single-run mode)")
    parser.add_argument("--seed", type=int,  help="Seed (single-run mode)")
    parser.add_argument("--checkpoint_root", help="Root directory for batch mode")
    parser.add_argument("--strategies", nargs='+')
    parser.add_argument("--seeds",      nargs='*', type=int)
    parser.add_argument("--preset", default="tiny", choices=list(PRESETS.keys()))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = PRESETS[args.preset]

    if args.checkpoint_root and args.strategies:
        run_list = list(iter_checkpoint_runs(args.checkpoint_root,
                                             args.strategies, args.seeds))
        if not run_list:
            raise SystemExit("No checkpoints found.")
    elif args.checkpoint:
        if not args.strategy or args.seed is None:
            raise SystemExit("Single-run mode requires --strategy and --seed.")
        run_list = [(args.strategy, args.seed, args.checkpoint)]
    else:
        raise SystemExit("Provide --checkpoint_root/--strategies or --checkpoint.")

    sleep_root   = get_dataset_root('sleep_edf')
    train_loader = DataLoader(SleepEDFDataset(sleep_root, 'TrainFold',
                              cfg['n_channels'], cfg['sampling_rate']),
                              batch_size=256, shuffle=True,  num_workers=1)
    val_loader   = DataLoader(SleepEDFDataset(sleep_root, 'ValidFold',
                              cfg['n_channels'], cfg['sampling_rate']),
                              batch_size=256, shuffle=False, num_workers=1)
    test_loader  = DataLoader(SleepEDFDataset(sleep_root, 'TestFold',
                              cfg['n_channels'], cfg['sampling_rate']),
                              batch_size=256, shuffle=False, num_workers=1)

    stage_pairs = list(combinations(range(len(STAGE_NAMES)), 2))
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {}
    for strat, seed, ckpt in run_list:
        print(f"Evaluating {strat} seed={seed} → {ckpt}")
        torch.manual_seed(seed); np.random.seed(seed)

        backbone  = load_backbone(ckpt, cfg, device)
        probe, _  = train_linear_probe(backbone, train_loader, val_loader,
                                       cfg['embed_dim'], device)
        model     = DownstreamClassifier(backbone, probe)
        preds, labels = get_predictions(model, test_loader, device)

        # save raw predictions for confusion analysis
        pred_dir  = os.path.join(results_dir, f"{strat}_seed{seed}")
        os.makedirs(pred_dir, exist_ok=True)
        np.savez(os.path.join(pred_dir, "test_predictions.npz"),
                 predictions=preds, true_labels=labels)

        pairwise_acc = compute_pairwise_accuracy(preds, labels,
                                                 stage_pairs, STAGE_NAMES)
        results.setdefault(strat, {'per_seed': {}})
        results[strat]['per_seed'][str(seed)] = {'pairwise_accuracy': pairwise_acc}

    # aggregate across seeds 
    aggregated = {}
    for strat, info in results.items():
        per       = info['per_seed']
        pair_keys = sorted({k for v in per.values()
                            for k in v['pairwise_accuracy']})
        aggregated[strat] = {
            'n_seeds': len(per),
            'pairwise_accuracy_mean': {
                k: float(np.mean([v['pairwise_accuracy'][k]
                                  for v in per.values()
                                  if k in v['pairwise_accuracy']]))
                for k in pair_keys
            },
        }

    out_path = os.path.join(results_dir, 'downstream_pairwise_aggregated.json')
    with open(out_path, 'w') as f:
        json.dump({'per_run': results, 'aggregated': aggregated}, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()