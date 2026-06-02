"""Evaluate pairwise downstream classification accuracy.

Usage:
    python experiments/downstream_pairwise_accuracy.py \
        --checkpoint_root checkpoints/significance_sleep_model2 \
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
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.eeg_dino_model import EEGTransformer
from model.dpe_module import DecoupledPositionalEmbedding
from model.tfe_module import TimeFrequencyEmbedding
from configs import PRESETS
from datasets import SleepEDFDataset, get_dataset_root


STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting", leave=False):
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(backbone, train_loader, val_loader, n_classes, embed_dim, device, epochs=50, lr=1e-3):
    tr_f, tr_y = extract_features(backbone, train_loader, device)
    va_f, va_y = extract_features(backbone, val_loader, device)

    probe = LinearProbe(embed_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for _ in range(1, epochs + 1):
        probe.train()
        idx = torch.randperm(len(tr_f))
        for i in range(0, len(tr_f), 256):
            b = idx[i:i + 256]
            logits = probe(tr_f[b].to(device))
            loss = criterion(logits, tr_y[b].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(va_f.to(device))
            preds = logits.argmax(1).cpu().numpy()
            gt = va_y.numpy()
            acc = accuracy_score(gt, preds)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_acc


def get_predictions(model, loader, device):
    """Run inference and collect predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Getting predictions", leave=False):
            logits = model(x.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(preds)

            if isinstance(y, torch.Tensor):
                all_labels.append(y.detach().cpu().numpy())
            else:
                all_labels.append(np.array(y))

    return np.concatenate(all_predictions), np.concatenate(all_labels)


class FrozenBackbone(nn.Module):
    """Backbone with frozen SSL weights."""

    def __init__(self, n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens, _ = self.tfe(x)
        tokens = self.dpe(tokens, channel_indices)
        cls, _ = self.transformer(tokens)
        return cls


class LinearProbe(nn.Module):
    """Linear classification head."""

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.head(x)


class DownstreamClassifier(nn.Module):
    """Frozen backbone followed by a trained linear probe."""

    def __init__(self, backbone, probe):
        super().__init__()
        self.backbone = backbone
        self.probe = probe

    def forward(self, x, channel_indices=None):
        features = self.backbone(x, channel_indices)
        logits = self.probe(features)
        return logits


def compute_pairwise_accuracy(predictions, labels, stage_pairs, stage_names):
    results = {}

    for (i, j) in stage_pairs:
        mask = (labels == i) | (labels == j)
        pair_predictions = predictions[mask]
        pair_labels = labels[mask]

        if len(pair_labels) == 0:
            results[f"{stage_names[i]}_vs_{stage_names[j]}"] = np.nan
            continue

        accuracy = (pair_predictions == pair_labels).mean()
        
        pair_name = f"{stage_names[i]}_vs_{stage_names[j]}"
        results[pair_name] = float(accuracy)
    
    return results


def compute_confusion_rates(predictions, labels, stage_pairs, stage_names):
    confusion = {}

    for (i, j) in stage_pairs:
        mask_i = (labels == i)
        if mask_i.sum() > 0:
            i_as_j = (predictions[mask_i] == j).sum() / mask_i.sum()
        else:
            i_as_j = np.nan

        mask_j = (labels == j)
        if mask_j.sum() > 0:
            j_as_i = (predictions[mask_j] == i).sum() / mask_j.sum()
        else:
            j_as_i = np.nan

        pair_name = f"{stage_names[i]}_vs_{stage_names[j]}"
        confusion[pair_name] = {
            f"{stage_names[i]}_as_{stage_names[j]}": float(i_as_j),
            f"{stage_names[j]}_as_{stage_names[i]}": float(j_as_i)
        }
    
    return confusion


def load_backbone(checkpoint_path, cfg, device):
    backbone = FrozenBackbone(
        n_channels=cfg["n_channels"],
        sampling_rate=cfg["sampling_rate"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        mlp_dim=cfg["mlp_dim"],
    ).to(device)

    ssl_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone_state = {}
    for key, value in ssl_ckpt["student"].items():
        if key.startswith(("tfe.", "dpe.", "transformer.")):
            backbone_state[key] = value

    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    return backbone


def load_probe(probe_path, embed_dim, device):
    probe = LinearProbe(embed_dim, n_classes=5).to(device)
    probe_ckpt = torch.load(probe_path, map_location=device, weights_only=True)
    probe.load_state_dict(probe_ckpt["probe_state_dict"])
    probe.eval()
    return probe


def iter_checkpoint_runs(checkpoint_root, strategies, seeds=None):
    root = checkpoint_root
    if not os.path.isabs(root):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', root))

    seed_set = set(seeds) if seeds is not None and len(seeds) > 0 else None
    for strategy in strategies:
        for run_dir in sorted(glob.glob(os.path.join(root, f"{strategy}_seed*"))):
            if not os.path.isdir(run_dir):
                continue
            base = os.path.basename(run_dir)
            if '_seed' not in base:
                continue
            try:
                seed = int(base.split('_seed')[-1])
            except ValueError:
                continue
            if seed_set is not None and seed not in seed_set:
                continue
            ckpt_path = os.path.join(run_dir, 'best_model.pth')
            if os.path.exists(ckpt_path):
                yield strategy, seed, ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate downstream pairwise classification accuracy")
    parser.add_argument("--checkpoint", help="Path to the SSL checkpoint (single-run mode)")
    parser.add_argument("--strategy", help="Masking strategy name (single-run mode)")
    parser.add_argument("--seed", type=int, help="Random seed (single-run mode)")
    parser.add_argument("--checkpoint_root", help="Root checkpoints directory for batch mode, e.g. checkpoints/significance_sleep_model2")
    parser.add_argument("--strategies", nargs='+', help="List of strategies to evaluate in batch mode")
    parser.add_argument("--seeds", nargs='*', type=int, help="Optional list of seeds to restrict batch mode")
    parser.add_argument("--preset", default="tiny", choices=list(PRESETS.keys()))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PRESETS[args.preset]

    if args.checkpoint_root and args.strategies:
        run_list = list(iter_checkpoint_runs(args.checkpoint_root, args.strategies, args.seeds))
        if not run_list:
            raise SystemExit('No checkpoints found for the given checkpoint_root/strategies/seeds')
    elif args.checkpoint:
        if args.strategy is None or args.seed is None:
            raise SystemExit('Single-run mode requires --strategy and --seed')
        run_list = [(args.strategy, args.seed, args.checkpoint)]
    else:
        raise SystemExit('Provide either --checkpoint_root and --strategies, or --checkpoint with --strategy/--seed')

    sleep_root = get_dataset_root('sleep_edf')
    train_ds = SleepEDFDataset(sleep_root, 'TrainFold', cfg['n_channels'], cfg['sampling_rate'])
    val_ds = SleepEDFDataset(sleep_root, 'ValidFold', cfg['n_channels'], cfg['sampling_rate'])
    test_ds = SleepEDFDataset(sleep_root, 'TestFold', cfg['n_channels'], cfg['sampling_rate'])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=1)

    stage_pairs = list(combinations(range(len(STAGE_NAMES)), 2))

    results = {}
    for strat, seed, ckpt in run_list:
        print(f"Evaluating {strat} seed={seed} -> ckpt={ckpt}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        backbone = load_backbone(ckpt, cfg, device)
        probe_model, _ = train_linear_probe(
            backbone, train_loader, val_loader, n_classes=5,
            embed_dim=cfg['embed_dim'], device=device,
            epochs=50, lr=1e-3,
        )

        preds, labels = get_predictions(DownstreamClassifier(backbone, probe_model), test_loader, device)
        pairwise_acc = compute_pairwise_accuracy(preds, labels, stage_pairs, STAGE_NAMES)
        out = {
            'pairwise_accuracy': pairwise_acc,
        }

        results.setdefault(strat, {'per_seed': {}})
        results[strat]['per_seed'][str(seed)] = out

    aggregated = {}
    for strat, info in results.items():
        per = info['per_seed']
        pair_keys = sorted({k for v in per.values() for k in v['pairwise_accuracy'].keys()})
        aggregated[strat] = {
            'n_seeds': len(per),
            'pairwise_accuracy_mean': {
                k: float(np.mean([v['pairwise_accuracy'][k] for v in per.values() if k in v['pairwise_accuracy']]))
                for k in pair_keys
            },
        }

    # save combined JSON
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'downstream_pairwise_aggregated.json')
    with open(out_path, 'w') as f:
        json.dump({'per_run': results, 'aggregated': aggregated}, f, indent=2)

    print(f"\n✓ Aggregated results saved to: {out_path}\n")


if __name__ == "__main__":
    main()