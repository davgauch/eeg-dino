"""Analyse embedding space geometry across masking strategies.

Usage:
    python experiments/analyze_embedding_geometry.py \
        --checkpoint_root checkpoints/myrun \
        --strategies none theta beta delta alpha \
        --preset tiny
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.eeg_dino_model import EEGTransformer
from model.dpe_module import DecoupledPositionalEmbedding
from model.tfe_module import TimeFrequencyEmbedding
from configs import PRESETS
from datasets import SleepEDFDataset, get_dataset_root


STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
N_CLASSES   = len(STAGE_NAMES)

STAGE_COLORS = {
    0: '#e74c3c',
    1: '#e67e22',
    2: '#2ecc71',
    3: '#2980b9',
    4: '#9b59b6',
}


class FrozenBackbone(torch.nn.Module):
    def __init__(self, n_channels, sampling_rate, embed_dim,
                 n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe         = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe         = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens, _ = self.tfe(x)
        tokens    = self.dpe(tokens, channel_indices)
        cls, _    = self.transformer(tokens)
        return cls


def load_backbone(checkpoint_path, cfg, device):
    backbone = FrozenBackbone(
        n_channels=cfg['n_channels'], sampling_rate=cfg['sampling_rate'],
        embed_dim=cfg['embed_dim'],   n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],       mlp_dim=cfg['mlp_dim'],
    ).to(device)
    ckpt           = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone_state = {k: v for k, v in ckpt['student'].items()
                      if k.startswith(('tfe.', 'dpe.', 'transformer.'))}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    return backbone


def extract_embeddings(backbone, loader, device):
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting", leave=False):
            embeddings.append(backbone(x.to(device)).cpu().numpy())
            labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.vstack(embeddings), np.concatenate(labels)



def compute_centroids(embeddings, labels):
    return {s: embeddings[labels == s].mean(axis=0) for s in range(N_CLASSES)}


def compute_centroid_distances(centroids):
    dist = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j:
                dist[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    return dist


def compute_within_class_scatter(embeddings, labels, centroids):
    scatter = {}
    for s in range(N_CLASSES):
        mask        = labels == s
        dists       = np.linalg.norm(embeddings[mask] - centroids[s], axis=1)
        scatter[s]  = {'mean': float(dists.mean()), 'std': float(dists.std())}
    return scatter


def print_centroid_table(distances, strategies, baseline='none'):
    key_pairs = [(2, 3, 'N2--N3'), (0, 3, 'Wake--N3'),
                 (0, 4, 'Wake--REM'), (2, 4, 'N2--REM'), (1, 4, 'N1--REM')]

    col_w  = 12
    header = f"{'Pair':<14}" + ''.join(f"{s.capitalize():>{col_w}}" for s in strategies)
    print('\n── Centroid distances ──────────────────────────────────────')
    print(header); print('─' * len(header))
    for i, j, name in key_pairs:
        row = f"{name:<14}"
        for s in strategies:
            row += f"{distances[s][i, j]:>{col_w}.3f}"
        print(row)

    if baseline not in distances: return
    print(f'\n── Δ vs {baseline} ─────────────────────────────────────────')
    print(header); print('─' * len(header))
    for i, j, name in key_pairs:
        row  = f"{name:<14}"
        base = distances[baseline][i, j]
        for s in strategies:
            if s == baseline:
                row += f"{'---':>{col_w}}"
            else:
                d    = distances[s][i, j]
                pct  = (d - base) / base * 100
                row += f"{pct:>+{col_w}.1f}%"[:-1] + ' '
        print(row)


def print_scatter_table(scatter, strategies):
    col_w  = 12
    header = f"{'Stage':<10}" + ''.join(f"{s.capitalize():>{col_w}}" for s in strategies)
    print('\n── Within-class scatter ────────────────────────────────────')
    print(header); print('─' * len(header))
    for s_idx, name in enumerate(STAGE_NAMES):
        row = f"{name:<10}"
        for strat in strategies:
            row += f"{scatter[strat][s_idx]['mean']:>{col_w}.3f}"
        print(row)



def plot_centroid_distances(distances, strategies, results_dir):
    """Grouped bar chart of key centroid distances across strategies."""
    key_pairs  = [(2, 3, 'N2--N3'), (0, 3, 'Wake--N3'),
                  (0, 4, 'Wake--REM'), (2, 4, 'N2--REM'), (1, 4, 'N1--REM')]
    pair_names = [p[2] for p in key_pairs]
    n_pairs    = len(key_pairs)
    n_strat    = len(strategies)

    x      = np.arange(n_pairs)
    width  = 0.8 / n_strat
    colors = ['#95a5a6', '#e74c3c', '#2980b9', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(11, 5))
    for k, strat in enumerate(strategies):
        vals = [distances[strat][i, j] for i, j, _ in key_pairs]
        ax.bar(x + k * width - 0.4 + width / 2, vals, width,
               label=strat.capitalize(), color=colors[k % len(colors)],
               edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=10)
    ax.set_ylabel('Centroid Distance (Euclidean)', fontsize=10)
    ax.set_xlabel('Stage Pair', fontsize=10)
    ax.set_title('Centroid Distances by Stage Pair and Masking Strategy',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'centroid_distances.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def plot_scatter_comparison(scatter, strategies, results_dir):
    """Grouped bar chart of within-class scatter per stage."""
    n_stages = N_CLASSES
    n_strat  = len(strategies)
    x        = np.arange(n_stages)
    width    = 0.8 / n_strat
    colors   = ['#95a5a6', '#e74c3c', '#2980b9', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, strat in enumerate(strategies):
        vals = [scatter[strat][s]['mean'] for s in range(N_CLASSES)]
        errs = [scatter[strat][s]['std']  for s in range(N_CLASSES)]
        ax.bar(x + k * width - 0.4 + width / 2, vals, width,
               yerr=errs, label=strat.capitalize(),
               color=colors[k % len(colors)],
               edgecolor='black', linewidth=0.7,
               error_kw={'linewidth': 1.0, 'capsize': 2})

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES, fontsize=10)
    ax.set_ylabel('Mean Distance to Centroid', fontsize=10)
    ax.set_xlabel('Sleep Stage', fontsize=10)
    ax.set_title('Within-Class Scatter by Stage and Masking Strategy',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'within_class_scatter.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()



def iter_checkpoints(checkpoint_root, strategies, seeds):
    root     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', checkpoint_root)) \
               if not os.path.isabs(checkpoint_root) else checkpoint_root
    seed_set = set(seeds) if seeds else None
    for strat in strategies:
        for run_dir in sorted(glob.glob(os.path.join(root, f'{strat}_seed*'))):
            if not os.path.isdir(run_dir): continue
            try:
                seed = int(os.path.basename(run_dir).split('_seed')[-1])
            except ValueError:
                continue
            if seed_set and seed not in seed_set: continue
            ckpt = os.path.join(run_dir, 'best_model.pth')
            if os.path.exists(ckpt):
                yield strat, seed, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_root', required=True)
    parser.add_argument('--strategies', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='*', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--preset', default='tiny', choices=list(PRESETS.keys()))
    args = parser.parse_args()

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg         = PRESETS[args.preset]
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    sleep_root  = get_dataset_root('sleep_edf')
    test_loader = DataLoader(
        SleepEDFDataset(sleep_root, 'TestFold', cfg['n_channels'], cfg['sampling_rate']),
        batch_size=256, shuffle=False, num_workers=1)


    per_seed = defaultdict(lambda: {'distances': [], 'scatter': []})

    for strat, seed, ckpt in iter_checkpoints(args.checkpoint_root,
                                               args.strategies, args.seeds):
        print(f"Processing {strat} seed={seed}")
        backbone          = load_backbone(ckpt, cfg, device)
        embeddings, labels = extract_embeddings(backbone, test_loader, device)
        centroids         = compute_centroids(embeddings, labels)
        per_seed[strat]['distances'].append(compute_centroid_distances(centroids))
        per_seed[strat]['scatter'].append(
            compute_within_class_scatter(embeddings, labels, centroids))

    # aggregate across seeds 
    strategies = [s for s in args.strategies if s in per_seed]
    distances  = {}
    scatter    = {}

    for strat in strategies:
        distances[strat] = np.mean(per_seed[strat]['distances'], axis=0)
        scatter[strat]   = {
            s: {
                'mean': float(np.mean([d[s]['mean'] for d in per_seed[strat]['scatter']])),
                'std':  float(np.std( [d[s]['mean'] for d in per_seed[strat]['scatter']],
                                      ddof=1)),
            }
            for s in range(N_CLASSES)
        }

    print_centroid_table(distances, strategies)
    print_scatter_table(scatter, strategies)
    plot_centroid_distances(distances, strategies, results_dir)
    plot_scatter_comparison(scatter, strategies, results_dir)

    out = {
        'centroid_distances': {s: distances[s].tolist() for s in strategies},
        'within_class_scatter': {
            s: {STAGE_NAMES[i]: scatter[s][i] for i in range(N_CLASSES)}
            for s in strategies
        },
    }
    out_path = os.path.join(results_dir, 'embedding_geometry.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == '__main__':
    main()