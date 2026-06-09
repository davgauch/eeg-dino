"""Compute pairwise AUC for each frequency band using log band power.
Usage:
    python compute_pairwise_AUC.py
"""
import os
import sys
import json
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from datasets import SleepEDFDataset, get_dataset_root


def extract_band_power(x, sampling_rate=200, band='theta'):
    band_ranges = {
        'delta':      (1,  4),
        'theta':      (4,  8),
        'alpha':      (8,  13),
        'beta':       (13, 30),
        'beta_upper': (20, 30),
        'theta_bw2':  (4,  6),
        'theta_bw3':  (4,  7),
        'theta_bw6':  (4,  10),
        'beta_bw2':   (18, 20),
        'beta_bw4':   (18, 22),
    }
    low_hz, high_hz = band_ranges[band]
    spectrum   = torch.fft.rfft(x, dim=-1)
    power      = torch.abs(spectrum) ** 2
    freqs      = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sampling_rate)
    mask       = (freqs >= low_hz) & (freqs < high_hz)
    band_power = power[:, :, mask]
    mean_power = band_power.mean(dim=2)
    log_power  = torch.log(mean_power + 1e-10)
    return log_power.cpu().numpy()


def plot_pairwise_auc(pairwise_auc, plot_bands, band_labels, stage_names,
                      results_dir):
    """Save one heatmap per band and one combined figure."""
    pairs     = [f"{a}_vs_{b}" for a, b in combinations(stage_names, 2)]
    n_pairs   = len(pairs)
    n_bands   = len(plot_bands)

    # ── build matrix (pairs × bands) ─────────────────────────────
    matrix = np.zeros((n_pairs, n_bands))
    for i, pair in enumerate(pairs):
        for j, band in enumerate(plot_bands):
            matrix[i, j] = pairwise_auc[pair][band]

    pair_labels = [p.replace('_vs_', ' vs ') for p in pairs]

    # ── combined heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_bands * 1.1), max(5, n_pairs * 0.55)))

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im   = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=9)

    # annotate cells
    for i in range(n_pairs):
        for j in range(n_bands):
            val   = matrix[i, j]
            color = 'black' if 0.65 < val < 0.92 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('AUC', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Pairwise AUC — Sleep Stage Separability by Frequency Band',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Stage Pair',     fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'pairwise_auc_heatmap.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()

    # ── print summary table ───────────────────────────────────────
    col_w  = 13
    header = f"{'Pair':<18}" + ''.join(f"{b:>{col_w}}" for b in band_labels)
    print('\n── Pairwise AUC summary ────────────────────────────────────')
    print(header)
    print('─' * len(header))
    for i, label in enumerate(pair_labels):
        row = f"{label:<18}"
        for j in range(n_bands):
            row += f"{matrix[i, j]:>{col_w}.3f}"
        print(row)


def main():
    print("Loading Sleep-EDF test set...")
    sleep_root = get_dataset_root('sleep_edf')
    test_ds    = SleepEDFDataset(sleep_root, 'TestFold',
                                 n_channels=2, sampling_rate=200)
    test_loader = DataLoader(test_ds, batch_size=256,
                             shuffle=False, num_workers=0)

    bands       = ['delta', 'theta', 'alpha', 'beta', 'beta_upper',
                   'theta_bw2', 'theta_bw3', 'theta_bw6',
                   'beta_bw2', 'beta_bw4']
    band_powers = {band: [] for band in bands}
    all_labels  = []

    print("Extracting band power...")
    for x, y in tqdm(test_loader, desc="Processing"):
        for band in bands:
            power = extract_band_power(x, sampling_rate=200, band=band)
            band_powers[band].append(power)
        all_labels.append(y.numpy())

    for band in bands:
        band_powers[band] = np.concatenate(band_powers[band], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    stage_names  = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pairs        = list(combinations(range(5), 2))
    pairwise_auc = {}

    for i, j in pairs:
        mask      = (all_labels == i) | (all_labels == j)
        y_bin     = (all_labels[mask] == i).astype(int)
        pair_key  = f"{stage_names[i]}_vs_{stage_names[j]}"
        pairwise_auc[pair_key] = {}
        for band in bands:
            X   = band_powers[band][mask]
            score = X.mean(axis=1)
            auc = roc_auc_score(y_bin, score)
            auc = max(auc, 1 - auc)
            pairwise_auc[pair_key][band] = float(auc)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, 'pairwise_auc_results.json')
    with open(output_path, 'w') as f:
        json.dump(pairwise_auc, f, indent=2)
    print(f"\nPairwise AUC results saved to {output_path}")

    # ── plot heatmap for the four canonical bands ─────────────────
    plot_bands  = ['delta', 'theta', 'alpha', 'beta', 'beta_upper']
    band_labels = ['Delta\n(1–4 Hz)', 'Theta\n(4–8 Hz)', 'Alpha\n(8–13 Hz)',
                   'Beta\n(13–30 Hz)', 'Upper Beta\n(20–30 Hz)']
    plot_pairwise_auc(pairwise_auc, plot_bands, band_labels,
                      stage_names, results_dir)

    print("Done.")


if __name__ == '__main__':
    main()