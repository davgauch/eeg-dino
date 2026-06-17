"""
Compute pairwise AUC for each frequency band using log band power.

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
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from datasets import SleepEDFDataset, get_dataset_root


BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'beta_upper': (20, 30),
    'theta_bw2': (4, 6),
    'theta_bw3': (4, 7),
    'theta_bw6': (4, 10),
    'beta_bw2': (18, 20),
    'beta_bw4': (18, 22),
}


PLOT_BANDS = [
    'delta', 'theta', 'alpha', 'beta'
]

PLOT_LABELS = [
    'Delta\n(1–4 Hz)',
    'Theta\n(4–8 Hz)',
    'Alpha\n(8–13 Hz)',
    'Beta\n(13–30 Hz)',

]


def extract_band_power(x, sampling_rate=200, band='theta'):
    low, high = BANDS[band]

    spectrum = torch.fft.rfft(x, dim=-1)
    power = torch.abs(spectrum) ** 2
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sampling_rate)

    mask = (freqs >= low) & (freqs < high)
    band_power = power[:, :, mask].mean(dim=2)

    return torch.log(band_power + 1e-10).cpu().numpy()


def main():
    print("Loading Sleep-EDF test set...")
    root = get_dataset_root('sleep_edf')
    ds = SleepEDFDataset(root, 'TestFold', n_channels=2, sampling_rate=200)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    band_powers = {b: [] for b in BANDS}
    all_labels = []

    print("Extracting band power...")
    for x, y in tqdm(loader, desc="Processing"):
        for b in BANDS:
            band_powers[b].append(extract_band_power(x, 200, b))
        all_labels.append(y.numpy())

    band_powers = {b: np.concatenate(v) for b, v in band_powers.items()}
    all_labels = np.concatenate(all_labels)

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pairs = list(combinations(range(5), 2))

    pairwise_auc = {}

    for i, j in pairs:
        mask = (all_labels == i) | (all_labels == j)
        y_bin = (all_labels[mask] == i).astype(int)

        key = f"{stage_names[i]}_vs_{stage_names[j]}"
        pairwise_auc[key] = {}

        for b in BANDS:
            X = band_powers[b][mask]
            score = X.mean(axis=1)

            auc = roc_auc_score(y_bin, score)
            pairwise_auc[key][b] = float(max(auc, 1 - auc))

    os.makedirs("results", exist_ok=True)
    out_path = "results/pairwise_auc_results.json"

    with open(out_path, "w") as f:
        json.dump(pairwise_auc, f, indent=2)

    print(f"Saved: {out_path}")

    # ---- heatmap only on selected bands (NO beta_upper) ----
    matrix = np.zeros((len(pairs), len(PLOT_BANDS)))

    pair_labels = [f"{stage_names[i]} vs {stage_names[j]}" for i, j in pairs]

    for i, key in enumerate(pairwise_auc):
        for j, b in enumerate(PLOT_BANDS):
            matrix[i, j] = pairwise_auc[key][b]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(matrix, cmap="RdYlGn",
                   norm=mcolors.Normalize(vmin=0.5, vmax=1.0),
                   aspect='auto')

    ax.set_xticks(range(len(PLOT_BANDS)))
    ax.set_xticklabels(PLOT_LABELS, fontsize=8)

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha='center', va='center',
                    fontsize=7,
                    color='black')

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Pairwise AUC", fontsize=11)

    plt.tight_layout()
    plt.savefig("results/pairwise_auc_heatmap.png", dpi=200)
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()