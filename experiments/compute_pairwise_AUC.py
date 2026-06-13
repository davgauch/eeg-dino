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
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from datasets import SleepEDFDataset, get_dataset_root

def extract_band_power(x, sampling_rate=200, band='theta'):
    band_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'beta_upper': (20, 30),
        'theta_bw2':(4,6),
        'theta_bw3':(4,7),
        'theta_bw6':(4,10),
        'beta_bw2':(18,20),
        'beta_bw4':(18,22),

    }
    low_hz, high_hz = band_ranges[band]
    spectrum = torch.fft.rfft(x, dim=-1)
    power = torch.abs(spectrum) ** 2
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/sampling_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    band_power = power[:, :, mask]
    mean_power = band_power.mean(dim=2)
    log_power = torch.log(mean_power + 1e-10)
    return log_power.cpu().numpy()


def main():
    print("Loading Sleep-EDF test set...")
    sleep_root = get_dataset_root('sleep_edf')
    test_ds = SleepEDFDataset(sleep_root, 'TestFold', n_channels=2, sampling_rate=200)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    bands = ['delta', 'theta', 'alpha', 'beta', 'beta_upper', 'theta_bw2', 'theta_bw3', 'theta_bw6', 'beta_bw2', 'beta_bw4']
    band_powers = {band: [] for band in bands}
    all_labels = []

    print("Extracting band power...")
    for x, y in tqdm(test_loader, desc="Processing"):
        for band in bands:
            power = extract_band_power(x, sampling_rate=200, band=band)
            band_powers[band].append(power)
        all_labels.append(y.numpy())

    for band in bands:
        band_powers[band] = np.concatenate(band_powers[band], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pairs = list(combinations(range(5), 2))

    pairwise_auc = {}
    for i, j in pairs:
        mask = (all_labels == i) | (all_labels == j)
        y_bin = (all_labels[mask] == i).astype(int)
        pair_key = f"{stage_names[i]}_vs_{stage_names[j]}"
        pairwise_auc[pair_key] = {}

        for band in bands:
            X = band_powers[band][mask]
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
    print("Done.")


if __name__ == '__main__':
    main()