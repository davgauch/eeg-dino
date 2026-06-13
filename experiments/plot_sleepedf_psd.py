"""Plots the average Power Spectral Density (PSD) per sleep stage for Sleep-EDF.

Usage:
    python plot_sleepedf_psd.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import SleepEDFDataset, get_dataset_root

SFREQ         = 200
FREQ_MAX      = 35
CHANNEL_NAMES = ['Fpz-Cz', 'Pz-Oz']

STAGE_NAMES = {
    0: 'Wake (W)',
    1: 'N1',
    2: 'N2',
    3: 'N3 (Deep)',
    4: 'REM',
}

STAGE_COLORS = {
    0: '#e74c3c',
    1: '#e67e22',
    2: '#2ecc71',
    3: '#2980b9',
    4: '#9b59b6',
}

BANDS = {
    'delta\n(1–4 Hz)':  (1,  4,  '#3498db', 0.13),
    'theta\n(4–8 Hz)':  (4,  8,  '#e74c3c', 0.13),
    'alpha\n(8–13 Hz)': (8,  13, '#2ecc71', 0.10),
    'beta\n(13–30 Hz)': (13, 30, '#f39c12', 0.08),
}

print("Loading Sleep-EDF TrainFold...")
sleep_root = get_dataset_root('sleep_edf')
dataset    = SleepEDFDataset(sleep_root, 'TrainFold', n_channels=2, sampling_rate=SFREQ)
loader     = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

psds_per_stage = defaultdict(list)

print("Computing PSDs...")
for x, y in tqdm(loader, desc="Welch PSD"):
    x_np = x.numpy()  
    y_np = y.numpy()   
    for i in range(x_np.shape[0]):
        label = int(y_np[i])
        freqs, psd = welch(x_np[i], fs=SFREQ, nperseg=512, axis=-1)
        psds_per_stage[label].append(psd)

avg_psd, p20_psd, p80_psd = {}, {}, {}
for stage, psds in psds_per_stage.items():
    arr = np.stack(psds, axis=0)
    avg_psd[stage] = arr.mean(axis=0) 
    p20_psd[stage] = np.percentile(arr, 20, axis=0)
    p80_psd[stage] = np.percentile(arr, 80, axis=0)

freq_mask = freqs <= FREQ_MAX
print(f"Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz | "
      f"Plotting up to {FREQ_MAX} Hz ({freq_mask.sum()} bins)")

# Plotting average PSDs per stage
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
    fig, ax = plt.subplots(figsize=(8, 5))

    ymin_approx, ymax_approx = -60, 10
    for band_name, (low, high, color, alpha) in BANDS.items():
        ax.axvspan(low, high, color=color, alpha=alpha, zorder=0)

    for stage in sorted(avg_psd.keys()):
        psd_ch  = avg_psd[stage][ch_idx]
        p20_ch  = p20_psd[stage][ch_idx]
        p80_ch  = p80_psd[stage][ch_idx]
        color   = STAGE_COLORS[stage]
        label   = STAGE_NAMES.get(stage, str(stage))


        psd_mean  = psd_ch[freq_mask]
        lower_lin = p20_ch[freq_mask]
        upper_lin = p80_ch[freq_mask]

        # Convert to Decibels
        psd_db    = 10 * np.log10(psd_mean  + 1e-12)
        upper     = 10 * np.log10(upper_lin + 1e-12)
        lower     = 10 * np.log10(lower_lin + 1e-12)

        ax.plot(freqs[freq_mask], psd_db, color=color,
                linewidth=2.0, label=label, zorder=2)
        ax.fill_between(freqs[freq_mask], lower, upper,
                        color=color, alpha=0.15, zorder=1)

    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(ymin_approx, ymax_approx)
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.02

    for band_name, (low, high, color, alpha) in BANDS.items():
        ax.text((low + high) / 2, label_y, band_name,
                ha='center', va='top', fontsize=7.5,
                color=color, fontweight='bold', zorder=3)

    ax.set_title(
        f'Sleep-EDF — Average PSD by Sleep Stage\n'
        f'Channel: {ch_name}  (Welch, {SFREQ} Hz)',
        fontsize=11, fontweight='bold'
    )
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=10)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_name = f'psd_sleepedf_{ch_name.replace("-", "_")}.png'
    out_path = os.path.join(results_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()

band_ranges = {
    'delta (1–4)':  (1,  4),
    'theta (4–8)':  (4,  8),
    'alpha (8–13)': (8,  13),
    'beta (13–30)': (13, 30),
}

col_w = 16

for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
    print(f'\n── Mean band power (dB) — channel: {ch_name} ────────')
    header = f"{'Stage':<14}" + ''.join(f"{b:>{col_w}}" for b in band_ranges)
    print(header)
    print('─' * len(header))

    for stage in sorted(avg_psd.keys()):
        psd_ch = avg_psd[stage][ch_idx]   # (n_freqs,) — this channel only
        row    = f"{STAGE_NAMES.get(stage, str(stage)):<14}"
        for band, (lo, hi) in band_ranges.items():
            mask     = (freqs >= lo) & (freqs < hi)
            power_db = 10 * np.log10(psd_ch[mask].mean() + 1e-12)
            row     += f"{power_db:>{col_w}.2f}"
        print(row)