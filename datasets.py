"""Dataset helpers for EEG-DINO."""

import os
from glob import glob

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_env_file(path='.env'):
    """Load KEY=VALUE pairs from a local .env file without extra dependencies."""
    if not os.path.exists(path):
        return

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def get_data_roots():
    """Return dataset roots, overridable via environment variables."""
    _load_env_file()
    return {
        'sleep_edf': os.environ.get(
            'SLEEP_EDF_PATH',
            '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/'
        ),
        'bci_2a': os.environ.get(
            'BCI_2A_PATH',
            '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2a_gdf/'
        ),
        'bci_2b': os.environ.get(
            'BCI_2B_PATH',
            '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2b_gdf/'
        ),
    }


def get_dataset_root(name):
    root = get_data_roots()[name]
    if not root:
        raise ValueError(f"Dataset root for '{name}' is empty. Set it in .env or environment variables.")
    return root


class SleepEDFDataset(Dataset):
    def __init__(self, root, fold='TrainFold', n_channels=2, sampling_rate=200):
        import torch.nn.functional as F

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_len = sampling_rate * 30

        pt_files = sorted(glob(os.path.join(root, fold, '**/*.pt'), recursive=True))
        print(f"[SleepEDF/{fold}] {len(pt_files)} files, loading...")

        self.data, self.labels = [], []
        for fp in tqdm(pt_files, desc=f"SleepEDF/{fold}"):
            try:
                label = int(os.path.basename(os.path.dirname(fp)))
                trial = torch.load(fp, map_location='cpu', weights_only=True).float()

                trial = F.interpolate(
                    trial.unsqueeze(0), size=self.epoch_len,
                    mode='linear', align_corners=False
                ).squeeze(0)

                real = trial[:min(trial.shape[0], n_channels)]
                trial = (trial - real.mean()) / (real.std() + 1e-8)

                if trial.shape[0] >= n_channels:
                    trial = trial[:n_channels]
                else:
                    pad = torch.zeros(n_channels - trial.shape[0], trial.shape[1])
                    trial = torch.cat([trial, pad], dim=0)

                self.data.append(trial)
                self.labels.append(label)
            except Exception as exc:
                print(f"  Skip {fp}: {exc}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class BCITrialBasedDataset(Dataset):
    """BCI dataset extracting clean motor imagery periods using event markers."""

    def __init__(self, gdf_paths, n_channels, sampling_rate, epoch_duration,
                 mi_offset=2.0):
        import mne

        mne.set_log_level('WARNING')

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.samples_per_epoch = int(epoch_duration * sampling_rate)
        self.trials = []

        for gdf_path in tqdm(gdf_paths, desc='Loading BCI trials'):
            try:
                mat_path = gdf_path.replace('.gdf', '.mat')
                if not os.path.exists(mat_path):
                    print(f"  Warning: No .mat file for {os.path.basename(gdf_path)}")
                    continue

                raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
                raw.pick_types(eeg=True, exclude=[])

                if len(raw.ch_names) > n_channels:
                    raw.pick(raw.ch_names[:n_channels])
                if raw.info['sfreq'] != sampling_rate:
                    raw.resample(sampling_rate)

                signal = raw.get_data()
                mat = scipy.io.loadmat(mat_path)
                labels = mat['classlabel'].flatten().astype(int) - 1

                events, event_id = mne.events_from_annotations(raw, verbose=False)
                onset_code = event_id.get('768')
                if onset_code is None:
                    print(f"  Warning: Event 768 not found in {os.path.basename(gdf_path)}")
                    continue

                trial_starts = [ev[0] for ev in events if ev[2] == onset_code]
                if len(trial_starts) != len(labels):
                    print(f"  Warning: Trial count mismatch in {os.path.basename(gdf_path)}")

                mi_start_offset = int(mi_offset * sampling_rate)
                for trial_start in trial_starts:
                    mi_start = trial_start + mi_start_offset
                    mi_end = mi_start + self.samples_per_epoch

                    if mi_end > signal.shape[1] or mi_start < 0:
                        continue

                    trial = signal[:, mi_start:mi_end]
                    if np.isnan(trial).any():
                        continue

                    trial = (trial - trial.mean(axis=1, keepdims=True)) / (
                        trial.std(axis=1, keepdims=True) + 1e-8
                    )
                    self.trials.append(torch.from_numpy(trial).float())

            except Exception as exc:
                print(f"  Error loading {os.path.basename(gdf_path)}: {exc}")
                continue

        print(f"  → {len(self.trials)} clean MI trials ({n_channels} ch, {epoch_duration}s)")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]
        n_ch = trial.shape[0]
        if n_ch == self.n_channels:
            return trial
        if n_ch > self.n_channels:
            return trial[:self.n_channels]
        pad = torch.zeros(self.n_channels - n_ch, trial.shape[1])
        return torch.cat([trial, pad], dim=0)


class UnlabeledWrapper(Dataset):
    """Wrap a dataset to return signals only."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item[0] if isinstance(item, (list, tuple)) else item
