"""EEG-DINO Linear Probing on BCI Competition IV 2a/2b.

Leave-One-Subject-Out (LOSO) evaluation for cross-subject generalization:
  - For each held-out subject: train probe on all other subjects, test on held-out
  - Reports per-subject and mean +/- std across subjects

Also supports within-subject evaluation (--eval_mode within):
  - Train probe on subject's T session, test on E session

Usage:
  python evaluate_bci.py --checkpoint checkpoints/bci_all_theta/best_model.pth --bci 2a
  python evaluate_bci.py --checkpoint checkpoints/bci_all_theta/best_model.pth --bci 2b
  python evaluate_bci.py --checkpoint checkpoints/bci_all_theta/best_model.pth --bci 2a --eval_mode within
"""

import os, argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

from eeg_dino_model import EEGTransformer
from tfe_module import TimeFrequencyEmbedding
from dpe_module import DecoupledPositionalEmbedding
from train import CONFIG, BCI_2A_PATH, BCI_2B_PATH


# Motor cortex channels
CHANNELS_2A = ['C3', 'C4']
CHANNELS_2B = ['C3', 'C4']


class FrozenBackbone(nn.Module):
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
        tokens, _ = self.tfe(x)
        tokens = self.dpe(tokens, channel_indices)
        cls, _ = self.transformer(tokens)
        return cls


class BCITrialDataset(Dataset):
    """Labeled motor imagery trials from a single BCI GDF session."""

    def __init__(self, gdf_path, n_channels=2, sampling_rate=200,
                 channels=None, trial_duration=4.0,
                 label_source='events', mat_path=None):
        import mne
        mne.set_log_level('WARNING')

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_len = sampling_rate * 30

        trial_samples = int(trial_duration * sampling_rate)
        cue_offset = int(0.5 * sampling_rate)

        raw = mne.io.read_raw_gdf(gdf_path, preload=True)

        if channels:
            pick_idx = []
            for target in channels:
                for i, ch in enumerate(raw.ch_names):
                    if target in ch:
                        pick_idx.append(i)
                        break
            raw.pick(pick_idx)
        else:
            eeg_idx = [i for i, ch in enumerate(raw.ch_names) if 'EOG' not in ch]
            raw.pick(eeg_idx)

        if raw.info['sfreq'] != sampling_rate:
            raw.resample(sampling_rate)

        data = torch.from_numpy(raw.get_data().copy()).float()
        events, event_id = mne.events_from_annotations(raw)

        trials = []
        if label_source == 'events':
            class_codes = {v: int(k) - 769
                           for k, v in event_id.items()
                           if k in ('769', '770', '771', '772')}
            for ev in events:
                if ev[2] in class_codes:
                    trials.append((ev[0], class_codes[ev[2]]))
        elif label_source == 'mat':
            import scipy.io
            mat = scipy.io.loadmat(mat_path)
            labels = mat['classlabel'].flatten().astype(int) - 1
            onset_code = event_id.get('768')
            onsets = [ev[0] for ev in events if ev[2] == onset_code]
            trials = list(zip(onsets, labels))

        self.data, self.labels = [], []
        for onset, label in trials:
            start = onset + cue_offset
            end = start + trial_samples
            if end > data.shape[1]:
                continue

            trial = data[:, start:end]
            if trial.shape[0] > n_channels:
                trial = trial[:n_channels]
            elif trial.shape[0] < n_channels:
                pad = torch.zeros(n_channels - trial.shape[0], trial_samples)
                trial = torch.cat([trial, pad], dim=0)

            n_rep = (self.epoch_len + trial_samples - 1) // trial_samples
            trial = trial.repeat(1, n_rep)[:, :self.epoch_len]
            trial = (trial - trial.mean()) / (trial.std() + 1e-8)

            self.data.append(trial)
            self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class _ListDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────
# Subject loading
# ─────────────────────────────────────────────────────────────

def load_subject_data_2a(root, sid, nc, sr, channels, td):
    """Load all labeled trials for one 2a subject (T + E sessions)."""
    t_gdf = os.path.join(root, f'{sid}T.gdf')
    e_gdf = os.path.join(root, f'{sid}E.gdf')
    e_mat = os.path.join(root, f'{sid}E.mat')

    datasets = []
    if os.path.exists(t_gdf):
        datasets.append(BCITrialDataset(t_gdf, nc, sr, channels, td, 'events'))
    if os.path.exists(e_gdf) and os.path.exists(e_mat):
        datasets.append(BCITrialDataset(e_gdf, nc, sr, channels, td, 'mat', e_mat))

    data, labels = [], []
    for ds in datasets:
        data.extend(ds.data)
        labels.append(ds.labels)
    return data, torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)


def load_subject_data_2b(root, sid, nc, sr, channels, td):
    """Load all labeled trials for one 2b subject (sessions 1-5)."""
    s_num = int(sid[1:])
    data, labels = [], []

    for sess in (1, 2, 3):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}T.gdf')
        if os.path.exists(gdf):
            ds = BCITrialDataset(gdf, nc, sr, channels, td, 'events')
            data.extend(ds.data)
            labels.append(ds.labels)

    for sess in (4, 5):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}E.gdf')
        mat = os.path.join(root, f'B{s_num:02d}{sess:02d}E.mat')
        if os.path.exists(gdf) and os.path.exists(mat):
            ds = BCITrialDataset(gdf, nc, sr, channels, td, 'mat', mat)
            data.extend(ds.data)
            labels.append(ds.labels)

    return data, torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# Feature extraction & linear probe
# ─────────────────────────────────────────────────────────────

def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_and_eval(backbone, train_ds, test_ds, embed_dim, device,
                   epochs=100, lr=1e-3, batch_size=64):
    tr_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0)

    tr_f, tr_y = extract_features(backbone, tr_loader, device)
    te_f, te_y = extract_features(backbone, te_loader, device)

    n_classes = int(tr_y.max().item()) + 1
    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        probe.train()
        idx = torch.randperm(len(tr_f))
        for i in range(0, len(tr_f), batch_size):
            b = idx[i:i+batch_size]
            loss = criterion(probe(tr_f[b].to(device)), tr_y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(te_f.to(device)).argmax(1).cpu().numpy()
    gt = te_y.numpy()
    return (accuracy_score(gt, preds),
            f1_score(gt, preds, average='macro'),
            cohen_kappa_score(gt, preds))


# ─────────────────────────────────────────────────────────────
# LOSO evaluation
# ─────────────────────────────────────────────────────────────

def eval_loso(backbone, bci, cfg, args):
    """Leave-One-Subject-Out: train probe on 8 subjects, test on 1."""
    nc, sr, td = cfg['n_channels'], cfg['sampling_rate'], args.trial_duration
    device = next(backbone.parameters()).device

    if bci == '2a':
        channels = CHANNELS_2A
        subjects = [f'A{s:02d}' for s in range(1, 10)]
        load_fn = lambda sid: load_subject_data_2a(BCI_2A_PATH, sid, nc, sr, channels, td)
    else:
        channels = CHANNELS_2B
        subjects = [f'B{s:02d}' for s in range(1, 10)]
        load_fn = lambda sid: load_subject_data_2b(BCI_2B_PATH, sid, nc, sr, channels, td)

    print(f"Loading all {len(subjects)} subjects...")
    all_data = {}
    for sid in tqdm(subjects, desc="Subjects"):
        data, labels = load_fn(sid)
        all_data[sid] = (data, labels)
        print(f"  {sid}: {len(data)} trials")

    results = []
    for test_sid in subjects:
        print(f"\n--- LOSO: held-out {test_sid} ---")

        train_data, train_labels = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            train_data.extend(all_data[sid][0])
            train_labels.append(all_data[sid][1])

        train_ds = _ListDataset(train_data, torch.cat(train_labels))
        test_ds = _ListDataset(*all_data[test_sid])

        print(f"  Train: {len(train_ds)} trials ({len(subjects)-1} subj) | "
              f"Test: {len(test_ds)} trials (subj {test_sid})")

        acc, f1, kappa = train_and_eval(
            backbone, train_ds, test_ds, cfg['embed_dim'], device,
            args.probe_epochs, args.probe_lr)
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
        results.append({'subject': test_sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    return results


# ─────────────────────────────────────────────────────────────
# Within-subject evaluation
# ─────────────────────────────────────────────────────────────

def eval_within(backbone, bci, cfg, args):
    """Within-subject: train probe on T session(s), test on E session(s)."""
    nc, sr, td = cfg['n_channels'], cfg['sampling_rate'], args.trial_duration
    device = next(backbone.parameters()).device

    results = []

    if bci == '2a':
        channels = CHANNELS_2A
        for s in range(1, 10):
            sid = f'A{s:02d}'
            t_gdf = os.path.join(BCI_2A_PATH, f'{sid}T.gdf')
            e_gdf = os.path.join(BCI_2A_PATH, f'{sid}E.gdf')
            e_mat = os.path.join(BCI_2A_PATH, f'{sid}E.mat')
            if not (os.path.exists(t_gdf) and os.path.exists(e_gdf)):
                continue

            print(f"\n--- Subject {sid} (within) ---")
            train_ds = BCITrialDataset(t_gdf, nc, sr, channels, td, 'events')
            test_ds = BCITrialDataset(e_gdf, nc, sr, channels, td, 'mat', e_mat)
            print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

            acc, f1, kappa = train_and_eval(
                backbone, train_ds, test_ds, cfg['embed_dim'], device,
                args.probe_epochs, args.probe_lr)
            print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
            results.append({'subject': sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    elif bci == '2b':
        channels = CHANNELS_2B
        for s in range(1, 10):
            sid = f'B{s:02d}'
            print(f"\n--- Subject {sid} (within) ---")

            train_data, train_labels = [], []
            for sess in (1, 2, 3):
                gdf = os.path.join(BCI_2B_PATH, f'B{s:02d}{sess:02d}T.gdf')
                if os.path.exists(gdf):
                    ds = BCITrialDataset(gdf, nc, sr, channels, td, 'events')
                    train_data.extend(ds.data)
                    train_labels.append(ds.labels)

            test_data, test_labels = [], []
            for sess in (4, 5):
                gdf = os.path.join(BCI_2B_PATH, f'B{s:02d}{sess:02d}E.gdf')
                mat = os.path.join(BCI_2B_PATH, f'B{s:02d}{sess:02d}E.mat')
                if os.path.exists(gdf) and os.path.exists(mat):
                    ds = BCITrialDataset(gdf, nc, sr, channels, td, 'mat', mat)
                    test_data.extend(ds.data)
                    test_labels.append(ds.labels)

            if not train_data or not test_data:
                continue

            train_ds = _ListDataset(train_data, torch.cat(train_labels))
            test_ds = _ListDataset(test_data, torch.cat(test_labels))
            print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

            acc, f1, kappa = train_and_eval(
                backbone, train_ds, test_ds, cfg['embed_dim'], device,
                args.probe_epochs, args.probe_lr)
            print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
            results.append({'subject': sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='EEG-DINO BCI Evaluation')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bci', required=True, choices=['2a', '2b'])
    p.add_argument('--eval_mode', default='loso', choices=['loso', 'within'],
                   help='loso = cross-subject (default), within = per-subject T->E')
    p.add_argument('--probe_epochs', type=int, default=100)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--trial_duration', type=float, default=4.0)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = CONFIG

    backbone = FrozenBackbone(
        cfg['n_channels'], cfg['sampling_rate'],
        cfg['embed_dim'], cfg['n_layers'], cfg['n_heads'], cfg['mlp_dim']
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    bb_sd = {}
    for k, v in ckpt['student'].items():
        for prefix in ('tfe.', 'dpe.', 'transformer.'):
            if k.startswith(prefix):
                bb_sd[k] = v
                break
    backbone.load_state_dict(bb_sd, strict=True)
    print(f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    print(f"Eval mode: {args.eval_mode} | BCI-IV {args.bci}\n")

    if args.eval_mode == 'loso':
        results = eval_loso(backbone, args.bci, cfg, args)
    else:
        results = eval_within(backbone, args.bci, cfg, args)

    # Summary
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    kappas = [r['kappa'] for r in results]
    print(f"\n{'='*50}")
    print(f"BCI-IV {args.bci} — {args.eval_mode.upper()} — {len(results)} subjects")
    print(f"  Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1 Macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Kappa    : {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['subject']}: acc={r['acc']:.4f} f1={r['f1']:.4f} kappa={r['kappa']:.4f}")


if __name__ == '__main__':
    main()
