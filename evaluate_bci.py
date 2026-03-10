"""EEG-DINO Linear Probing on BCI Competition IV 2a/2b.

LOSO (Leave-One-Subject-Out) — cross-subject generalization:
  Train probe on 8 subjects' T sessions, test on held-out subject's E session.
  No data leakage: backbone pretrained on T sessions only, E sessions are unseen.

  2a: 22 EEG channels, 250 Hz, 4 classes (left hand, right hand, feet, tongue)
  2b:  3 EEG channels, 250 Hz, 2 classes (left hand, right hand)

Usage:
  python evaluate_bci.py --checkpoint checkpoints/bci_2a_theta/best_model.pth --bci 2a
  python evaluate_bci.py --checkpoint checkpoints/bci_2b_theta/best_model.pth --bci 2b --preset bci_2b
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
from train import PRESETS, BCI_2A_PATH, BCI_2B_PATH


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
    """Labeled motor imagery trials from a single BCI GDF session.

    Extracts trial epochs after cue onset. NO tiling — returns the raw
    trial duration (e.g. 4s at 250Hz = 1000 samples → 4 TFE tokens).
    Labels always from .mat file (2a GDF has no class-label events).
    Keeps all EEG channels, excludes EOG.
    """

    def __init__(self, gdf_path, n_channels, sampling_rate, trial_duration,
                 mat_path):
        import mne
        import scipy.io
        mne.set_log_level('WARNING')

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate

        trial_samples = int(trial_duration * sampling_rate)
        cue_offset = int(0.5 * sampling_rate)

        raw = mne.io.read_raw_gdf(gdf_path, preload=True)
        eeg_idx = [i for i, ch in enumerate(raw.ch_names) if 'EOG' not in ch]
        raw.pick(eeg_idx)
        if raw.info['sfreq'] != sampling_rate:
            raw.resample(sampling_rate)

        data = torch.from_numpy(raw.get_data().copy()).float()
        events, event_id = mne.events_from_annotations(raw)

        # Labels always from .mat
        mat = scipy.io.loadmat(mat_path)
        labels = mat['classlabel'].flatten().astype(int) - 1
        onset_code = event_id.get('768')
        onsets = [ev[0] for ev in events if ev[2] == onset_code]
        trials = list(zip(onsets, labels[:len(onsets)]))

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
# Feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ─────────────────────────────────────────────────────────────
# Linear probe with validation + early stopping
# ─────────────────────────────────────────────────────────────

def train_and_eval(backbone, train_ds, test_ds, embed_dim, device,
                   epochs=100, lr=1e-3, batch_size=64):
    tr_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0)

    tr_f, tr_y = extract_features(backbone, tr_loader, device)
    te_f, te_y = extract_features(backbone, te_loader, device)

    # 80/20 train/val split for early stopping
    n = len(tr_f)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    n_classes = int(tr_y.max().item()) + 1
    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0, None
    patience, wait = 15, 0

    for ep in range(epochs):
        probe.train()
        idx = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(idx), batch_size):
            b = idx[i:i+batch_size]
            loss = criterion(probe(tr_f[b].to(device)), tr_y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(tr_f[val_idx].to(device)).argmax(1).cpu()
            val_acc = (val_preds == tr_y[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        preds = probe(te_f.to(device)).argmax(1).cpu().numpy()
    gt = te_y.numpy()
    return (accuracy_score(gt, preds),
            f1_score(gt, preds, average='macro'),
            cohen_kappa_score(gt, preds))


# ─────────────────────────────────────────────────────────────
# Subject loading — separate T (train) and E (test) sessions
# ─────────────────────────────────────────────────────────────

def load_subject_2a(root, sid, nc, sr, td):
    """Return (train_ds from T session, test_ds from E session)."""
    t_gdf = os.path.join(root, f'{sid}T.gdf')
    t_mat = os.path.join(root, f'{sid}T.mat')
    e_gdf = os.path.join(root, f'{sid}E.gdf')
    e_mat = os.path.join(root, f'{sid}E.mat')
    train_ds = BCITrialDataset(t_gdf, nc, sr, td, t_mat)
    test_ds = BCITrialDataset(e_gdf, nc, sr, td, e_mat)
    return train_ds, test_ds


def load_subject_2b(root, sid, nc, sr, td):
    """Return (train_ds from T sessions 1-3, test_ds from E sessions 4-5)."""
    s_num = int(sid[1:])

    train_data, train_labels = [], []
    for sess in (1, 2, 3):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}T.gdf')
        mat = os.path.join(root, f'B{s_num:02d}{sess:02d}T.mat')
        if os.path.exists(gdf) and os.path.exists(mat):
            ds = BCITrialDataset(gdf, nc, sr, td, mat)
            train_data.extend(ds.data)
            train_labels.append(ds.labels)

    test_data, test_labels = [], []
    for sess in (4, 5):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}E.gdf')
        mat = os.path.join(root, f'B{s_num:02d}{sess:02d}E.mat')
        if os.path.exists(gdf) and os.path.exists(mat):
            ds = BCITrialDataset(gdf, nc, sr, td, mat)
            test_data.extend(ds.data)
            test_labels.append(ds.labels)

    train_ds = _ListDataset(train_data, torch.cat(train_labels))
    test_ds = _ListDataset(test_data, torch.cat(test_labels))
    return train_ds, test_ds


# ─────────────────────────────────────────────────────────────
# LOSO: train probe on 8 subj T sessions, test on held-out E session
# ─────────────────────────────────────────────────────────────

def eval_loso(backbone, bci, cfg, args):
    nc, sr, td = cfg['n_channels'], cfg['sampling_rate'], args.trial_duration
    device = next(backbone.parameters()).device

    if bci == '2a':
        subjects = [f'A{s:02d}' for s in range(1, 10)]
    else:
        subjects = [f'B{s:02d}' for s in range(1, 10)]

    # Load T and E sessions separately per subject
    print(f"Loading {len(subjects)} subjects (T=probe train, E=probe test)...")
    all_train, all_test = {}, {}
    for sid in tqdm(subjects, desc="Subjects"):
        if bci == '2a':
            tr, te = load_subject_2a(BCI_2A_PATH, sid, nc, sr, td)
        else:
            tr, te = load_subject_2b(BCI_2B_PATH, sid, nc, sr, td)
        all_train[sid] = tr
        all_test[sid] = te
        print(f"  {sid}: T={len(tr)} trials, E={len(te)} trials")

    # LOSO loop
    results = []
    for test_sid in subjects:
        print(f"\n--- LOSO: held-out {test_sid} ---")

        # Probe train: T sessions from other 8 subjects
        train_data, train_labels = [], []
        for sid in subjects:
            if sid == test_sid:
                continue
            ds = all_train[sid]
            if isinstance(ds, _ListDataset):
                train_data.extend(ds.data)
                train_labels.append(ds.labels)
            else:
                train_data.extend(ds.data)
                train_labels.append(ds.labels)

        train_ds = _ListDataset(train_data, torch.cat(train_labels))

        # Probe test: E session from held-out subject
        te = all_test[test_sid]
        if isinstance(te, _ListDataset):
            test_ds = te
        else:
            test_ds = _ListDataset(te.data, te.labels)

        print(f"  Train: {len(train_ds)} trials ({len(subjects)-1} subj T) | "
              f"Test: {len(test_ds)} trials ({test_sid} E)")

        acc, f1, kappa = train_and_eval(
            backbone, train_ds, test_ds, cfg['embed_dim'], device,
            args.probe_epochs, args.probe_lr)
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
        results.append({'subject': test_sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='EEG-DINO BCI Evaluation (LOSO)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--bci', required=True, choices=['2a', '2b'])
    p.add_argument('--preset', default='bci_2a', choices=['tiny', 'bci_2a', 'bci_2b'],
                   help='Model config matching pretraining preset')
    p.add_argument('--probe_epochs', type=int, default=100)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--trial_duration', type=float, default=4.0)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = PRESETS[args.preset]

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
    print(f"LOSO | BCI-IV {args.bci} | preset: {args.preset}")
    print(f"  n_channels={cfg['n_channels']}, sampling_rate={cfg['sampling_rate']}Hz, "
          f"trial_duration={args.trial_duration}s\n")

    results = eval_loso(backbone, args.bci, cfg, args)

    # Summary
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    kappas = [r['kappa'] for r in results]
    print(f"\n{'='*50}")
    print(f"BCI-IV {args.bci} -- LOSO -- {len(results)} subjects")
    print(f"  Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1 Macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Kappa    : {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['subject']}: acc={r['acc']:.4f} f1={r['f1']:.4f} kappa={r['kappa']:.4f}")


if __name__ == '__main__':
    main()
