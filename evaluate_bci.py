"""EEG-DINO Linear Probing on BCI Competition IV 2a/2b.

Evaluation modes:
  within  — Within-subject: per-subject T→E probe
  loso    — Cross-subject LOSO: train probe on 8 subjects' T, test on held-out E

Usage:
  python evaluate_bci.py --checkpoint best_model.pth --bci 2a --mode within
  python evaluate_bci.py --checkpoint random --bci 2a --mode within
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
        self.n_channels = n_channels
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        if x.shape[1] < self.n_channels:
            ci = channel_indices[0].clamp(0, self.n_channels - 1)
            full = torch.zeros(x.shape[0], self.n_channels, x.shape[2],
                               device=x.device, dtype=x.dtype)
            full[:, ci, :] = x
            x = full
        tokens, _ = self.tfe(x)
        tokens = self.dpe(tokens, channel_indices)
        cls, _ = self.transformer(tokens)
        return cls


def get_mi_offset(bci_type, trial_duration):
    """Calculate MI extraction offset based on official BCI Competition IV timing.
    
    Official 2a timing (desc_2a.pdf):
      t=0s: Event 768 (fixation cross)
      t=2s: Cue onset (arrow) for 1.25s
      t=6s: Fixation disappears (end of MI)
      MI period: [2.0s, 6.0s] = 4s
    
    Official 2b timing (desc_2b.pdf):
      Screening (01T, 02T): Similar to 2a
      Feedback (03T-05E): t=3s cue, t=7.5s end
      MI period: [3.0s, 7.0s] or [3.0s, 7.5s]
    
    For 6s extraction matching pretraining epoch_duration=6:
      Extract window that includes trial context (fixation+cue+MI)
      to match pretraining's sliding windows over mixed content.
    """
    if bci_type == '2a':
        if trial_duration <= 4.0:
            return 2.0
        elif trial_duration <= 5.0:
            return 1.5
        else:  
            return 0.5
    else:  # 2b
        if trial_duration <= 4.0:
            return 3.0
        elif trial_duration <= 5.0:
            return 2.5
        else:  
            return 1.5


class BCITrialDataset(Dataset):
    def __init__(self, gdf_path, n_channels, sampling_rate, trial_duration,
                 mat_path, mi_offset):
        import mne
        import scipy.io
        mne.set_log_level('WARNING')

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate

        trial_samples = int(trial_duration * sampling_rate)
        offset_samples = int(mi_offset * sampling_rate)

        raw = mne.io.read_raw_gdf(gdf_path, preload=True)
        eeg_idx = [i for i, ch in enumerate(raw.ch_names) if 'EOG' not in ch]
        raw.pick(eeg_idx)
        if raw.info['sfreq'] != sampling_rate:
            raw.resample(sampling_rate)

        data = torch.from_numpy(raw.get_data().copy()).float()
        events, event_id = mne.events_from_annotations(raw)

        mat = scipy.io.loadmat(mat_path)
        labels = mat['classlabel'].flatten().astype(int) - 1
        onset_code = event_id.get('768')
        onsets = [ev[0] for ev in events if ev[2] == onset_code]
        trials = list(zip(onsets, labels[:len(onsets)]))

        self.data, self.labels = [], []
        for onset, label in trials:
            start = onset + offset_samples
            end = start + trial_samples
            if end > data.shape[1]:
                continue

            trial = data[:, start:end]
            if torch.isnan(trial).any():
                continue
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


def load_subject_2a(root, sid, nc, sr, td, mi_off):
    t_gdf = os.path.join(root, f'{sid}T.gdf')
    t_mat = os.path.join(root, f'{sid}T.mat')
    e_gdf = os.path.join(root, f'{sid}E.gdf')
    e_mat = os.path.join(root, f'{sid}E.mat')
    train_ds = BCITrialDataset(t_gdf, nc, sr, td, t_mat, mi_off)
    test_ds = BCITrialDataset(e_gdf, nc, sr, td, e_mat, mi_off)
    return train_ds, test_ds


def load_subject_2b(root, sid, nc, sr, td, mi_off):
    s_num = int(sid[1:])
    train_data, train_labels = [], []
    for sess in (1, 2, 3):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}T.gdf')
        mat = os.path.join(root, f'B{s_num:02d}{sess:02d}T.mat')
        if os.path.exists(gdf) and os.path.exists(mat):
            ds = BCITrialDataset(gdf, nc, sr, td, mat, mi_off)
            train_data.extend(ds.data)
            train_labels.append(ds.labels)

    test_data, test_labels = [], []
    for sess in (4, 5):
        gdf = os.path.join(root, f'B{s_num:02d}{sess:02d}E.gdf')
        mat = os.path.join(root, f'B{s_num:02d}{sess:02d}E.mat')
        if os.path.exists(gdf) and os.path.exists(mat):
            ds = BCITrialDataset(gdf, nc, sr, td, mat, mi_off)
            test_data.extend(ds.data)
            test_labels.append(ds.labels)

    train_ds = _ListDataset(train_data, torch.cat(train_labels))
    test_ds = _ListDataset(test_data, torch.cat(test_labels))
    return train_ds, test_ds


def eval_within_subject(backbone, bci, cfg, args, mi_off):
    nc, sr, td = cfg['n_channels'], cfg['sampling_rate'], args.trial_duration
    device = next(backbone.parameters()).device
    subjects = [f'A{s:02d}' for s in range(1, 10)] if bci == '2a' else [f'B{s:02d}' for s in range(1, 10)]

    results = []
    for sid in subjects:
        print(f"\n--- Within-subject: {sid} ---")
        if bci == '2a':
            train_ds, test_ds = load_subject_2a(BCI_2A_PATH, sid, nc, sr, td, mi_off)
        else:
            train_ds, test_ds = load_subject_2b(BCI_2B_PATH, sid, nc, sr, td, mi_off)

        print(f"  Train: {len(train_ds)} trials (T) | Test: {len(test_ds)} trials (E)")
        acc, f1, kappa = train_and_eval(
            backbone, train_ds, test_ds, cfg['embed_dim'], device,
            args.probe_epochs, args.probe_lr)
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
        results.append({'subject': sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    return results


def eval_loso(backbone, bci, cfg, args, mi_off):
    nc, sr, td = cfg['n_channels'], cfg['sampling_rate'], args.trial_duration
    device = next(backbone.parameters()).device
    subjects = [f'A{s:02d}' for s in range(1, 10)] if bci == '2a' else [f'B{s:02d}' for s in range(1, 10)]

    print(f"Loading {len(subjects)} subjects (T=probe train, E=probe test)...")
    all_train, all_test = {}, {}
    for sid in tqdm(subjects, desc="Subjects"):
        if bci == '2a':
            tr, te = load_subject_2a(BCI_2A_PATH, sid, nc, sr, td, mi_off)
        else:
            tr, te = load_subject_2b(BCI_2B_PATH, sid, nc, sr, td, mi_off)
        all_train[sid] = tr
        all_test[sid] = te
        print(f"  {sid}: T={len(tr)} trials, E={len(te)} trials")

    results = []
    for test_sid in subjects:
        print(f"\n--- LOSO: held-out {test_sid} ---")
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
        te = all_test[test_sid]
        test_ds = te if isinstance(te, _ListDataset) else _ListDataset(te.data, te.labels)

        print(f"  Train: {len(train_ds)} trials ({len(subjects)-1} subj T) | "
              f"Test: {len(test_ds)} trials ({test_sid} E)")

        acc, f1, kappa = train_and_eval(
            backbone, train_ds, test_ds, cfg['embed_dim'], device,
            args.probe_epochs, args.probe_lr)
        print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")
        results.append({'subject': test_sid, 'acc': acc, 'f1': f1, 'kappa': kappa})

    return results


def main():
    p = argparse.ArgumentParser(description='EEG-DINO BCI Evaluation')
    p.add_argument('--checkpoint', required=True,
                   help='Path to checkpoint, or "random" for untrained baseline')
    p.add_argument('--bci', required=True, choices=['2a', '2b'])
    p.add_argument('--preset', default='bci_2a', choices=['tiny', 'bci_2a', 'bci_2b'],
                   help='Model config matching pretraining preset')
    p.add_argument('--mode', default='within', choices=['within', 'loso'],
                   help='within = per-subject T→E, loso = cross-subject')
    p.add_argument('--probe_epochs', type=int, default=100)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--trial_duration', type=float, default=None,
                   help='Trial duration in seconds (default: auto-detect from preset)')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = PRESETS[args.preset]

    if args.trial_duration is None:
        args.trial_duration = cfg.get('epoch_duration', 6.0)
        print(f"Auto-detected trial_duration={args.trial_duration}s from preset "
              f"(matches pretraining epoch_duration)")

    mi_offset = get_mi_offset(args.bci, args.trial_duration)

    backbone = FrozenBackbone(
        cfg['n_channels'], cfg['sampling_rate'],
        cfg['embed_dim'], cfg['n_layers'], cfg['n_heads'], cfg['mlp_dim']
    ).to(device)

    if args.checkpoint == 'random':
        print("Using RANDOM (untrained) backbone as baseline")
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        bb_sd = {}
        for k, v in ckpt['student'].items():
            for prefix in ('tfe.', 'dpe.', 'transformer.'):
                if k.startswith(prefix):
                    bb_sd[k] = v
                    break
        backbone.load_state_dict(bb_sd, strict=True)
        print(f"Loaded backbone from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # ========================================
    # ADD DIAGNOSTIC PRINTS HERE
    # ========================================
    
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS")
    print("="*60)
    
    # Check 1: Model architecture
    print(f"\n1. Model Architecture:")
    print(f"   Backbone expects: {backbone.n_channels} channels")
    print(f"   Sampling rate: {cfg['sampling_rate']} Hz")
    print(f"   Embed dim: {cfg['embed_dim']}")
    
    # Check 2: Load one trial to verify data format
    print(f"\n2. Loading Test Trial:")
    test_subject = 'A01' if args.bci == '2a' else 'B01'
    if args.bci == '2a':
        test_gdf = os.path.join(BCI_2A_PATH, f'{test_subject}E.gdf')
        test_mat = os.path.join(BCI_2A_PATH, f'{test_subject}E.mat')
    else:
        test_gdf = os.path.join(BCI_2B_PATH, f'B0104E.gdf')
        test_mat = os.path.join(BCI_2B_PATH, f'B0104E.mat')
    
    test_ds = BCITrialDataset(test_gdf, cfg['n_channels'], cfg['sampling_rate'], 
                              args.trial_duration, test_mat, mi_offset)
    
    print(f"   Dataset: {len(test_ds)} trials")
    print(f"   Trial duration: {args.trial_duration}s")
    print(f"   MI offset: {mi_offset}s")
    print(f"   Extraction window: [{mi_offset}, {mi_offset + args.trial_duration}]s from event 768")
    
    # Check 3: Inspect one trial
    x_sample, y_sample = test_ds[0]
    print(f"\n3. Trial Data Shape:")
    print(f"   Input shape: {x_sample.shape}  (expected: [22, {int(args.trial_duration * cfg['sampling_rate'])}])")
    print(f"   Channels provided: {x_sample.shape[0]}")
    print(f"   Time samples: {x_sample.shape[1]} (= {x_sample.shape[1] / cfg['sampling_rate']:.1f}s)")
    
    # Check 4: Forward pass through backbone
    print(f"\n4. Forward Pass Test:")
    with torch.no_grad():
        x_batch = x_sample.unsqueeze(0).to(device)
        
        # Check TFE output
        tokens, raw_features = backbone.tfe(x_batch)
        print(f"   TFE input: {x_batch.shape}")
        print(f"   TFE tokens: {tokens.shape}  (expected: [1, {int(args.trial_duration)}, {cfg['embed_dim']}])")
        print(f"   → Got {tokens.shape[1]} tokens (should be {int(args.trial_duration)} for {args.trial_duration}s)")
        
        # Check full backbone output
        cls_output = backbone(x_batch)
        print(f"   Backbone CLS output: {cls_output.shape}  (expected: [1, {cfg['embed_dim']}])")
    
    # Check 5: Summary
    print(f"\n5. Validation Summary:")
    issues = []
    
    if x_sample.shape[0] != cfg['n_channels']:
        issues.append(f"❌ Channel mismatch: got {x_sample.shape[0]}, expected {cfg['n_channels']}")
    else:
        print(f"   ✓ Channels: {x_sample.shape[0]} = {cfg['n_channels']}")
    
    expected_samples = int(args.trial_duration * cfg['sampling_rate'])
    if x_sample.shape[1] != expected_samples:
        issues.append(f"❌ Duration mismatch: got {x_sample.shape[1]} samples, expected {expected_samples}")
    else:
        print(f"   ✓ Duration: {x_sample.shape[1]} samples = {args.trial_duration}s @ {cfg['sampling_rate']}Hz")
    
    if tokens.shape[1] != int(args.trial_duration):
        issues.append(f"❌ Token count mismatch: got {tokens.shape[1]}, expected {int(args.trial_duration)}")
    else:
        print(f"   ✓ Tokens: {tokens.shape[1]} = {int(args.trial_duration)}s @ 1 token/sec")
    
    if issues:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✓ All checks passed!")
    
    print("="*60 + "\n")
    
    # ========================================
    # END DIAGNOSTIC PRINTS
    # ========================================
    if args.mode == 'within':
        results = eval_within_subject(backbone, args.bci, cfg, args, mi_offset)
    else:
        results = eval_loso(backbone, args.bci, cfg, args, mi_offset)

    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    kappas = [r['kappa'] for r in results]
    print(f"\n{'='*50}")
    print(f"BCI-IV {args.bci} -- {mode_label} -- {len(results)} subjects")
    print(f"  Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1 Macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Kappa    : {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['subject']}: acc={r['acc']:.4f} f1={r['f1']:.4f} kappa={r['kappa']:.4f}")


if __name__ == '__main__':
    main()