"""EEG-DINO Pre-Training on Sleep-EDF and BCI-IV datasets."""

import os, math, argparse, random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from eeg_dino_model import StudentModel, TeacherModel
from channel_aware_sampling import ChannelAwareSampling
from losses import DINOLoss, PatchLoss

SLEEP_EDF_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/'
BCI_2A_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2a_gdf/'
BCI_2B_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/BCICIV_2b_gdf/'

CONFIG = dict(
    n_channels=2, sampling_rate=200, epoch_duration=30,
    embed_dim=64, n_layers=2, n_heads=4, mlp_dim=128,
    out_dim=4096, head_hidden_dim=256, head_bottleneck_dim=64,
    n_local_views=4, n_masked_views=1, batch_size=64,
    learning_rate=1.25e-4, warmup_epochs=10,
    weight_decay_start=0.04, weight_decay_end=0.40,
    momentum_start=0.996, momentum_end=1.0,
    n_epochs=100, mask_strategy='alpha',
)

CONFIG_BCI_2A = dict(
    n_channels=22, sampling_rate=250, epoch_duration=6,
    embed_dim=64, n_layers=2, n_heads=4, mlp_dim=128,
    out_dim=4096, head_hidden_dim=256, head_bottleneck_dim=64,
    n_local_views=4, n_masked_views=1, batch_size=64,
    learning_rate=1.25e-4, warmup_epochs=10,
    weight_decay_start=0.04, weight_decay_end=0.40,
    momentum_start=0.996, momentum_end=1.0,
    n_epochs=100, mask_strategy='spatiotemporal',
)

CONFIG_BCI_2B = dict(
    n_channels=3, sampling_rate=250, epoch_duration=6,
    embed_dim=64, n_layers=2, n_heads=4, mlp_dim=128,
    out_dim=4096, head_hidden_dim=256, head_bottleneck_dim=64,
    n_local_views=4, n_masked_views=1, batch_size=64,
    learning_rate=1.25e-4, warmup_epochs=10,
    weight_decay_start=0.04, weight_decay_end=0.40,
    momentum_start=0.996, momentum_end=1.0,
    n_epochs=100, mask_strategy='spatiotemporal',
)

PRESETS = {'tiny': CONFIG, 'bci_2a': CONFIG_BCI_2A, 'bci_2b': CONFIG_BCI_2B}


class SleepEDFDataset(Dataset):

    def __init__(self, root, fold='TrainFold', n_channels=2,
                 sampling_rate=200, max_samples=None):
        from glob import glob
        import torch.nn.functional as F

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_len = sampling_rate * 30

        pt_files = sorted(glob(os.path.join(root, fold, '**/*.pt'), recursive=True))
        if max_samples is not None:
            pt_files = pt_files[:max_samples]
        print(f"[SleepEDF/{fold}] {len(pt_files)} files, loading...")

        self.data, self.labels = [], []
        for fp in tqdm(pt_files, desc=f"SleepEDF/{fold}"):
            try:
                label = int(os.path.basename(os.path.dirname(fp)))
                t = torch.load(fp, map_location='cpu', weights_only=True).float()

                t = F.interpolate(t.unsqueeze(0), size=self.epoch_len,
                                  mode='linear', align_corners=False).squeeze(0)

                real = t[:min(t.shape[0], n_channels)]
                t = (t - real.mean()) / (real.std() + 1e-8)
                if t.shape[0] >= n_channels:
                    t = t[:n_channels]
                else:
                    pad = torch.zeros(n_channels - t.shape[0], t.shape[1])
                    t = torch.cat([t, pad], dim=0)

                self.data.append(t)
                self.labels.append(label)
            except Exception as e:
                print(f"  Skip {fp}: {e}")
        print(f"  → {len(self.data)} epochs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class BCIDataset(Dataset):
    """BCI dataset using sliding windows over entire sessions (includes non-MI periods)."""

    def __init__(self, root, dataset='2a', n_channels=22, sampling_rate=250,
                 epoch_duration=6, sessions='T'):
        import mne
        from glob import glob
        mne.set_log_level('WARNING')

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_len = int(sampling_rate * epoch_duration)

        gdf_files = sorted(glob(os.path.join(root, '*.gdf')))
        if sessions != 'all':
            gdf_files = [f for f in gdf_files if f.endswith(f'{sessions}.gdf')]
        print(f"[BCI-{dataset}] {len(gdf_files)} GDF files ({sessions} sessions), loading...")

        self.windows = []
        for fp in tqdm(gdf_files, desc=f"BCI-{dataset}"):
            try:
                raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)
                raw.pick_types(eeg=True, exclude=[])
                if len(raw.ch_names) > n_channels:
                    raw.pick(raw.ch_names[:n_channels])
                if raw.info['sfreq'] != sampling_rate:
                    raw.resample(sampling_rate)

                data = torch.from_numpy(raw.get_data().copy()).float()
                n_times = data.shape[1]

                for i in range(n_times // self.epoch_len):
                    window = data[:, i * self.epoch_len:(i + 1) * self.epoch_len]
                    if torch.isnan(window).any():
                        continue
                    window = (window - window.mean()) / (window.std() + 1e-8)
                    self.windows.append(window)
            except Exception as e:
                print(f"  Skip {fp}: {e}")

        n_ch = self.windows[0].shape[0] if self.windows else '?'
        print(f"  → {len(self.windows)} windows ({n_ch} ch, {epoch_duration}s)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        n_ch = window.shape[0]
        if n_ch == self.n_channels:
            return window
        if n_ch > self.n_channels:
            return window[:self.n_channels]
        pad = torch.zeros(self.n_channels - n_ch, self.epoch_len)
        return torch.cat([window, pad], dim=0)


class BCITrialBasedDataset(Dataset):
    """BCI dataset extracting only clean motor imagery periods using event markers."""

    def __init__(self, gdf_paths, n_channels, sampling_rate, epoch_duration,
                 mi_offset=2.0):
        import mne
        mne.set_log_level('WARNING')
        
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.samples_per_epoch = int(epoch_duration * sampling_rate)
        self.trials = []
        
        for gdf_path in tqdm(gdf_paths, desc="Loading BCI trials"):
            try:
                raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
                raw.pick_types(eeg=True, exclude=[])
                if len(raw.ch_names) > n_channels:
                    raw.pick(raw.ch_names[:n_channels])
                if raw.info['sfreq'] != sampling_rate:
                    raw.resample(sampling_rate)
                
                signal = raw.get_data()
                
                # Use annotations to get events (GDF files store events as annotations)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                
                # Find trial start events (event code 768 = 0x300)
                # event_id maps annotation descriptions to integer codes
                trial_start_key = None
                for key, code in event_id.items():
                    if '768' in str(code) or 'T0' in key or code == 768:
                        trial_start_key = key
                        break
                
                if trial_start_key is None:
                    # Fallback: look for cue onset events (769-772)
                    cue_keys = [k for k, v in event_id.items() 
                               if any(x in str(v) or x in k for x in ['769', '770', '771', '772', 'T1', 'T2'])]
                    
                    if cue_keys:
                        # Use cue events but adjust timing (cue at t=2s after trial start)
                        trial_events = []
                        for key in cue_keys:
                            cue_events = events[events[:, 2] == event_id[key]]
                            for evt in cue_events:
                                # Subtract 2s to get trial start
                                trial_start = evt[0] - int(2.0 * sampling_rate)
                                if trial_start >= 0:
                                    trial_events.append(trial_start)
                        trial_starts = np.array(trial_events)
                    else:
                        print(f"  Warning: No trial/cue events in {os.path.basename(gdf_path)}")
                        print(f"    Available events: {event_id}")
                        continue
                else:
                    trial_events = events[events[:, 2] == event_id[trial_start_key]]
                    trial_starts = trial_events[:, 0]
                
                if len(trial_starts) == 0:
                    print(f"  Warning: No trials found in {os.path.basename(gdf_path)}")
                    continue
                
                mi_start_offset = int(mi_offset * sampling_rate)
                
                for trial_start in trial_starts:
                    mi_start = trial_start + mi_start_offset
                    mi_end = mi_start + self.samples_per_epoch
                    
                    if mi_end <= signal.shape[1] and mi_start >= 0:
                        trial_data = signal[:, mi_start:mi_end]
                        if not np.isnan(trial_data).any():
                            trial_data = (trial_data - trial_data.mean()) / (trial_data.std() + 1e-8)
                            self.trials.append(torch.from_numpy(trial_data).float())
            except Exception as e:
                print(f"  Error in {os.path.basename(gdf_path)}: {e}")
                import traceback
                traceback.print_exc()
        
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
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x = self.ds[idx]
        return x[0] if isinstance(x, (list, tuple)) else x


def cosine_schedule(base, final, total, step):
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / total))


def warmup_cosine_lr(base_lr, warmup, total, step):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


class EEGDINOTrainer:

    def __init__(self, n_channels=2, sampling_rate=200, embed_dim=64,
                 n_layers=2, n_heads=4, mlp_dim=128, out_dim=4096,
                 head_hidden_dim=256, head_bottleneck_dim=64,
                 n_local_views=4, n_masked_views=1, batch_size=64,
                 learning_rate=1.25e-4, weight_decay_start=0.04,
                 weight_decay_end=0.40, momentum_start=0.996,
                 momentum_end=1.0, warmup_epochs=10, n_epochs=100,
                 mask_strategy='alpha', device='cuda'):

        self.device = 'cuda:0' if (device != 'cpu' and torch.cuda.is_available()) else 'cpu'
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.wd_start, self.wd_end = weight_decay_start, weight_decay_end
        self.mom_start, self.mom_end = momentum_start, momentum_end
        self.warmup_epochs = warmup_epochs
        self.n_epochs = n_epochs

        self.student = StudentModel(
            n_channels, sampling_rate, embed_dim,
            n_layers, n_heads, mlp_dim, out_dim,
            head_hidden_dim, head_bottleneck_dim
        ).to(self.device)
        self.teacher = TeacherModel(self.student).to(self.device)

        self.sampler = ChannelAwareSampling(
            n_channels, sampling_rate, n_local_views, n_masked_views,
            mask_strategy=mask_strategy)
        self.signal_loss_fn = DINOLoss(out_dim=out_dim).to(self.device)
        self.patch_loss_fn = PatchLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=learning_rate,
            weight_decay=weight_decay_start)

        print(f"Mask strategy: '{mask_strategy}'")

        self.total_steps = None
        self.step = 0
        self.momentum = momentum_start

        n_p = sum(p.numel() for p in self.student.parameters())
        print(f"Model: {n_p/1e6:.2f}M params | device: {self.device}")

    @staticmethod
    def _expand_ci(ci, B):
        return ci.unsqueeze(0).expand(B, -1).contiguous()

    def _update_schedules(self):
        s, S = self.step, self.total_steps
        W = self.warmup_epochs * (S // self.n_epochs)

        lr = warmup_cosine_lr(self.base_lr, W, S, s)
        wd = cosine_schedule(self.wd_start, self.wd_end, S, s)
        for g in self.optimizer.param_groups:
            g['lr'], g['weight_decay'] = lr, wd

        self.momentum = cosine_schedule(self.mom_start, self.mom_end, S, s)

    @torch.no_grad()
    def _ema_update(self):
        m = self.momentum
        for ps, pt in zip(self.student.parameters(), self.teacher.model.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    def _forward(self, x):
        views = self.sampler(x)
        s_out, s_pat, t_out, t_pat = {}, {}, {}, {}

        for name, v in views.items():
            vt = v['view'].to(self.device)
            ci = self._expand_ci(v['channels'].to(self.device), vt.shape[0])
            if 'masked' in name:
                sf, pf = self.student(vt, ci, return_patch=True)
                s_out[name], s_pat[name] = sf, pf
            else:
                s_out[name] = self.student(vt, ci, return_patch=False)

        for name in ['global_0', 'global_1']:
            vt = views[name]['view'].to(self.device)
            ci = self._expand_ci(views[name]['channels'].to(self.device), vt.shape[0])
            sf, pf = self.teacher(vt, ci, return_patch=True)
            t_out[name], t_pat[name] = sf, pf

        return s_out, t_out, s_pat, t_pat

    def _train_epoch(self, loader, epoch):
        self.student.train()
        self.teacher.eval()
        self.signal_loss_fn.set_epoch(epoch)
        self.patch_loss_fn.set_epoch(epoch)
        tot, sig_t, pat_t = 0., 0., 0.
        diag = {'s_std': 0., 't_std': 0., 'c_norm': 0., 'grad_norm': 0.}

        for x in loader:
            self._update_schedules()
            self.step += 1

            s_out, t_out, s_pat, t_pat = self._forward(x)
            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.patch_center)
            loss = l_sig + l_pat

            self.optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
            self.optimizer.step()
            self._ema_update()

            with torch.no_grad():
                tg = torch.cat([t_out['global_0'], t_out['global_1']])
                self.teacher.update_center(tg)
                tp = torch.cat([t_pat['global_0'], t_pat['global_1']])
                self.teacher.update_patch_center(tp)

                s_cat = torch.cat([v for v in s_out.values()])
                diag['s_std'] += s_cat.std(dim=0).mean().item()
                diag['t_std'] += tg.std(dim=0).mean().item()
                diag['c_norm'] += self.teacher.center.norm().item()
                diag['grad_norm'] += gn.item() if isinstance(gn, torch.Tensor) else gn

            tot += loss.item()
            sig_t += l_sig.item()
            pat_t += l_pat.item()

        n = len(loader)
        for k in diag:
            diag[k] /= n
        return {'loss': tot/n, 'signal': sig_t/n, 'patch': pat_t/n, 'diag': diag}

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.student.eval()
        self.teacher.eval()
        tot, sig_t, pat_t = 0., 0., 0.

        for x in loader:
            s_out, t_out, s_pat, t_pat = self._forward(x)
            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.patch_center)
            tot += (l_sig + l_pat).item()
            sig_t += l_sig.item()
            pat_t += l_pat.item()

        n = len(loader)
        return {'loss': tot/n, 'signal': sig_t/n, 'patch': pat_t/n}

    def _checkpoint(self, epoch, metrics):
        return {
            'epoch': epoch,
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'metrics': metrics,
        }

    def train(self, train_loader, val_loader=None, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        spe = len(train_loader)
        self.total_steps = spe * self.n_epochs
        self.step = 0

        print(f"Epochs: {self.n_epochs} | Steps/epoch: {spe} | "
              f"Total: {self.total_steps} | LR: {self.base_lr} | BS: {self.batch_size}\n")

        collapse_threshold = math.log(self.signal_loss_fn.out_dim)
        best = float('inf')

        for epoch in range(1, self.n_epochs + 1):
            tr = self._train_epoch(train_loader, epoch)
            d = tr['diag']
            lr = self.optimizer.param_groups[0]['lr']
            wd = self.optimizer.param_groups[0]['weight_decay']
            t_temp = self.signal_loss_fn.teacher_temp

            print(f"Ep {epoch:3d} | loss:{tr['loss']:.4f} sig:{tr['signal']:.4f} pat:{tr['patch']:.4f}"
                  f" | lr:{lr:.2e} wd:{wd:.3f} mom:{self.momentum:.5f} t_temp:{t_temp:.4f}")
            print(f"        | s_std:{d['s_std']:.6f} t_std:{d['t_std']:.6f}"
                  f" c_norm:{d['c_norm']:.4f} grad:{d['grad_norm']:.4f}")

            if tr['signal'] > 0.95 * collapse_threshold:
                print(f"  ⚠ COLLAPSE WARNING: sig_loss={tr['signal']:.4f} ~ {collapse_threshold:.2f}")
            if d['s_std'] < 1e-4:
                print(f"  ⚠ STUDENT OUTPUTS COLLAPSED: std={d['s_std']:.8f}")

            monitor = tr['loss']
            if val_loader:
                va = self._val_epoch(val_loader)
                print(f"   Val  | loss:{va['loss']:.4f} sig:{va['signal']:.4f} pat:{va['patch']:.4f}")
                monitor = va['loss']

            if monitor < best:
                best = monitor
                torch.save(self._checkpoint(epoch, tr),
                           os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ Best ({best:.4f})")

            if epoch % 10 == 0:
                torch.save(self._checkpoint(epoch, tr),
                           os.path.join(save_dir, f'ckpt_ep{epoch}.pth'))

        print(f"\nDone! Best loss: {best:.4f} → {save_dir}/")


def main():
    p = argparse.ArgumentParser(description='EEG-DINO Pre-Training')
    p.add_argument('--max_train', type=int, default=None)
    p.add_argument('--max_val', type=int, default=None)
    p.add_argument('--n_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--mask_strategy', default=None,
                   help='alpha, alpha+beta, random, none, spatiotemporal, all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', default='sleep_edf',
                   choices=['sleep_edf', 'bci_2a', 'bci_2b'])
    p.add_argument('--preset', default='tiny', choices=['tiny', 'bci_2a', 'bci_2b'])
    p.add_argument('--bci_trial_based', action='store_true',
                   help='Use trial-based extraction (clean MI only) for BCI datasets')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = {**PRESETS[args.preset], 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    if args.n_epochs:    cfg['n_epochs'] = args.n_epochs
    if args.batch_size:  cfg['batch_size'] = args.batch_size
    if args.lr:          cfg['learning_rate'] = args.lr
    if args.mask_strategy: cfg['mask_strategy'] = args.mask_strategy

    print(f"\n{'='*60}")
    print(f"EEG-DINO | {args.dataset} | preset: {args.preset}")
    print(f"{'='*60}")

    nc, sr = cfg['n_channels'], cfg['sampling_rate']
    epoch_dur = cfg.get('epoch_duration', 30)
    va_ds = None

    if args.dataset == 'sleep_edf':
        tr_ds = UnlabeledWrapper(SleepEDFDataset(
            SLEEP_EDF_PATH, 'TrainFold', nc, sr, args.max_train))
        va_ds = UnlabeledWrapper(SleepEDFDataset(
            SLEEP_EDF_PATH, 'ValidFold', nc, sr, args.max_val))

    elif args.dataset == 'bci_2a':
        from glob import glob
        if args.bci_trial_based:
            gdf_paths = sorted(glob(os.path.join(BCI_2A_PATH, '*T.gdf')))
            tr_ds = BCITrialBasedDataset(gdf_paths, nc, sr, epoch_dur, mi_offset=2.0)
        else:
            tr_ds = BCIDataset(BCI_2A_PATH, '2a', nc, sr, epoch_dur, sessions='T')

    elif args.dataset == 'bci_2b':
        from glob import glob
        if args.bci_trial_based:
            sessions = ['01T', '02T', '03T']
            gdf_paths = []
            for i in range(1, 10):
                for s in sessions:
                    path = os.path.join(BCI_2B_PATH, f'B{i:02d}{s}.gdf')
                    if os.path.exists(path):
                        gdf_paths.append(path)
            tr_ds = BCITrialBasedDataset(gdf_paths, nc, sr, epoch_dur, mi_offset=3.0)
        else:
            tr_ds = BCIDataset(BCI_2B_PATH, '2b', nc, sr, epoch_dur, sessions='T')

    val_str = f" | Val: {len(va_ds)}" if va_ds else ""
    print(f"Train: {len(tr_ds)}{val_str}\n")

    cuda = cfg['device'] != 'cpu'
    tr_loader = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,
                           num_workers=4, pin_memory=cuda, drop_last=True)
    va_loader = None
    if va_ds:
        va_loader = DataLoader(va_ds, cfg['batch_size'], shuffle=False,
                               num_workers=4, pin_memory=cuda)

    trainer_cfg = {k: v for k, v in cfg.items() if k != 'epoch_duration'}
    trainer = EEGDINOTrainer(**trainer_cfg)
    trainer.train(tr_loader, va_loader, save_dir=args.save_dir)


if __name__ == '__main__':
    main()