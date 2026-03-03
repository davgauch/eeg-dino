"""
EEG-DINO Pre-Training
=====================
Usage:
  python train.py --preset tiny --dataset sleep
  python train.py --preset small --dataset sleep --max_train 5000
  python train.py --preset base --dataset both --n_channels 2

For server:
  CUDA_VISIBLE_DEVICES=0,2 nohup python train.py --preset tiny --dataset sleep > train.log 2>&1 &
"""

import os, json, math, argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from eeg_dino_model import StudentModel, TeacherModel
from channel_aware_sampling import ChannelAwareSampling
from losses import DINOLoss, PatchLoss

SLEEP_EDF_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/'
BCI_PATH       = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/'
SPLITS_FILE    = './splits.json'


# ── Datasets ─────────────────────────────────────────────────────────────────

class SleepEDFDataset(Dataset):
    """Sleep-EDF .pt files → z-scored 30s epochs. Native: 2 channels at 100 Hz."""

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

                # Resample 100→200 Hz
                t = F.interpolate(t.unsqueeze(0), size=self.epoch_len,
                                  mode='linear', align_corners=False).squeeze(0)

                # Z-score, then ensure exactly n_channels (trim or pad)
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
    """BCI-IV-2a .gdf files → z-scored 30s epochs. Native: 22 EEG at 250 Hz."""

    def __init__(self, root, split='train', n_channels=2,
                 sampling_rate=200, splits_file=SPLITS_FILE,
                 seed=42, max_samples=None):
        from glob import glob
        import mne, warnings

        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_len = sampling_rate * 30

        gdf_files = sorted(glob(os.path.join(root, '**/*.gdf'), recursive=True))

        # Deterministic train/val/test split by file
        if os.path.exists(splits_file):
            with open(splits_file) as f:
                splits = json.load(f)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(gdf_files)).tolist()
            nv = max(1, int(0.1 * len(idx)))
            splits = {'train': idx[nv*2:], 'val': idx[:nv], 'test': idx[nv:nv*2]}
            with open(splits_file, 'w') as f:
                json.dump(splits, f, indent=2)

        files = [gdf_files[i] for i in splits[split]]
        print(f"[BCI/{split}] {len(files)} files, loading...")

        self.data, self.labels = [], []
        for fp in tqdm(files, desc=f"BCI/{split}"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)

                picks = mne.pick_types(raw.info, eeg=True, exclude='bads')[:n_channels]
                raw.resample(sampling_rate)
                data = raw.get_data(picks=picks)

                if data.shape[0] < n_channels:
                    pad = np.zeros((n_channels - data.shape[0], data.shape[1]))
                    data = np.concatenate([data, pad], axis=0)

                for s in range(0, data.shape[1] - self.epoch_len, self.epoch_len):
                    seg = torch.FloatTensor(data[:, s:s + self.epoch_len])
                    seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                    self.data.append(seg)
                    self.labels.append(-1)
                    if max_samples and len(self.data) >= max_samples:
                        break
            except Exception as e:
                print(f"  Skip {fp}: {e}")
        print(f"  → {len(self.data)} epochs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class UnlabeledWrapper(Dataset):
    """Strip labels for pre-training."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds[idx][0]


# ── Schedules ────────────────────────────────────────────────────────────────

def cosine_schedule(base, final, total, step):
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / total))

def warmup_cosine_lr(base_lr, warmup, total, step):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


# ── Trainer ──────────────────────────────────────────────────────────────────

class EEGDINOTrainer:

    def __init__(self, n_channels=2, sampling_rate=200, embed_dim=64,
                 n_layers=2, n_heads=4, mlp_dim=128, out_dim=64,
                 n_local_views=4, n_masked_views=1, batch_size=64,
                 learning_rate=5e-4, weight_decay_start=0.04,
                 weight_decay_end=0.20, momentum_start=0.996,
                 momentum_end=1.0, warmup_epochs=5, n_epochs=100,
                 device='cuda'):

        self.device = device
        self.batch_size = batch_size
        self.base_lr = learning_rate
        self.wd_start, self.wd_end = weight_decay_start, weight_decay_end
        self.mom_start, self.mom_end = momentum_start, momentum_end
        self.warmup_epochs = warmup_epochs
        self.n_epochs = n_epochs

        # Resolve device — single GPU by default, safe multi-GPU only
        # via CUDA_VISIBLE_DEVICES=0,1 (set before launching)
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda:0'

        self.student = StudentModel(
            n_channels, sampling_rate, embed_dim,
            n_layers, n_heads, mlp_dim, out_dim
        ).to(self.device)
        self.teacher = TeacherModel(self.student).to(self.device)

        self.sampler = ChannelAwareSampling(
            n_channels, sampling_rate, n_local_views, n_masked_views)
        self.signal_loss_fn = DINOLoss(out_dim=out_dim).to(self.device)
        self.patch_loss_fn = PatchLoss().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self._raw().parameters(), lr=learning_rate,
            weight_decay=weight_decay_start)

        self.total_steps = None
        self.step = 0
        self.momentum = momentum_start

        n_p = sum(p.numel() for p in self._raw().parameters())
        print(f"Model: {n_p/1e6:.2f}M params | device: {self.device}")

    def _raw(self):
        return self.student.module if isinstance(self.student, nn.DataParallel) else self.student

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
        for ps, pt in zip(self._raw().parameters(), self.teacher.model.parameters()):
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

        for i, x in enumerate(loader):
            if isinstance(x, (list, tuple)):
                x = x[0]
            self._update_schedules()
            self.step += 1

            s_out, t_out, s_pat, t_pat = self._forward(x)
            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.center)
            loss = l_sig + l_pat

            self.optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self._raw().parameters(), 3.0)
            self.optimizer.step()
            self._ema_update()

            with torch.no_grad():
                tg = torch.cat([t_out['global_0'], t_out['global_1']])
                self.teacher.update_center(tg)

                # Diagnostics
                s_cat = torch.cat([v for v in s_out.values()])
                t_cat = tg
                diag['s_std'] += s_cat.std(dim=0).mean().item()
                diag['t_std'] += t_cat.std(dim=0).mean().item()
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            s_out, t_out, s_pat, t_pat = self._forward(x)
            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.center)
            tot += (l_sig + l_pat).item()
            sig_t += l_sig.item()
            pat_t += l_pat.item()

        n = len(loader)
        return {'loss': tot/n, 'signal': sig_t/n, 'patch': pat_t/n}

    def _checkpoint(self, epoch, metrics):
        return {
            'epoch': epoch,
            'student': self._raw().state_dict(),
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
        self.momentum = self.mom_start

        print(f"\nEpochs: {self.n_epochs} | Steps/epoch: {spe} | "
              f"Total: {self.total_steps} | LR: {self.base_lr} | BS: {self.batch_size}")

        collapse_threshold = 3 * math.log(self.signal_loss_fn.out_dim if hasattr(self.signal_loss_fn, 'out_dim') else 256)

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

            # Collapse warning
            if tr['signal'] > 0.95 * collapse_threshold:
                print(f"  ⚠ COLLAPSE WARNING: sig_loss={tr['signal']:.4f} ~ {collapse_threshold:.2f} (uniform)")
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


# ── Presets ──────────────────────────────────────────────────────────────────

PRESETS = {
    'tiny': dict(
        embed_dim=64, n_layers=2, n_heads=4, mlp_dim=128, out_dim=256,
        n_local_views=4, n_masked_views=1, batch_size=64,
        learning_rate=5e-4, warmup_epochs=5,
        weight_decay_start=0.04, weight_decay_end=0.20,
    ),
    'small': dict(
        embed_dim=128, n_layers=4, n_heads=4, mlp_dim=256, out_dim=256,
        n_local_views=6, n_masked_views=2, batch_size=64,
        learning_rate=3e-4, warmup_epochs=5,
        weight_decay_start=0.04, weight_decay_end=0.30,
    ),
    'base': dict(
        embed_dim=200, n_layers=12, n_heads=8, mlp_dim=512, out_dim=256,
        n_local_views=8, n_masked_views=2, batch_size=256,
        learning_rate=1e-4, warmup_epochs=10,
        weight_decay_start=0.04, weight_decay_end=0.40,
    ),
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='EEG-DINO Pre-Training')
    p.add_argument('--preset', default='tiny', choices=PRESETS)
    p.add_argument('--dataset', default='sleep', choices=['sleep', 'bci', 'both'])
    p.add_argument('--n_channels', type=int, default=2)
    p.add_argument('--max_train', type=int, default=None)
    p.add_argument('--max_val', type=int, default=None)
    p.add_argument('--n_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save_dir', default='checkpoints')
    args = p.parse_args()

    cfg = {
        'n_channels': args.n_channels,
        'sampling_rate': 200,
        'momentum_start': 0.996,
        'momentum_end': 1.0,
        'n_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        **PRESETS[args.preset],
    }
    if args.n_epochs:  cfg['n_epochs'] = args.n_epochs
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.lr:        cfg['learning_rate'] = args.lr

    print(f"EEG-DINO | preset={args.preset} | dataset={args.dataset}")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")

    # Load data
    tr_parts, va_parts = [], []
    if args.dataset in ('sleep', 'both'):
        tr_parts.append(SleepEDFDataset(
            SLEEP_EDF_PATH, 'TrainFold', cfg['n_channels'],
            cfg['sampling_rate'], args.max_train))
        va_parts.append(SleepEDFDataset(
            SLEEP_EDF_PATH, 'ValidFold', cfg['n_channels'],
            cfg['sampling_rate'], args.max_val))

    if args.dataset in ('bci', 'both'):
        tr_parts.append(BCIDataset(
            BCI_PATH, 'train', cfg['n_channels'],
            cfg['sampling_rate'], max_samples=args.max_train))
        va_parts.append(BCIDataset(
            BCI_PATH, 'val', cfg['n_channels'],
            cfg['sampling_rate'], max_samples=args.max_val))

    tr_ds = UnlabeledWrapper(ConcatDataset(tr_parts) if len(tr_parts) > 1 else tr_parts[0])
    va_ds = UnlabeledWrapper(ConcatDataset(va_parts) if len(va_parts) > 1 else va_parts[0])
    print(f"Train: {len(tr_ds)} | Val: {len(va_ds)}")

    cuda = cfg['device'] != 'cpu'
    tr_loader = DataLoader(tr_ds, cfg['batch_size'], shuffle=True,
                           num_workers=4, pin_memory=cuda, drop_last=True)
    va_loader = DataLoader(va_ds, cfg['batch_size'], shuffle=False,
                           num_workers=4, pin_memory=cuda)

    trainer = EEGDINOTrainer(**cfg)
    trainer.train(tr_loader, va_loader, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
