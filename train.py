"""EEG-DINO pre-training on Sleep-EDF and BCI-IV datasets."""

import argparse
import math
import os
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
import logging
import sys

from model.eeg_dino_model import StudentModel, TeacherModel
from model.channel_aware_sampling import ChannelAwareSampling
from model.losses import DINOLoss, PatchLoss
from configs import PRESETS
from datasets import BCITrialBasedDataset, SleepEDFDataset, UnlabeledWrapper, get_dataset_root


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

        logging.getLogger(__name__).info(f"Mask strategy: '{mask_strategy}'")

        self.total_steps = None
        self.step = 0
        self.momentum = momentum_start

        n_p = sum(p.numel() for p in self.student.parameters())
        logging.getLogger(__name__).info(f"Model: {n_p/1e6:.2f}M params | device: {self.device}")

    @staticmethod
    def _expand_ci(ci, B):
        # bradcast channel indices to match batch size for loss computation
        return ci.unsqueeze(0).expand(B, -1).contiguous()

    def _update_schedules(self):
        s, S = self.step, self.total_steps
        W = self.warmup_epochs * (S // self.n_epochs) # warmup steps

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
        views = self.sampler(x) # generate augmented views  
        s_out, s_pat, t_out, t_pat = {}, {}, {}, {}

        # student forward pass for all views
        for name, v in views.items():
            vt = v['view'].to(self.device)
            ci = self._expand_ci(v['channels'].to(self.device), vt.shape[0])

            if 'masked' in name:
                sf, pf = self.student(vt, ci, return_patch=True) # for masked views, also return patch features for patch loss
                s_out[name], s_pat[name] = sf, pf 
            else:
                s_out[name] = self.student(vt, ci, return_patch=False) 

        # teacher forward pass on global views only
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

        has_masked_views = (self.sampler.n_masked_views > 0)  # ← ADD THIS

        for x in loader:

            self._update_schedules()
            self.step += 1

            s_out, t_out, s_pat, t_pat = self._forward(x)

            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            if has_masked_views:
                l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.patch_center)
                loss = l_sig + l_pat
            else:
                l_pat = torch.tensor(0.0, device=self.device)
                loss = l_sig

            self.optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.0)
            self.optimizer.step()

            self._ema_update()

            with torch.no_grad():
                tg = torch.cat([t_out['global_0'], t_out['global_1']])
                self.teacher.update_center(tg)
                
                if has_masked_views:
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
        has_masked_views = (self.sampler.n_masked_views > 0)

        for x in loader:
            s_out, t_out, s_pat, t_pat = self._forward(x)
            l_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)

            if has_masked_views:
                l_pat = self.patch_loss_fn(s_pat, t_pat, self.teacher.patch_center)
                total_loss = l_sig + l_pat
            else:
                l_pat = torch.tensor(0.0, device=self.device)
                total_loss = l_sig 
                      
            tot += total_loss.item()
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
        self.total_steps = len(train_loader) * self.n_epochs
        self.step = 0

        logging.getLogger(__name__).info(
            f"Epochs: {self.n_epochs} | Steps/epoch: {len(train_loader)} | "
            f"Total: {self.total_steps} | LR: {self.base_lr} | BS: {self.batch_size}\n")

        collapse_threshold = math.log(self.signal_loss_fn.out_dim)
        best = float('inf')

        for epoch in range(1, self.n_epochs + 1):
            tr = self._train_epoch(train_loader, epoch)
            d = tr['diag']
            lr = self.optimizer.param_groups[0]['lr']
            wd = self.optimizer.param_groups[0]['weight_decay']
            t_temp = self.signal_loss_fn.teacher_temp

            logging.getLogger(__name__).info(
                f"Ep {epoch:3d} | loss:{tr['loss']:.4f} sig:{tr['signal']:.4f} pat:{tr['patch']:.4f}"
                f" | lr:{lr:.2e} wd:{wd:.3f} mom:{self.momentum:.5f} t_temp:{t_temp:.4f}")
            logging.getLogger(__name__).info(
                f"        | s_std:{d['s_std']:.6f} t_std:{d['t_std']:.6f}"
                f" c_norm:{d['c_norm']:.4f} grad:{d['grad_norm']:.4f}")
            
            # check for collapse (signal loss close to log(out_dim) and very low student std)
            if tr['signal'] > 0.95 * collapse_threshold:
                logging.getLogger(__name__).warning(
                    f"⚠ COLLAPSE WARNING: sig_loss={tr['signal']:.4f} ~ {collapse_threshold:.2f}")
            if d['s_std'] < 1e-4:
                logging.getLogger(__name__).warning(
                    f"⚠ STUDENT OUTPUTS COLLAPSED: std={d['s_std']:.8f}")

            monitor = tr['loss']
            if val_loader:
                va = self._val_epoch(val_loader)
                logging.getLogger(__name__).info(
                    f"   Val  | loss:{va['loss']:.4f} sig:{va['signal']:.4f} pat:{va['patch']:.4f}")
                monitor = va['loss']

            # save best 
            if monitor < best:
                best = monitor
                torch.save(self._checkpoint(epoch, tr),
                           os.path.join(save_dir, 'best_model.pth'))
                logging.getLogger(__name__).info(f"✓ Best ({best:.4f})")

            if epoch % 10 == 0:
                torch.save(self._checkpoint(epoch, tr),
                           os.path.join(save_dir, f'ckpt_ep{epoch}.pth'))

        logging.getLogger(__name__).info(f"\nDone! Best loss: {best:.4f} → {save_dir}/")


def main():
    p = argparse.ArgumentParser(description='EEG-DINO Pre-Training')
    p.add_argument('--n_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--mask_strategy', default=None,
                   help='alpha, beta, delta, alpha+beta, random, none, spatiotemporal')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', default='sleep_edf',
                   choices=['sleep_edf', 'bci_2a', 'bci_2b'])
    p.add_argument('--preset', default='tiny', choices=['tiny', 'bci_2a', 'bci_2b'])
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Load preset config and override with command-line args
    cfg = {**PRESETS[args.preset], 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    if args.n_epochs:    cfg['n_epochs'] = args.n_epochs
    if args.batch_size:  cfg['batch_size'] = args.batch_size
    if args.lr:          cfg['learning_rate'] = args.lr
    if args.mask_strategy: cfg['mask_strategy'] = args.mask_strategy
    if cfg['mask_strategy'] == 'none': cfg['n_masked_views'] = 0 # override to ensure no masked views if strategy is 'none'

    run_leaf = f"{cfg['mask_strategy']}_seed{args.seed}"
    save_dir = args.save_dir
    if os.path.basename(os.path.normpath(save_dir)) != run_leaf:
        save_dir = os.path.join(save_dir, run_leaf)

    print(f"\n{'='*60}")
    print(f"EEG-DINO | {args.dataset} | preset: {args.preset} | seed: {args.seed}")
    print(f"{'='*60}")

    nc, sr = cfg['n_channels'], cfg['sampling_rate']
    epoch_dur = cfg.get('epoch_duration', 30)

    if args.dataset == 'sleep_edf':
        sleep_root = get_dataset_root('sleep_edf')
        tr_ds = UnlabeledWrapper(SleepEDFDataset(
            sleep_root, 'TrainFold', nc, sr))
        va_ds = UnlabeledWrapper(SleepEDFDataset(
            sleep_root, 'ValidFold', nc, sr))
    
    elif args.dataset == 'bci_2a':
        from glob import glob
        bci_2a_root = get_dataset_root('bci_2a')
        gdf_paths = sorted(glob(os.path.join(bci_2a_root, '*T.gdf')))
        tr_ds = BCITrialBasedDataset(gdf_paths, nc, sr, epoch_dur, mi_offset=2.0)
        va_ds = None
    
    elif args.dataset == 'bci_2b':
        from glob import glob
        bci_2b_root = get_dataset_root('bci_2b')
        sessions = ['01T', '02T', '03T']
        gdf_paths = []
        for i in range(1, 10):
            for s in sessions:
                path = os.path.join(bci_2b_root, f'B{i:02d}{s}.gdf')
                if os.path.exists(path):
                    gdf_paths.append(path)
        tr_ds = BCITrialBasedDataset(gdf_paths, nc, sr, epoch_dur, mi_offset=3.0)
        va_ds = None

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
    trainer.train(tr_loader, va_loader, save_dir=save_dir)


if __name__ == '__main__':
    main()