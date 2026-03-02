"""
EEG-DINO Pre-Training
=====================
Launch:
  CUDA_VISIBLE_DEVICES=0,2 nohup python train.py > train.log 2>&1 &
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from eeg_dino_model import StudentModel, TeacherModel
from channel_aware_sampling import ChannelAwareSampling
from losses import DINOLoss, PatchLoss


# ── Paths ─────────────────────────────────────────────────────────────────────
SLEEP_EDF_PATH = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/'
BCI_PATH       = '/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/'
SPLITS_FILE    = './splits.json'   # saved locally, never touches the data server


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class SleepEDFDataset(Dataset):
    """
    Loads pre-epoched .pt tensors from Sleep-EDF.
    The server already provides TrainFold / ValidFold / TestFold directories,
    so we respect that split rather than creating our own.

    Each .pt file: [2, 3000]  (2 channels, 3000 samples @ 100 Hz = 30 seconds)
    We resample to 200 Hz → [2, 6000] and zero-pad to [19, 6000].

    The subfolder name (0–4) encodes the sleep stage label, stored in self.labels
    so the evaluation script can use it for linear probing.
    """
    def __init__(self, root, fold='TrainFold', n_channels=19, sampling_rate=200):
        from glob import glob
        import torch.nn.functional as F

        self.n_channels   = n_channels
        self.sampling_rate = sampling_rate
        self.samples_per_epoch = sampling_rate * 30   # 6000

        fold_path = os.path.join(root, fold)
        pt_files  = sorted(glob(os.path.join(fold_path, '**/*.pt'), recursive=True))
        print(f"[SleepEDF/{fold}] Found {len(pt_files)} files, loading...")

        self.data   = []
        self.labels = []

        for fp in tqdm(pt_files, desc=f"SleepEDF/{fold}"):
            try:
                # Label = subfolder name (sleep stage 0–4)
                label = int(os.path.basename(os.path.dirname(fp)))

                tensor = torch.load(fp, map_location='cpu', weights_only=True)  # [2,3000]

                # Resample 100 Hz → 200 Hz
                tensor = F.interpolate(
                    tensor.unsqueeze(0).float(),
                    size=self.samples_per_epoch,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)                                                      # [2,6000]

                # Zero-pad channels to n_channels
                if tensor.shape[0] < n_channels:
                    pad = torch.zeros(n_channels - tensor.shape[0], tensor.shape[1])
                    tensor = torch.cat([tensor, pad], dim=0)                      # [19,6000]

                self.data.append(tensor)
                self.labels.append(label)

            except Exception as e:
                print(f"  Skipping {fp}: {e}")

        print(f"  → Loaded {len(self.data)} epochs")

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


class BCIDataset(Dataset):
    """
    Loads raw .gdf files from BCI Competition IV 2a/2b.
    We do NOT modify or write anything to the data server.

    A random 80/10/10 train/val/test split is computed once and saved to
    splits.json in the local working directory. Subsequent runs reload it,
    ensuring reproducibility without touching the original files.

    Each sample: [19, 6000] (zero-padded to 19 ch, 200 Hz, 30 s)
    """
    def __init__(self, root, split='train', n_channels=19, sampling_rate=200,
                 splits_file=SPLITS_FILE, seed=42):
        from glob import glob
        import mne, warnings

        self.n_channels    = n_channels
        self.sampling_rate = sampling_rate
        self.samples_per_epoch = sampling_rate * 30

        gdf_files = sorted(glob(os.path.join(root, '**/*.gdf'), recursive=True))

        # ── Load or create split indices ──────────────────────────────────
        if os.path.exists(splits_file):
            with open(splits_file) as f:
                splits = json.load(f)
            print(f"[BCI] Loaded existing splits from {splits_file}")
        else:
            rng   = np.random.default_rng(seed)
            idx   = rng.permutation(len(gdf_files)).tolist()
            n_val = max(1, int(0.1 * len(idx)))
            splits = {
                'train': idx[n_val*2:],
                'val':   idx[:n_val],
                'test':  idx[n_val:n_val*2]
            }
            with open(splits_file, 'w') as f:
                json.dump(splits, f, indent=2)
            print(f"[BCI] Created new splits → saved to {splits_file}")

        selected_files = [gdf_files[i] for i in splits[split]]
        print(f"[BCI/{split}] {len(selected_files)} files, loading...")

        self.data   = []
        self.labels = []   # trial labels from GDF events (if available)

        for fp in tqdm(selected_files, desc=f"BCI/{split}"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = mne.io.read_raw_gdf(fp, preload=True, verbose=False)

                eeg_idx = mne.pick_types(raw.info, eeg=True, exclude='bads')
                eeg_idx = eeg_idx[:n_channels]

                raw.resample(sampling_rate)
                epoch_data = raw.get_data(picks=eeg_idx)   # [n_ch, n_times]

                if epoch_data.shape[0] < n_channels:
                    pad = np.zeros((n_channels - epoch_data.shape[0], epoch_data.shape[1]))
                    epoch_data = np.concatenate([epoch_data, pad], axis=0)

                for start in range(0, epoch_data.shape[1] - self.samples_per_epoch,
                                   self.samples_per_epoch):
                    seg = epoch_data[:, start:start + self.samples_per_epoch]
                    self.data.append(torch.FloatTensor(seg))
                    self.labels.append(-1)   # no per-epoch label for pre-training

            except Exception as e:
                print(f"  Skipping {fp}: {e}")

        print(f"  → Loaded {len(self.data)} epochs")

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


class UnlabeledWrapper(Dataset):
    """Strip labels so DataLoader returns plain tensors during pre-training."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):  return len(self.dataset)
    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler helpers  (DINO uses per-step schedules, not per-epoch)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_schedule(base_val, final_val, total_steps, step):
    """Cosine interpolation from base_val to final_val over total_steps."""
    return final_val + 0.5 * (base_val - final_val) * (
        1 + math.cos(math.pi * step / total_steps)
    )

def warmup_cosine_lr(base_lr, warmup_steps, total_steps, step):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr * 0.5 * (
        1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class EEGDINOTrainer:
    """
    EEG-DINO-S pre-training with collapse-prevention measures.

    Key differences from the collapsed run:
      • LR = 1e-4 (not 2e-4)
      • Linear warmup for warmup_epochs then cosine decay
      • Gradient clipping max_norm=3.0
      • Teacher momentum: cosine schedule 0.996 → 1.0
      • Weight decay: cosine schedule 0.04 → 0.4
    """
    def __init__(
        self,
        n_channels=19,
        sampling_rate=200,
        embed_dim=200,
        n_layers=12,
        n_heads=8,
        mlp_dim=512,
        batch_size=256,
        learning_rate=1e-4,       # paper default — do NOT increase
        weight_decay_start=0.04,
        weight_decay_end=0.40,
        momentum_start=0.996,
        momentum_end=1.0,
        warmup_epochs=10,
        n_epochs=100,
        device='cuda'
    ):
        self.device            = device
        self.batch_size        = batch_size
        self.base_lr           = learning_rate
        self.wd_start          = weight_decay_start
        self.wd_end            = weight_decay_end
        self.mom_start         = momentum_start
        self.mom_end           = momentum_end
        self.warmup_epochs     = warmup_epochs
        self.n_epochs          = n_epochs

        # ── Models ──────────────────────────────────────────────────────────
        print("Initializing student model...")
        self.student = StudentModel(
            n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim
        ).to(device)

        # Teacher MUST be built from raw StudentModel BEFORE DataParallel
        print("Initializing teacher model...")
        self.teacher = TeacherModel(self.student).to(device)

        # GPU selection: CUDA_VISIBLE_DEVICES=0,2 remaps the two L40S to 0,1
        n_visible = torch.cuda.device_count()
        if n_visible >= 2:
            print(f"Using {n_visible} GPUs via DataParallel (device_ids=[0,1])")
            self.student = nn.DataParallel(self.student, device_ids=[0, 1])
            self.device  = 'cuda:0'
        elif n_visible == 1:
            print("Single GPU mode on cuda:0")
            self.device = 'cuda:0'
        else:
            raise RuntimeError("No CUDA GPUs visible.")

        # ── Sampling & losses ────────────────────────────────────────────────
        self.sampler        = ChannelAwareSampling(n_channels, sampling_rate)
        self.signal_loss_fn = DINOLoss().to(self.device)
        self.patch_loss_fn  = PatchLoss().to(self.device)

        # ── Optimizer  (AdamW, no scheduler — we update LR/WD manually) ─────
        self.optimizer = torch.optim.AdamW(
            self._raw_student().parameters(),
            lr=learning_rate,
            weight_decay=weight_decay_start
        )

        # total_steps filled in train() once we know the dataloader length
        self.total_steps   = None
        self.current_step  = 0

        n_params = sum(p.numel() for p in self._raw_student().parameters())
        print(f"✓ Initialized EEG-DINO with {n_params/1e6:.1f}M parameters")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _raw_student(self):
        if isinstance(self.student, nn.DataParallel):
            return self.student.module
        return self.student

    @staticmethod
    def _batch_channel_indices(channel_indices, batch_size):
        return channel_indices.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _update_schedules(self):
        """Called every step: update LR, weight decay, and teacher momentum."""
        s, S = self.current_step, self.total_steps
        W    = self.warmup_epochs * (S // self.n_epochs)  # warmup steps

        # LR: linear warmup → cosine decay
        lr = warmup_cosine_lr(self.base_lr, W, S, s)
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        # Weight decay: cosine 0.04 → 0.40
        wd = cosine_schedule(self.wd_start, self.wd_end, S, s)
        for g in self.optimizer.param_groups:
            g['weight_decay'] = wd

        # Teacher momentum: cosine 0.996 → 1.0
        self.current_momentum = cosine_schedule(
            self.mom_start, self.mom_end, S, s
        )

    # ── EMA update ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_teacher(self):
        m = self.current_momentum
        for p_s, p_t in zip(self._raw_student().parameters(),
                             self.teacher.model.parameters()):
            p_t.data = p_t.data * m + p_s.data * (1.0 - m)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward_pass(self, x):
        views = self.sampler(x)
        s_out, s_patch, t_out, t_patch = {}, {}, {}, {}

        for vname, vdata in views.items():
            vt  = vdata['view'].to(self.device)
            ci  = vdata['channels'].to(self.device)
            cib = self._batch_channel_indices(ci, vt.shape[0])
            if 'masked' in vname:
                sf, pf = self.student(vt, cib, return_patch=True)
                s_out[vname]   = sf
                s_patch[vname] = pf
            else:
                s_out[vname] = self.student(vt, cib, return_patch=False)

        for vname in ['global_0', 'global_1']:
            vt  = views[vname]['view'].to(self.device)
            ci  = views[vname]['channels'].to(self.device)
            cib = self._batch_channel_indices(ci, vt.shape[0])
            sf, pf = self.teacher(vt, cib, return_patch=True)
            t_out[vname]   = sf
            t_patch[vname] = pf

        return s_out, t_out, s_patch, t_patch

    # ── One epoch ────────────────────────────────────────────────────────────

    def train_epoch(self, dataloader, epoch):
        self.student.train()
        self.teacher.eval()

        tot, sig_tot, pat_tot = 0.0, 0.0, 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for x in pbar:
            # x may be a plain tensor (UnlabeledWrapper) or (tensor, label) tuple
            if isinstance(x, (list, tuple)):
                x = x[0]

            self._update_schedules()
            self.current_step += 1

            s_out, t_out, s_patch, t_patch = self.forward_pass(x)

            loss_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            loss_pat    = self.patch_loss_fn(s_patch, t_patch, self.teacher.center)
            loss        = loss_sig + loss_pat

            self.optimizer.zero_grad()
            loss.backward()

            # ── Gradient clipping: critical for DINO stability ───────────
            torch.nn.utils.clip_grad_norm_(
                self._raw_student().parameters(), max_norm=3.0
            )

            self.optimizer.step()
            self.update_teacher()

            with torch.no_grad():
                tg = torch.cat([t_out['global_0'], t_out['global_1']], dim=0)
                self.teacher.update_center(tg)

            tot     += loss.item()
            sig_tot += loss_sig.item()
            pat_tot += loss_pat.item()

            pbar.set_postfix({
                'loss':   f'{loss.item():.4f}',
                'signal': f'{loss_sig.item():.4f}',
                'patch':  f'{loss_pat.item():.4f}',
                'mom':    f'{self.current_momentum:.5f}',
                'lr':     f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        n = len(dataloader)
        return {'loss': tot/n, 'signal_loss': sig_tot/n, 'patch_loss': pat_tot/n}

    @torch.no_grad()
    def val_epoch(self, dataloader):
        """
        Validation: compute average DINO loss on held-out data.
        Student is in eval mode — no gradient, no EMA update.
        """
        self.student.eval()
        self.teacher.eval()

        tot, sig_tot, pat_tot = 0.0, 0.0, 0.0
        for x in tqdm(dataloader, desc="  Val", leave=False):
            if isinstance(x, (list, tuple)):
                x = x[0]

            s_out, t_out, s_patch, t_patch = self.forward_pass(x)
            loss_sig, _ = self.signal_loss_fn(s_out, t_out, self.teacher.center)
            loss_pat    = self.patch_loss_fn(s_patch, t_patch, self.teacher.center)
            loss        = loss_sig + loss_pat

            tot     += loss.item()
            sig_tot += loss_sig.item()
            pat_tot += loss_pat.item()

        n = len(dataloader)
        return {'loss': tot/n, 'signal_loss': sig_tot/n, 'patch_loss': pat_tot/n}

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _build_checkpoint(self, epoch, metrics):
        return {
            'epoch':                epoch,
            'student_state_dict':   self._raw_student().state_dict(),
            'teacher_state_dict':   self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step':         self.current_step,
            'metrics':              metrics
        }

    # ── Full training loop ───────────────────────────────────────────────────

    def train(self, train_loader, val_loader=None, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        steps_per_epoch  = len(train_loader)
        self.total_steps = steps_per_epoch * self.n_epochs
        self.current_step = 0
        self.current_momentum = self.mom_start

        print(f"\n{'='*60}")
        print(f"EEG-DINO-S Pre-training")
        print(f"{'='*60}")
        print(f"Epochs:          {self.n_epochs}")
        print(f"Steps/epoch:     {steps_per_epoch}")
        print(f"Total steps:     {self.total_steps}")
        print(f"Warmup epochs:   {self.warmup_epochs}")
        print(f"Base LR:         {self.base_lr}")
        print(f"Batch size:      {self.batch_size}")
        print(f"Grad clip:       3.0")
        print(f"Momentum:        {self.mom_start} → {self.mom_end} (cosine)")
        print(f"Weight decay:    {self.wd_start} → {self.wd_end} (cosine)")
        print(f"Device:          {self.device}")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(1, self.n_epochs + 1):

            train_m = self.train_epoch(train_loader, epoch)

            print(f"\nEpoch {epoch}/{self.n_epochs}")
            print(f"  Train  — loss: {train_m['loss']:.4f}  "
                  f"signal: {train_m['signal_loss']:.4f}  "
                  f"patch: {train_m['patch_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}  "
                  f"WD: {self.optimizer.param_groups[0]['weight_decay']:.4f}  "
                  f"Mom: {self.current_momentum:.5f}")

            # Validation
            if val_loader is not None:
                val_m = self.val_epoch(val_loader)
                print(f"  Val    — loss: {val_m['loss']:.4f}  "
                      f"signal: {val_m['signal_loss']:.4f}  "
                      f"patch: {val_m['patch_loss']:.4f}")
                monitor = val_m['loss']
            else:
                monitor = train_m['loss']

            # Save best
            if monitor < best_val_loss:
                best_val_loss = monitor
                torch.save(
                    self._build_checkpoint(epoch, train_m),
                    os.path.join(save_dir, 'best_model.pth')
                )
                print(f"  ✓ Saved best model (val loss: {best_val_loss:.4f})")

            # Periodic checkpoint
            if epoch % 10 == 0:
                torch.save(
                    self._build_checkpoint(epoch, train_m),
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                )

        print("\n✓ Pre-training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # EEG-DINO-S config from Table 1 of the paper
    config = {
        'n_channels':        19,
        'sampling_rate':     200,
        'embed_dim':         200,    # hidden size
        'n_layers':          12,
        'n_heads':           8,
        'mlp_dim':           512,
        'batch_size':        256,
        'learning_rate':     1e-4,   # do NOT scale up — caused collapse before
        'weight_decay_start': 0.04,
        'weight_decay_end':   0.40,
        'momentum_start':    0.996,
        'momentum_end':      1.0,
        'warmup_epochs':     10,
        'n_epochs':          100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    # Sleep-EDF: server already provides train/val/test folds
    sleep_train = SleepEDFDataset(SLEEP_EDF_PATH, fold='TrainFold',
                                   n_channels=config['n_channels'],
                                   sampling_rate=config['sampling_rate'])
    sleep_val   = SleepEDFDataset(SLEEP_EDF_PATH, fold='ValidFold',
                                   n_channels=config['n_channels'],
                                   sampling_rate=config['sampling_rate'])

    # BCI: random split saved to splits.json (reproducible, non-destructive)
    bci_train = BCIDataset(BCI_PATH, split='train',
                            n_channels=config['n_channels'],
                            sampling_rate=config['sampling_rate'])
    bci_val   = BCIDataset(BCI_PATH, split='val',
                            n_channels=config['n_channels'],
                            sampling_rate=config['sampling_rate'])

    # Combine and strip labels for pre-training
    train_dataset = UnlabeledWrapper(ConcatDataset([sleep_train, bci_train]))
    val_dataset   = UnlabeledWrapper(ConcatDataset([sleep_val,   bci_val]))

    print(f"\nTrain: {len(train_dataset)} epochs")
    print(f"Val:   {len(val_dataset)} epochs")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=4,
        pin_memory=(config['device'] == 'cuda'), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4,
        pin_memory=(config['device'] == 'cuda')
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    n_epochs = config.pop('n_epochs')
    trainer  = EEGDINOTrainer(**config)
    trainer.train(train_loader, val_loader=val_loader,
                  save_dir='checkpoints')


if __name__ == "__main__":
    main()