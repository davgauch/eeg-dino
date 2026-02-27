"""
Complete EEG-DINO Training Loop
Multi-GPU (2x) ready via DataParallel

Launch: CUDA_VISIBLE_DEVICES=0,2 python train.py   (GPU 1 is a GT 1030, unusable)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import os

from eeg_dino_model import StudentModel, TeacherModel
from channel_aware_sampling import ChannelAwareSampling
from losses import DINOLoss, PatchLoss


class EEGDataset(Dataset):
    """
    Dataset for EEG samples.
    Supports .edf (Sleep-EDF) and .gdf (BCI Competition) formats.
    Files are opened read-only and never written to.
    """
    def __init__(self, data_path=None, n_channels=19, sampling_rate=200, epoch_length=30):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.samples_per_epoch = sampling_rate * epoch_length

        if data_path is None:
            print("⚠️  Using synthetic data for testing")
            self.data = [
                torch.randn(n_channels, self.samples_per_epoch)
                for _ in range(1000)
            ]
        else:
            self.data = self.load_real_data(data_path)

    def load_real_data(self, data_path):
        from glob import glob
        import torch.nn.functional as F

        data = []

        pt_files  = glob(os.path.join(data_path, '**/*.pt'),  recursive=True)
        gdf_files = glob(os.path.join(data_path, '**/*.gdf'), recursive=True)

        # ── Sleep-EDF: pre-epoched .pt tensors, shape [2, 3000] ─────────────
        # 2 channels (Fpz-Cz, Pz-Oz), 3000 samples @ 100 Hz = 30 seconds.
        # We resample to self.sampling_rate (200 Hz) → [2, 6000],
        # then zero-pad to [n_channels, samples_per_epoch] = [19, 6000].
        if pt_files:
            print(f"Found {len(pt_files)} .pt files (Sleep-EDF) in {data_path}")
            src_rate   = 100
            tgt_rate   = self.sampling_rate                    # 200
            tgt_samples = self.samples_per_epoch               # 6000

            for file_path in pt_files:
                try:
                    # torch.load is read-only — the file is never modified
                    tensor = torch.load(file_path, map_location='cpu')  # [2, 3000]

                    # Resample: interpolate expects [batch, channels, time]
                    if src_rate != tgt_rate:
                        tensor = F.interpolate(
                            tensor.unsqueeze(0).float(),       # [1, 2, 3000]
                            size=tgt_samples,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0)                           # [2, 6000]

                    # Zero-pad channels: [2, 6000] → [19, 6000]
                    n_ch = tensor.shape[0]
                    if n_ch < self.n_channels:
                        pad = torch.zeros(self.n_channels - n_ch, tensor.shape[1])
                        tensor = torch.cat([tensor, pad], dim=0)

                    data.append(tensor)

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

        # ── BCI Competition: raw .gdf files, multiple channels @ 250 Hz ─────
        # BCI-IV 2a: 22 EEG channels + 3 EOG. BCI-IV 2b: 3 EEG channels.
        # MNE reads them, we resample to self.sampling_rate, take up to
        # n_channels EEG channels, then split into 30s epochs.
        if gdf_files:
            import mne
            print(f"Found {len(gdf_files)} .gdf files (BCI Competition) in {data_path}")

            for file_path in gdf_files:
                try:
                    # Read-only — never calls raw.save() or any write method
                    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

                    # Keep only EEG channels (drop EOG, stim, etc.)
                    raw.pick_types(eeg=True)

                    raw.resample(self.sampling_rate)

                    # Take up to n_channels channels
                    available = raw.ch_names[:self.n_channels]
                    raw.pick_channels(available)

                    epoch_data = raw.get_data()   # [n_ch, n_times]
                    n_ch = epoch_data.shape[0]

                    # Zero-pad channels if fewer than expected
                    if n_ch < self.n_channels:
                        pad = np.zeros((self.n_channels - n_ch, epoch_data.shape[1]))
                        epoch_data = np.concatenate([epoch_data, pad], axis=0)

                    # Split into non-overlapping 30s epochs
                    for start in range(0, epoch_data.shape[1] - self.samples_per_epoch,
                                       self.samples_per_epoch):
                        epoch = epoch_data[:, start:start + self.samples_per_epoch]
                        data.append(torch.FloatTensor(epoch))

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

        if not data:
            raise RuntimeError(f"No .pt or .gdf files found under {data_path}")

        print(f"Loaded {len(data)} epochs total")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EEGDINOTrainer:
    """
    Complete training pipeline for EEG-DINO.
    Implements Algorithm from Paper Section 2.2.

    DataParallel notes:
    - Teacher is built BEFORE student is wrapped, preserving self.teacher.model reference.
    - channel_indices is 1D and shared across the batch. It is expanded to [batch, n_ch]
      before every model call so DataParallel splits it correctly along dim=0.
      StudentModel.forward handles both 1D and 2D channel_indices.
    - EMA update and checkpoints always use _raw_student() to unwrap DataParallel.
    """
    def __init__(
        self,
        n_channels=19,
        sampling_rate=200,
        embed_dim=200,
        n_layers=12,
        n_heads=8,
        mlp_dim=512,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.04,
        teacher_momentum=0.996,
        device='cuda'
    ):
        self.device = device
        self.batch_size = batch_size
        self.teacher_momentum = teacher_momentum

        # ── Models ──────────────────────────────────────────────────────────
        print("Initializing student model...")
        self.student = StudentModel(
            n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim
        ).to(device)

        # Teacher MUST be built from the raw StudentModel BEFORE DataParallel
        # wraps it. Otherwise teacher.model is a DataParallel object and all
        # EMA parameter iterations and state_dict calls break.
        print("Initializing teacher model...")
        self.teacher = TeacherModel(self.student).to(device)

        # Now safe to wrap student for multi-GPU training
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs via DataParallel")
            self.student = nn.DataParallel(self.student)

        # ── Channel-aware sampling ───────────────────────────────────────────
        self.sampler = ChannelAwareSampling(n_channels, sampling_rate)

        # ── Loss functions ───────────────────────────────────────────────────
        self.signal_loss_fn = DINOLoss().to(device)
        self.patch_loss_fn  = PatchLoss().to(device)

        # ── Optimizer (uses raw student params, no DataParallel wrapper) ─────
        self.optimizer = torch.optim.AdamW(
            self._raw_student().parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # ── Cosine annealing scheduler ───────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        n_params = sum(p.numel() for p in self._raw_student().parameters())
        print(f"✓ Initialized EEG-DINO with {n_params / 1e6:.1f}M parameters")

    # ── Helper: safely unwrap DataParallel ───────────────────────────────────
    def _raw_student(self):
        """Returns the underlying StudentModel, stripping DataParallel if present."""
        if isinstance(self.student, nn.DataParallel):
            return self.student.module
        return self.student

    # ── Helper: expand 1D channel_indices to batch dim for DataParallel ──────
    @staticmethod
    def _batch_channel_indices(channel_indices, batch_size):
        """
        DataParallel splits all positional args along dim=0.
        channel_indices is 1D [n_ch] and identical for every sample in the batch.
        Expanding to [batch, n_ch] lets DataParallel split it safely along the
        batch dimension; each GPU replica receives [batch/n_gpu, n_ch].
        StudentModel.forward handles both 1D and 2D channel_indices.
        """
        return channel_indices.unsqueeze(0).expand(batch_size, -1).contiguous()

    # ── EMA update ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def update_teacher(self):
        """Exponential moving average update of teacher weights from student."""
        for param_s, param_t in zip(
            self._raw_student().parameters(),
            self.teacher.model.parameters()
        ):
            param_t.data = (
                param_t.data * self.teacher_momentum
                + param_s.data * (1.0 - self.teacher_momentum)
            )

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward_pass(self, x):
        """
        Build 12 channel-aware views, run student on all and teacher on globals.

        Returns:
            student_outputs:       signal features for all views
            teacher_outputs:       signal features for global views only
            student_patch_outputs: patch features for masked views only
            teacher_patch_outputs: patch features for global views
        """
        views = self.sampler(x)

        student_outputs       = {}
        student_patch_outputs = {}
        teacher_outputs       = {}
        teacher_patch_outputs = {}

        # Student processes ALL views
        for view_name, view_data in views.items():
            view_tensor     = view_data['view'].to(self.device)
            channel_indices = view_data['channels'].to(self.device)
            batch_size      = view_tensor.shape[0]

            # Expand 1D channel_indices → [batch, n_ch] for DataParallel safety
            ch_batched = self._batch_channel_indices(channel_indices, batch_size)

            if 'masked' in view_name:
                signal_feat, patch_feat = self.student(
                    view_tensor, ch_batched, return_patch=True
                )
                student_outputs[view_name]       = signal_feat
                student_patch_outputs[view_name] = patch_feat
            else:
                signal_feat = self.student(
                    view_tensor, ch_batched, return_patch=False
                )
                student_outputs[view_name] = signal_feat

        # Teacher processes ONLY the two global views
        for view_name in ['global_0', 'global_1']:
            view_tensor     = views[view_name]['view'].to(self.device)
            channel_indices = views[view_name]['channels'].to(self.device)
            batch_size      = view_tensor.shape[0]

            # Teacher is not wrapped in DataParallel, but we keep the same
            # interface so StudentModel.forward works identically in both.
            ch_batched = self._batch_channel_indices(channel_indices, batch_size)

            signal_feat, patch_feat = self.teacher(
                view_tensor, ch_batched, return_patch=True
            )
            teacher_outputs[view_name]       = signal_feat
            teacher_patch_outputs[view_name] = patch_feat

        return student_outputs, teacher_outputs, student_patch_outputs, teacher_patch_outputs

    # ── One epoch ────────────────────────────────────────────────────────────
    def train_epoch(self, dataloader, epoch):
        self.student.train()
        self.teacher.eval()

        total_loss        = 0.0
        total_signal_loss = 0.0
        total_patch_loss  = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for x in pbar:
            student_out, teacher_out, student_patch, teacher_patch = self.forward_pass(x)

            loss_signal, _ = self.signal_loss_fn(
                student_out, teacher_out, self.teacher.center
            )
            loss_patch = self.patch_loss_fn(
                student_patch, teacher_patch, self.teacher.center
            )
            loss = loss_signal + loss_patch

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EMA teacher update (uses _raw_student, DataParallel safe)
            self.update_teacher()

            # Update teacher centering vector
            with torch.no_grad():
                teacher_global = torch.cat([
                    teacher_out['global_0'],
                    teacher_out['global_1']
                ], dim=0)
                self.teacher.update_center(teacher_global)

            total_loss        += loss.item()
            total_signal_loss += loss_signal.item()
            total_patch_loss  += loss_patch.item()

            pbar.set_postfix({
                'loss':   f'{loss.item():.4f}',
                'signal': f'{loss_signal.item():.4f}',
                'patch':  f'{loss_patch.item():.4f}'
            })

        self.scheduler.step()

        n = len(dataloader)
        return {
            'loss':        total_loss        / n,
            'signal_loss': total_signal_loss / n,
            'patch_loss':  total_patch_loss  / n
        }

    # ── Checkpoint helpers ───────────────────────────────────────────────────
    def _build_checkpoint(self, epoch, loss):
        """
        Always saves raw (unwrapped) state dicts.
        Without this, DataParallel adds a 'module.' prefix to every key,
        making checkpoints impossible to load without DataParallel at inference.
        """
        return {
            'epoch':                epoch,
            'student_state_dict':   self._raw_student().state_dict(),
            'teacher_state_dict':   self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss':                 loss
        }

    # ── Full training loop ───────────────────────────────────────────────────
    def train(self, dataloader, n_epochs=100, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Starting EEG-DINO Pre-training")
        print(f"{'='*60}")
        print(f"Epochs:     {n_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device:     {self.device}")
        print(f"{'='*60}\n")

        best_loss = float('inf')

        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)

            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Total Loss:  {metrics['loss']:.4f}")
            print(f"  Signal Loss: {metrics['signal_loss']:.4f}")
            print(f"  Patch Loss:  {metrics['patch_loss']:.4f}")
            print(f"  LR:          {self.optimizer.param_groups[0]['lr']:.6f}")

            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save(
                    self._build_checkpoint(epoch, metrics['loss']),
                    os.path.join(save_dir, 'best_model.pth')
                )
                print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

            if epoch % 10 == 0:
                torch.save(
                    self._build_checkpoint(epoch, metrics['loss']),
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                )

        print("\n✓ Training complete!")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    config = {
        'n_channels':       19,
        'sampling_rate':    200,
        'embed_dim':        200,
        'n_layers':         12,
        'n_heads':          8,
        'mlp_dim':          512,
        'batch_size':       32,
        'learning_rate':    1e-4,
        'weight_decay':     0.04,
        'teacher_momentum': 0.996,
        'device':           'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # n_epochs is NOT a parameter of EEGDINOTrainer.__init__, keep it separate
    n_epochs = 100

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  n_epochs: {n_epochs}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_sleep = EEGDataset(
        data_path='/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/sleep_edf/',
        n_channels=config['n_channels'],
        sampling_rate=config['sampling_rate']
    )
    dataset_bci = EEGDataset(
        data_path='/net/inltitan2.epfl.ch/scratch2/tzhu/EEGPT/datasets/downstream/Raw_data/',
        n_channels=config['n_channels'],
        sampling_rate=config['sampling_rate']
    )
    dataset = ConcatDataset([dataset_sleep, dataset_bci])

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=(config['device'] == 'cuda')
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = EEGDINOTrainer(**config)
    trainer.train(dataloader, n_epochs=n_epochs)


if __name__ == "__main__":
    main()