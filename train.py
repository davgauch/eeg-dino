"""
Complete EEG-DINO Training Loop
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
# import mne
# from glob import glob

from eeg_dino_model import StudentModel, TeacherModel
from channel_aware_sampling import ChannelAwareSampling
from losses import DINOLoss, PatchLoss

class EEGDataset(Dataset):
    """
    Simple dataset for EEG samples
    """
    def __init__(self, data_path=None, n_channels=19, sampling_rate=200, epoch_length=30):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.samples_per_epoch = sampling_rate * epoch_length
        
        # For testing: generate synthetic data
        # Replace with real data loading
        if data_path is None:
            print("⚠️  Using synthetic data for testing")
            self.data = [
                torch.randn(n_channels, self.samples_per_epoch) 
                for _ in range(1000)
            ]
        else:
            self.data = self.load_real_data(data_path)
    
    def load_real_data(self, data_path):
        """
        Load TUEG dataset
        """
    
        # data = []
        # edf_files = glob(os.path.join(data_path, '**/*.edf'), recursive=True)
        
        # for edf_file in edf_files[:100]:  # Start with 100 files
        #     try:
        #         raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        #         raw.resample(200)  # Resample to 200Hz
                
        #         # Select 19 channels
        #         channel_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        #                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        #                     'FZ', 'CZ', 'PZ']
        #         raw.pick_channels(channel_names, ordered=True)
                
        #         # Split into 30-second epochs
        #         epoch_data = raw.get_data()
        #         n_samples = epoch_data.shape[1]
        #         epoch_length = 30 * 200  # 30 seconds
                
        #         for start in range(0, n_samples - epoch_length, epoch_length):
        #             epoch = epoch_data[:, start:start + epoch_length]
        #             data.append(torch.FloatTensor(epoch))
                    
        #     except Exception as e:
        #         print(f"Error loading {edf_file}: {e}")
        #         continue
        
        # return data
        raise NotImplementedError("Implement real data loading here")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class EEGDINOTrainer:
    """
    Complete training pipeline for EEG-DINO
    Implements Algorithm from Paper Section 2.2
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
        
        # Models
        print("Initializing student model...")
        self.student = StudentModel(
            n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim
        ).to(device)
        
        print("Initializing teacher model...")
        self.teacher = TeacherModel(self.student).to(device)
        
        # Channel-aware sampling
        self.sampler = ChannelAwareSampling(n_channels, sampling_rate)
        
        # Loss functions
        self.signal_loss_fn = DINOLoss().to(device)
        self.patch_loss_fn = PatchLoss().to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler: cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        print(f"✓ Initialized EEG-DINO with {sum(p.numel() for p in self.student.parameters())/1e6:.1f}M parameters")
    
    @torch.no_grad()
    def update_teacher(self):
        """
        EMA update of teacher
        """
        for param_s, param_t in zip(self.student.parameters(), self.teacher.model.parameters()):
            param_t.data = param_t.data * self.teacher_momentum + \
                          param_s.data * (1 - self.teacher_momentum)
    
    def forward_pass(self, x):
        """
        Forward pass through multi-view pipeline
        
        Returns:
            student_outputs: dict of signal features for all views
            teacher_outputs: dict of signal features for global views
            student_patch_outputs: dict of patch features for masked views
            teacher_patch_outputs: dict of patch features for global views
        """
        # Create 12 views
        views = self.sampler(x)
        
        student_outputs = {}
        student_patch_outputs = {}
        teacher_outputs = {}
        teacher_patch_outputs = {}
        
        # Student: process ALL views
        for view_name, view_data in views.items():
            view_tensor = view_data['view'].to(self.device)
            channel_indices = view_data['channels'].to(self.device)
            
            if 'masked' in view_name:
                # Masked views: need patch tokens
                signal_feat, patch_feat = self.student(
                    view_tensor, channel_indices, return_patch=True
                )
                student_outputs[view_name] = signal_feat
                student_patch_outputs[view_name] = patch_feat
            else:
                # Global/Local views: only signal features
                signal_feat = self.student(view_tensor, channel_indices, return_patch=False)
                student_outputs[view_name] = signal_feat
        
        # Teacher: process ONLY global views
        for view_name in ['global_0', 'global_1']:
            view_tensor = views[view_name]['view'].to(self.device)
            channel_indices = views[view_name]['channels'].to(self.device)
            
            signal_feat, patch_feat = self.teacher(
                view_tensor, channel_indices, return_patch=True
            )
            teacher_outputs[view_name] = signal_feat
            teacher_patch_outputs[view_name] = patch_feat
        
        return student_outputs, teacher_outputs, student_patch_outputs, teacher_patch_outputs
    
    def train_epoch(self, dataloader, epoch):
        """
        Train one epoch
        """
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0
        total_signal_loss = 0
        total_patch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, x in enumerate(pbar):
            # Forward pass
            student_out, teacher_out, student_patch, teacher_patch = self.forward_pass(x)
            
            # Compute losses
            loss_signal, loss_dict = self.signal_loss_fn(
                student_out, teacher_out, self.teacher.center
            )
            
            loss_patch = self.patch_loss_fn(
                student_patch, teacher_patch, self.teacher.center
            )
            
            # Total loss (
            loss = loss_signal + loss_patch
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update teacher (EMA)
            self.update_teacher()
            
            # Update teacher center
            with torch.no_grad():
                teacher_global_outputs = torch.cat([
                    teacher_out['global_0'], 
                    teacher_out['global_1']
                ], dim=0)
                self.teacher.update_center(teacher_global_outputs)
            
            # Logging
            total_loss += loss.item()
            total_signal_loss += loss_signal.item()
            total_patch_loss += loss_patch.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'signal': f'{loss_signal.item():.4f}',
                'patch': f'{loss_patch.item():.4f}'
            })
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(dataloader),
            'signal_loss': total_signal_loss / len(dataloader),
            'patch_loss': total_patch_loss / len(dataloader)
        }
    
    def train(self, dataloader, n_epochs=100, save_dir='checkpoints'):
        """
        Complete training loop
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting EEG-DINO Pre-training")
        print(f"{'='*60}")
        print(f"Epochs: {n_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        best_loss = float('inf')
        
        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)
            
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Total Loss: {metrics['loss']:.4f}")
            print(f"  Signal Loss: {metrics['signal_loss']:.4f}")
            print(f"  Patch Loss: {metrics['patch_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'student_state_dict': self.student.state_dict(),
                    'teacher_state_dict': self.teacher.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': metrics['loss']
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'student_state_dict': self.student.state_dict(),
                    'teacher_state_dict': self.teacher.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': metrics['loss']
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print("\n✓ Training complete!")

# launch with python train.py --data_path /path/to/TUEG/dataset
def main():
    """
    Main training script
    """
    # Configuration 
    config = {
        'n_channels': 19,
        'sampling_rate': 200,
        'embed_dim': 200,      # Hidden Size 
        'n_layers': 12,       
        'n_heads': 8,
        'mlp_dim': 512,        # MLP Size
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.04,
        'n_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Dataset
    dataset = EEGDataset(
        data_path=None,  # Use synthetic data for testing
        n_channels=config['n_channels'],
        sampling_rate=config['sampling_rate']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Trainer
    trainer = EEGDINOTrainer(**config)
    
    # Train
    trainer.train(dataloader, n_epochs=config['n_epochs'])


if __name__ == "__main__":
    main()