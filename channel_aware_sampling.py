# channel_aware_sampling.py
"""
Channel-Aware Sampling
Creates 12 views: 2 global, 8 local, 2 masked
"""
import torch
import numpy as np

class ChannelAwareSampling:
    def __init__(self, n_channels=19, sampling_rate=200):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
    def create_global_view(self, x):
        """
        Global: 70% channels, 80% time
        """
        batch_size, n_channels, n_samples = x.shape
        
        n_keep_channels = int(0.7 * n_channels)
        n_keep_samples = int(0.8 * n_samples)
        
        # Random channel selection
        channel_idx = np.sort(np.random.choice(n_channels, n_keep_channels, replace=False))
        
        # Random temporal window
        start_sample = np.random.randint(0, n_samples - n_keep_samples + 1)
        
        view = x[:, channel_idx, start_sample:start_sample + n_keep_samples]
        return view, torch.LongTensor(channel_idx)
    
    def create_local_view(self, x):
        """
        Local: 30% channels, 50% time
        """
        batch_size, n_channels, n_samples = x.shape
        
        n_keep_channels = int(0.3 * n_channels)
        n_keep_samples = int(0.5 * n_samples)
        
        channel_idx = np.sort(np.random.choice(n_channels, n_keep_channels, replace=False))
        start_sample = np.random.randint(0, n_samples - n_keep_samples + 1)
        
        view = x[:, channel_idx, start_sample:start_sample + n_keep_samples]
        return view, torch.LongTensor(channel_idx)
    
    def create_masked_view(self, x):
        """
        Masked: Start with global, then mask 20% channels and 20% temporal patches
        """
        view, channel_idx = self.create_global_view(x)
        batch_size, n_channels, n_samples = view.shape
        
        # Channel masking (20%)
        n_mask_channels = max(1, int(0.2 * n_channels))
        mask_ch_idx = np.random.choice(n_channels, n_mask_channels, replace=False)
        view[:, mask_ch_idx, :] = 0
        
        # Temporal patch masking (20% of 1-second patches)
        patch_size = self.sampling_rate  # 1 second
        n_patches = n_samples // patch_size
        n_mask_patches = max(1, int(0.2 * n_patches))
        mask_patch_idx = np.random.choice(n_patches, n_mask_patches, replace=False)
        
        for p_idx in mask_patch_idx:
            start = p_idx * patch_size
            end = start + patch_size
            view[:, :, start:end] = 0
        
        return view, channel_idx
    
    def __call__(self, x):
        """
        Create 12 views per sample
        Returns dict: {'global_0': {'view': tensor, 'channels': indices}, ...}
        """
        views = {}
        
        # 2 global
        for i in range(2):
            view, ch = self.create_global_view(x)
            views[f'global_{i}'] = {'view': view, 'channels': ch}
        
        # 8 local
        for i in range(8):
            view, ch = self.create_local_view(x)
            views[f'local_{i}'] = {'view': view, 'channels': ch}
        
        # 2 masked
        for i in range(2):
            view, ch = self.create_masked_view(x)
            views[f'masked_{i}'] = {'view': view, 'channels': ch}
        
        return views
