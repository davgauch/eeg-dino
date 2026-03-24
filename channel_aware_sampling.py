"""Channel-Aware Sampling — multi-scale view generation for DINO.

Global views: ceil(70%) channels, 80% temporal window.
Local views:  floor(30%) channels, 50% temporal window.
Masked views: global crop + bandstop filter on a physiological band.

Adaptive channel sampling:
- Low-channel datasets (≤5): Enable channel subsampling for robustness
- High-channel datasets (>5): Disable channel subsampling, preserve spatial info
"""
import math
import torch
import numpy as np


BAND_RANGES = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

ALL_BANDS = list(BAND_RANGES.keys())


def parse_mask_strategy(strategy: str):
    """Parse mask strategy: 'alpha', 'alpha+beta', 'all', 'random', 'none', 'spatiotemporal'."""
    if strategy in ('none', 'random', 'spatiotemporal'):
        return strategy
    bands = [b.strip() for b in strategy.split('+')]
    if bands == ['all']:
        return ALL_BANDS
    for b in bands:
        if b not in BAND_RANGES:
            raise ValueError(
                f"Unknown band '{b}'. Choose from {ALL_BANDS} "
                f"or combine with '+' (e.g. 'alpha+beta'), or use "
                f"'random'/'none'/'spatiotemporal'/'all'.")
    return bands


class ChannelAwareSampling:

    def __init__(self, n_channels=2, sampling_rate=200,
                 n_local_views=4, n_masked_views=1,
                 mask_strategy='alpha',
                 force_channel_sampling=None):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_local_views = n_local_views
        self.n_masked_views = n_masked_views
        self.mask_strategy_raw = mask_strategy
        self.mask_strategy = parse_mask_strategy(mask_strategy)
        
        if force_channel_sampling is not None:
            self.use_channel_sampling = force_channel_sampling
            override_msg = "FORCED by user"
        else:
            self.use_channel_sampling = (n_channels <= 5)
            override_msg = "auto-detected"
        
        if self.use_channel_sampling:
            print(f"[ChannelAwareSampling] Channel subsampling ENABLED "
                  f"({override_msg}, n_channels={n_channels} ≤ 5)")
            print(f"  → Global views: ~70% channels, Local views: ~30% channels")
        else:
            print(f"[ChannelAwareSampling] Channel subsampling DISABLED "
                  f"({override_msg}, n_channels={n_channels} > 5)")
            print(f"  → All views use all {n_channels} channels (only temporal cropping)")

    def _random_crop(self, x, ch_frac, time_frac, ceil_ch=False):
        B, C, T = x.shape
        
        # Temporal cropping
        n_t = int(time_frac * T)
        t_start = np.random.randint(0, T - n_t + 1)
        
        # Channel selection
        if self.use_channel_sampling:
            C_eff = min(C, self.n_channels)
            n_ch = max(1, math.ceil(ch_frac * C_eff) if ceil_ch else int(ch_frac * C_eff))
            n_ch = min(n_ch, C_eff)
            ch_idx = np.sort(np.random.choice(C_eff, n_ch, replace=False))
        else:
            ch_idx = np.arange(C) # keep all channels if subsampling is disabled
        
        return x[:, ch_idx, t_start:t_start + n_t], torch.LongTensor(ch_idx)

    def _bandstop(self, x, low_hz, high_hz):
        """Zero frequency bins in [low_hz, high_hz) via FFT."""
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate)
        mask = (freqs >= low_hz) & (freqs < high_hz)
        spectrum[:, :, mask] = 0 # zero out selected frequency bins
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1) # convert back to time domain

    def _random_bandstop_constrained(self, x, ratio=0.08):
        """Zero random frequency bins within physiological range (1-50 Hz).
        
        This ensures fair comparison with targeted masking strategies that
        operate within specific frequency bands.
        """
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate)
        
        # Constrain to physiological range (1-50 Hz, matching TFE)
        phys_mask = (freqs >= 1) & (freqs <= 50)
        phys_bins = torch.where(phys_mask)[0]
        
        # Randomly select ratio% of bins in this range
        n_mask = max(1, int(ratio * len(phys_bins)))
        
        # Create mask for each sample in batch (different random selection per sample)
        for b in range(x.shape[0]):
            mask_indices = phys_bins[torch.randperm(len(phys_bins))[:n_mask]]
            spectrum[b, :, mask_indices] = 0
        
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)

    def _spatiotemporal_mask(self, view, ch_ratio=0.5, time_patch_ratio=0.15):
        """Zero random channels (if enabled) + random temporal patches."""
        B, C, T = view.shape
        view = view.clone()
        
        if self.use_channel_sampling:
            n_zero_ch = max(1, int(ch_ratio * C))
            if n_zero_ch < C:
                zero_ch = np.random.choice(C, n_zero_ch, replace=False)
                view[:, zero_ch, :] = 0
        
        patch_len = self.sampling_rate
        n_patches = T // patch_len
        if n_patches > 0:
            n_zero_patches = max(1, int(time_patch_ratio * n_patches))
            zero_patches = np.random.choice(n_patches, n_zero_patches, replace=False)
            for p in zero_patches:
                view[:, :, p * patch_len:(p + 1) * patch_len] = 0
        
        return view

    def create_global_view(self, x):
        return self._random_crop(x, 0.7, 0.8, ceil_ch=True)

    def create_local_view(self, x):
        return self._random_crop(x, 0.3, 0.5, ceil_ch=False)

    def create_masked_view(self, x):
        view, ch_idx = self.create_global_view(x)
        
        if isinstance(self.mask_strategy, list):
            for band in self.mask_strategy:
                low, high = BAND_RANGES[band]
                view = self._bandstop(view, low, high)
        elif self.mask_strategy == 'random':
            view = self._random_bandstop_constrained(view, ratio=0.08)
        elif self.mask_strategy == 'spatiotemporal':
            view = self._spatiotemporal_mask(view)
        
        return view, ch_idx

    def __call__(self, x):
        views = {}
        
        for i in range(2):
            v, ch = self.create_global_view(x)
            views[f'global_{i}'] = {'view': v, 'channels': ch}
        
        for i in range(self.n_local_views):
            v, ch = self.create_local_view(x)
            views[f'local_{i}'] = {'view': v, 'channels': ch}
        
        for i in range(self.n_masked_views):
            v, ch = self.create_masked_view(x)
            views[f'masked_{i}'] = {'view': v, 'channels': ch}
        
        return views