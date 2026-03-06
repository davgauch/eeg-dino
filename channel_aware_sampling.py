"""Channel-Aware Sampling — multi-scale view generation for DINO.

Global views: ceil(70%) channels, 80% temporal window.
Local views:  floor(30%) channels, 50% temporal window.
Masked views: global crop + bandstop filter on a physiological band.
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
    """Parse a mask strategy string into a list of band names.

    Examples:
        'alpha'          → ['alpha']
        'alpha+beta'     → ['alpha', 'beta']
        'all'            → ['delta', 'theta', 'alpha', 'beta', 'gamma']
        'random'         → 'random'
        'none'           → 'none'
    """
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
                 mask_strategy='alpha'):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_local_views = n_local_views
        self.n_masked_views = n_masked_views
        self.mask_strategy_raw = mask_strategy
        self.mask_strategy = parse_mask_strategy(mask_strategy)

    def _random_crop(self, x, ch_frac, time_frac, ceil_ch=False):
        B, C, T = x.shape
        n_t = int(time_frac * T)
        t_start = np.random.randint(0, T - n_t + 1)
        C_eff = min(C, self.n_channels)
        n_ch = max(1, math.ceil(ch_frac * C_eff) if ceil_ch else int(ch_frac * C_eff))
        n_ch = min(n_ch, C_eff)
        ch_idx = np.sort(np.random.choice(C_eff, n_ch, replace=False))
        return x[:, ch_idx, t_start:t_start + n_t], torch.LongTensor(ch_idx)

    def _bandstop(self, x, low_hz, high_hz):
        """Zero frequency bins in [low_hz, high_hz) via FFT."""
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate)
        mask = (freqs >= low_hz) & (freqs < high_hz)
        spectrum[:, :, mask] = 0
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)

    def _random_bandstop(self, x, ratio=0.20):
        """Zero a random 20% of frequency bins via FFT."""
        spectrum = torch.fft.rfft(x, dim=-1)
        n_bins = spectrum.shape[-1]
        mask = torch.rand(n_bins) < ratio
        spectrum[:, :, mask] = 0
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)

    def _spatiotemporal_mask(self, view, ch_ratio=0.5, time_patch_ratio=0.15):
        """Paper-style masking: zero random channels + random temporal patches."""
        B, C, T = view.shape
        # Zero random channels
        n_zero_ch = max(1, int(ch_ratio * C))
        if n_zero_ch < C:
            zero_ch = np.random.choice(C, n_zero_ch, replace=False)
            view = view.clone()
            view[:, zero_ch, :] = 0
        # Zero random 1-second temporal patches
        patch_len = self.sampling_rate  # 1 second = 200 samples
        n_patches = T // patch_len
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
            view = self._random_bandstop(view)
        elif self.mask_strategy == 'spatiotemporal':
            view = self._spatiotemporal_mask(view)
        # 'none' → no filtering
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
