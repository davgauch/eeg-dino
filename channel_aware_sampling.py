"""Channel-Aware Sampling with Adaptive Peak-Based Frequency Masking.

Global views: ceil(70%) channels, 80% temporal window.
Local views:  floor(30%) channels, 50% temporal window.
Masked views: global crop + adaptive bandstop around detected spectral peaks.

Masking strategies:
- 'theta', 'alpha', etc.: Peak-based masking (finds individual peak, masks around it)
- 'spatiotemporal': Random channel + temporal patch masking (baseline)
- 'none': No masking (ablation)
"""
import math
import torch
import numpy as np


# Peak detection configuration: search range and masking bandwidth
BAND_CONFIGS = {
    'delta': {'search': (1, 4),   'bandwidth': 1.0},
    'theta': {'search': (4, 8),   'bandwidth': 1.5},
    'alpha': {'search': (7, 14),  'bandwidth': 2.0},
    'beta':  {'search': (13, 35), 'bandwidth': 3.0},
    'gamma': {'search': (30, 50), 'bandwidth': 5.0},
}


def parse_mask_strategy(strategy: str):
    """Parse masking strategy.
    
    Options:
    - 'theta', 'alpha', etc.: Adaptive peak-based masking
    - 'spatiotemporal': Baseline random masking
    - 'none': No frequency masking
    """
    if strategy in ('none', 'spatiotemporal'):
        return strategy
    
    if strategy in BAND_CONFIGS:
        return [strategy]
    
    # Allow combinations like 'alpha+beta'
    bands = [b.strip() for b in strategy.split('+')]
    for b in bands:
        if b not in BAND_CONFIGS:
            raise ValueError(
                f"Unknown band '{b}'. Choose from {list(BAND_CONFIGS.keys())} "
                f"or use 'spatiotemporal' or 'none'.")
    return bands


class ChannelAwareSampling:

    def __init__(self, n_channels=2, sampling_rate=200,
                 n_local_views=4, n_masked_views=1,
                 mask_strategy='theta',
                 force_channel_sampling=None):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_local_views = n_local_views
        self.n_masked_views = n_masked_views
        self.mask_strategy_raw = mask_strategy
        self.mask_strategy = parse_mask_strategy(mask_strategy)
        
        # Adaptive channel sampling: enabled for low-channel datasets
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
        """Random spatial and temporal cropping."""
        B, C, T = x.shape
        
        # Temporal cropping
        n_t = int(time_frac * T)
        t_start = np.random.randint(0, T - n_t + 1)
        
        # Channel selection (if enabled)
        if self.use_channel_sampling:
            C_eff = min(C, self.n_channels)
            n_ch = max(1, math.ceil(ch_frac * C_eff) if ceil_ch else int(ch_frac * C_eff))
            n_ch = min(n_ch, C_eff)
            ch_idx = np.sort(np.random.choice(C_eff, n_ch, replace=False))
        else:
            ch_idx = np.arange(C)
        
        return x[:, ch_idx, t_start:t_start + n_t], torch.LongTensor(ch_idx)

    def _find_peak_frequency(self, x, search_low, search_high):
        """Find frequency with maximum power in the search range.
        
        Args:
            x: Input signal [B, C, T]
            search_low: Lower bound of search range (Hz)
            search_high: Upper bound of search range (Hz)
            
        Returns:
            peak_freqs: Detected peak frequencies per sample [B]
        """
        # Compute power spectrum via FFT
        spectrum = torch.fft.rfft(x, dim=-1)
        power = torch.abs(spectrum) ** 2
        
        # Average power across channels
        power_avg = power.mean(dim=1)  # [B, freq_bins]
        
        # Get frequency axis
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate)
        
        # Mask for search range
        search_mask = (freqs >= search_low) & (freqs < search_high)
        
        # Find peak within search range for each sample
        peak_freqs = []
        for b in range(x.shape[0]):
            power_in_range = power_avg[b, search_mask]
            if len(power_in_range) == 0:
                # Fallback to center if no frequencies in range
                peak_freqs.append((search_low + search_high) / 2)
            else:
                # Get frequency with max power
                local_peak_idx = torch.argmax(power_in_range)
                freqs_in_range = freqs[search_mask]
                peak_freqs.append(freqs_in_range[local_peak_idx].item())
        
        return torch.tensor(peak_freqs)
    
    def _adaptive_bandstop(self, x, peak_freqs, bandwidth):
        """Zero out frequencies around individual peak frequencies.
        
        Args:
            x: Input signal [B, C, T]
            peak_freqs: Peak frequency per sample [B]
            bandwidth: Half-width of notch filter (Hz)
            
        Returns:
            Filtered signal [B, C, T]
        """
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate)
        
        # Create per-sample masks
        for b in range(x.shape[0]):
            peak = peak_freqs[b].item()
            low = peak - bandwidth
            high = peak + bandwidth
            mask = (freqs >= low) & (freqs < high)
            spectrum[b, :, mask] = 0
        
        return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)

    def _spatiotemporal_mask(self, view, ch_ratio=0.5, time_patch_ratio=0.15):
        """Baseline spatiotemporal masking: random channels + temporal patches.
        
        Args:
            view: Input view [B, C, T]
            ch_ratio: Fraction of channels to zero (if subsampling enabled)
            time_patch_ratio: Fraction of 1-second patches to zero
            
        Returns:
            Masked view [B, C, T]
        """
        B, C, T = view.shape
        view = view.clone()
        
        # Zero random channels (if channel subsampling is enabled)
        if self.use_channel_sampling:
            n_zero_ch = max(1, int(ch_ratio * C))
            if n_zero_ch < C:
                zero_ch = np.random.choice(C, n_zero_ch, replace=False)
                view[:, zero_ch, :] = 0
        
        # Zero random 1-second temporal patches
        patch_len = self.sampling_rate
        n_patches = T // patch_len
        if n_patches > 0:
            n_zero_patches = max(1, int(time_patch_ratio * n_patches))
            zero_patches = np.random.choice(n_patches, n_zero_patches, replace=False)
            for p in zero_patches:
                view[:, :, p * patch_len:(p + 1) * patch_len] = 0
        
        return view

    def create_global_view(self, x):
        """70% channels, 80% temporal duration."""
        return self._random_crop(x, 0.7, 0.8, ceil_ch=True)

    def create_local_view(self, x):
        """30% channels, 50% temporal duration."""
        return self._random_crop(x, 0.3, 0.5, ceil_ch=False)

    def create_masked_view(self, x):
        """Global crop + frequency-domain masking."""
        view, ch_idx = self.create_global_view(x)
        
        if isinstance(self.mask_strategy, list):
            # Adaptive peak-based masking
            for band in self.mask_strategy:
                if band in BAND_CONFIGS:
                    config = BAND_CONFIGS[band]
                    search_low, search_high = config['search']
                    bandwidth = config['bandwidth']
                    
                    # Find individual spectral peaks
                    peak_freqs = self._find_peak_frequency(
                        view, search_low, search_high)
                    
                    # Mask around detected peaks
                    view = self._adaptive_bandstop(view, peak_freqs, bandwidth)
        
        elif self.mask_strategy == 'spatiotemporal':
            # Baseline: random spatiotemporal masking
            view = self._spatiotemporal_mask(view)
        
        # 'none' strategy: no masking, return view as-is
        
        return view, ch_idx

    def __call__(self, x):
        """Generate multi-scale views for DINO.
        
        Returns:
            Dictionary with keys: 'global_0', 'global_1', 'local_0'...'local_3', 'masked_0'
        """
        views = {}
        
        # 2 global views (for teacher)
        for i in range(2):
            v, ch = self.create_global_view(x)
            views[f'global_{i}'] = {'view': v, 'channels': ch}
        
        # 4 local views (for student)
        for i in range(self.n_local_views):
            v, ch = self.create_local_view(x)
            views[f'local_{i}'] = {'view': v, 'channels': ch}
        
        # 1 masked view (for reconstruction)
        for i in range(self.n_masked_views):
            v, ch = self.create_masked_view(x)
            views[f'masked_{i}'] = {'view': v, 'channels': ch}
        
        return views