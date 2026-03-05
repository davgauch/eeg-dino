"""Physiological Frequency Masking.

Operates on per-Hz PSD features from TFE: [B, n_tokens, C × n_freq_bins].
Flat index for channel c, bin f = c * n_freq_bins + f.  Bin f = (f + 1) Hz.

Strategies (all mask across ALL channels — no spatial confound):
  'delta'|'theta'|'alpha'|'beta'|'gamma' — fixed band, all channels
  'iaf'    — per-sample peak detection in 8-13 Hz, mask peak ± 2 Hz
  'random' — 20% random entries
"""
import torch


class PhysiologicalMasker:

    BAND_RANGES = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 50),
    }
    FREQ_MIN = 1

    def __init__(self, n_channels=2, n_freq_bins=50):
        self.n_channels = n_channels
        self.n_freq_bins = n_freq_bins

    def _hz_to_bin(self, hz):
        return round(hz) - self.FREQ_MIN

    def _bin_range(self, low_hz, high_hz):
        lo = max(0, self._hz_to_bin(low_hz))
        hi = min(self.n_freq_bins, self._hz_to_bin(high_hz))
        return lo, hi

    def _zero_bins_all_channels(self, raw_features, lo_bin, hi_bin):
        masked = raw_features.clone()
        for c in range(self.n_channels):
            off = c * self.n_freq_bins
            masked[:, :, off + lo_bin : off + hi_bin] = 0.0
        return masked

    def mask_band(self, raw_features, band='alpha'):
        low, high = self.BAND_RANGES[band]
        lo, hi = self._bin_range(low, high)
        masked = self._zero_bins_all_channels(raw_features, lo, hi)
        return masked, {'strategy': band, 'freq_range_hz': (low, high),
                        'bins_per_ch': hi - lo}

    def mask_iaf(self, raw_features, raw_signal,
                 sampling_rate=200, bandwidth=2):
        B = raw_features.shape[0]
        masked = raw_features.clone()
        peak_freqs = []

        for b in range(B):
            sig = raw_signal[b].float().mean(dim=0)
            spectrum = torch.fft.rfft(sig)
            psd = spectrum.abs().pow(2) / sig.shape[0]
            freqs = torch.linspace(0, sampling_rate / 2, psd.shape[0],
                                   device=sig.device)

            alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
            alpha_psd = psd.clone()
            alpha_psd[~alpha_mask] = -1.0
            peak_hz = freqs[alpha_psd.argmax()].item()
            peak_freqs.append(round(peak_hz, 2))

            lo_hz = max(peak_hz - bandwidth, self.FREQ_MIN)
            hi_hz = min(peak_hz + bandwidth + 1, self.FREQ_MIN + self.n_freq_bins)
            lo, hi = self._bin_range(lo_hz, hi_hz)
            for c in range(self.n_channels):
                off = c * self.n_freq_bins
                masked[b, :, off + lo : off + hi] = 0.0

        return masked, {'strategy': 'iaf', 'bandwidth': bandwidth,
                        'peak_freqs': peak_freqs}

    def random_baseline(self, raw_features, mask_ratio=0.20):
        mask = torch.rand_like(raw_features) < mask_ratio
        masked = raw_features.clone()
        masked[mask] = 0.0
        return masked, {'strategy': 'random', 'n_masked': mask.sum().item()}

    def apply_strategy(self, raw_features, strategy='alpha', **kwargs):
        if strategy in self.BAND_RANGES:
            return self.mask_band(raw_features, band=strategy)
        if strategy == 'iaf':
            return self.mask_iaf(raw_features, **kwargs)
        if strategy == 'random':
            return self.random_baseline(raw_features)
        raise ValueError(f"Unknown strategy: {strategy}")
