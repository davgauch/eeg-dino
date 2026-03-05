"""Time-Frequency Embedding (TFE) — per-Hz PSD tokenizer.

Input:  [B, C, T]
Output: tokens       [B, n_tokens, embed_dim]
        raw_features [B, n_tokens, C × 50]  (log-PSD at 1..50 Hz)
"""
import torch
import torch.nn as nn


class TimeFrequencyEmbedding(nn.Module):

    FREQ_MIN = 1
    FREQ_MAX = 50
    BAND_RANGES = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 50),
    }

    def __init__(self, n_channels=2, sampling_rate=200, embed_dim=64):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.samples_per_token = sampling_rate
        self.n_freq_bins = self.FREQ_MAX - self.FREQ_MIN + 1

        n_rfft = self.samples_per_token // 2 + 1
        select = torch.zeros(n_rfft, dtype=torch.bool)
        select[self.FREQ_MIN : self.FREQ_MAX + 1] = True
        self.register_buffer('_bin_select', select)

        self.projection = nn.Linear(n_channels * self.n_freq_bins, embed_dim)

    def extract_psd(self, segment):
        window = torch.hann_window(segment.shape[-1], device=segment.device)
        spectrum = torch.fft.rfft(segment * window, dim=-1)
        psd = (spectrum.real ** 2 + spectrum.imag ** 2) / segment.shape[-1]
        return torch.log1p(psd[:, :, self._bin_select])

    def forward(self, x):
        B, C, T = x.shape
        n_tok = T // self.samples_per_token
        x = x[:, :, :n_tok * self.samples_per_token]
        x = x.reshape(B, C, n_tok, self.samples_per_token).permute(0, 2, 1, 3)
        flat = x.reshape(B * n_tok, C, self.samples_per_token)
        psd = self.extract_psd(flat)
        raw_features = psd.reshape(B, n_tok, C * self.n_freq_bins)
        return self.projection(raw_features), raw_features
