"""Time-Frequency Embedding — tokenizes EEG into 1-second windows."""
import torch.nn as nn


class TimeFrequencyEmbedding(nn.Module):
    """Each token = 1 second of all channels, linearly projected."""

    def __init__(self, n_channels=2, sampling_rate=200, embed_dim=64):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.samples_per_token = sampling_rate
        self.projection = nn.Linear(n_channels * sampling_rate, embed_dim)

    def forward(self, x):
        # x: [B, C, T] → [B, n_tokens, embed_dim]
        B, C, T = x.shape
        n_tok = T // self.samples_per_token
        x = x[:, :, :n_tok * self.samples_per_token]
        x = x.reshape(B, C, n_tok, self.samples_per_token)
        x = x.permute(0, 2, 1, 3).reshape(B, n_tok, -1)
        return self.projection(x)
