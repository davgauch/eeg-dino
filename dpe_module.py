"""Decoupled Positional Embedding — Embed(X) = Pc + Pt + E."""
import torch
import torch.nn as nn


class DecoupledPositionalEmbedding(nn.Module):
    """Adds learnable spatial (channel) and temporal (conv) embeddings."""

    def __init__(self, n_channels=2, embed_dim=64):
        super().__init__()
        self.n_channels = n_channels
        self.channel_embedding = nn.Embedding(n_channels, embed_dim)
        # Channel-wise 1D conv on token embeddings (input-dependent)
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3,
                                       padding=1, groups=1)

    def forward(self, tokens, channel_indices=None):
        # tokens: [B, T, D]
        B, T, D = tokens.shape

        if channel_indices is None:
            channel_indices = torch.arange(self.n_channels, device=tokens.device)

        # Pc: average spatial embedding over present channels
        channel_indices = channel_indices.clamp(0, self.n_channels - 1)
        if channel_indices.dim() == 2:
            spatial = self.channel_embedding(channel_indices).mean(dim=1)
            spatial = spatial.unsqueeze(1).expand(B, T, -1)
        else:
            spatial = self.channel_embedding(channel_indices).mean(dim=0)
            spatial = spatial.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        # Pt: dynamic temporal embedding via 1D conv on E itself
        temporal = self.temporal_conv(tokens.transpose(1, 2)).transpose(1, 2)

        return tokens + spatial + temporal
