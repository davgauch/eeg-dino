"""
Decoupled Positional Embedding (DPE)
Equation 1: Embed(X) = Pc + Pt + E
"""
import torch
import torch.nn as nn

class DecoupledPositionalEmbedding(nn.Module):
    def __init__(self, n_channels=19, embed_dim=200):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        # Spatial (channel) embedding - learnable
        self.channel_embedding = nn.Embedding(n_channels, embed_dim)
        
        # Temporal embedding - 1D conv along temporal axis
        self.temporal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, tokens, channel_indices=None):
        """
        Args:
            tokens: [batch, n_tokens, embed_dim] (E in equation)
            channel_indices: [n_channels_present] - which channels are in this view
        Returns:
            embedded: [batch, n_tokens, embed_dim] (Pc + Pt + E)
        """
        batch_size, n_tokens, _ = tokens.shape
        
        # Default: all channels present
        if channel_indices is None:
            channel_indices = torch.arange(self.n_channels, device=tokens.device)
        
        # Pc: Spatial embedding (average over present channels)
        spatial_embed = self.channel_embedding(channel_indices).mean(dim=0)  # [embed_dim]
        spatial_embed = spatial_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
        spatial_embed = spatial_embed.expand(batch_size, n_tokens, -1)
        
        # Pt: Temporal embedding
        # Create temporal input: [batch, 1, n_tokens]
        temporal_input = torch.arange(n_tokens, device=tokens.device).float()
        temporal_input = temporal_input.unsqueeze(0).unsqueeze(0)  # [1, 1, n_tokens]
        temporal_input = temporal_input.expand(batch_size, -1, -1)
        
        temporal_embed = self.temporal_conv(temporal_input)  # [batch, embed_dim, n_tokens]
        temporal_embed = temporal_embed.permute(0, 2, 1)  # [batch, n_tokens, embed_dim]
        
        # Equation 1: Embed(X) = Pc + Pt + E
        embedded = tokens + spatial_embed + temporal_embed
        
        return embedded
