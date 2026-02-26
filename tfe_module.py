
"""
Time-Frequency Embedding (TFE)
"""
import torch
import torch.nn as nn

class TimeFrequencyEmbedding(nn.Module):
    """
    Tokenizes EEG: 1 token = 1 second across all channels
    """
    def __init__(self, n_channels=19, sampling_rate=200, embed_dim=200):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.samples_per_token = sampling_rate  # 1 second = 200 samples
        
        # Linear projection: flatten [channels × samples] → embed_dim
        input_dim = n_channels * self.samples_per_token  # 19 * 200 = 3800
        self.projection = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time_samples]
        Returns:
            tokens: [batch, n_tokens, embed_dim]
        """
        batch_size, n_channels, n_samples = x.shape
        n_tokens = n_samples // self.samples_per_token
        
        # Truncate to exact multiple of token size
        x = x[:, :, :n_tokens * self.samples_per_token]
        
        # Reshape: [batch, channels, n_tokens, samples_per_token]
        x = x.reshape(batch_size, n_channels, n_tokens, self.samples_per_token)
        x = x.permute(0, 2, 1, 3)  # [batch, n_tokens, channels, samples_per_token]
        
        # Flatten each token
        x = x.reshape(batch_size, n_tokens, -1)  # [batch, n_tokens, channels*samples]
        
        # Project to embedding dimension
        tokens = self.projection(x)
        
        return tokens