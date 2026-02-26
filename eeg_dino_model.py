"""
Complete EEG-DINO model with teacher-student architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tfe_module import TimeFrequencyEmbedding
from dpe_module import DecoupledPositionalEmbedding

class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG-DINO
    Table 1: EEG-DINO-S has 12 layers, hidden=200
    """
    def __init__(self, embed_dim=200, n_layers=12, n_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm like in modern transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, n_tokens, embed_dim]
        Returns:
            cls_token: [batch, embed_dim]
            patch_tokens: [batch, n_tokens, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Split CLS and patches
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        return cls_token, patch_tokens


class StudentModel(nn.Module):
    """
    Student model for EEG-DINO
    """
    def __init__(self, n_channels=19, sampling_rate=200, embed_dim=200, 
                 n_layers=12, n_heads=8, mlp_dim=512):
        super().__init__()
        
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        
        # Signal-level projection head
        self.signal_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 256)
        )
        
        # Patch-level projection head
        self.patch_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 256)
        )
        
    def forward(self, x, channel_indices=None, return_patch=False):
        """
        Args:
            x: [batch, channels, samples]
            channel_indices: which channels are present
            return_patch: whether to return patch tokens
        Returns:
            signal_features: [batch, 256]
            patch_features: [batch, n_tokens, 256] (if return_patch=True)
        """
        # Pad to full channel count if a subset was selected
        if channel_indices is not None and x.shape[1] < self.tfe.n_channels:
            full_x = torch.zeros(x.shape[0], self.tfe.n_channels, x.shape[2],
                                 device=x.device, dtype=x.dtype)
            full_x[:, channel_indices, :] = x
            x = full_x

        # TFE: raw signal → tokens
        tokens = self.tfe(x)
        
        # DPE: add positional embeddings
        tokens = self.dpe(tokens, channel_indices)
        
        # Transformer
        cls_token, patch_tokens = self.transformer(tokens)
        
        # Project
        signal_features = self.signal_head(cls_token)
        
        if return_patch:
            patch_features = self.patch_head(patch_tokens)
            return signal_features, patch_features
        
        return signal_features


class TeacherModel(nn.Module):
    """
    Teacher model (EMA of student)
    """
    def __init__(self, student):
        super().__init__()
        self.model = StudentModel(
            n_channels=student.tfe.n_channels,
            sampling_rate=student.tfe.sampling_rate,
            embed_dim=student.transformer.cls_token.shape[-1],
            n_layers=len(student.transformer.transformer.layers),
            n_heads=student.transformer.transformer.layers[0].self_attn.num_heads,
            mlp_dim=student.signal_head[0].out_features
        )
        
        # Copy student weights
        self.model.load_state_dict(student.state_dict())
        
        # Freeze teacher
        for param in self.parameters():
            param.requires_grad = False
        
        # Center for teacher outputs (Equation 2 - centering)
        self.register_buffer('center', torch.zeros(1, 256))
        self.center_momentum = 0.9
    
    @torch.no_grad()
    def forward(self, x, channel_indices=None, return_patch=False):
        return self.model(x, channel_indices, return_patch)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        EMA update of center
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)
