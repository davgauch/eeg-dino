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
    Transformer encoder for EEG-DINO.
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
            norm_first=True  # Pre-norm like modern transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, n_tokens, embed_dim]
        Returns:
            cls_token:    [batch, embed_dim]
            patch_tokens: [batch, n_tokens, embed_dim]
        """
        batch_size = x.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Split CLS and patch tokens
        cls_token    = x[:, 0]
        patch_tokens = x[:, 1:]

        return cls_token, patch_tokens


class StudentModel(nn.Module):
    """
    Student model for EEG-DINO.

    channel_indices handling:
    - When running without DataParallel: 1D tensor of shape [n_selected_ch]
    - When running under DataParallel:   2D tensor of shape [batch_slice, n_selected_ch]
      (train.py expands 1D → 2D before the call so DataParallel can split along dim=0)
    In both cases we extract a 1D view via _resolve_channel_indices() before use.
    """
    def __init__(self, n_channels=19, sampling_rate=200, embed_dim=200,
                 n_layers=12, n_heads=8, mlp_dim=512):
        super().__init__()

        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)

        # Signal-level projection head (CLS token → 256-d)
        self.signal_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 256)
        )

        # Patch-level projection head (patch tokens → 256-d)
        self.patch_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 256)
        )

    @staticmethod
    def _resolve_channel_indices(channel_indices):
        """
        Normalise channel_indices to a guaranteed 1D tensor.

        train.py expands channel_indices to [batch, n_ch] so that DataParallel
        can split it along dim=0 (each GPU gets [batch/n_gpu, n_ch]).
        All rows are identical within a batch, so we just take row 0.
        A plain 1D tensor (single-GPU / teacher path) is returned as-is.
        """
        if channel_indices is None:
            return None
        if channel_indices.dim() == 2:
            return channel_indices[0]   # all rows are the same, pick first
        return channel_indices          # already 1D

    def forward(self, x, channel_indices=None, return_patch=False):
        """
        Args:
            x:               [batch, n_selected_ch, n_samples]
            channel_indices: 1D [n_selected_ch]  or  2D [batch_slice, n_selected_ch]
                             Identifies which of the full n_channels are present in x.
            return_patch:    whether to also return patch-level features

        Returns:
            signal_features: [batch, 256]
            patch_features:  [batch, n_tokens, 256]  (only if return_patch=True)
        """
        # Normalise to 1D regardless of DataParallel batching
        ci = self._resolve_channel_indices(channel_indices)

        # If a subset of channels was passed, pad back to the full channel count
        # so TFE and DPE always see a consistent input shape.
        if ci is not None and x.shape[1] < self.tfe.n_channels:
            full_x = torch.zeros(
                x.shape[0], self.tfe.n_channels, x.shape[2],
                device=x.device, dtype=x.dtype
            )
            full_x[:, ci, :] = x
            x = full_x

        # TFE: raw signal → patch tokens
        tokens = self.tfe(x)

        # DPE: add positional embeddings (needs to know which channels are active)
        tokens = self.dpe(tokens, ci)

        # Transformer
        cls_token, patch_tokens = self.transformer(tokens)

        # Project to 256-d space
        signal_features = self.signal_head(cls_token)

        if return_patch:
            patch_features = self.patch_head(patch_tokens)
            return signal_features, patch_features

        return signal_features


class TeacherModel(nn.Module):
    """
    Teacher model: EMA copy of the student.
    Always receives the raw (unwrapped) StudentModel — never a DataParallel object.
    train.py guarantees this by constructing TeacherModel before wrapping the
    student in DataParallel.
    """
    def __init__(self, student: StudentModel):
        super().__init__()

        # Build a fresh StudentModel with the same hyper-parameters
        self.model = StudentModel(
            n_channels=student.tfe.n_channels,
            sampling_rate=student.tfe.sampling_rate,
            embed_dim=student.transformer.cls_token.shape[-1],
            n_layers=len(student.transformer.transformer.layers),
            n_heads=student.transformer.transformer.layers[0].self_attn.num_heads,
            mlp_dim=student.signal_head[0].out_features
        )

        # Initialise with student weights
        self.model.load_state_dict(student.state_dict())

        # Freeze — teacher is never updated by gradient
        for param in self.parameters():
            param.requires_grad = False

        # Centering buffer (Equation 2)
        self.register_buffer('center', torch.zeros(1, 256))
        self.center_momentum = 0.9

    @torch.no_grad()
    def forward(self, x, channel_indices=None, return_patch=False):
        return self.model(x, channel_indices, return_patch)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """EMA update of the centering buffer."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )
