"""
EEG-DINO Model — Teacher/Student Architecture
==============================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tfe_module import TimeFrequencyEmbedding
from dpe_module import DecoupledPositionalEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Projection head with L2 normalization
# ─────────────────────────────────────────────────────────────────────────────

class DINOHead(nn.Module):
    """
    MLP projection head following DINO convention:
        Linear → GELU → Linear → L2-normalize → weight-normalized linear

    The L2 normalization step is the critical anti-collapse component.
    It ensures all feature vectors live on a unit hypersphere so the model
    cannot collapse by growing magnitudes — it must learn directions instead.

    The final weight-normalized linear layer (the "prototypes" layer) is also
    standard in DINO and further stabilizes training by keeping the prototype
    norms bounded.
    """
    def __init__(self, in_dim, hidden_dim, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Weight-normalized prototype layer — no bias
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        # Fix the weight_g norm to 1 at init (standard DINO practice)
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False  # only direction is learned

    def forward(self, x):
        x = self.mlp(x)
        # L2 normalize onto unit hypersphere — the anti-collapse core
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer backbone
# ─────────────────────────────────────────────────────────────────────────────

class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG-DINO.
    Table 1 — EEG-DINO-S: 12 layers, hidden=200, MLP=512
    """
    def __init__(self, embed_dim=200, n_layers=12, n_heads=8,
                 mlp_dim=512, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True   # pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [batch, n_tokens, embed_dim]
        Returns: cls_token [batch, embed_dim], patch_tokens [batch, n_tokens, embed_dim]
        """
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.transformer(x)
        x   = self.norm(x)
        return x[:, 0], x[:, 1:]


# ─────────────────────────────────────────────────────────────────────────────
# Student model
# ─────────────────────────────────────────────────────────────────────────────

class StudentModel(nn.Module):
    """
    Student model for EEG-DINO.

    channel_indices handling:
      - 1D [n_selected_ch]:          standard single-GPU / teacher path
      - 2D [batch_slice, n_ch]:      DataParallel path (train.py expands to 2D
                                     before the call so DP can split on dim=0)
    _resolve_channel_indices() normalises both cases to 1D.
    """
    def __init__(self, n_channels=19, sampling_rate=200, embed_dim=200,
                 n_layers=12, n_heads=8, mlp_dim=512):
        super().__init__()

        self.tfe         = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe         = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)

        # Both heads use the L2-normalizing DINOHead
        self.signal_head = DINOHead(embed_dim, mlp_dim, out_dim=256)
        self.patch_head  = DINOHead(embed_dim, mlp_dim, out_dim=256)

    @staticmethod
    def _resolve_channel_indices(channel_indices):
        """
        Normalise to guaranteed 1D tensor.
        DataParallel expands [n_ch] → [batch, n_ch]; we take row 0 (all identical).
        """
        if channel_indices is None:
            return None
        if channel_indices.dim() == 2:
            return channel_indices[0]
        return channel_indices

    def forward(self, x, channel_indices=None, return_patch=False):
        """
        x:               [batch, n_selected_ch, n_samples]
        channel_indices: 1D or 2D (see class docstring)
        return_patch:    if True also return patch-level features
        """
        ci = self._resolve_channel_indices(channel_indices)

        # Pad back to full channel count if a subset was passed
        if ci is not None and x.shape[1] < self.tfe.n_channels:
            full_x = torch.zeros(
                x.shape[0], self.tfe.n_channels, x.shape[2],
                device=x.device, dtype=x.dtype
            )
            full_x[:, ci, :] = x
            x = full_x

        tokens = self.tfe(x)
        tokens = self.dpe(tokens, ci)
        cls_token, patch_tokens = self.transformer(tokens)

        signal_features = self.signal_head(cls_token)

        if return_patch:
            # Apply head to each patch token independently
            B, T, D = patch_tokens.shape
            patch_flat     = patch_tokens.reshape(B * T, D)
            patch_features = self.patch_head(patch_flat).reshape(B, T, -1)
            return signal_features, patch_features

        return signal_features


# ─────────────────────────────────────────────────────────────────────────────
# Teacher model
# ─────────────────────────────────────────────────────────────────────────────

class TeacherModel(nn.Module):
    """
    EMA copy of the student.
    Always receives the raw (unwrapped) StudentModel — never a DataParallel obj.
    train.py guarantees this by constructing TeacherModel before wrapping the
    student in DataParallel.
    """
    def __init__(self, student: StudentModel):
        super().__init__()

        self.model = StudentModel(
            n_channels   = student.tfe.n_channels,
            sampling_rate= student.tfe.sampling_rate,
            embed_dim    = student.transformer.cls_token.shape[-1],
            n_layers     = len(student.transformer.transformer.layers),
            n_heads      = student.transformer.transformer.layers[0].self_attn.num_heads,
            mlp_dim      = student.signal_head.mlp[0].out_features
        )
        self.model.load_state_dict(student.state_dict())

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer('center', torch.zeros(1, 256))
        self.center_momentum = 0.9

    @torch.no_grad()
    def forward(self, x, channel_indices=None, return_patch=False):
        return self.model(x, channel_indices, return_patch)

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center  = (self.center * self.center_momentum
                        + batch_center * (1 - self.center_momentum))