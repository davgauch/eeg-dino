"""EEG-DINO Model — Student/Teacher with DINO projection heads.

Head dims are scaled proportionally to backbone size (hidden=4*embed_dim,
bottleneck=embed_dim) to maintain a healthy backbone-to-head gradient ratio.
The original paper uses hidden=2048, bottleneck=256 for ViT-S/B (~22-86M).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tfe_module import TimeFrequencyEmbedding
from dpe_module import DecoupledPositionalEmbedding


class DINOHead(nn.Module):
    """3-layer MLP → L2-norm → weight-normalized projection."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=4096, bottleneck_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class EEGTransformer(nn.Module):
    """Pre-norm Transformer encoder with learnable CLS token."""

    def __init__(self, embed_dim=64, n_layers=2, n_heads=4,
                 mlp_dim=128, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.norm(self.transformer(x))
        return x[:, 0], x[:, 1:]  # (cls, patches)


class StudentModel(nn.Module):
    """TFE → DPE → Transformer → signal head + patch head."""

    def __init__(self, n_channels=2, sampling_rate=200, embed_dim=64,
                 n_layers=2, n_heads=4, mlp_dim=128, out_dim=4096,
                 head_hidden_dim=256, head_bottleneck_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_bottleneck_dim = head_bottleneck_dim
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        self.signal_head = DINOHead(embed_dim, head_hidden_dim, out_dim, head_bottleneck_dim)
        self.patch_head = DINOHead(embed_dim, head_hidden_dim, out_dim, head_bottleneck_dim)

    @staticmethod
    def _resolve_ci(ci):
        if ci is None:
            return None
        return ci[0] if ci.dim() == 2 else ci

    def forward(self, x, channel_indices=None, return_patch=False):
        ci = self._resolve_ci(channel_indices)

        if ci is not None and x.shape[1] < self.tfe.n_channels:
            ci = ci.clamp(0, self.tfe.n_channels - 1)
            full = torch.zeros(x.shape[0], self.tfe.n_channels, x.shape[2],
                               device=x.device, dtype=x.dtype)
            full[:, ci, :] = x
            x = full

        tokens = self.dpe(self.tfe(x), ci)
        cls, patches = self.transformer(tokens)
        sig = self.signal_head(cls)

        if return_patch:
            B, T, D = patches.shape
            pat = self.patch_head(patches.reshape(B * T, D)).reshape(B, T, -1)
            return sig, pat
        return sig


class TeacherModel(nn.Module):
    """EMA copy of the student — no gradients, updated externally."""

    def __init__(self, student: StudentModel):
        super().__init__()
        self.model = StudentModel(
            n_channels=student.tfe.n_channels,
            sampling_rate=student.tfe.sampling_rate,
            embed_dim=student.transformer.cls_token.shape[-1],
            n_layers=len(student.transformer.transformer.layers),
            n_heads=student.transformer.transformer.layers[0].self_attn.num_heads,
            mlp_dim=student.transformer.transformer.layers[0].linear1.out_features,
            out_dim=student.out_dim,
            head_hidden_dim=student.head_hidden_dim,
            head_bottleneck_dim=student.head_bottleneck_dim,
        )
        self.model.load_state_dict(student.state_dict())
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer('center', torch.zeros(1, student.out_dim))
        self.center_momentum = 0.9

    @torch.no_grad()
    def forward(self, x, channel_indices=None, return_patch=False):
        return self.model(x, channel_indices, return_patch)

    @torch.no_grad()
    def update_center(self, teacher_output):
        bc = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + bc * (1 - self.center_momentum)
