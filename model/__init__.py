"""Model package for EEG-DINO."""

from .channel_aware_sampling import ChannelAwareSampling
from .dpe_module import DecoupledPositionalEmbedding
from .eeg_dino_model import DINOHead, EEGTransformer, StudentModel, TeacherModel
from .losses import DINOLoss, PatchLoss
from .tfe_module import TimeFrequencyEmbedding

__all__ = [
    'ChannelAwareSampling',
    'DecoupledPositionalEmbedding',
    'DINOHead',
    'DINOLoss',
    'EEGTransformer',
    'PatchLoss',
    'StudentModel',
    'TeacherModel',
    'TimeFrequencyEmbedding',
]
