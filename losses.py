"""Hierarchical Self-Distillation Losses for EEG-DINO.

Signal loss: standard DINO cross-entropy between teacher and student CLS tokens.
Patch loss:  distillation between masked student patches and global teacher patches.
All cross-view pairs are averaged with equal weight (per-term normalization).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Signal-level DINO loss — cross-entropy over all (teacher, student) pairs."""

    def __init__(self, out_dim=4096, teacher_temp_base=0.04,
                 teacher_temp_final=0.05, temp_warmup_epochs=10,
                 student_temp=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_final = teacher_temp_final
        self.temp_warmup_epochs = temp_warmup_epochs
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp_base

    def set_epoch(self, epoch):
        """Teacher temperature warmup (epoch is 1-indexed)."""
        e = epoch - 1
        if e < self.temp_warmup_epochs:
            self.teacher_temp = (self.teacher_temp_base +
                (self.teacher_temp_final - self.teacher_temp_base)
                * e / self.temp_warmup_epochs)
        else:
            self.teacher_temp = self.teacher_temp_final

    def forward(self, student_outputs, teacher_outputs, teacher_center):
        t_probs = {k: F.softmax((v - teacher_center) / self.teacher_temp, dim=-1)
                   for k, v in teacher_outputs.items()}
        s_probs = {k: F.log_softmax(v / self.student_temp, dim=-1)
                   for k, v in student_outputs.items()}

        total_loss, n_terms = 0.0, 0
        for t_key in sorted(t_probs):
            for s_key in sorted(s_probs):
                if s_key == t_key:
                    continue
                total_loss += -torch.sum(t_probs[t_key] * s_probs[s_key], dim=-1).mean()
                n_terms += 1

        return total_loss / max(1, n_terms), {'n_terms': n_terms}


class PatchLoss(nn.Module):
    """Patch-level distillation: masked student patches vs global teacher patches."""
    def __init__(self, teacher_temp_base=0.04, teacher_temp_final=0.05,
                 temp_warmup_epochs=10, student_temp=0.1):
        super().__init__()
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_final = teacher_temp_final
        self.temp_warmup_epochs = temp_warmup_epochs
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp_base
        
    def set_epoch(self, epoch):
        e = epoch - 1
        if e < self.temp_warmup_epochs:
            self.teacher_temp = (self.teacher_temp_base +
                                (self.teacher_temp_final - self.teacher_temp_base)
                                * e / self.temp_warmup_epochs)
        else:
            self.teacher_temp = self.teacher_temp_final
            
    def forward(self, student_patches, teacher_patches, teacher_center):
        # Handle case with no masked views (e.g., strategy='none' with n_masked_views=0)
        masked_keys = [k for k in student_patches if k.startswith('masked_')]
        if len(masked_keys) == 0:
            return torch.tensor(0.0, device=teacher_center.device)
        
        total, n = 0.0, 0
        for sk in sorted(masked_keys):
            for tk in ['global_0', 'global_1']:
                sp, tp = student_patches[sk], teacher_patches[tk]
                min_t = min(sp.shape[1], tp.shape[1])
                sp = sp[:, :min_t].reshape(-1, sp.shape[-1])
                tp = tp[:, :min_t].reshape(-1, tp.shape[-1])
                t_p = F.softmax((tp - teacher_center) / self.teacher_temp, dim=-1)
                s_p = F.log_softmax(sp / self.student_temp, dim=-1)
                total += -torch.sum(t_p * s_p, dim=-1).mean()
                n += 1
        return total / n
