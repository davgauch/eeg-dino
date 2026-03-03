"""Hierarchical Self-Distillation Losses for EEG-DINO.

Signal loss follows the exact DINO formulation: for each teacher global
view t and each student view s (where s ≠ t), compute H(P_t, P_s).
All terms are averaged with equal weight (not per-group).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Signal-level DINO loss with paper-faithful per-term normalization.

    For n_local=4, n_masked=1: 12 total cross-view pairs, each weight 1/12.
    This avoids overweighting the global↔global terms (which are easiest
    to minimize by collapse since both views are very similar).
    """

    def __init__(self, out_dim=4096, teacher_temp_base=0.04,
                 teacher_temp_final=0.07, temp_warmup_epochs=30,
                 student_temp=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_final = teacher_temp_final
        self.temp_warmup_epochs = temp_warmup_epochs
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp_base

    def set_epoch(self, epoch):
        """Teacher temperature warmup: base → final over warmup epochs.
        epoch is 1-indexed (first training epoch = 1)."""
        e = epoch - 1  # 0-indexed for schedule math
        if e < self.temp_warmup_epochs:
            self.teacher_temp = (self.teacher_temp_base +
                (self.teacher_temp_final - self.teacher_temp_base)
                * e / self.temp_warmup_epochs)
        else:
            self.teacher_temp = self.teacher_temp_final

    def forward(self, student_outputs, teacher_outputs, teacher_center):
        # Teacher: centered + sharpened probabilities
        t_probs = {}
        for k, v in teacher_outputs.items():
            t_probs[k] = F.softmax((v - teacher_center) / self.teacher_temp, dim=-1)

        # Student: log-softmax probabilities
        s_probs = {}
        for k, v in student_outputs.items():
            s_probs[k] = F.log_softmax(v / self.student_temp, dim=-1)

        # DINO loss: equal weight per (teacher, student) pair, skip self
        total_loss = 0.0
        n_terms = 0
        for t_key in sorted(t_probs.keys()):       # global_0, global_1
            for s_key in sorted(s_probs.keys()):    # all views
                if s_key == t_key:
                    continue  # skip self-distillation
                total_loss += -torch.sum(t_probs[t_key] * s_probs[s_key],
                                         dim=-1).mean()
                n_terms += 1

        loss_signal = total_loss / max(1, n_terms)
        return loss_signal, {'n_terms': n_terms}


class PatchLoss(nn.Module):
    """Patch-level distillation between masked student and global teacher."""

    def __init__(self, teacher_temp_base=0.04, teacher_temp_final=0.07,
                 temp_warmup_epochs=30, student_temp=0.1):
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
        total, n = 0.0, 0
        masked_keys = sorted(k for k in student_patches if k.startswith('masked_'))

        for sk in masked_keys:
            for tk in ['global_0', 'global_1']:
                sp = student_patches[sk]
                tp = teacher_patches[tk]
                min_t = min(sp.shape[1], tp.shape[1])
                sp = sp[:, :min_t].reshape(-1, sp.shape[-1])
                tp = tp[:, :min_t].reshape(-1, tp.shape[-1])

                t_p = F.softmax((tp - teacher_center) / self.teacher_temp, dim=-1)
                s_p = F.log_softmax(sp / self.student_temp, dim=-1)
                total += -torch.sum(t_p * s_p, dim=-1).mean()
                n += 1

        return total / max(1, n)
