"""Hierarchical Self-Distillation Losses for EEG-DINO."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Signal-level loss: L_global + L_local + L_masked."""

    def __init__(self, out_dim=64, teacher_temp_base=0.04,
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
        """Teacher temperature warmup: base → final over warmup epochs."""
        if epoch < self.temp_warmup_epochs:
            self.teacher_temp = (self.teacher_temp_base +
                (self.teacher_temp_final - self.teacher_temp_base)
                * epoch / self.temp_warmup_epochs)
        else:
            self.teacher_temp = self.teacher_temp_final

    def forward(self, student_outputs, teacher_outputs, teacher_center):
        # Teacher: centered softmax; Student: log-softmax
        t_probs = {}
        for k, v in teacher_outputs.items():
            t_probs[k] = F.softmax((v - teacher_center) / self.teacher_temp, dim=-1)

        s_probs = {}
        for k, v in student_outputs.items():
            s_probs[k] = F.log_softmax(v / self.student_temp, dim=-1)

        def _ce(t_key, s_key):
            return -torch.sum(t_probs[t_key] * s_probs[s_key], dim=-1).mean()

        # Global cross-view
        loss_global = (_ce('global_0', 'global_1') + _ce('global_1', 'global_0')) / 2

        # Local → global
        local_keys = sorted(k for k in s_probs if k.startswith('local_'))
        loss_local = sum(_ce(tk, sk) for sk in local_keys
                         for tk in ['global_0', 'global_1'])
        loss_local = loss_local / max(1, len(local_keys) * 2)

        # Masked → global
        masked_keys = sorted(k for k in s_probs if k.startswith('masked_'))
        loss_masked = sum(_ce(tk, sk) for sk in masked_keys
                          for tk in ['global_0', 'global_1'])
        loss_masked = loss_masked / max(1, len(masked_keys) * 2)

        loss_signal = loss_global + loss_local + loss_masked
        return loss_signal, {
            'global': loss_global.item(),
            'local': loss_local.item(),
            'masked': loss_masked.item(),
        }


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
        if epoch < self.temp_warmup_epochs:
            self.teacher_temp = (self.teacher_temp_base +
                (self.teacher_temp_final - self.teacher_temp_base)
                * epoch / self.temp_warmup_epochs)
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
