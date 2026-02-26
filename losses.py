"""
Hierarchical Self-Distillation Losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    """
    - LSignal = LGlobal + LLocal + LMasked  
    - L = LSignal + LPatch 
    """
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.out_dim = out_dim
        
    def forward(self, student_outputs, teacher_outputs, teacher_center):
        """
        Args:
            student_outputs: dict with keys 'global_0', 'global_1', 'local_0-7', 'masked_0-1'
                            each value is [batch, 256]
            teacher_outputs: dict with keys 'global_0', 'global_1'
                            each value is [batch, 256]
            teacher_center: [1, 256] - running center for teacher
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses for logging
        """
        # Apply temperature and softmax
        # Teacher: center outputs then softmax
        teacher_probs = {}
        for key, output in teacher_outputs.items():
            centered = output - teacher_center
            teacher_probs[key] = F.softmax(centered / self.teacher_temp, dim=-1)
        
        # Student: just softmax
        student_probs = {}
        for key, output in student_outputs.items():
            student_probs[key] = F.log_softmax(output / self.student_temp, dim=-1)
        
        # Signal-level distillation 
        loss_global = 0
        loss_local = 0
        loss_masked = 0
        
        # Global views: student global ↔ teacher global
        for s_key in ['global_0', 'global_1']:
            for t_key in ['global_0', 'global_1']:
                if s_key != t_key:  # Don't match same view
                    loss_global += -torch.sum(
                        teacher_probs[t_key] * student_probs[s_key], 
                        dim=-1
                    ).mean()
        
        # Local views: student local ↔ teacher global
        for s_key in [f'local_{i}' for i in range(8)]:
            for t_key in ['global_0', 'global_1']:
                loss_local += -torch.sum(
                    teacher_probs[t_key] * student_probs[s_key],
                    dim=-1
                ).mean()
        
        # Masked views: student masked ↔ teacher global
        for s_key in ['masked_0', 'masked_1']:
            for t_key in ['global_0', 'global_1']:
                loss_masked += -torch.sum(
                    teacher_probs[t_key] * student_probs[s_key],
                    dim=-1
                ).mean()
        
        # Normalize
        loss_global = loss_global / 2  # 2 comparisons
        loss_local = loss_local / 16   # 8 local × 2 global
        loss_masked = loss_masked / 4  # 2 masked × 2 global
        
        # Total signal loss 
        loss_signal = loss_global + loss_local + loss_masked
        
        return loss_signal, {
            'loss_global': loss_global.item(),
            'loss_local': loss_local.item(),
            'loss_masked': loss_masked.item()
        }


class PatchLoss(nn.Module):
    """
    Patch-level distillation
    """
    def __init__(self, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
    
    def forward(self, student_patch_outputs, teacher_patch_outputs, teacher_center):
        """
        Args:
            student_patch_outputs: dict with 'masked_0', 'masked_1'
                                  each value is [batch, n_tokens, 256]
            teacher_patch_outputs: dict with 'global_0', 'global_1'
                                  each value is [batch, n_tokens, 256]
            teacher_center: [1, 256]
        Returns:
            patch_loss: scalar
        """
        total_loss = 0
        n_comparisons = 0
        
        for s_key in ['masked_0', 'masked_1']:
            for t_key in ['global_0', 'global_1']:
                student_patches = student_patch_outputs[s_key]  # [batch, n_tokens, 256]
                teacher_patches = teacher_patch_outputs[t_key]  # [batch, n_tokens, 256]
                
                # Align number of tokens (student masked might have fewer)
                min_tokens = min(student_patches.shape[1], teacher_patches.shape[1])
                student_patches = student_patches[:, :min_tokens, :]
                teacher_patches = teacher_patches[:, :min_tokens, :]
                
                # Flatten batch and tokens: [batch*n_tokens, 256]
                student_flat = student_patches.reshape(-1, student_patches.shape[-1])
                teacher_flat = teacher_patches.reshape(-1, teacher_patches.shape[-1])
                
                # Center teacher and apply temperature
                teacher_centered = teacher_flat - teacher_center
                teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)
                student_log_probs = F.log_softmax(student_flat / self.student_temp, dim=-1)
                
                # Cross-entropy
                loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
                total_loss += loss
                n_comparisons += 1
        
        return total_loss / n_comparisons
