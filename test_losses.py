# test_losses.py
import torch
from losses import DINOLoss, PatchLoss

# Test signal loss
signal_loss_fn = DINOLoss()

# Build full student output dicts (global + local + masked)
def make_student_outputs(base):
    outputs = {f'global_{i}': base for i in range(2)}
    outputs.update({f'local_{i}': base for i in range(8)})
    outputs.update({f'masked_{i}': base for i in range(2)})
    return outputs

# Same outputs = low loss
base_tensor = torch.randn(4, 256)
same_student = make_student_outputs(base_tensor)
same_teacher = {f'global_{i}': base_tensor for i in range(2)}
loss_same, _ = signal_loss_fn(same_student, same_teacher, torch.zeros(1, 256))

# Different outputs = high loss
diff_student = make_student_outputs(torch.randn(4, 256))
diff_teacher = {f'global_{i}': torch.randn(4, 256) for i in range(2)}
loss_diff, _ = signal_loss_fn(diff_student, diff_teacher, torch.zeros(1, 256))

print(f"Same outputs loss: {loss_same.item():.4f}")
print(f"Different outputs loss: {loss_diff.item():.4f}")
print(f"✓ Test passed!" if loss_diff > loss_same else "✗ Test failed!")