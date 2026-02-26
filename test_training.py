# test_training.py
"""
Quick test with minimal epochs
"""
import torch
from train import EEGDataset, EEGDINOTrainer
from torch.utils.data import DataLoader


# Small config for testing
config = {
    'n_channels': 19,
    'sampling_rate': 200,
    'embed_dim': 128,  # Smaller for speed
    'n_layers': 4,     # Fewer layers
    'n_heads': 4,
    'mlp_dim': 256,
    'batch_size': 4,   # Small batch
    'learning_rate': 1e-4,
    'weight_decay': 0.04,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Synthetic data
dataset = EEGDataset(data_path=None, n_channels=19, sampling_rate=200)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train 2 epochs
trainer = EEGDINOTrainer(**config)
trainer.train(dataloader, n_epochs=2, save_dir='test_checkpoints')

print("\n✓ Test passed! Training loop works.")