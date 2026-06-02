"""Analyze intrinsic representation quality across masking strategies.

Tests whether masking strategies create fundamentally better representations.

Usage:
    python analyze_representation_quality.py \
        --strategies none random theta delta alpha beta \
        --seeds 42 43 44 45 46 \
        --checkpoint_dir checkpoints/significance_sleep_model \
        --preset tiny
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.eeg_dino_model import EEGTransformer
from model.tfe_module import TimeFrequencyEmbedding
from model.dpe_module import DecoupledPositionalEmbedding
from datasets import SleepEDFDataset, get_dataset_root
from configs import PRESETS


class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']


class FrozenBackbone(torch.nn.Module):
    """Backbone for extracting embeddings."""
    
    def __init__(self, n_channels, sampling_rate, embed_dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.tfe = TimeFrequencyEmbedding(n_channels, sampling_rate, embed_dim)
        self.dpe = DecoupledPositionalEmbedding(n_channels, embed_dim)
        self.transformer = EEGTransformer(embed_dim, n_layers, n_heads, mlp_dim)
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, channel_indices=None):
        if channel_indices is None:
            channel_indices = torch.arange(x.shape[1], device=x.device)
            channel_indices = channel_indices.unsqueeze(0).expand(x.shape[0], -1)
        tokens, _ = self.tfe(x)
        tokens = self.dpe(tokens, channel_indices)
        cls, _ = self.transformer(tokens)
        return cls


def extract_embeddings(checkpoint_path, cfg, device):
    """Extract embeddings from frozen SSL backbone."""
    
    # Build backbone
    backbone = FrozenBackbone(
        n_channels=cfg['n_channels'],
        sampling_rate=cfg['sampling_rate'],
        embed_dim=cfg['embed_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        mlp_dim=cfg['mlp_dim']
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    backbone_state = {}
    for key, value in ckpt['student'].items():
        for prefix in ('tfe.', 'dpe.', 'transformer.'):
            if key.startswith(prefix):
                backbone_state[key] = value
                break
    backbone.load_state_dict(backbone_state, strict=True)
    
    # Load test set
    sleep_root = get_dataset_root('sleep_edf')
    test_ds = SleepEDFDataset(
        sleep_root, 'TestFold',
        n_channels=cfg['n_channels'],
        sampling_rate=cfg['sampling_rate']
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=1, pin_memory=False)
    
    # Extract embeddings
    backbone.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Extracting", leave=False):
            emb = backbone(x.to(device)).cpu().numpy()
            all_embeddings.append(emb)
            
            if isinstance(y, torch.Tensor):
                all_labels.append(y.numpy())
            else:
                all_labels.append(np.array(y))
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels


def compute_silhouette(embeddings, labels):
    """Compute silhouette score (cluster quality without using labels for training)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return silhouette_score(embeddings, labels, metric='euclidean', sample_size=min(10000, len(labels)))
    except:
        return np.nan


def compute_knn_accuracy(embeddings, labels, k=5, test_fraction=0.3, random_state=42):
    """Compute k-NN accuracy (non-parametric, non-linear)."""
    np.random.seed(random_state)
    n = len(labels)
    indices = np.random.permutation(n)
    n_test = int(n * test_fraction)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def compute_within_between_ratio(embeddings, labels):
    """Compute within-class variance / between-class variance."""
    centroids = {}
    within_var = 0
    
    for stage in np.unique(labels):
        mask = labels == stage
        stage_embs = embeddings[mask]
        centroid = stage_embs.mean(axis=0)
        centroids[stage] = centroid
        within_var += np.sum((stage_embs - centroid) ** 2)
    
    # Between-class variance
    global_centroid = embeddings.mean(axis=0)
    between_var = 0
    for stage in np.unique(labels):
        n_stage = (labels == stage).sum()
        between_var += n_stage * np.sum((centroids[stage] - global_centroid) ** 2)
    
    if between_var == 0:
        return np.inf
    return within_var / between_var


def compute_collapse_metric(embeddings):
    """Measure embedding collapse (higher std = less collapse)."""
    return np.std(embeddings, axis=0).mean()


def compute_class_separation_score(embeddings, labels):
    """Average distance from each class centroid to other centroids."""
    centroids = {}
    for stage in np.unique(labels):
        mask = labels == stage
        centroids[stage] = embeddings[mask].mean(axis=0)
    
    separation_scores = []
    for stage in np.unique(labels):
        centroid = centroids[stage]
        other_centroids = [centroids[s] for s in centroids if s != stage]
        avg_dist = np.mean([np.linalg.norm(centroid - other) for other in other_centroids])
        separation_scores.append(avg_dist)
    
    return np.mean(separation_scores)


def analyze_strategy(strategy, seeds, checkpoint_dir, cfg, device):
    """Analyze representation quality across seeds for one strategy."""
    
    metrics = defaultdict(list)
    
    logger = logging.getLogger(__name__)
    for seed in seeds:
        checkpoint_path = Path(checkpoint_dir) / f"{strategy}_seed{seed}" / "best_model.pth"

        logger.info(f"  Seed {seed}: checking {checkpoint_path}")

        try:
            embeddings, labels = extract_embeddings(checkpoint_path, cfg, device)
            logger.info(f"✓ Extracted {len(embeddings)} embeddings")
            
            # Compute metrics
            metrics['silhouette'].append(compute_silhouette(embeddings, labels))
            metrics['knn_accuracy'].append(compute_knn_accuracy(embeddings, labels, k=5))
            metrics['within_between_ratio'].append(compute_within_between_ratio(embeddings, labels))
            metrics['collapse_metric'].append(compute_collapse_metric(embeddings))
            metrics['separation_score'].append(compute_class_separation_score(embeddings, labels))
            
        except FileNotFoundError:
            logger.warning("✗ Checkpoint not found")
            continue
        except Exception as e:
            logger.exception(f"✗ Error: {e}")
            continue
    
    # Aggregate across seeds
    summary = {}
    for metric_name, values in metrics.items():
        if values:
            summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values],
                'n_seeds': len(values)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--checkpoint_dir', default='checkpoints/significance_sleep_model2')
    parser.add_argument('--preset', default='tiny', choices=list(PRESETS.keys()))
    args = parser.parse_args()
    
    # configure logging
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = PRESETS[args.preset]

    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("REPRESENTATION QUALITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info("="*80)
    
    # Analyze each strategy
    results = {}
    for strategy in args.strategies:
        logger.info(f"\n[{strategy.upper()}]")
        results[strategy] = analyze_strategy(strategy, args.seeds, args.checkpoint_dir, cfg, device)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    metrics_info = [
        ('silhouette', 'SILHOUETTE SCORE (cluster quality)', 'higher = better', False),
        ('knn_accuracy', 'kNN ACCURACY (non-linear separability)', 'higher = better', True),
        ('within_between_ratio', 'WITHIN/BETWEEN VARIANCE RATIO', 'lower = better', False),
        ('collapse_metric', 'COLLAPSE METRIC (embedding std)', 'higher = less collapse', False),
        ('separation_score', 'CLASS SEPARATION SCORE', 'higher = better', False),
    ]
    
    for metric_key, metric_title, direction, is_pct in metrics_info:
        logger.info(f"\n{metric_title} ({direction}):")
        logger.info(f"{'Strategy':<12} {'Mean':<15} {'Std':<15} {'N':<5}")
        logger.info("-"*50)
        
        for strategy in sorted(results.keys()):
            if metric_key not in results[strategy]:
                continue
            data = results[strategy][metric_key]
            mean_val = data['mean']
            std_val = data['std']
            n = data['n_seeds']
            
            if is_pct:
                logger.info(f"{strategy:<12} {mean_val:>14.1%} {std_val:>14.1%} {n:<5}")
            else:
                logger.info(f"{strategy:<12} {mean_val:>14.4f} {std_val:>14.4f} {n:<5}")
    
    # Done: results printed above. No further interpretation is added here.
    
    # Save results
    output = {'strategies': results}
    
    with open('representation_quality_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("✓ Saved: representation_quality_analysis.json")
    logger.info("="*80)


if __name__ == '__main__':
    main()