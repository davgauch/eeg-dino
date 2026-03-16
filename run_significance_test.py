"""Run significance testing across multiple seeds for BCI-IV 2a masking strategies.

Usage:
    python run_significance_test.py --strategies spatiotemporal alpha beta --seeds 5
"""

import argparse
import subprocess
import json
import numpy as np
from scipy import stats
import os
from pathlib import Path


def train_model(strategy, seed, save_dir):
    """Train a single model with given strategy and seed."""
    cmd = [
        'python', 'train.py',
        '--preset', 'bci_2a',
        '--dataset', 'bci_2a',
        '--mask_strategy', strategy,
        '--seed', str(seed),
        '--save_dir', save_dir,
        '--n_epochs', '100'  # ← FIXED: was --epochs
    ]
    
    print(f"\n{'='*60}")
    print(f"Training: {strategy} | Seed {seed}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def evaluate_model(checkpoint, strategy, seed):
    """Evaluate a trained model."""
    cmd = [
        'python', 'evaluate_bci.py',
        '--checkpoint', checkpoint,
        '--bci', '2a',
        '--preset', 'bci_2a',
        '--mode', 'within'
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {strategy} | Seed {seed}")
    print(f"{'='*60}")
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to extract metrics
    lines = result.stdout.split('\n')
    metrics = {'subjects': {}}
    
    for line in lines:
        if 'Accuracy :' in line:
            parts = line.split()
            metrics['accuracy_mean'] = float(parts[2])
            metrics['accuracy_std'] = float(parts[4])
        elif 'F1 Macro :' in line:
            parts = line.split()
            metrics['f1_mean'] = float(parts[3])
            metrics['f1_std'] = float(parts[5])
        elif 'Kappa    :' in line:
            parts = line.split()
            metrics['kappa_mean'] = float(parts[2])
            metrics['kappa_std'] = float(parts[4])
        elif line.strip().startswith('A0'):  # Subject results
            parts = line.split()
            subject = parts[0].rstrip(':')
            acc = float(parts[1].split('=')[1])
            f1 = float(parts[2].split('=')[1])
            kappa = float(parts[3].split('=')[1])
            metrics['subjects'][subject] = {
                'accuracy': acc, 'f1': f1, 'kappa': kappa
            }
    
    return metrics


def run_paired_test(results_a, results_b, metric='accuracy'):
    """Run paired t-test between two strategies."""
    # Extract per-subject scores for each seed
    scores_a = []
    scores_b = []
    
    for seed_results in results_a:
        for subject in sorted(seed_results['subjects'].keys()):
            scores_a.append(seed_results['subjects'][subject][metric])
    
    for seed_results in results_b:
        for subject in sorted(seed_results['subjects'].keys()):
            scores_b.append(seed_results['subjects'][subject][metric])
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_pvalue = stats.wilcoxon(scores_a, scores_b)
    
    return {
        'mean_a': np.mean(scores_a),
        'mean_b': np.mean(scores_b),
        'std_a': np.std(scores_a),
        'std_b': np.std(scores_b),
        't_statistic': t_stat,
        'p_value': p_value,
        'w_statistic': w_stat,
        'w_pvalue': w_pvalue,
        'n_samples': len(scores_a)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', nargs='+', required=True,
                       help='Masking strategies to compare (e.g., spatiotemporal alpha beta)')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds to run')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed value')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--results_dir', default='significance_results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Store all results
    all_results = {strategy: [] for strategy in args.strategies}
    
    # Run experiments
    for seed_offset in range(args.seeds):
        seed = args.start_seed + seed_offset
        
        for strategy in args.strategies:
            save_dir = f"checkpoints/significance/{strategy}_seed{seed}"
            checkpoint = f"{save_dir}/best_model.pth"
            
            # Train if needed
            if not args.skip_training:
                success = train_model(strategy, seed, save_dir)
                if not success:
                    print(f"Training failed for {strategy} seed {seed}")
                    continue
            
            # Evaluate
            if os.path.exists(checkpoint):
                metrics = evaluate_model(checkpoint, strategy, seed)
                metrics['seed'] = seed
                all_results[strategy].append(metrics)
                
                # Save intermediate results
                with open(f"{args.results_dir}/results_{strategy}.json", 'w') as f:
                    json.dump(all_results[strategy], f, indent=2)
            else:
                print(f"Checkpoint not found: {checkpoint}")
    
    # Statistical analysis
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    # Compare all pairs
    for i, strategy_a in enumerate(args.strategies):
        for strategy_b in args.strategies[i+1:]:
            print(f"\n{strategy_a.upper()} vs {strategy_b.upper()}")
            print("-" * 60)
            
            for metric in ['accuracy', 'f1', 'kappa']:
                test_results = run_paired_test(
                    all_results[strategy_a],
                    all_results[strategy_b],
                    metric
                )
                
                print(f"\n{metric.upper()}:")
                print(f"  {strategy_a}: {test_results['mean_a']:.4f} ± {test_results['std_a']:.4f}")
                print(f"  {strategy_b}: {test_results['mean_b']:.4f} ± {test_results['std_b']:.4f}")
                print(f"  Difference: {test_results['mean_b'] - test_results['mean_a']:.4f}")
                print(f"  t-test: t={test_results['t_statistic']:.4f}, p={test_results['p_value']:.4f}", end='')
                if test_results['p_value'] < 0.05:
                    print(" ***")
                elif test_results['p_value'] < 0.10:
                    print(" *")
                else:
                    print()
                print(f"  Wilcoxon: W={test_results['w_statistic']:.4f}, p={test_results['w_pvalue']:.4f}", end='')
                if test_results['w_pvalue'] < 0.05:
                    print(" ***")
                elif test_results['w_pvalue'] < 0.10:
                    print(" *")
                else:
                    print()
    
    # Save full results
    with open(f"{args.results_dir}/full_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.results_dir}/")


if __name__ == '__main__':
    main()