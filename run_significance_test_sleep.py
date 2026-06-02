"""Run significance testing for Sleep-EDF masking strategies.
Usage:
    python run_significance_test_sleep.py \
        --strategies spatiotemporal theta \
        --seeds 5 \
        --n_epochs 30 \
        --results_dir significance_results_sleep
"""

import argparse
import json
import numpy as np
from scipy import stats
import os
import sys
import re
import subprocess
import logging
import sys


def train_model(strategy, seed, save_dir, n_epochs):
    """Train a single model with given strategy and seed."""
    cmd = [
        sys.executable, 'train.py',
        '--preset', 'tiny',
        '--dataset', 'sleep_edf',
        '--mask_strategy', strategy,
        '--seed', str(seed),
        '--save_dir', save_dir,
        '--n_epochs', str(n_epochs)
    ]
    
    logging.getLogger(__name__).info(f"\n{'='*60}")
    logging.getLogger(__name__).info(f"Training: {strategy} | Seed {seed} | Epochs {n_epochs}")
    logging.getLogger(__name__).info(f"Command: {' '.join(cmd)}")
    logging.getLogger(__name__).info(f"{'='*60}")
    
    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        logging.getLogger(__name__).error(f"❌ Training FAILED for {strategy} seed {seed}")
        return False
    
    checkpoint = os.path.join(save_dir, 'best_model.pth')
    if not os.path.exists(checkpoint):
        logging.getLogger(__name__).error(f"❌ Checkpoint not created: {checkpoint}")
        return False
    
    logging.getLogger(__name__).info(f"✓ Training completed successfully")
    logging.getLogger(__name__).info(f"✓ Checkpoint saved: {checkpoint}")
    return True


def evaluate_model(checkpoint, strategy, seed):
    """Evaluate a trained model on sleep stage classification."""
    cmd = [
        sys.executable, 'evaluate.py',
        '--checkpoint', checkpoint,
        '--preset', 'tiny',
        '--n_classes', '5',
        '--probe_epochs', '50',
        '--probe_lr', '1e-3',
        '--seed', str(seed)
    ]
    
    logging.getLogger(__name__).info(f"\n{'='*60}")
    logging.getLogger(__name__).info(f"Evaluating: {strategy} | Seed {seed}")
    logging.getLogger(__name__).info(f"{'='*60}")
    
    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logging.getLogger(__name__).error(f"❌ Evaluation FAILED")
        if result.stdout:
            logging.getLogger(__name__).error(f"STDOUT:\n{result.stdout}")
        logging.getLogger(__name__).error(f"STDERR:\n{result.stderr}")
        return None
    
    combined_output = f"{result.stdout}\n{result.stderr}"
    lines = combined_output.split('\n')
    metrics = {}
    
    for line in lines:
        if 'Accuracy :' in line:
            match = re.search(r'Accuracy\s*:\s*([\d.]+)', line)
            if match:
                metrics['accuracy'] = float(match.group(1))
        elif 'F1 Macro :' in line:
            match = re.search(r'F1 Macro\s*:\s*([\d.]+)', line)
            if match:
                metrics['f1_macro'] = float(match.group(1))
        elif 'Kappa    :' in line:
            match = re.search(r'Kappa\s*:\s*([\d.]+)', line)
            if match:
                metrics['kappa'] = float(match.group(1))
    
    if not metrics:
        logging.getLogger(__name__).error(f"❌ Failed to parse evaluation results")
        logging.getLogger(__name__).error(f"STDOUT:\n{result.stdout}")
        logging.getLogger(__name__).error(f"STDERR:\n{result.stderr}")
        return None
    
    logging.getLogger(__name__).info(f"✓ Evaluation completed:")
    logging.getLogger(__name__).info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    logging.getLogger(__name__).info(f"  F1 Macro: {metrics.get('f1_macro', 'N/A'):.4f}")
    logging.getLogger(__name__).info(f"  Kappa:    {metrics.get('kappa', 'N/A'):.4f}")
    
    return metrics


def run_paired_test(results_a, results_b, metric='accuracy'):
    """Run paired t-test between two strategies."""
    scores_a = [r[metric] for r in results_a if r and metric in r]
    scores_b = [r[metric] for r in results_b if r and metric in r]
    
    if len(scores_a) == 0 or len(scores_b) == 0:
        return None
    
    if len(scores_a) != len(scores_b):
        logging.getLogger(__name__).warning(
            f"⚠ Warning: Unequal sample sizes ({len(scores_a)} vs {len(scores_b)})")
        min_len = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
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
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser(
        description='Multi-seed significance testing for Sleep-EDF')
    parser.add_argument('--strategies', nargs='+', required=True,
                       help='Masking strategies to compare')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds to run')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed value')
    parser.add_argument('--n_epochs', type=int, default=30,
                       help='Number of pre-training epochs')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--retrain_strategies', nargs='+', default=[],
                       help='Strategies to force retraining even if checkpoint exists')
    parser.add_argument('--checkpoint_root', default='checkpoints/myrun',
                       help='Base directory containing per-strategy checkpoint folders')
    parser.add_argument('--results_dir', default='significance_results_sleep',
                       help='Directory to save results')
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    all_results = {strategy: [] for strategy in args.strategies}
    
    total_experiments = len(args.strategies) * args.seeds
    completed = 0
    
    for seed_offset in range(args.seeds):
        seed = args.start_seed + seed_offset
        
        for strategy in args.strategies:
            completed += 1
            logging.getLogger(__name__).info(f"\n\n{'#'*80}")
            logging.getLogger(__name__).info(f"# Experiment {completed}/{total_experiments}: {strategy} | Seed {seed}")
            logging.getLogger(__name__).info(f"{'#'*80}")
            
            save_dir = os.path.join(args.checkpoint_root, f"{strategy}_seed{seed}")
            checkpoint = f"{save_dir}/best_model.pth"
            

            if not args.skip_training:
                should_retrain = strategy in args.retrain_strategies
                
                if os.path.exists(checkpoint) and not should_retrain:
                    logging.getLogger(__name__).info(f"⏭  Checkpoint exists, skipping training: {checkpoint}")
                else:
                    if should_retrain and os.path.exists(checkpoint):
                        logging.getLogger(__name__).info(f"🔁 Retraining forced for {strategy}, overwriting checkpoint")
                    
                    success = train_model(strategy, seed, save_dir, args.n_epochs)
                    if not success:
                        logging.getLogger(__name__).error(f"❌ Skipping evaluation due to training failure")
                        continue
            
            if not os.path.exists(checkpoint):
                logging.getLogger(__name__).error(f"❌ Checkpoint not found: {checkpoint}")
                continue
            
            metrics = evaluate_model(checkpoint, strategy, seed)
            if metrics is None:
                logging.getLogger(__name__).error(f"❌ Evaluation failed")
                continue
            
            metrics['seed'] = seed
            metrics['strategy'] = strategy
            all_results[strategy].append(metrics)
            
            result_file = f"{args.results_dir}/results_{strategy}.json"
            with open(result_file, 'w') as f:
                json.dump(all_results[strategy], f, indent=2)
            logging.getLogger(__name__).info(f"✓ Results saved: {result_file}")
    
    logging.getLogger(__name__).info(f"\n\n{'='*80}")
    logging.getLogger(__name__).info("STATISTICAL ANALYSIS - SLEEP STAGE CLASSIFICATION")
    logging.getLogger(__name__).info(f"{'='*80}\n")
    
    total_results = sum(len(results) for results in all_results.values())
    if total_results == 0:
        logging.getLogger(__name__).error("❌ No results collected! All experiments failed.")
        sys.exit(1)
    
    logging.getLogger(__name__).info("Results collected:")
    for strategy, results in all_results.items():
        logging.getLogger(__name__).info(f"  {strategy}: {len(results)} / {args.seeds} seeds")
    logging.getLogger(__name__).info("")
    
    for i, strategy_a in enumerate(args.strategies):
        for strategy_b in args.strategies[i+1:]:
            if not all_results[strategy_a] or not all_results[strategy_b]:
                logging.getLogger(__name__).warning(
                    f"\n⚠ Skipping {strategy_a} vs {strategy_b} - insufficient data")
                continue
            
            logging.getLogger(__name__).info(f"\n{strategy_a.upper()} vs {strategy_b.upper()}")
            logging.getLogger(__name__).info("-" * 60)
            
            for metric in ['accuracy', 'f1_macro', 'kappa']:
                test_results = run_paired_test(
                    all_results[strategy_a],
                    all_results[strategy_b],
                    metric
                )
                
                if test_results is None:
                    logging.getLogger(__name__).info(f"\n{metric.upper()}: Insufficient data")
                    continue
                
                logging.getLogger(__name__).info(f"\n{metric.upper()}:")
                logging.getLogger(__name__).info(
                    f"  {strategy_a}: {test_results['mean_a']:.4f} ± {test_results['std_a']:.4f}")
                logging.getLogger(__name__).info(
                    f"  {strategy_b}: {test_results['mean_b']:.4f} ± {test_results['std_b']:.4f}")
                logging.getLogger(__name__).info(
                    f"  Difference: {test_results['mean_b'] - test_results['mean_a']:.4f}")
                t_line = f"  t-test: t={test_results['t_statistic']:.4f}, p={test_results['p_value']:.4f}"
                if test_results['p_value'] < 0.05:
                    t_line += " ***"
                elif test_results['p_value'] < 0.10:
                    t_line += " *"
                logging.getLogger(__name__).info(t_line)

                w_line = f"  Wilcoxon: W={test_results['w_statistic']:.4f}, p={test_results['w_pvalue']:.4f}"
                if test_results['w_pvalue'] < 0.05:
                    w_line += " ***"
                elif test_results['w_pvalue'] < 0.10:
                    w_line += " *"
                logging.getLogger(__name__).info(w_line)
    
    full_results_file = f"{args.results_dir}/full_results.json"
    with open(full_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.getLogger(__name__).info(f"\n✓ All results saved to {args.results_dir}/")
    logging.getLogger(__name__).info(f"✓ Full results: {full_results_file}")


if __name__ == '__main__':
    main()