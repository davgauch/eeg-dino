"""Compute and analyse confusion matrices across masking strategies.

Loads test_predictions.npz files saved by downstream_pairwise_accuracy.py
and computes per-strategy confusion matrices and N3 breakdown.

Usage:
    python experiments/analyze_confusion_matrices.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
N_CLASSES   = len(STAGE_NAMES)

STRATEGIES  = ['none', 'random', 'theta', 'delta', 'alpha', 'beta']
SEEDS       = [42, 43, 44, 45, 46]

STAGE_COLORS = {
    'Correct': '#2ecc71',
    'N2':      '#e74c3c',
    'Wake':    '#3498db',
    'N1':      '#f39c12',
    'REM':     '#9b59b6',
}


def load_predictions(results_dir, strategies, seeds):
    data = defaultdict(dict)
    for strategy in strategies:
        for seed in seeds:
            npz_path = os.path.join(results_dir, f"{strategy}_seed{seed}",
                                    "test_predictions.npz")
            if not os.path.exists(npz_path):
                print(f"  [Missing] {npz_path}")
                continue
            arr = np.load(npz_path)
            data[strategy][seed] = {
                'predictions': arr['predictions'],
                'labels':      arr['true_labels'],
            }
    return data


def compute_confusion_matrix(predictions, labels):
    """Row-normalised confusion matrix (rows = true class)."""
    cm = np.zeros((N_CLASSES, N_CLASSES))
    for t, p in zip(labels, predictions):
        cm[t, p] += 1
    return cm / (cm.sum(axis=1, keepdims=True) + 1e-10)


def aggregate_confusion_matrices(data):
    aggregated = {}
    for strategy, seeds_data in data.items():
        cms = [compute_confusion_matrix(v['predictions'], v['labels'])
               for v in seeds_data.values()]
        aggregated[strategy] = {
            'mean': np.mean(cms, axis=0),
            'std':  np.std(cms,  axis=0, ddof=1),
            'n':    len(cms),
        }
    return aggregated


def print_n3_table(aggregated, baseline='none'):
    n3 = 3
    col_w = 14
    header = f"{'Strategy':<12}" + ''.join(f"{f'{STAGE_NAMES[j]}':>{col_w}}"
                                            for j in range(N_CLASSES))
    print('\n── N3 confusion breakdown ──────────────────────────────────')
    print(header)
    print('─' * len(header))
    for strategy in STRATEGIES:
        if strategy not in aggregated: continue
        row = f"{strategy:<12}"
        for j in range(N_CLASSES):
            val  = aggregated[strategy]['mean'][n3, j]
            mark = ' ✓' if j == n3 else ''
            row += f"{val*100:>{col_w-2}.2f}%{mark:2}"
        print(row)

    # differences from baseline
    if baseline not in aggregated: return
    base_row = aggregated[baseline]['mean'][n3]
    print(f'\n── Δ vs {baseline} ─────────────────────────────────────────')
    print(header)
    print('─' * len(header))
    for strategy in STRATEGIES:
        if strategy == baseline or strategy not in aggregated: continue
        diff = aggregated[strategy]['mean'][n3] - base_row
        row  = f"{strategy:<12}"
        for j in range(N_CLASSES):
            row += f"{diff[j]*100:>+{col_w}.2f}%"[:-1] + ' '
        print(row)


def print_large_changes(aggregated, threshold=0.02, baseline='none'):
    if baseline not in aggregated: return
    base_cm = aggregated[baseline]['mean']
    print(f'\n── Largest confusion changes vs {baseline} (|Δ| > {threshold:.0%}) ──')
    for strategy in STRATEGIES:
        if strategy == baseline or strategy not in aggregated: continue
        diff    = aggregated[strategy]['mean'] - base_cm
        changes = [(i, j, diff[i, j]) for i in range(N_CLASSES)
                   for j in range(N_CLASSES)
                   if i != j and abs(diff[i, j]) > threshold]
        if not changes:
            print(f"  {strategy}: no changes above threshold")
            continue
        changes.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n  {strategy.upper()}:")
        for i, j, d in changes:
            arrow = '↑' if d > 0 else '↓'
            print(f"    {STAGE_NAMES[i]}→{STAGE_NAMES[j]}: {d*100:+.2f}% {arrow}")


def plot_n3_breakdown(aggregated, results_dir):
    strategies = [s for s in STRATEGIES if s in aggregated]
    n3         = 3
    categories = ['Correct', 'N2', 'Wake', 'N1', 'REM']
    stage_map  = {'Correct': 3, 'Wake': 0, 'N1': 1, 'N2': 2, 'REM': 4}

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(strategies))
    bottom = np.zeros(len(strategies))

    for cat in categories:
        j      = stage_map[cat]
        values = [aggregated[s]['mean'][n3, j] for s in strategies]
        ax.bar(x, values, 0.6, label=cat, bottom=bottom,
               color=STAGE_COLORS[cat], edgecolor='black', linewidth=0.8)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in strategies], fontsize=10)
    ax.set_ylabel('Fraction of N3 Samples', fontsize=10)
    ax.set_xlabel('Masking Strategy',        fontsize=10)
    ax.set_title('N3 Classification Breakdown by Masking Strategy',
                 fontsize=11, fontweight='bold')
    ax.legend(title='Predicted as', fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'n3_confusion_breakdown.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def save_summary(aggregated, results_dir):
    summary = {
        strategy: {
            'confusion_mean': info['mean'].tolist(),
            'confusion_std':  info['std'].tolist(),
            'n_seeds':        info['n'],
        }
        for strategy, info in aggregated.items()
    }
    out_path = os.path.join(results_dir, 'confusion_matrices_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    print("Loading predictions...")
    data = load_predictions(results_dir, STRATEGIES, SEEDS)
    if not data:
        raise SystemExit("No prediction files found. Run downstream_pairwise_accuracy.py first.")
    print(f"Loaded predictions for: {sorted(data.keys())}")

    print("Computing confusion matrices...")
    aggregated = aggregate_confusion_matrices(data)

    print_n3_table(aggregated)
    print_large_changes(aggregated)

    print("\nGenerating plots...")
    plot_n3_breakdown(aggregated, results_dir)
    save_summary(aggregated, results_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()