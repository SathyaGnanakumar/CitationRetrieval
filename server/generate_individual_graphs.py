#!/usr/bin/env python3
"""
Generate individual graphs from baseline reranking results.
Creates separate, focused visualizations instead of one large dashboard.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("baseline_reranking_results")
OUTPUT_DIR = RESULTS_DIR / "individual_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
with open(RESULTS_DIR / "full_results.json") as f:
    full_results = json.load(f)

summary_df = pd.read_csv(RESULTS_DIR / "summary_table.csv")

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'BM25': '#FF6B6B',
    'E5': '#4ECDC4',
    'SPECTER': '#95E1D3',
    'baseline': '#95A5A6',
    'reranked': '#E67E22'
}

# ============================================================================
# 1. Individual Method Comparisons (Before/After Reranking)
# ============================================================================

def plot_method_comparison(method_name):
    """Create bar chart comparing baseline vs reranked for one method."""
    baseline_row = summary_df[summary_df['Method'] == method_name].iloc[0]
    reranked_row = summary_df[summary_df['Method'] == f"{method_name} + LLM RERANKER"].iloc[0]

    metrics = ['R@5', 'R@10', 'R@20', 'MRR']
    baseline_vals = [baseline_row[m] for m in metrics]
    reranked_vals = [reranked_row[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_vals, width, label='+ LLM Reranker',
                   color=COLORS['reranked'], alpha=0.8)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{method_name} Performance: Baseline vs LLM Reranking',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{method_name.lower()}_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {method_name.lower()}_comparison.png")


# Generate for each method
for method in ['BM25', 'E5', 'SPECTER']:
    plot_method_comparison(method)


# ============================================================================
# 2. Recall@K Curves (Individual for Each Method)
# ============================================================================

def plot_recall_curve(method_name):
    """Plot recall@k curve for one method."""
    baseline_row = summary_df[summary_df['Method'] == method_name].iloc[0]
    reranked_row = summary_df[summary_df['Method'] == f"{method_name} + LLM RERANKER"].iloc[0]

    k_values = [5, 10, 20]
    baseline_recalls = [baseline_row[f'R@{k}'] for k in k_values]
    reranked_recalls = [reranked_row[f'R@{k}'] for k in k_values]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, baseline_recalls, 'o-', linewidth=2.5, markersize=8,
            label='Baseline', color=COLORS['baseline'])
    ax.plot(k_values, reranked_recalls, 's-', linewidth=2.5, markersize=8,
            label='+ LLM Reranker', color=COLORS['reranked'])

    ax.set_xlabel('K (Top-K Results)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall@K', fontsize=11, fontweight='bold')
    ax.set_title(f'{method_name} Recall@K Curve', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{method_name.lower()}_recall_curve.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {method_name.lower()}_recall_curve.png")


for method in ['BM25', 'E5', 'SPECTER']:
    plot_recall_curve(method)


# ============================================================================
# 3. MRR Comparison (All Methods)
# ============================================================================

def plot_mrr_comparison():
    """Compare MRR across all methods."""
    methods = ['BM25', 'E5', 'SPECTER']
    baseline_mrrs = []
    reranked_mrrs = []

    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        baseline_mrrs.append(baseline_row['MRR'])
        reranked_mrrs.append(reranked_row['MRR'])

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, baseline_mrrs, width, label='Baseline',
                   color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_mrrs, width, label='+ LLM Reranker',
                   color=COLORS['reranked'], alpha=0.8)

    ax.set_ylabel('MRR (Mean Reciprocal Rank)', fontsize=11, fontweight='bold')
    ax.set_title('MRR Comparison: Baseline vs LLM Reranking',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mrr_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: mrr_comparison_all.png")


plot_mrr_comparison()


# ============================================================================
# 4. Improvement Percentages (Individual per Method)
# ============================================================================

def plot_improvement(method_name):
    """Show percentage improvement from reranking."""
    baseline_row = summary_df[summary_df['Method'] == method_name].iloc[0]
    reranked_row = summary_df[summary_df['Method'] == f"{method_name} + LLM RERANKER"].iloc[0]

    metrics = ['R@5', 'R@10', 'R@20', 'MRR']
    improvements = []

    for metric in metrics:
        baseline_val = baseline_row[metric]
        reranked_val = reranked_row[metric]
        if baseline_val > 0:
            improvement = ((reranked_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0 if reranked_val == 0 else 100
        improvements.append(improvement)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax.bar(metrics, improvements, color=colors, alpha=0.7)

    ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{method_name}: LLM Reranking Improvement',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{method_name.lower()}_improvement.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {method_name.lower()}_improvement.png")


for method in ['BM25', 'E5', 'SPECTER']:
    plot_improvement(method)


# ============================================================================
# 5. Side-by-Side Metric Comparison (One per Metric)
# ============================================================================

def plot_metric_comparison(metric):
    """Compare all methods for one metric."""
    methods = ['BM25', 'E5', 'SPECTER']
    baseline_vals = []
    reranked_vals = []

    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        baseline_vals.append(baseline_row[metric])
        reranked_vals.append(reranked_row[metric])

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_vals, width, label='+ LLM Reranker',
                   color=COLORS['reranked'], alpha=0.8)

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Across All Methods', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'metric_{metric.replace("@", "_at_")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: metric_{metric.replace('@', '_at_')}.png")


for metric in ['R@5', 'R@10', 'R@20', 'MRR']:
    plot_metric_comparison(metric)


# ============================================================================
# 6. Comprehensive All-Methods Grouped Bar Chart (Like the provided image)
# ============================================================================

def plot_all_methods_grouped():
    """Create grouped bar chart showing all metrics for all methods."""
    # Get all methods in order
    methods = ['BM25', 'BM25 + LLM RERANKER', 'E5', 'E5 + LLM RERANKER',
               'SPECTER', 'SPECTER + LLM RERANKER']

    metrics = ['R@5', 'R@10', 'R@20', 'MRR']

    # Extract data
    data = {metric: [] for metric in metrics}
    for method in methods:
        row = summary_df[summary_df['Method'] == method]
        if len(row) > 0:
            row = row.iloc[0]
            for metric in metrics:
                data[metric].append(row[metric])
        else:
            for metric in metrics:
                data[metric].append(0)

    # Set up the plot
    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    # Colors matching the image style
    colors = {
        'R@5': '#E8989C',    # Pink/salmon
        'R@10': '#D4A574',   # Orange/tan
        'R@20': '#82C68C',   # Green
        'MRR': '#6DB5B8'     # Teal/cyan
    }

    # Create bars
    bars = []
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bar = ax.bar(x + offset, data[metric], width, label=metric,
                     color=colors[metric], alpha=0.85)
        bars.append(bar)

    # Styling
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Performance Metrics', fontsize=13, fontweight='bold')
    ax.set_title('Baseline Retrieval with LLM Reranking - All Methods',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(data[m]) for m in metrics) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_methods_grouped.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: all_methods_grouped.png")


plot_all_methods_grouped()


# ============================================================================
# 7. Comprehensive Comparison Dashboard (Compact Version)
# ============================================================================

def plot_comprehensive_dashboard():
    """Create a comprehensive 2x2 dashboard with key comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Reranking Comprehensive Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    methods = ['BM25', 'E5', 'SPECTER']

    # 1. Top-left: Recall@5 comparison
    ax = axes[0, 0]
    baseline_r5 = []
    reranked_r5 = []
    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        baseline_r5.append(baseline_row['R@5'])
        reranked_r5.append(reranked_row['R@5'])

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, baseline_r5, width, label='Baseline',
                   color='#95A5A6', alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_r5, width, label='+ LLM Reranker',
                   color='#E67E22', alpha=0.8)
    ax.set_ylabel('Recall@5', fontsize=11, fontweight='bold')
    ax.set_title('Recall@5 Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Top-right: Recall@20 comparison
    ax = axes[0, 1]
    baseline_r20 = []
    reranked_r20 = []
    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        baseline_r20.append(baseline_row['R@20'])
        reranked_r20.append(reranked_row['R@20'])

    bars1 = ax.bar(x - width/2, baseline_r20, width, label='Baseline',
                   color='#95A5A6', alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_r20, width, label='+ LLM Reranker',
                   color='#E67E22', alpha=0.8)
    ax.set_ylabel('Recall@20', fontsize=11, fontweight='bold')
    ax.set_title('Recall@20 Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Bottom-left: MRR comparison
    ax = axes[1, 0]
    baseline_mrr = []
    reranked_mrr = []
    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        baseline_mrr.append(baseline_row['MRR'])
        reranked_mrr.append(reranked_row['MRR'])

    bars1 = ax.bar(x - width/2, baseline_mrr, width, label='Baseline',
                   color='#95A5A6', alpha=0.8)
    bars2 = ax.bar(x + width/2, reranked_mrr, width, label='+ LLM Reranker',
                   color='#E67E22', alpha=0.8)
    ax.set_ylabel('MRR', fontsize=11, fontweight='bold')
    ax.set_title('Mean Reciprocal Rank', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Bottom-right: Recall curves for all methods
    ax = axes[1, 1]
    k_values = [5, 10, 20]

    for method in methods:
        baseline_row = summary_df[summary_df['Method'] == method].iloc[0]
        baseline_recalls = [baseline_row[f'R@{k}'] for k in k_values]
        ax.plot(k_values, baseline_recalls, 'o--', linewidth=2, markersize=6,
                label=f'{method} (Baseline)', alpha=0.6)

        reranked_row = summary_df[summary_df['Method'] == f"{method} + LLM RERANKER"].iloc[0]
        reranked_recalls = [reranked_row[f'R@{k}'] for k in k_values]
        ax.plot(k_values, reranked_recalls, 'o-', linewidth=2.5, markersize=7,
                label=f'{method} (Reranked)')

    ax.set_xlabel('K (Top-K Results)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall@K', fontsize=11, fontweight='bold')
    ax.set_title('Recall@K Curves', fontsize=12, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: comprehensive_dashboard.png")


plot_comprehensive_dashboard()


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("✅ Generated individual graphs in:", OUTPUT_DIR)
print("="*60)
print("\nFiles created:")
print("  All Methods Grouped: all_methods_grouped.png")
print("  Comprehensive Dashboard: comprehensive_dashboard.png")
print("  Method Comparisons: bm25_comparison.png, e5_comparison.png, specter_comparison.png")
print("  Recall Curves: bm25_recall_curve.png, e5_recall_curve.png, specter_recall_curve.png")
print("  Improvements: bm25_improvement.png, e5_improvement.png, specter_improvement.png")
print("  Metric Views: metric_R_at_5.png, metric_R_at_10.png, metric_R_at_20.png, metric_MRR.png")
print("  Overall MRR: mrr_comparison_all.png")
print("="*60)
