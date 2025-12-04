"""Visualization and results saving for citation retrieval."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


class ResultsVisualizer:
    """Visualizes and saves citation retrieval results."""

    def __init__(self, results_dir: str = "./results/"):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory to save results and plots
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def save_results(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save results DataFrame to CSV.

        Args:
            df: Results DataFrame
            filename: Output filename
        """
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)
        print(f"💾 Saved results to {output_path}")

    def print_summary(self, method_name: str, metrics: Dict[str, float]) -> None:
        """
        Print formatted metrics summary.

        Args:
            method_name: Name of the retrieval method
            metrics: Dictionary of metric names and values
        """
        print(f"\n📊 {method_name} Summary:")
        print(f"  Recall@5 : {metrics.get('recall@5', 0.0):.3f}")
        print(f"  Recall@10: {metrics.get('recall@10', 0.0):.3f}")
        print(f"  MRR      : {metrics.get('mrr', 0.0):.3f}")
        print(f"  Top-1    : {metrics.get('top1', 0.0):.3f}")

    def plot_metric_comparison(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        output_filename: str = "metric_comparison.png"
    ) -> None:
        """
        Create bar chart comparing metrics across methods.

        Args:
            all_metrics: Dict mapping method names to their metrics
            output_filename: Output plot filename
        """
        if not all_metrics:
            print("⚠️  No metrics to plot")
            return

        # Prepare data for plotting
        methods = list(all_metrics.keys())
        metric_names = ["recall@5", "recall@10", "mrr", "top1"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle("Citation Retrieval: Method Comparison", fontsize=14, fontweight='bold')

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            values = [all_metrics[method].get(metric_name, 0.0) for method in methods]

            bars = ax.bar(methods, values, color=sns.color_palette("husl", len(methods)))
            ax.set_title(metric_name.upper().replace("@", " @ "), fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_path = self.results_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved comparison plot to {output_path}")
        plt.close()

    def plot_recall_curves(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        output_filename: str = "recall_curves.png"
    ) -> None:
        """
        Create line plot showing recall progression.

        Args:
            all_metrics: Dict mapping method names to their metrics
            output_filename: Output plot filename
        """
        if not all_metrics:
            print("⚠️  No metrics to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        k_values = [5, 10]
        for method_name, metrics in all_metrics.items():
            recall_values = [metrics.get(f"recall@{k}", 0.0) for k in k_values]
            ax.plot(k_values, recall_values, marker='o', linewidth=2,
                   markersize=8, label=method_name)

        ax.set_xlabel("k (Top-K)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Recall", fontsize=12, fontweight='bold')
        ax.set_title("Recall@K Comparison Across Methods", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(k_values)

        plt.tight_layout()
        output_path = self.results_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved recall curves to {output_path}")
        plt.close()

    def plot_method_heatmap(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        output_filename: str = "method_heatmap.png"
    ) -> None:
        """
        Create heatmap showing all metrics across methods.

        Args:
            all_metrics: Dict mapping method names to their metrics
            output_filename: Output plot filename
        """
        if not all_metrics:
            print("⚠️  No metrics to plot")
            return

        # Prepare data
        methods = list(all_metrics.keys())
        metric_names = ["Recall@5", "Recall@10", "MRR", "Top-1"]
        data = []

        for method in methods:
            row = [
                all_metrics[method].get("recall@5", 0.0),
                all_metrics[method].get("recall@10", 0.0),
                all_metrics[method].get("mrr", 0.0),
                all_metrics[method].get("top1", 0.0),
            ]
            data.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(methods) * 1.5))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=metric_names, yticklabels=methods,
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                   linewidths=0.5, ax=ax)

        ax.set_title("Performance Heatmap: All Methods & Metrics",
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_path = self.results_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved heatmap to {output_path}")
        plt.close()

    def save_summary_text(
        self,
        all_metrics: Dict[str, Dict[str, float]],
        output_filename: str = "summary.txt"
    ) -> None:
        """
        Save text summary of all metrics.

        Args:
            all_metrics: Dict mapping method names to their metrics
            output_filename: Output text filename
        """
        output_path = self.results_dir / output_filename

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CITATION RETRIEVAL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            for method_name, metrics in all_metrics.items():
                f.write(f"{method_name.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Recall@5  : {metrics.get('recall@5', 0.0):.4f}\n")
                f.write(f"  Recall@10 : {metrics.get('recall@10', 0.0):.4f}\n")
                f.write(f"  MRR       : {metrics.get('mrr', 0.0):.4f}\n")
                f.write(f"  Top-1     : {metrics.get('top1', 0.0):.4f}\n")
                f.write("\n")

        print(f"📄 Saved summary to {output_path}")

    def create_all_visualizations(
        self,
        all_metrics: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Generate all visualization plots.

        Args:
            all_metrics: Dict mapping method names to their metrics
        """
        print("\n🎨 Generating visualizations...")
        self.plot_metric_comparison(all_metrics)
        self.plot_recall_curves(all_metrics)
        self.plot_method_heatmap(all_metrics)
        self.save_summary_text(all_metrics)
        print("✅ All visualizations complete")
