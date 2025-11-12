"""
Visualization script for comparing citation retrieval models.

This script:
1. Scans results directory for all model evaluation results
2. Aggregates metrics from JSON files
3. Generates comprehensive comparison graphs
4. Saves plots to results/visualizations/

Usage:
    uv run visualization.py [--results-dir DIR] [--output-dir DIR]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class ResultsVisualizer:
    """Visualize and compare citation retrieval model results"""

    def __init__(self, results_dir: str = "results/example", output_dir: str = "results/visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_data = {}

    def load_all_results(self):
        """Load all metric and prediction JSON files from results directory"""
        print(f"🔍 Scanning {self.results_dir} for results...")

        # Find all metrics files
        metrics_files = list(self.results_dir.glob("*_metrics.json"))

        if not metrics_files:
            print(f"❌ No metrics files found in {self.results_dir}")
            return False

        print(f"   Found {len(metrics_files)} metrics file(s)")

        for metrics_file in metrics_files:
            # Extract model name from filename
            # Format: {model_name}_demo_{timestamp}_metrics.json
            filename = metrics_file.stem
            parts = filename.split("_")

            # Get model name (first part before _demo)
            model_name = parts[0]
            timestamp = "_".join(parts[2:-1]) if len(parts) > 3 else "unknown"

            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                metrics = metrics_data.get('metrics', metrics_data)  # Handle nested structure

            # Try to load corresponding predictions and failures
            base_name = filename.replace("_metrics", "")
            predictions_file = self.results_dir / f"{base_name}_predictions.json"
            failures_file = self.results_dir / f"{base_name}_failures.json"

            predictions = []
            failures = []

            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    predictions = json.load(f)

            if failures_file.exists():
                with open(failures_file, 'r') as f:
                    failures = json.load(f)

            # Store data
            key = f"{model_name}_{timestamp}"
            self.model_data[key] = {
                'model_name': model_name,
                'timestamp': timestamp,
                'metrics': metrics,
                'predictions': predictions,
                'failures': failures,
                'metrics_file': str(metrics_file),
                'avg_latency_ms': metrics_data.get('avg_latency_ms', 0)
            }

            print(f"   ✓ Loaded {model_name} ({timestamp})")

        print(f"\n✅ Loaded data for {len(self.model_data)} model run(s)")
        return True

    def aggregate_by_model(self) -> Dict[str, List[Dict]]:
        """Group results by model name (ignoring timestamps)"""
        aggregated = {}

        for key, data in self.model_data.items():
            model_name = data['model_name']
            if model_name not in aggregated:
                aggregated[model_name] = []
            aggregated[model_name].append(data)

        return aggregated

    def get_latest_results(self) -> Dict[str, Dict]:
        """Get the most recent results for each model"""
        aggregated = self.aggregate_by_model()
        latest = {}

        for model_name, runs in aggregated.items():
            # Sort by timestamp and get latest
            runs_sorted = sorted(runs, key=lambda x: x['timestamp'], reverse=True)
            latest[model_name] = runs_sorted[0]

        return latest

    def plot_recall_comparison(self, latest_results: Dict[str, Dict]):
        """Plot Recall@K comparison across models"""
        print("\n📊 Generating Recall@K comparison plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(latest_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for idx, (model_name, data) in enumerate(latest_results.items()):
            metrics = data['metrics']

            # Extract recall@k values
            k_values = []
            recall_values = []

            for key, value in sorted(metrics.items()):
                if key.startswith('recall@'):
                    k = int(key.split('@')[1])
                    k_values.append(k)
                    recall_values.append(value)

            if k_values:
                ax.plot(k_values, recall_values, marker='o', linewidth=2,
                       label=model_name.upper(), color=colors[idx])

        ax.set_xlabel('K (Number of Retrieved Documents)', fontsize=12)
        ax.set_ylabel('Recall@K', fontsize=12)
        ax.set_title('Recall@K Comparison Across Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        output_file = self.output_dir / "recall_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_mrr_comparison(self, latest_results: Dict[str, Dict]):
        """Plot MRR comparison across models"""
        print("\n📊 Generating MRR comparison plot...")

        fig, ax = plt.subplots(figsize=(8, 6))

        models = list(latest_results.keys())
        mrr_values = [data['metrics']['mrr'] for data in latest_results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        bars = ax.bar(models, mrr_values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Mean Reciprocal Rank (MRR)', fontsize=12)
        ax.set_title('MRR Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(mrr_values) * 1.15])
        ax.grid(True, alpha=0.3, axis='y')

        # Uppercase model names on x-axis
        ax.set_xticklabels([m.upper() for m in models])

        output_file = self.output_dir / "mrr_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_precision_comparison(self, latest_results: Dict[str, Dict]):
        """Plot Precision@K comparison across models"""
        print("\n📊 Generating Precision@K comparison plot...")

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(latest_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for idx, (model_name, data) in enumerate(latest_results.items()):
            metrics = data['metrics']

            # Extract precision@k values
            k_values = []
            precision_values = []

            for key, value in sorted(metrics.items()):
                if key.startswith('precision@'):
                    k = int(key.split('@')[1])
                    k_values.append(k)
                    precision_values.append(value)

            if k_values:
                ax.plot(k_values, precision_values, marker='s', linewidth=2,
                       label=model_name.upper(), color=colors[idx])

        ax.set_xlabel('K (Number of Retrieved Documents)', fontsize=12)
        ax.set_ylabel('Precision@K', fontsize=12)
        ax.set_title('Precision@K Comparison Across Models', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max([max([v for k, v in data['metrics'].items() if k.startswith('precision@')])
                             for data in latest_results.values()]) * 1.1])

        output_file = self.output_dir / "precision_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_latency_comparison(self, latest_results: Dict[str, Dict]):
        """Plot latency comparison across models"""
        print("\n📊 Generating latency comparison plot...")

        fig, ax = plt.subplots(figsize=(8, 6))

        models = []
        latencies = []

        for model_name, data in latest_results.items():
            if data.get('avg_latency_ms', 0) > 0:
                models.append(model_name)
                latencies.append(data['avg_latency_ms'])

        if not latencies:
            print("   ⚠️  No latency data available, skipping plot")
            return

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, latencies, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Average Latency (ms)', fontsize=12)
        ax.set_title('Latency Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(latencies) * 1.15])
        ax.grid(True, alpha=0.3, axis='y')

        # Uppercase model names on x-axis
        ax.set_xticklabels([m.upper() for m in models])

        output_file = self.output_dir / "latency_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_success_rate_comparison(self, latest_results: Dict[str, Dict]):
        """Plot success rate (exact match rate) comparison"""
        print("\n📊 Generating exact match rate comparison plot...")

        fig, ax = plt.subplots(figsize=(8, 6))

        models = list(latest_results.keys())
        exact_match_rates = [data['metrics'].get('exact_match_rate', 0) for data in latest_results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        bars = ax.bar(models, exact_match_rates, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Exact Match Rate', fontsize=12)
        ax.set_title('Exact Match Rate Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(exact_match_rates) * 1.15 if exact_match_rates else 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Uppercase model names on x-axis
        ax.set_xticklabels([m.upper() for m in models])

        output_file = self.output_dir / "exact_match_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_metrics_heatmap(self, latest_results: Dict[str, Dict]):
        """Create a heatmap of all key metrics"""
        print("\n📊 Generating metrics heatmap...")

        # Select key metrics to display
        key_metrics = ['recall@1', 'recall@5', 'recall@10', 'recall@20',
                      'precision@1', 'precision@5', 'precision@10', 'mrr', 'exact_match_rate']

        models = list(latest_results.keys())
        data_matrix = []
        available_metrics = []

        # Build data matrix
        for metric in key_metrics:
            row = []
            has_data = False
            for model_name in models:
                metrics = latest_results[model_name]['metrics']
                if metric in metrics:
                    row.append(metrics[metric])
                    has_data = True
                else:
                    row.append(0)

            if has_data:
                data_matrix.append(row)
                available_metrics.append(metric)

        if not data_matrix:
            print("   ⚠️  No metrics available for heatmap, skipping plot")
            return

        data_matrix = np.array(data_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(available_metrics)))
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_yticklabels(available_metrics)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(available_metrics)):
            for j in range(len(models)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        ax.set_title("Metrics Heatmap Comparison", fontsize=14, fontweight='bold', pad=20)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)

        output_file = self.output_dir / "metrics_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def plot_comprehensive_overview(self, latest_results: Dict[str, Dict]):
        """Create a comprehensive overview with multiple subplots"""
        print("\n📊 Generating comprehensive overview...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        models = list(latest_results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        # 1. Recall@K (top-left, large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for idx, (model_name, data) in enumerate(latest_results.items()):
            metrics = data['metrics']
            k_values = []
            recall_values = []
            for key, value in sorted(metrics.items()):
                if key.startswith('recall@'):
                    k = int(key.split('@')[1])
                    k_values.append(k)
                    recall_values.append(value)
            if k_values:
                ax1.plot(k_values, recall_values, marker='o', linewidth=2,
                        label=model_name.upper(), color=colors[idx])
        ax1.set_xlabel('K')
        ax1.set_ylabel('Recall@K')
        ax1.set_title('Recall@K Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # 2. MRR (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        mrr_values = [data['metrics']['mrr'] for data in latest_results.values()]
        bars = ax2.bar(range(len(models)), mrr_values, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
        ax2.set_ylabel('MRR')
        ax2.set_title('MRR Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 3. Precision@K (middle-right)
        ax3 = fig.add_subplot(gs[1, 2])
        # Show precision@10 as bar chart
        prec_values = []
        for model_name in models:
            metrics = latest_results[model_name]['metrics']
            prec_values.append(metrics.get('precision@10', 0))
        bars = ax3.bar(range(len(models)), prec_values, color=colors, edgecolor='black')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
        ax3.set_ylabel('Precision@10')
        ax3.set_title('Precision@10', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. Exact Match Rate (bottom-left)
        ax4 = fig.add_subplot(gs[2, 0])
        exact_match_values = []
        for model_name in models:
            metrics = latest_results[model_name]['metrics']
            exact_match_values.append(metrics.get('exact_match_rate', 0))
        bars = ax4.bar(range(len(models)), exact_match_values, color=colors, edgecolor='black')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
        ax4.set_ylabel('Exact Match Rate')
        ax4.set_title('Exact Match Rate', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1.05])
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # 5. Latency (bottom-middle)
        ax5 = fig.add_subplot(gs[2, 1])
        latencies = []
        latency_models = []
        for model_name in models:
            if latest_results[model_name].get('avg_latency_ms', 0) > 0:
                latencies.append(latest_results[model_name]['avg_latency_ms'])
                latency_models.append(model_name)
        if latencies:
            model_colors = [colors[models.index(m)] for m in latency_models]
            bars = ax5.bar(range(len(latency_models)), latencies, color=model_colors, edgecolor='black')
            ax5.set_xticks(range(len(latency_models)))
            ax5.set_xticklabels([m.upper() for m in latency_models], rotation=45, ha='right')
            ax5.set_ylabel('Latency (ms)')
            ax5.set_title('Avg Latency', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=9)

        # 6. Failure count (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2])
        failure_counts = []
        total_counts = []
        for model_name in models:
            failure_counts.append(len(latest_results[model_name]['failures']))
            total_counts.append(len(latest_results[model_name]['predictions']))
        failure_rates = [f/t if t > 0 else 0 for f, t in zip(failure_counts, total_counts)]
        bars = ax6.bar(range(len(models)), failure_rates, color=colors, edgecolor='black')
        ax6.set_xticks(range(len(models)))
        ax6.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
        ax6.set_ylabel('Failure Rate')
        ax6.set_title('Failure Rate', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0, 1.05])
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        fig.suptitle('Citation Retrieval Models - Comprehensive Comparison',
                    fontsize=16, fontweight='bold', y=0.995)

        output_file = self.output_dir / "comprehensive_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved to {output_file}")
        plt.close()

    def generate_summary_report(self, latest_results: Dict[str, Dict]):
        """Generate a text summary report"""
        print("\n📊 Generating summary report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CITATION RETRIEVAL EVALUATION - SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models evaluated: {len(latest_results)}")
        report_lines.append("\n" + "=" * 80)

        for model_name, data in latest_results.items():
            metrics = data['metrics']

            report_lines.append(f"\n{model_name.upper()}")
            report_lines.append("-" * 40)
            report_lines.append(f"Timestamp: {data['timestamp']}")
            report_lines.append(f"Total predictions: {len(data['predictions'])}")
            report_lines.append(f"Failures: {len(data['failures'])}")
            report_lines.append(f"Failure rate: {len(data['failures'])/len(data['predictions'])*100:.1f}%")
            report_lines.append("\nKey Metrics:")
            report_lines.append(f"  MRR: {metrics['mrr']:.4f}")

            # Recall metrics
            for k in [1, 5, 10, 20]:
                key = f'recall@{k}'
                if key in metrics:
                    report_lines.append(f"  Recall@{k}: {metrics[key]:.4f}")

            # Precision metrics
            for k in [5, 10]:
                key = f'precision@{k}'
                if key in metrics:
                    report_lines.append(f"  Precision@{k}: {metrics[key]:.4f}")

            # Exact match rate
            if 'exact_match_rate' in metrics:
                report_lines.append(f"  Exact Match Rate: {metrics['exact_match_rate']:.4f}")

            # Latency
            if data.get('avg_latency_ms', 0) > 0:
                report_lines.append(f"  Avg Latency: {data['avg_latency_ms']:.2f}ms")

            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("\nBest performing model by metric:")
        report_lines.append("-" * 40)

        # Find best for each metric
        metrics_to_compare = ['mrr', 'recall@10', 'recall@20', 'precision@10']
        for metric in metrics_to_compare:
            best_model = None
            best_value = -1
            for model_name, data in latest_results.items():
                if metric in data['metrics']:
                    value = data['metrics'][metric]
                    if value > best_value:
                        best_value = value
                        best_model = model_name
            if best_model:
                report_lines.append(f"  {metric}: {best_model.upper()} ({best_value:.4f})")

        report_lines.append("\n" + "=" * 80)

        report_text = "\n".join(report_lines)

        # Save report
        output_file = self.output_dir / "summary_report.txt"
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n   ✓ Report saved to {output_file}")

    def visualize_all(self):
        """Run all visualizations"""
        print("\n" + "=" * 70)
        print("CITATION RETRIEVAL VISUALIZATION")
        print("=" * 70)

        # Load results
        if not self.load_all_results():
            return

        # Get latest results for each model
        latest_results = self.get_latest_results()

        if not latest_results:
            print("❌ No results to visualize")
            return

        print(f"\n📈 Generating visualizations for {len(latest_results)} model(s)...")

        # Generate all plots
        self.plot_recall_comparison(latest_results)
        self.plot_mrr_comparison(latest_results)
        self.plot_precision_comparison(latest_results)
        self.plot_success_rate_comparison(latest_results)
        self.plot_latency_comparison(latest_results)
        self.plot_metrics_heatmap(latest_results)
        self.plot_comprehensive_overview(latest_results)

        # Generate summary report
        self.generate_summary_report(latest_results)

        print("\n" + "=" * 70)
        print("✅ VISUALIZATION COMPLETE!")
        print("=" * 70)
        print(f"\nAll visualizations saved to: {self.output_dir}/")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize citation retrieval evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/example",
        help="Directory containing result JSON files (default: results/example)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualization outputs (default: results/visualizations)"
    )

    args = parser.parse_args()

    visualizer = ResultsVisualizer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )

    visualizer.visualize_all()


if __name__ == "__main__":
    main()
