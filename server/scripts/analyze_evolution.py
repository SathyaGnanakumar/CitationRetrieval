#!/usr/bin/env python3
"""
Evolution analysis and reporting script.

Generates reports and visualizations from evolution data.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
_server_dir = Path(__file__).resolve().parents[1]
_root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_server_dir))
sys.path.insert(0, str(_root_dir))

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_evaluation_data(data_dir: str = "./data/evaluations") -> List[Dict[str, Any]]:
    """
    Load evaluation data from JSON files.

    Args:
        data_dir: Directory containing evaluation data

    Returns:
        List of evaluation dicts
    """
    data_path = Path(data_dir)
    summary_file = data_path / "summary.json"

    if not summary_file.exists():
        logger.warning(f"No summary file found at {summary_file}")
        return []

    with open(summary_file, "r") as f:
        data = json.load(f)

    return data


def load_version_metadata(data_dir: str = "./data/module_versions") -> Dict[str, Any]:
    """
    Load module version metadata.

    Args:
        data_dir: Directory containing module versions

    Returns:
        Dict with version metadata per module
    """
    data_path = Path(data_dir)
    versions = {}

    for module_dir in data_path.iterdir():
        if not module_dir.is_dir():
            continue

        metadata_file = module_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                versions[module_dir.name] = json.load(f)

    return versions


def plot_score_progression(evaluations: List[Dict[str, Any]], output_dir: str = "./reports"):
    """
    Plot score progression over time.

    Args:
        evaluations: List of evaluation dicts
        output_dir: Output directory for plots
    """
    if not evaluations:
        logger.warning("No evaluations to plot")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract scores and timestamps
    df = pd.DataFrame(evaluations)
    if "score" not in df.columns:
        logger.warning("No score data in evaluations")
        return

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["score"], marker="o", linewidth=2, markersize=4)
    plt.axhline(
        y=df["score"].mean(), color="r", linestyle="--", label=f'Average: {df["score"].mean():.4f}'
    )
    plt.xlabel("Query Number")
    plt.ylabel("Score")
    plt.title("Score Progression Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_file = output_path / "score_progression.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"✓ Saved score progression plot to {plot_file}")


def plot_metrics_distribution(evaluations: List[Dict[str, Any]], output_dir: str = "./reports"):
    """
    Plot distribution of metrics.

    Args:
        evaluations: List of evaluation dicts
        output_dir: Output directory for plots
    """
    if not evaluations:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    scores = [e["score"] for e in evaluations if "score" in e]

    if not scores:
        return

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(
        x=sum(scores) / len(scores),
        color="r",
        linestyle="--",
        label=f"Mean: {sum(scores)/len(scores):.4f}",
    )
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_path / "score_distribution.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"✓ Saved score distribution plot to {plot_file}")


def generate_summary_report(
    evaluations: List[Dict[str, Any]], versions: Dict[str, Any], output_dir: str = "./reports"
):
    """
    Generate text summary report.

    Args:
        evaluations: List of evaluation dicts
        versions: Module version metadata
        output_dir: Output directory for report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / "evolution_report.txt"

    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EVOLUTION SYSTEM REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Evaluation statistics
        f.write("EVALUATION STATISTICS\n")
        f.write("-" * 70 + "\n")

        if evaluations:
            scores = [e["score"] for e in evaluations if "score" in e]
            f.write(f"Total evaluations: {len(evaluations)}\n")
            f.write(f"Average score: {sum(scores)/len(scores):.4f}\n")
            f.write(f"Min score: {min(scores):.4f}\n")
            f.write(f"Max score: {max(scores):.4f}\n")
            f.write(f"Std deviation: {pd.Series(scores).std():.4f}\n")

            # Score ranges
            failures = len([s for s in scores if s < 0.3])
            moderate = len([s for s in scores if 0.3 <= s < 0.7])
            successes = len([s for s in scores if s >= 0.7])

            f.write(f"\nScore Distribution:\n")
            f.write(f"  Failures (<0.3):   {failures} ({failures/len(scores)*100:.1f}%)\n")
            f.write(f"  Moderate (0.3-0.7): {moderate} ({moderate/len(scores)*100:.1f}%)\n")
            f.write(f"  Successes (≥0.7):  {successes} ({successes/len(scores)*100:.1f}%)\n")
        else:
            f.write("No evaluation data available\n")

        f.write("\n")

        # Module versions
        f.write("MODULE VERSIONS\n")
        f.write("-" * 70 + "\n")

        if versions:
            for module_name, module_versions in versions.items():
                f.write(f"\n{module_name}:\n")

                if isinstance(module_versions, list) and module_versions:
                    f.write(f"  Total versions: {len(module_versions)}\n")

                    # Latest version
                    latest = module_versions[-1]
                    f.write(
                        f"  Latest: v{latest.get('version')} (score: {latest.get('score', 0):.4f})\n"
                    )

                    # Best version
                    best = max(module_versions, key=lambda v: v.get("score", 0))
                    f.write(f"  Best: v{best.get('version')} (score: {best.get('score', 0):.4f})\n")

                    # Version history
                    f.write(f"  Version history:\n")
                    for ver in module_versions[-5:]:  # Last 5 versions
                        f.write(
                            f"    v{ver.get('version')}: {ver.get('score', 0):.4f} ({ver.get('timestamp', 'unknown')})\n"
                        )
                else:
                    f.write("  No versions available\n")
        else:
            f.write("No version data available\n")

        f.write("\n")
        f.write("=" * 70 + "\n")

    logger.info(f"✓ Saved summary report to {report_file}")


def main():
    """Generate evolution analysis reports."""
    logger.info("=" * 70)
    logger.info("Evolution Analysis")
    logger.info("=" * 70)

    # Load data
    logger.info("\n📊 Loading evaluation data...")
    evaluations = load_evaluation_data()
    logger.info(f"✓ Loaded {len(evaluations)} evaluations")

    logger.info("\n📦 Loading module versions...")
    versions = load_version_metadata()
    logger.info(f"✓ Loaded versions for {len(versions)} modules")

    # Generate plots
    logger.info("\n📈 Generating plots...")
    plot_score_progression(evaluations)
    plot_metrics_distribution(evaluations)

    # Generate summary report
    logger.info("\n📄 Generating summary report...")
    generate_summary_report(evaluations, versions)

    logger.info("\n" + "=" * 70)
    logger.info("✅ Analysis complete!")
    logger.info("=" * 70)
    logger.info("Reports saved to ./reports/")
    logger.info("  - score_progression.png")
    logger.info("  - score_distribution.png")
    logger.info("  - evolution_report.txt")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
