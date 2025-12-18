#!/usr/bin/env python3
"""
One-time optimization script for testing and development.

Runs baseline evaluation, optimizes with GEPA, and compares results.
"""

import os
import sys
import warnings

# Suppress tokenizer warnings early before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

import argparse
from pathlib import Path

# Add parent directories to path for imports
_server_dir = Path(__file__).resolve().parents[1]
_root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_server_dir))
sys.path.insert(0, str(_root_dir))

import logging
from dotenv import load_dotenv

from src.workflow import RetrievalWorkflow
from src.agents.self_evolve.evolution_engine import SelfEvolvingRetrievalSystem
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources, save_resources
from corpus_loaders.scholarcopilot import load_dataset, build_citation_corpus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="One-time GEPA optimization run")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Number of queries for baseline and post-optimization evals",
    )
    parser.add_argument(
        "--use-llm-reranker",
        action="store_true",
        help="Use LLM-based reranker instead of cross-encoder",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save eval checkpoint every N queries (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="data/evaluations/optimize_once_checkpoint.json",
        help="Path to checkpoint JSON file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving cached resources (BM25/E5/SPECTER)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: only show progress bars, loader info, and final results",
    )
    return parser.parse_args()


def main():
    """Run one-time optimization."""
    args = parse_args()
    load_dotenv()

    # Quiet mode: silence per-query and workflow logs, keep loaders and summary
    if args.quiet:
        warnings.filterwarnings("ignore")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        try:
            import transformers

            transformers.logging.set_verbosity_error()
        except ImportError:
            pass
        for noisy in [
            "src.agents",
            "src.workflow",
            "src.agents.self_evolve.evolution_engine",
            "src.agents.formulators",
            "src.agents.retrievers",
            "langchain",
            "httpx",
            "httpcore",
            "transformers",
        ]:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # Temporarily enable evolution for this script
    os.environ["ENABLE_DSPY_EVOLUTION"] = "true"

    logger.info("=" * 70)
    logger.info("DSPy One-Time Optimization")
    logger.info("=" * 70)

    # Get configuration
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        logger.error("DATASET_DIR not set in .env")
        return 1

    # Check for OpenAI API key (needed for GPT-5 Mini teacher)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set - required for GPT-5 Mini teacher model")
        return 1

    logger.info(f"Dataset: {dataset_path}")

    # Load dataset and build/load resources
    logger.info("\n📚 Loading dataset and building resources...")
    try:
        dataset = load_dataset(dataset_path)
        logger.info(f"✓ Loaded {len(dataset)} papers")

        corpus = build_citation_corpus(dataset)
        logger.info(f"✓ Built corpus with {len(corpus)} documents")

        resources = None
        if not args.no_cache:
            resources = load_resources(
                dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True
            )
            if resources:
                logger.info("✓ Loaded resources from cache")

        if not resources:
            resources = build_inmemory_resources(
                corpus, enable_bm25=True, enable_e5=True, enable_specter=True
            )
            logger.info("✓ Built retrieval resources")
            if not args.no_cache:
                save_resources(resources, dataset_path)
                logger.info("✓ Saved resources to cache")
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        return 1

    # Initialize workflow with evolution enabled
    logger.info("\n🔧 Initializing workflow...")
    workflow = RetrievalWorkflow(use_llm_reranker=args.use_llm_reranker, enable_evolution=True)
    logger.info("✓ Workflow initialized")

    # Create self-evolving system
    logger.info("\n⚡ Creating self-evolving system...")
    try:
        evolving_system = SelfEvolvingRetrievalSystem(
            workflow=workflow,
            resources=resources,
            dataset_path=dataset_path,
            local_model=os.getenv("STUDENT_MODEL", "ollama_chat/gemma:2b"),
            teacher_model=os.getenv("TEACHER_MODEL", "openai/gpt-5-mini"),
        )
        logger.info("✓ Self-evolving system created")
    except Exception as e:
        logger.error(f"Failed to create self-evolving system: {e}")
        return 1

    # Run baseline evaluation
    logger.info("\n" + "=" * 70)
    logger.info("📊 BASELINE EVALUATION")
    logger.info("=" * 70)

    try:
        baseline_slice = dataset[: args.max_queries]
        baseline_results = evolving_system.evaluate_batch(
            baseline_slice,
            max_queries=args.max_queries,
            checkpoint_every=args.checkpoint_every or None,
            checkpoint_file=str(Path(args.checkpoint_file)),
        )
        baseline_score = baseline_results["score"]
        logger.info(f"\n✓ Baseline weighted score: {baseline_score:.4f}")
        logger.info(f"  R@5:  {baseline_results['R@5']:.4f}")
        logger.info(f"  R@10: {baseline_results['R@10']:.4f}")
        logger.info(f"  MRR:  {baseline_results['MRR']:.4f}")
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        return 1

    # Run optimization
    logger.info("\n" + "=" * 70)
    logger.info("⚡ OPTIMIZATION WITH GEPA")
    logger.info("=" * 70)

    try:
        optimized = evolving_system.optimize_module(
            module_name="picker", min_score=0.5, auto_budget="medium"
        )

        if optimized:
            logger.info("✓ Optimization completed successfully")
        else:
            logger.warning("⚠️  Optimization did not produce a result")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1

    # Post-optimization evaluation
    logger.info("\n" + "=" * 70)
    logger.info("📊 POST-OPTIMIZATION EVALUATION")
    logger.info("=" * 70)

    try:
        start = args.max_queries
        end = start + args.max_queries
        optimized_results = evolving_system.evaluate_batch(
            dataset[start:end],
            max_queries=args.max_queries,
            checkpoint_every=args.checkpoint_every or None,
            checkpoint_file=str(Path(args.checkpoint_file)),
        )
        optimized_score = optimized_results["score"]
        logger.info(f"\n✓ Optimized weighted score: {optimized_score:.4f}")
        logger.info(f"  R@5:  {optimized_results['R@5']:.4f}")
        logger.info(f"  R@10: {optimized_results['R@10']:.4f}")
        logger.info(f"  MRR:  {optimized_results['MRR']:.4f}")

        improvement = optimized_score - baseline_score
        if baseline_score != 0:
            pct_improvement = improvement / baseline_score * 100
            pct_text = f"{pct_improvement:+.1f}%"
        else:
            pct_text = "n/a (baseline=0)"
        logger.info(f"✓ Improvement: {improvement:+.4f} ({pct_text})")
    except Exception as e:
        logger.error(f"Post-optimization evaluation failed: {e}")
        return 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Individual metrics comparison
    logger.info("")
    logger.info(f"{'Metric':<12} {'Baseline':>10} {'Optimized':>10} {'Change':>12}")
    logger.info("-" * 46)

    for metric in ["R@5", "R@10", "MRR"]:
        base_val = baseline_results[metric]
        opt_val = optimized_results[metric]
        diff = opt_val - base_val
        if base_val != 0:
            pct = diff / base_val * 100
            change_str = f"{diff:+.4f} ({pct:+.1f}%)"
        else:
            change_str = f"{diff:+.4f}"
        logger.info(f"{metric:<12} {base_val:>10.4f} {opt_val:>10.4f} {change_str:>12}")

    logger.info("-" * 46)

    # Weighted score
    if baseline_score != 0:
        pct_improvement = improvement / baseline_score * 100
        pct_text = f"{pct_improvement:+.1f}%"
    else:
        pct_text = "n/a (baseline=0)"

    logger.info(
        f"{'Weighted':<12} {baseline_score:>10.4f} {optimized_score:>10.4f} {improvement:+.4f} ({pct_text})"
    )
    logger.info("")
    logger.info("(Weighted = 0.4×R@5 + 0.3×R@10 + 0.3×MRR)")
    logger.info("=" * 70)

    # Check module versions
    versions = workflow.get_module_versions()
    if versions:
        logger.info("\nModule Versions:")
        for module_name, info in versions.items():
            logger.info(
                f"  {module_name}: v{info.get('current_version')} (best: v{info.get('best_version')}, score: {info.get('best_score', 0):.4f})"
            )

    logger.info("\n✅ One-time optimization complete!")
    logger.info("\nTo enable continuous evolution, set ENABLE_DSPY_EVOLUTION=true in .env")

    return 0


if __name__ == "__main__":
    sys.exit(main())
