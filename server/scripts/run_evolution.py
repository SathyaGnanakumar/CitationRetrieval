#!/usr/bin/env python3
"""
Continuous evolution script for production use.

Runs continuous learning loop with automated optimization.
"""

import os
import sys
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


def main():
    """Run continuous evolution loop."""
    load_dotenv()

    # Check if evolution is enabled
    evolution_enabled = os.getenv("ENABLE_DSPY_EVOLUTION", "false").lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    if not evolution_enabled:
        logger.error("=" * 70)
        logger.error("ERROR: ENABLE_DSPY_EVOLUTION is not enabled")
        logger.error("=" * 70)
        logger.error("Set ENABLE_DSPY_EVOLUTION=true in .env to enable evolution")
        logger.error("Current mode: Standard retrieval only")
        logger.error("=" * 70)
        return 1

    logger.info("=" * 70)
    logger.info("DSPy Continuous Evolution Loop")
    logger.info("=" * 70)
    logger.info("🔄 DSPy Evolution ENABLED - Starting evolution loop...")

    # Get configuration
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        logger.error("DATASET_DIR not set in .env")
        return 1

    # Check for OpenAI API key (needed for GPT-5 Mini teacher)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set - required for GPT-5 Mini teacher model")
        return 1

    # Get evolution configuration
    eval_interval = int(os.getenv("EVOLUTION_EVAL_INTERVAL", "100"))
    optimize_interval = int(os.getenv("EVOLUTION_OPTIMIZE_INTERVAL", "1000"))
    min_improvement = float(os.getenv("EVOLUTION_MIN_IMPROVEMENT", "0.05"))

    logger.info(f"\nConfiguration:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Eval interval: {eval_interval} queries")
    logger.info(f"  Optimize interval: {optimize_interval} queries")
    logger.info(f"  Min improvement: {min_improvement:.2%}")
    logger.info(f"  Teacher model: {os.getenv('TEACHER_MODEL', 'openai/gpt-5-mini')}")
    logger.info(f"  Student model: {os.getenv('STUDENT_MODEL', 'ollama_chat/gemma:2b')}")

    # Load dataset and build resources
    logger.info("\n📚 Loading dataset and building resources...")
    try:
        dataset = load_dataset(dataset_path)
        logger.info(f"✓ Loaded {len(dataset)} papers")

        # Try to load from cache
        use_cache = not os.getenv("NO_CACHE", "false").lower() in {"true", "1"}
        resources = None

        if use_cache:
            resources = load_resources(
                dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True
            )

        if not resources:
            corpus = build_citation_corpus(dataset)
            logger.info(f"✓ Built corpus with {len(corpus)} documents")

            resources = build_inmemory_resources(
                corpus, enable_bm25=True, enable_e5=True, enable_specter=True
            )
            logger.info("✓ Built retrieval resources")

            if use_cache:
                save_resources(resources, dataset_path)
                logger.info("✓ Saved resources to cache")
        else:
            logger.info("✓ Loaded resources from cache")
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        return 1

    # Initialize workflow with evolution enabled
    logger.info("\n🔧 Initializing workflow...")
    workflow = RetrievalWorkflow(use_llm_reranker=False, enable_evolution=True)
    logger.info("✓ Workflow initialized with evolution enabled")

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

    # Run initial baseline evaluation
    logger.info("\n" + "=" * 70)
    logger.info("📊 INITIAL BASELINE EVALUATION")
    logger.info("=" * 70)

    try:
        baseline_score = evolving_system.evaluate_batch(
            dataset[:eval_interval], max_queries=eval_interval
        )
        logger.info(
            f"\n✓ Baseline evaluation: {eval_interval} queries, avg_score={baseline_score:.4f}"
        )
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        return 1

    # Start continuous evolution loop
    logger.info("\n" + "=" * 70)
    logger.info("🔄 STARTING CONTINUOUS EVOLUTION LOOP")
    logger.info("=" * 70)
    logger.info(f"Will evaluate every {eval_interval} queries")
    logger.info(f"Will optimize every {optimize_interval} queries")
    logger.info(f"Press Ctrl+C to stop")
    logger.info("=" * 70 + "\n")

    try:
        evolving_system.continuous_evolution_loop(
            eval_interval=eval_interval,
            optimize_interval=optimize_interval,
            improvement_threshold=min_improvement,
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Evolution loop failed: {e}")
        return 1

    # Show final statistics
    logger.info("\n" + "=" * 70)
    logger.info("EVOLUTION STATISTICS")
    logger.info("=" * 70)

    # Show evaluation statistics
    eval_stats = evolving_system.eval_store.get_statistics()
    logger.info(f"\nEvaluation Store:")
    logger.info(f"  Total evaluations: {eval_stats.get('count', 0)}")
    logger.info(f"  Average score: {eval_stats.get('avg_score', 0):.4f}")
    logger.info(f"  Successes (≥0.7): {eval_stats.get('successes', 0)}")
    logger.info(f"  Failures (<0.3): {eval_stats.get('failures', 0)}")

    # Show module versions
    versions = workflow.get_module_versions()
    if versions:
        logger.info(f"\nModule Versions:")
        for module_name, info in versions.items():
            logger.info(f"  {module_name}:")
            logger.info(f"    Current: v{info.get('current_version')}")
            logger.info(
                f"    Best: v{info.get('best_version')} (score: {info.get('best_score', 0):.4f})"
            )

    logger.info("\n" + "=" * 70)
    logger.info("✅ Evolution loop complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
