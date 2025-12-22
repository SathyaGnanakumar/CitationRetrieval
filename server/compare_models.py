#!/usr/bin/env python3
"""
Model Comparison Script for Citation Retrieval System

Compare the performance of different LLM models (OpenAI, Anthropic, etc.) for reranking.
Evaluates models on metrics like Recall@k, MRR, latency, and estimated cost.

Usage:
    # Compare default models
    uv run python compare_models.py --num-queries 20

    # Compare specific models
    uv run python compare_models.py --models gpt-5-mini-2025-08-07 gpt-5-2025-08-07 gpt-4o-mini --num-queries 50

    # Compare with custom k values
    uv run python compare_models.py --num-queries 30 --k 5 10 20

    # Save results to file
    uv run python compare_models.py --num-queries 100 --output results/model_comparison.json
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add parent directory to path for corpus import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus.scholarcopilot import build_citation_corpus, load_dataset
from src.evaluation.metrics import calculate_recall_at_k
from src.workflow import RetrievalWorkflow

load_dotenv()

# Model pricing (per 1M tokens) - approximate as of 2025
MODEL_PRICING = {
    "gpt-5-mini-2025-08-07": {"input": 0.15, "output": 0.60},  # GPT-5-mini pricing
    "gpt-5-2025-08-07": {"input": 2.50, "output": 10.00},  # GPT-5 pricing
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # GPT-4o-mini pricing
    "gpt-4o": {"input": 5.00, "output": 15.00},  # GPT-4o pricing
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},  # GPT-4 Turbo pricing
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},  # Claude 3.5 Sonnet
}

# Approximate token estimates
AVG_PROMPT_TOKENS = 1500  # Average tokens in reranking prompt
AVG_COMPLETION_TOKENS = 200  # Average tokens in completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def estimate_cost(model: str, num_queries: int) -> float:
    """Estimate the cost for running queries with a given model."""
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    input_cost = (AVG_PROMPT_TOKENS * num_queries / 1_000_000) * pricing["input"]
    output_cost = (AVG_COMPLETION_TOKENS * num_queries / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def run_evaluation_with_model(
    workflow: RetrievalWorkflow,
    queries: List[Dict[str, Any]],
    resources: Dict[str, Any],
    model: str,
    k: int = 20,
    use_openai: bool = True
) -> Tuple[Dict[str, List[float]], float]:
    """
    Run evaluation with a specific model.

    Returns:
        Tuple of (metrics_dict, avg_latency_seconds)
    """
    recalls_at_k = defaultdict(list)
    mrr_scores = []
    latencies = []

    # Set environment variable for the model
    original_model = os.environ.get("OPENAI_RERANKER_MODEL")
    original_use_openai = os.environ.get("USE_OPENAI_RERANKER")

    try:
        os.environ["OPENAI_RERANKER_MODEL"] = model
        os.environ["USE_OPENAI_RERANKER"] = "true" if use_openai else "false"

        for i, query_data in enumerate(queries):
            query = query_data.get("citation_context", "")
            gold_ids = set(query_data.get("positive_ids", []))

            if not query or not gold_ids:
                continue

            logger.info(f"[{model}] Processing query {i+1}/{len(queries)}")

            # Run workflow with timing
            start_time = time.time()

            try:
                result = workflow.run({
                    "messages": [HumanMessage(content=query)],
                    "resources": resources,
                    "config": {"k": k}
                })

                latency = time.time() - start_time
                latencies.append(latency)

                # Extract ranked papers
                ranked_papers = result.get("ranked_papers", [])
                retrieved_ids = [p.get("id") for p in ranked_papers if p.get("id")]

                # Calculate metrics
                for k_val in [5, 10, 20]:
                    recall = calculate_recall_at_k(retrieved_ids, gold_ids, k=k_val)
                    recalls_at_k[f"recall@{k_val}"].append(recall)

                # Calculate MRR
                mrr = 0.0
                for rank, paper_id in enumerate(retrieved_ids, 1):
                    if paper_id in gold_ids:
                        mrr = 1.0 / rank
                        break
                mrr_scores.append(mrr)

            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                continue

        # Calculate averages
        metrics = {
            "recall@5": recalls_at_k["recall@5"],
            "recall@10": recalls_at_k["recall@10"],
            "recall@20": recalls_at_k["recall@20"],
            "mrr": mrr_scores
        }

        avg_latency = np.mean(latencies) if latencies else 0.0

        return metrics, avg_latency

    finally:
        # Restore original environment
        if original_model:
            os.environ["OPENAI_RERANKER_MODEL"] = original_model
        elif "OPENAI_RERANKER_MODEL" in os.environ:
            del os.environ["OPENAI_RERANKER_MODEL"]

        if original_use_openai:
            os.environ["USE_OPENAI_RERANKER"] = original_use_openai
        elif "USE_OPENAI_RERANKER" in os.environ:
            del os.environ["USE_OPENAI_RERANKER"]


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 100)

    # Header
    print(f"{'Model':<35} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'MRR':>8} {'Latency (s)':>12} {'Est. Cost ($)':>15}")
    print("-" * 100)

    # Sort by Recall@20 (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["avg_metrics"]["recall@20"],
        reverse=True
    )

    for model, data in sorted_models:
        metrics = data["avg_metrics"]
        print(
            f"{model:<35} "
            f"{metrics['recall@5']:>8.4f} "
            f"{metrics['recall@10']:>8.4f} "
            f"{metrics['recall@20']:>8.4f} "
            f"{metrics['mrr']:>8.4f} "
            f"{data['avg_latency']:>12.2f} "
            f"{data['estimated_cost']:>15.4f}"
        )

    print("=" * 100)

    # Print best model
    best_model = sorted_models[0][0]
    best_metrics = sorted_models[0][1]["avg_metrics"]
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Recall@20: {best_metrics['recall@20']:.4f}")
    print(f"   MRR: {best_metrics['mrr']:.4f}")
    print(f"   Avg Latency: {sorted_models[0][1]['avg_latency']:.2f}s")
    print(f"   Est. Cost: ${sorted_models[0][1]['estimated_cost']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare different LLM models for citation retrieval reranking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (default: from DATASET_DIR env var)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5-mini-2025-08-07", "gpt-4o-mini"],
        help="Models to compare (default: gpt-5-mini-2025-08-07 gpt-4o-mini)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of queries to evaluate (default: 20)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of results to retrieve (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling (default: 42)"
    )

    args = parser.parse_args()

    # Load dataset
    dataset_path = args.dataset or os.getenv("DATASET_DIR", "corpus/scholarcopilot/scholar_copilot_eval_data_1k.json")
    logger.info(f"Loading dataset from: {dataset_path}")

    dataset = load_dataset(dataset_path)
    corpus = build_citation_corpus(dataset)

    logger.info(f"Loaded {len(corpus)} papers from corpus")

    # Sample queries
    np.random.seed(args.seed)
    queries = dataset[:args.num_queries] if len(dataset) >= args.num_queries else dataset

    logger.info(f"Evaluating on {len(queries)} queries")
    logger.info(f"Models to compare: {', '.join(args.models)}")

    # Initialize workflow
    workflow = RetrievalWorkflow()

    # Build resources (shared across all models)
    from corpus.builders import build_inmemory_resources
    logger.info("Building retrieval resources (this may take a few minutes)...")
    resources = build_inmemory_resources(corpus)

    # Run evaluation for each model
    results = {}

    for model in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model}")
        logger.info(f"{'='*60}")

        metrics, avg_latency = run_evaluation_with_model(
            workflow=workflow,
            queries=queries,
            resources=resources,
            model=model,
            k=args.k,
            use_openai=True
        )

        # Calculate averages
        avg_metrics = {
            metric_name: np.mean(values) if values else 0.0
            for metric_name, values in metrics.items()
        }

        # Estimate cost
        estimated_cost = estimate_cost(model, len(queries))

        results[model] = {
            "avg_metrics": avg_metrics,
            "avg_latency": avg_latency,
            "estimated_cost": estimated_cost,
            "num_queries": len(queries)
        }

        logger.info(f"Completed evaluation for {model}")
        logger.info(f"  Recall@20: {avg_metrics['recall@20']:.4f}")
        logger.info(f"  MRR: {avg_metrics['mrr']:.4f}")
        logger.info(f"  Avg Latency: {avg_latency:.2f}s")
        logger.info(f"  Est. Cost: ${estimated_cost:.4f}")

    # Print comparison table
    print_comparison_table(results)

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
