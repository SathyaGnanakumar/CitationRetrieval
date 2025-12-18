#!/usr/bin/env python3
"""
Simple test for LLM reranker without conditional routing.
Creates a minimal workflow to isolate the LLM reranker functionality.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loaders.scholarcopilot import load_dataset, build_citation_corpus
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources
from src.agents.formulators.query_reformulator import query_reformulator
from src.agents.retrievers.bm25_agent import bm25_agent
from src.agents.retrievers.e5_agent import e5_agent
from src.agents.retrievers.specter_agent import specter_agent
from src.agents.formulators.aggregator import aggregator
from src.agents.formulators.llm_agent import llm_reranker
from src.evaluation.metrics import evaluate_retrieval
from evaluate import get_ground_truth_ids, extract_query_from_paper

load_dotenv()


def process_single_paper(paper, resources, verbose=False):
    """Process a single paper through the pipeline."""
    # Extract query
    paper_text = paper.get("paper", "")
    query_str = extract_query_from_paper(paper_text, max_words=50)

    # Get ground truth
    relevant_ids = get_ground_truth_ids(paper)
    if not relevant_ids:
        return None  # Skip papers without ground truth

    # Build state manually step by step
    state = {
        "messages": [HumanMessage(content=query_str)],
        "resources": resources,
        "config": {"k": 20},
    }

    # Run pipeline
    ref_output = query_reformulator(state)
    state.update(ref_output)

    bm25_output = bm25_agent(state)
    state.update(bm25_output)

    e5_output = e5_agent(state)
    state.update(e5_output)

    specter_output = specter_agent(state)
    state.update(specter_output)

    agg_output = aggregator(state)
    state.update(agg_output)

    rerank_output = llm_reranker(state)
    state.update(rerank_output)

    ranked = state.get("ranked_papers", [])

    # Calculate metrics
    papers_for_eval = [item[0] if isinstance(item, tuple) else item for item in ranked]
    metrics = evaluate_retrieval(papers_for_eval, relevant_ids, k_values=[5, 10, 20])

    return {
        "paper_id": paper.get("paper_id", "unknown"),
        "title": paper.get("title", "No title"),
        "query": query_str[:100],
        "relevant_count": len(relevant_ids),
        "ranked_count": len(ranked),
        "metrics": metrics,
        "ranked_papers": ranked,
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM reranker on multiple examples")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1,
        help="Number of examples to run (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each example",
    )
    args = parser.parse_args()
    print("=" * 80)
    print(f"LLM RERANKER TEST - {args.num_examples} Examples")
    print("=" * 80)

    # Load dataset
    dataset_path = os.getenv("DATASET_DIR")
    if not dataset_path:
        print("❌ Error: DATASET_DIR not set in .env")
        return 1

    print(f"\n📚 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"   ✓ Loaded {len(dataset)} papers")

    # Build resources
    print(f"\n🔧 Loading resources...")
    resources = load_resources(dataset_path, enable_bm25=True, enable_e5=True, enable_specter=True)
    if not resources:
        print("   Cache miss, building from scratch...")
        corpus = build_citation_corpus(dataset)
        resources = build_inmemory_resources(
            corpus, enable_bm25=True, enable_e5=True, enable_specter=True
        )
    print(f"   ✓ Resources ready")

    # Process papers
    print(f"\n🚀 Processing {args.num_examples} papers...")
    print("-" * 80)

    results = []
    skipped = 0

    # Suppress verbose output during processing unless --verbose
    if not args.verbose:
        import logging

        logging.getLogger("src.agents.formulators.llm_agent").setLevel(logging.WARNING)

    for i, paper in enumerate(tqdm(dataset[: args.num_examples], desc="Processing", unit="paper")):
        result = process_single_paper(paper, resources, verbose=args.verbose)

        if result is None:
            skipped += 1
            continue

        results.append(result)

        if args.verbose:
            print(f"\n[{i+1}] {result['title'][:60]}...")
            print(
                f"    R@5={result['metrics']['R@5']:.3f}, R@10={result['metrics']['R@10']:.3f}, MRR={result['metrics']['MRR']:.3f}"
            )

    print("-" * 80)

    # Show aggregate results
    if not results:
        print("\n❌ No results - all papers skipped (no ground truth)")
        return 1

    print(f"\n📊 AGGREGATE RESULTS:")
    print("=" * 80)

    # Calculate averages
    avg_r5 = sum(r["metrics"]["R@5"] for r in results) / len(results)
    avg_r10 = sum(r["metrics"]["R@10"] for r in results) / len(results)
    avg_r20 = sum(r["metrics"]["R@20"] for r in results) / len(results)
    avg_mrr = sum(r["metrics"]["MRR"] for r in results) / len(results)

    print(f"\n✅ Successfully processed: {len(results)}/{args.num_examples} papers")
    if skipped > 0:
        print(f"⚠️  Skipped (no ground truth): {skipped}")

    print(f"\n📈 Average Metrics:")
    print(f"  • Recall@5:  {avg_r5:.4f} ({avg_r5*100:.1f}%)")
    print(f"  • Recall@10: {avg_r10:.4f} ({avg_r10*100:.1f}%)")
    print(f"  • Recall@20: {avg_r20:.4f} ({avg_r20*100:.1f}%)")
    print(f"  • MRR:       {avg_mrr:.4f}")

    # Show best and worst performers
    print(f"\n🏆 Best Performing (by R@10):")
    best = sorted(results, key=lambda r: r["metrics"]["R@10"], reverse=True)[:3]
    for i, r in enumerate(best, 1):
        print(f"  {i}. {r['title'][:60]}... (R@10={r['metrics']['R@10']:.3f})")

    print(f"\n💡 Worst Performing (by R@10):")
    worst = sorted(results, key=lambda r: r["metrics"]["R@10"])[:3]
    for i, r in enumerate(worst, 1):
        print(f"  {i}. {r['title'][:60]}... (R@10={r['metrics']['R@10']:.3f})")

    # Weighted score
    weighted = sum(
        0.4 * r["metrics"]["R@5"] + 0.3 * r["metrics"]["R@10"] + 0.3 * r["metrics"]["MRR"]
        for r in results
    ) / len(results)

    print(f"\n⚖️  Weighted Score: {weighted:.4f}")
    print(f"   (0.4×R@5 + 0.3×R@10 + 0.3×MRR)")

    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
