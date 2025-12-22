"""
Example: Batch LLM Reranking

Demonstrates how to use the batch_llm_reranker function for efficient
parallel processing of multiple queries.

Usage:
    uv run python examples/batch_reranking_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.formulators.llm_agent import batch_llm_reranker
from corpus.builders import build_inmemory_resources
from corpus.scholarcopilot import load_dataset, build_citation_corpus
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    """Run batch reranking example."""

    print("=" * 80)
    print("Batch LLM Reranking Example")
    print("=" * 80)

    # Load dataset and build corpus
    dataset_path = os.getenv("DATASET_DIR", "corpus/scholarcopilot/scholar_copilot_eval_data_1k.json")

    print(f"\n📂 Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)
    corpus = build_citation_corpus(dataset)

    print(f"✓ Loaded {len(corpus)} papers in corpus")

    # Build resources (including LLM reranker model)
    print("\n🔧 Building resources (this will cache the LLM model)...")
    resources = build_inmemory_resources(
        corpus,
        enable_bm25=False,  # Don't need BM25 for this example
        enable_e5=False,     # Don't need E5 for this example
        enable_specter=False, # Don't need SPECTER for this example
        enable_llm_reranker=True  # Enable LLM reranker!
    )

    print("✓ Resources built")

    # Example queries
    queries = [
        "Transformers revolutionized natural language processing",
        "Attention mechanisms enable models to focus on relevant parts",
        "BERT pretraining improved many NLP tasks",
        "Graph neural networks learn from graph-structured data",
        "Contrastive learning is effective for representation learning",
    ]

    # Create mock candidate papers for each query
    # In a real scenario, these would come from your retrieval system
    candidate_papers_list = []
    for _ in queries:
        # Get top 10 random papers from corpus as candidates
        candidates = [
            {
                "id": paper["id"],
                "title": paper["title"],
                "score": 0.5,  # Mock retriever score
            }
            for paper in list(corpus.values())[:10]
        ]
        candidate_papers_list.append(candidates)

    print(f"\n🚀 Running batch reranking on {len(queries)} queries...")
    print(f"   Each query has {len(candidate_papers_list[0])} candidate papers")

    # Run batch reranking
    results = batch_llm_reranker(
        queries=queries,
        candidate_papers_list=candidate_papers_list,
        resources=resources,
        max_workers=None,  # Auto-detect based on model context limits
        closed_source=False
    )

    # Display results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)

    for i, (query, ranked_papers) in enumerate(zip(queries, results), 1):
        print(f"\n{i}. Query: {query[:60]}...")
        print(f"   Top 3 ranked papers:")
        for j, paper in enumerate(ranked_papers[:3], 1):
            title = paper.get("title", "Unknown")[:50]
            score = paper.get("rerank_score", 0.0)
            print(f"      {j}. {title}... (score: {score:.3f})")

    print("\n" + "=" * 80)
    print("✅ Batch reranking example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
