"""
Tests for E5 and SPECTER retrievers with single and batch query processing.

This test suite verifies:
1. Single query processing for both E5 and SPECTER
2. Batch query processing (10 queries) for both E5 and SPECTER
3. CLI-based testing with configurable models

Uses the actual ScholarCopilot dataset to build the corpus once and reuse it across all tests.
"""

import sys
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# Ensure `import src...` works when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.retrievers.e5_agent import E5Retriever
from src.agents.retrievers.specter_agent import SPECTERRetriever
from src.resources.builders import build_e5_resources, build_specter_resources
from src.corpus.scholarcopilot import load_dataset, build_citation_corpus


# Module-level variables to cache corpus and resources
_corpus: Optional[List[Dict[str, Any]]] = None
_e5_resources: Optional[Dict[str, Any]] = None
_specter_resources: Optional[Dict[str, Any]] = None


def get_dataset_path() -> str:
    """Get dataset path from environment variable or default location."""
    dataset_path = os.getenv(
        "DATASET_DIR",
        str(Path(__file__).parent.parent.parent / "datasets" / "scholar_copilot_eval_data_1k.json"),
    )
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Set DATASET_DIR environment variable or place dataset in expected location."
        )
    return dataset_path


def build_corpus_once() -> List[Dict[str, Any]]:
    """Build corpus from dataset once and cache it."""
    global _corpus
    if _corpus is None:
        dataset_path = get_dataset_path()
        print(f"\n[Loading dataset] {dataset_path}")
        dataset = load_dataset(dataset_path)
        print("[Building citation corpus from bib_info]")
        _corpus = build_citation_corpus(dataset)
        print(f"[Corpus built] {len(_corpus)} documents")
    return _corpus


def get_e5_resources(model_name: str = "intfloat/e5-base-v2") -> Dict[str, Any]:
    """Build E5 resources once and cache them."""
    global _e5_resources
    if _e5_resources is None or _e5_resources.get("model_name") != model_name:
        corpus = build_corpus_once()
        print(f"[Building E5 resources] Model: {model_name}")
        _e5_resources = build_e5_resources(corpus, model_name=model_name, batch_size=16)
        print(f"[E5 resources built] Corpus size: {len(corpus)}")
    return _e5_resources


def get_specter_resources(model_name: str = "allenai/specter2_base") -> Dict[str, Any]:
    """Build SPECTER resources once and cache them."""
    global _specter_resources
    if _specter_resources is None or _specter_resources.get("model_name") != model_name:
        corpus = build_corpus_once()
        print(f"[Building SPECTER resources] Model: {model_name}")
        _specter_resources = build_specter_resources(corpus, model_name=model_name, batch_size=8)
        print(f"[SPECTER resources built] Corpus size: {len(corpus)}")
    return _specter_resources


def test_e5_single_query():
    """Test E5 retriever with a single query."""
    print("\n" + "=" * 70)
    print("Testing E5 Single Query")
    print("=" * 70)

    # Get resources (built once, cached)
    e5_resources = get_e5_resources()

    # Initialize retriever
    retriever = E5Retriever(e5_resources["model"], e5_resources.get("device"))

    # Single query
    query = "transformer architecture for natural language processing"
    results = retriever.single_query(
        query,
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=3,
    )

    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("id" in r for r in results), "All results should have 'id'"
    assert all("title" in r for r in results), "All results should have 'title'"
    assert all("score" in r for r in results), "All results should have 'score'"
    assert all(r["source"] == "e5" for r in results), "All results should have source 'e5'"
    assert all(isinstance(r["score"], float) for r in results), "All scores should be floats"
    assert results[0]["score"] >= results[-1]["score"], "Results should be sorted by score"

    print(f"✓ Query: {query}")
    print(f"✓ Retrieved {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title'][:60]}... (score: {r['score']:.4f})")

    return results


def test_e5_batch_query():
    """Test E5 retriever with 10 queries in batch."""
    print("\n" + "=" * 70)
    print("Testing E5 Batch Query (10 queries)")
    print("=" * 70)

    # Get resources (built once, cached)
    e5_resources = get_e5_resources()

    # Initialize retriever
    retriever = E5Retriever(e5_resources["model"], e5_resources.get("device"))

    # 10 queries
    queries = [
        "transformer architecture",
        "neural network models",
        "language representation learning",
        "deep learning for NLP",
        "attention mechanisms",
        "bidirectional transformers",
        "pre-training methods",
        "image recognition",
        "residual learning",
        "sequence transduction",
    ]

    results_per_query = retriever.batch_query(
        queries,
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=3,
    )

    # Verify results
    assert len(results_per_query) == 10, f"Expected 10 query results, got {len(results_per_query)}"
    for i, query_results in enumerate(results_per_query):
        assert len(query_results) == 3, f"Query {i} should return 3 results"
        assert all("id" in r for r in query_results), f"Query {i} results should have 'id'"
        assert all("title" in r for r in query_results), f"Query {i} results should have 'title'"
        assert all("score" in r for r in query_results), f"Query {i} results should have 'score'"
        assert all(
            r["source"] == "e5" for r in query_results
        ), f"Query {i} results should have source 'e5'"

    print(f"✓ Processed {len(queries)} queries in batch")
    for i, (query, query_results) in enumerate(zip(queries, results_per_query), 1):
        print(f"\n  Query {i}: {query}")
        print(
            f"    Top result: {query_results[0]['title'][:50]}... (score: {query_results[0]['score']:.4f})"
        )

    return results_per_query


def test_specter_single_query():
    """Test SPECTER retriever with a single query."""
    print("\n" + "=" * 70)
    print("Testing SPECTER Single Query")
    print("=" * 70)

    # Get resources (built once, cached)
    specter_resources = get_specter_resources()

    # Initialize retriever
    retriever = SPECTERRetriever(
        specter_resources["model"],
        specter_resources["tokenizer"],
        specter_resources.get("device"),
    )

    # Single query
    query = "transformer architecture for natural language processing"
    results = retriever.single_query(
        query,
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=3,
    )

    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("id" in r for r in results), "All results should have 'id'"
    assert all("title" in r for r in results), "All results should have 'title'"
    assert all("score" in r for r in results), "All results should have 'score'"
    assert all(
        r["source"] == "specter" for r in results
    ), "All results should have source 'specter'"
    assert all(isinstance(r["score"], float) for r in results), "All scores should be floats"
    assert results[0]["score"] >= results[-1]["score"], "Results should be sorted by score"

    print(f"✓ Query: {query}")
    print(f"✓ Retrieved {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title'][:60]}... (score: {r['score']:.4f})")

    return results


def test_specter_batch_query():
    """Test SPECTER retriever with 10 queries in batch."""
    print("\n" + "=" * 70)
    print("Testing SPECTER Batch Query (10 queries)")
    print("=" * 70)

    # Get resources (built once, cached)
    specter_resources = get_specter_resources()

    # Initialize retriever
    retriever = SPECTERRetriever(
        specter_resources["model"],
        specter_resources["tokenizer"],
        specter_resources.get("device"),
    )

    # 10 queries
    queries = [
        "transformer architecture",
        "neural network models",
        "language representation learning",
        "deep learning for NLP",
        "attention mechanisms",
        "bidirectional transformers",
        "pre-training methods",
        "image recognition",
        "residual learning",
        "sequence transduction",
    ]

    results_per_query = retriever.batch_query(
        queries,
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=3,
    )

    # Verify results
    assert len(results_per_query) == 10, f"Expected 10 query results, got {len(results_per_query)}"
    for i, query_results in enumerate(results_per_query):
        assert len(query_results) == 3, f"Query {i} should return 3 results"
        assert all("id" in r for r in query_results), f"Query {i} results should have 'id'"
        assert all("title" in r for r in query_results), f"Query {i} results should have 'title'"
        assert all("score" in r for r in query_results), f"Query {i} results should have 'score'"
        assert all(
            r["source"] == "specter" for r in query_results
        ), f"Query {i} results should have source 'specter'"

    print(f"✓ Processed {len(queries)} queries in batch")
    for i, (query, query_results) in enumerate(zip(queries, results_per_query), 1):
        print(f"\n  Query {i}: {query}")
        print(
            f"    Top result: {query_results[0]['title'][:50]}... (score: {query_results[0]['score']:.4f})"
        )

    return results_per_query


def test_consistency_single_vs_batch():
    """Test that single query and batch query (with one query) produce identical results."""
    print("\n" + "=" * 70)
    print("Testing Consistency: Single Query vs Batch Query (1 query)")
    print("=" * 70)

    # Test E5
    e5_resources = get_e5_resources()
    e5_retriever = E5Retriever(e5_resources["model"], e5_resources.get("device"))
    query = "transformer models"

    single_results = e5_retriever.single_query(
        query,
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=3,
    )
    batch_results = e5_retriever.batch_query(
        [query],
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=3,
    )[0]

    assert len(single_results) == len(batch_results), "Results should have same length"
    for s, b in zip(single_results, batch_results):
        assert s["id"] == b["id"], "Result IDs should match"
        assert abs(s["score"] - b["score"]) < 1e-5, "Scores should match (within float precision)"

    print("✓ E5: Single query and batch query (1 query) produce identical results")

    # Test SPECTER
    specter_resources = get_specter_resources()
    specter_retriever = SPECTERRetriever(
        specter_resources["model"],
        specter_resources["tokenizer"],
        specter_resources.get("device"),
    )

    single_results = specter_retriever.single_query(
        query,
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=3,
    )
    batch_results = specter_retriever.batch_query(
        [query],
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=3,
    )[0]

    assert len(single_results) == len(batch_results), "Results should have same length"
    for s, b in zip(single_results, batch_results):
        assert s["id"] == b["id"], "Result IDs should match"
        assert abs(s["score"] - b["score"]) < 1e-5, "Scores should match (within float precision)"

    print("✓ SPECTER: Single query and batch query (1 query) produce identical results")


def run_cli_test(
    e5_model: str = "intfloat/e5-base-v2",
    specter_model: str = "allenai/specter2_base",
    num_queries: int = 10,
    k: int = 3,
):
    """
    CLI-based test that allows specifying models from command line.

    Args:
        e5_model: E5 model name to use
        specter_model: SPECTER model name to use
        num_queries: Number of queries to test in batch
        k: Number of top results to retrieve per query
    """
    print("\n" + "=" * 70)
    print("CLI Test: Configurable Models")
    print("=" * 70)
    print(f"E5 Model: {e5_model}")
    print(f"SPECTER Model: {specter_model}")
    print(f"Number of queries: {num_queries}")
    print(f"Top-k: {k}")
    print("=" * 70)

    # Generate test queries
    base_queries = [
        "transformer architecture",
        "neural network models",
        "language representation learning",
        "deep learning for NLP",
        "attention mechanisms",
        "bidirectional transformers",
        "pre-training methods",
        "image recognition",
        "residual learning",
        "sequence transduction",
    ]
    queries = (base_queries * ((num_queries // len(base_queries)) + 1))[:num_queries]

    # Test E5
    print("\n[E5 Retriever]")
    print("-" * 70)
    e5_resources = get_e5_resources(e5_model)
    e5_retriever = E5Retriever(e5_resources["model"], e5_resources.get("device"))

    print(f"Testing single query...")
    single_result = e5_retriever.single_query(
        queries[0],
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=k,
    )
    print(f"  ✓ Single query: {len(single_result)} results")

    print(f"Testing batch query ({num_queries} queries)...")
    import time

    start_time = time.time()
    batch_results = e5_retriever.batch_query(
        queries,
        e5_resources["corpus_embeddings"],
        e5_resources["ids"],
        e5_resources["titles"],
        k=k,
    )
    batch_time = time.time() - start_time
    print(f"  ✓ Batch query: {len(batch_results)} query results in {batch_time:.2f}s")
    print(f"  ✓ Average time per query: {batch_time/num_queries:.3f}s")

    # Test SPECTER
    print("\n[SPECTER Retriever]")
    print("-" * 70)
    specter_resources = get_specter_resources(specter_model)
    specter_retriever = SPECTERRetriever(
        specter_resources["model"],
        specter_resources["tokenizer"],
        specter_resources.get("device"),
    )

    print(f"Testing single query...")
    single_result = specter_retriever.single_query(
        queries[0],
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=k,
    )
    print(f"  ✓ Single query: {len(single_result)} results")

    print(f"Testing batch query ({num_queries} queries)...")
    start_time = time.time()
    batch_results = specter_retriever.batch_query(
        queries,
        specter_resources["corpus_embeddings"],
        specter_resources["ids"],
        specter_resources["titles"],
        k=k,
    )
    batch_time = time.time() - start_time
    print(f"  ✓ Batch query: {len(batch_results)} query results in {batch_time:.2f}s")
    print(f"  ✓ Average time per query: {batch_time/num_queries:.3f}s")

    print("\n" + "=" * 70)
    print("✓ All CLI tests completed successfully!")
    print("=" * 70)


def main():
    """Main entry point for CLI testing."""
    parser = argparse.ArgumentParser(
        description="Test E5 and SPECTER retrievers with configurable models"
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default="intfloat/e5-base-v2",
        help="E5 model name (default: intfloat/e5-base-v2)",
    )
    parser.add_argument(
        "--specter-model",
        type=str,
        default="allenai/specter2_base",
        help="SPECTER model name (default: allenai/specter2_base)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to test in batch (default: 10)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["all", "single", "batch", "cli", "consistency"],
        default="all",
        help="Type of test to run (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset (overrides DATASET_DIR env var)",
    )

    args = parser.parse_args()

    # Override dataset path if provided
    if args.dataset:
        os.environ["DATASET_DIR"] = args.dataset

    if args.test_type == "all":
        test_e5_single_query()
        test_e5_batch_query()
        test_specter_single_query()
        test_specter_batch_query()
        test_consistency_single_vs_batch()
        run_cli_test(args.e5_model, args.specter_model, args.num_queries, args.k)
    elif args.test_type == "single":
        test_e5_single_query()
        test_specter_single_query()
    elif args.test_type == "batch":
        test_e5_batch_query()
        test_specter_batch_query()
    elif args.test_type == "cli":
        run_cli_test(args.e5_model, args.specter_model, args.num_queries, args.k)
    elif args.test_type == "consistency":
        test_consistency_single_vs_batch()


if __name__ == "__main__":
    main()
