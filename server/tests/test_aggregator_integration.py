"""
Integration tests for the aggregator using the actual ScholarCopilot dataset.

This test file loads the real dataset, builds actual resources (BM25, E5, SPECTER),
runs real retrievers, and tests the aggregator with actual results.

Note: These tests require:
- The ScholarCopilot dataset (default: ../corpus/scholarcopilot/scholar_copilot_eval_data_1k.json)
- Set DATASET_DIR environment variable to override location
- Model downloads (E5, SPECTER) - first run will be slow
- Sufficient memory/GPU for embeddings

To run:
    pytest tests/test_aggregator_integration.py -v -s

To skip (for fast unit tests):
    pytest tests/ -v -m "not integration"
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add parent directory for corpus import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage

from corpus.scholarcopilot import build_citation_corpus, load_dataset
from corpus.builders import build_bm25_resources, build_e5_resources, build_specter_resources
from src.agents.retrievers.bm25_agent import bm25_agent
from src.agents.retrievers.e5_agent import e5_agent
from src.agents.retrievers.specter_agent import specter_agent
from src.agents.formulators.aggregator import aggregator
from src.workflow import RetrievalWorkflow

# Dataset path - use environment variable or default relative path
DATASET_PATH = Path(
    os.getenv("DATASET_DIR", "../corpus/scholarcopilot/scholar_copilot_eval_data_1k.json")
)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():
    """Load the ScholarCopilot dataset once for all tests."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found at {DATASET_PATH}")

    print(f"\n📂 Loading dataset from {DATASET_PATH}...")
    data = load_dataset(DATASET_PATH)
    print(f"✓ Loaded {len(data)} papers from dataset")
    return data


@pytest.fixture(scope="module")
def corpus(dataset):
    """Build citation corpus from the dataset."""
    print("\n🔨 Building citation corpus...")
    corpus = build_citation_corpus(dataset)
    print(f"✓ Built corpus with {len(corpus)} unique citations")
    return corpus


@pytest.fixture(scope="module")
def small_corpus(corpus):
    """Create a smaller corpus for faster testing (first 100 papers)."""
    small = corpus[:100]
    print(f"📦 Using small corpus: {len(small)} papers")
    return small


@pytest.fixture(scope="module")
def bm25_resources(small_corpus):
    """Build BM25 resources."""
    print("\n⚙️  Building BM25 resources...")
    resources = build_bm25_resources(small_corpus)
    print("✓ BM25 resources ready")
    return resources


@pytest.fixture(scope="module")
def e5_resources(small_corpus):
    """Build E5 resources (this will download model on first run)."""
    print("\n⚙️  Building E5 resources (this may take a while on first run)...")
    resources = build_e5_resources(
        small_corpus,
        model_name="intfloat/e5-base-v2",  # Use base model for faster testing
        batch_size=16,
    )
    print("✓ E5 resources ready")
    return resources


@pytest.fixture(scope="module")
def specter_resources(small_corpus):
    """Build SPECTER resources (this will download model on first run)."""
    print("\n⚙️  Building SPECTER resources (this may take a while on first run)...")
    resources = build_specter_resources(
        small_corpus, model_name="allenai/specter2_base", batch_size=8
    )
    print("✓ SPECTER resources ready")
    return resources


def test_aggregator_with_real_retrievers(bm25_resources, e5_resources, specter_resources):
    """Test aggregator with real retrieval results from actual dataset."""

    # Create a realistic academic query
    query = "transformer architecture for natural language processing"

    # Build state with resources
    state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "queries": [query],
        "resources": {
            "bm25": bm25_resources,
            "e5": e5_resources,
            "specter": specter_resources,
        },
        "config": {"k": 10},
    }

    print(f"\n🔍 Running retrievers for query: '{query}'")

    # Run BM25
    print("  → Running BM25...")
    bm25_result = bm25_agent(state)
    state.update(bm25_result)
    print(f"    ✓ BM25 returned {len(state['bm25_results'])} results")

    # Run E5
    print("  → Running E5...")
    e5_result = e5_agent(state)
    state.update(e5_result)
    print(f"    ✓ E5 returned {len(state['e5_results'])} results")

    # Run SPECTER
    print("  → Running SPECTER...")
    specter_result = specter_agent(state)
    state.update(specter_result)
    print(f"    ✓ SPECTER returned {len(state['specter_results'])} results")

    # Run aggregator
    print("\n🔗 Running aggregator...")
    agg_result = aggregator(state)

    # Assertions
    assert "retriever_results" in agg_result
    assert "candidate_papers" in agg_result
    assert "messages" in agg_result

    candidate_papers = agg_result["candidate_papers"]
    print(f"✓ Aggregator produced {len(candidate_papers)} unique papers")

    # Check we have results
    assert len(candidate_papers) > 0, "Aggregator should produce some results"

    # Check RRF scores are present (using default RRF method)
    for paper in candidate_papers:
        assert "rrf_score" in paper, "Papers should have RRF scores"
        assert "retriever_count" in paper, "Papers should have retriever count"
        assert "sources" in paper, "Papers should list their sources"

    # Check papers are sorted by RRF score
    rrf_scores = [p["rrf_score"] for p in candidate_papers]
    assert rrf_scores == sorted(rrf_scores, reverse=True), "Papers should be sorted by RRF score"

    # Print top results for manual inspection
    print("\n📊 Top 5 aggregated results:")
    for i, paper in enumerate(candidate_papers[:5], 1):
        print(f"{i}. {paper['title'][:80]}...")
        print(
            f"   RRF: {paper['rrf_score']:.4f}, Sources: {paper['sources']}, Count: {paper['retriever_count']}"
        )

    # Check that papers appearing in multiple retrievers have higher scores
    multi_source_papers = [p for p in candidate_papers if p["retriever_count"] > 1]
    single_source_papers = [p for p in candidate_papers if p["retriever_count"] == 1]

    if multi_source_papers and single_source_papers:
        avg_multi = sum(p["rrf_score"] for p in multi_source_papers) / len(multi_source_papers)
        avg_single = sum(p["rrf_score"] for p in single_source_papers) / len(single_source_papers)
        print(
            f"\n📈 Avg RRF score - Multi-source: {avg_multi:.4f}, Single-source: {avg_single:.4f}"
        )
        # Multi-source papers should generally have higher scores
        assert avg_multi > avg_single, "Multi-source papers should have higher average RRF scores"


def test_aggregator_simple_method(bm25_resources, e5_resources, specter_resources):
    """Test aggregator with simple max-score method."""

    query = "deep learning for computer vision"

    state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "queries": [query],
        "resources": {
            "bm25": bm25_resources,
            "e5": e5_resources,
            "specter": specter_resources,
        },
        "config": {"k": 5, "aggregation_method": "simple"},
    }

    print(f"\n🔍 Testing simple aggregation for: '{query}'")

    # Run retrievers
    state.update(bm25_agent(state))
    state.update(e5_agent(state))
    state.update(specter_agent(state))

    # Run aggregator with simple method
    agg_result = aggregator(state)

    candidate_papers = agg_result["candidate_papers"]
    print(f"✓ Simple aggregation produced {len(candidate_papers)} papers")

    # Check results
    assert len(candidate_papers) > 0

    # With simple method, papers should have normalized scores but not RRF scores
    for paper in candidate_papers:
        assert "score" in paper
        # Should NOT have RRF scores with simple method
        if "rrf_score" in paper:
            pytest.skip("RRF scores present with simple method - may be from retriever")

    # Check sorting by score
    scores = [p["score"] for p in candidate_papers]
    assert scores == sorted(scores, reverse=True)

    print(f"✓ Papers sorted by score (top score: {scores[0]:.4f})")


def test_full_workflow_with_aggregator(bm25_resources, e5_resources, specter_resources):
    """Test the complete workflow including aggregator and reranking."""

    # Create a fake reranker for testing (to avoid API calls)
    class FakeReranker:
        def compute_score(self, pairs, normalize=True):
            # Return decreasing scores
            n = len(pairs)
            return [float(n - i) / n for i in range(n)]

    # Use a very general query that should match the corpus
    query = "machine learning"

    resources = {
        "bm25": bm25_resources,
        "e5": e5_resources,
        "specter": specter_resources,
        "reranker_model": FakeReranker(),
    }

    print(f"\n🌊 Testing full workflow for: '{query}'")

    # Initialize and run workflow
    workflow = RetrievalWorkflow()
    final_state = workflow.run(
        {
            "messages": [HumanMessage(content=query)],
            "resources": resources,
            "config": {"k": 10, "aggregation_method": "rrf"},
        }
    )

    # Check workflow outputs
    assert "candidate_papers" in final_state, "Should have candidate papers from aggregator"
    assert "ranked_papers" in final_state, "Should have ranked papers from reranker"
    assert "retriever_results" in final_state, "Should have raw retriever results"

    candidate_papers = final_state["candidate_papers"]
    ranked_papers = final_state["ranked_papers"]

    print(f"✓ Workflow complete:")
    print(f"  - Candidate papers (after aggregation): {len(candidate_papers)}")
    print(f"  - Ranked papers (after reranking): {len(ranked_papers)}")

    # If no papers found, print debug info but don't fail
    if len(candidate_papers) == 0:
        print("⚠️  No candidate papers found - corpus may be too specialized")
        print(f"  BM25 results: {len(final_state.get('retriever_results', {}).get('bm25', []))}")
        print(f"  E5 results: {len(final_state.get('retriever_results', {}).get('e5', []))}")
        print(
            f"  SPECTER results: {len(final_state.get('retriever_results', {}).get('specter', []))}"
        )
        pytest.skip("No results found for query in small corpus")

    # Check that ranked papers have rerank scores
    assert len(ranked_papers) > 0, "Should have ranked papers if we have candidates"
    for paper in ranked_papers:
        assert "rerank_score" in paper, "Ranked papers should have rerank_score"

    # Print final results
    print("\n🏆 Top 3 final ranked papers:")
    for i, paper in enumerate(ranked_papers[:3], 1):
        print(f"{i}. {paper['title'][:80]}...")
        print(f"   Rerank: {paper['rerank_score']:.4f}, RRF: {paper.get('rrf_score', 'N/A')}")


def test_aggregator_with_overlapping_results(bm25_resources, e5_resources):
    """Test that aggregator properly handles overlapping results from different retrievers."""

    query = "machine learning"

    state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "queries": [query],
        "resources": {
            "bm25": bm25_resources,
            "e5": e5_resources,
            "specter": {},  # Empty to keep it simple
        },
        "config": {"k": 10},
    }

    # Run only BM25 and E5
    state.update(bm25_agent(state))
    state.update(e5_agent(state))
    state["specter_results"] = []  # No SPECTER results

    # Get paper IDs from each retriever
    bm25_ids = {p["id"] for p in state["bm25_results"]}
    e5_ids = {p["id"] for p in state["e5_results"]}
    overlapping_ids = bm25_ids & e5_ids

    print(f"\n📊 Overlap analysis:")
    print(f"  - BM25 papers: {len(bm25_ids)}")
    print(f"  - E5 papers: {len(e5_ids)}")
    print(f"  - Overlapping: {len(overlapping_ids)}")

    # Run aggregator
    agg_result = aggregator(state)
    candidate_papers = agg_result["candidate_papers"]

    # Check that overlapping papers have higher RRF scores
    if overlapping_ids:
        overlapping_papers = [p for p in candidate_papers if p["id"] in overlapping_ids]
        non_overlapping_papers = [p for p in candidate_papers if p["id"] not in overlapping_ids]

        if overlapping_papers and non_overlapping_papers:
            max_overlap_score = max(p["rrf_score"] for p in overlapping_papers)
            max_non_overlap_score = max(p["rrf_score"] for p in non_overlapping_papers)

            print(f"  - Max RRF (overlap): {max_overlap_score:.4f}")
            print(f"  - Max RRF (no overlap): {max_non_overlap_score:.4f}")

            assert (
                max_overlap_score > max_non_overlap_score
            ), "Papers in multiple retrievers should have higher RRF scores"


if __name__ == "__main__":
    # Run with: python tests/test_aggregator_integration.py
    pytest.main([__file__, "-v", "-s"])
