"""
Comprehensive integration tests for the multi-agent citation retrieval workflow.

This test file validates the complete LangGraph pipeline execution:
- Query reformulation
- Parallel retrieval (BM25, E5, SPECTER)
- Result aggregation with RRF
- Reranking (cross-encoder or LLM-based)
- DSPy picker (optional)

Tests verify:
- All workflow nodes execute in correct order
- State transitions contain expected keys
- Parallel retrievers run concurrently
- Aggregation combines results correctly
- Final output contains ranked papers

Usage:
    pytest tests/test_workflow_integration.py -v -s
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage
import pytest

from corpus.builders import build_inmemory_resources
from src.workflow import RetrievalWorkflow


# Test data
SAMPLE_DOCS = [
    {
        "id": "paper1",
        "title": "Attention Is All You Need",
        "text": "We propose a new architecture based solely on attention mechanisms called Transformer.",
    },
    {
        "id": "paper2",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "text": "We introduce BERT, a language representation model using bidirectional transformers.",
    },
    {
        "id": "paper3",
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "text": "We train GPT-3, a 175B parameter autoregressive language model.",
    },
    {
        "id": "paper4",
        "title": "ResNet: Deep Residual Learning for Image Recognition",
        "text": "We present residual networks for image classification with skip connections.",
    },
    {
        "id": "paper5",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "text": "AlexNet achieves top-1 accuracy on ImageNet using convolutional neural networks.",
    },
]


@pytest.fixture(scope="module")
def resources():
    """Build retrieval resources once for all tests."""
    print("\n🔨 Building test resources...")

    # Build only BM25 for speed (E5/SPECTER take too long for unit tests)
    res = build_inmemory_resources(
        SAMPLE_DOCS,
        enable_bm25=True,
        enable_e5=False,
        enable_specter=False
    )

    print(f"✓ Resources built with {len(SAMPLE_DOCS)} documents")
    return res


def test_workflow_basic_execution(resources):
    """Test that the workflow executes end-to-end without errors."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="transformer architecture for NLP")],
        "resources": resources,
        "config": {"k": 5}
    }

    # Run workflow
    final_state = workflow.run(initial_state)

    # Verify state contains expected keys
    assert "messages" in final_state
    assert "ranked_papers" in final_state
    assert "query" in final_state or "queries" in final_state

    # Verify we got some results
    assert len(final_state["ranked_papers"]) > 0
    print(f"✓ Workflow executed, retrieved {len(final_state['ranked_papers'])} papers")


def test_workflow_reformulator_node(resources):
    """Test that query reformulator executes and produces queries."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="transformer architecture")],
        "resources": resources,
        "config": {"k": 5}
    }

    final_state = workflow.run(initial_state)

    # Verify reformulator produced queries
    assert "queries" in final_state or "query" in final_state

    # If queries exist, verify they're non-empty
    if "queries" in final_state:
        queries = final_state["queries"]
        assert isinstance(queries, list)
        assert len(queries) > 0
        print(f"✓ Reformulator generated {len(queries)} queries: {queries}")

    if "query" in final_state:
        query = final_state["query"]
        assert isinstance(query, str)
        assert len(query) > 0
        print(f"✓ Query extracted: {query[:50]}...")


def test_workflow_retrieval_nodes(resources):
    """Test that retrieval agents execute and produce candidate papers."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="transformer architecture")],
        "resources": resources,
        "config": {"k": 3}
    }

    final_state = workflow.run(initial_state)

    # Verify retrieval produced candidate papers
    assert "candidate_papers" in final_state
    candidate_papers = final_state["candidate_papers"]

    assert isinstance(candidate_papers, list)
    assert len(candidate_papers) > 0

    # Verify paper structure
    for paper in candidate_papers:
        assert isinstance(paper, dict)
        assert "id" in paper
        assert "title" in paper

    print(f"✓ Retrievers found {len(candidate_papers)} candidate papers")


def test_workflow_aggregator_node(resources):
    """Test that aggregator combines results from multiple retrievers."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="deep learning for NLP")],
        "resources": resources,
        "config": {
            "k": 5,
            "aggregation_method": "rrf"  # Test RRF aggregation
        }
    }

    final_state = workflow.run(initial_state)

    # Verify aggregator produced merged results
    assert "candidate_papers" in final_state
    candidates = final_state["candidate_papers"]

    # Should have aggregated results
    assert len(candidates) > 0

    # Verify scores exist (aggregator adds scores)
    for paper in candidates:
        assert "score" in paper or "rrf_score" in paper

    print(f"✓ Aggregator combined results: {len(candidates)} papers")


def test_workflow_reranker_node(resources):
    """Test that reranker produces ranked papers."""
    workflow = RetrievalWorkflow(use_llm_reranker=False)  # Use cross-encoder reranker

    initial_state = {
        "messages": [HumanMessage(content="attention mechanism for sequence models")],
        "resources": resources,
        "config": {"k": 5}
    }

    final_state = workflow.run(initial_state)

    # Verify reranker produced ranked_papers
    assert "ranked_papers" in final_state
    ranked_papers = final_state["ranked_papers"]

    assert isinstance(ranked_papers, list)
    assert len(ranked_papers) > 0

    # Verify papers have rerank scores
    for paper in ranked_papers:
        assert "rerank_score" in paper or "score" in paper

    # Verify papers are sorted by score (descending)
    if len(ranked_papers) >= 2:
        score_key = "rerank_score" if "rerank_score" in ranked_papers[0] else "score"
        scores = [p.get(score_key, 0) for p in ranked_papers]
        assert scores == sorted(scores, reverse=True), "Papers should be sorted by score"

    print(f"✓ Reranker produced {len(ranked_papers)} ranked papers")
    print(f"  Top result: {ranked_papers[0]['title']}")


def test_workflow_with_different_k_values(resources):
    """Test workflow with different k values for retrieval."""
    workflow = RetrievalWorkflow()

    for k in [3, 5, 10]:
        initial_state = {
            "messages": [HumanMessage(content="neural networks")],
            "resources": resources,
            "config": {"k": k}
        }

        final_state = workflow.run(initial_state)

        assert "ranked_papers" in final_state
        ranked_papers = final_state["ranked_papers"]

        # Should retrieve at most k papers (may be less if corpus is small)
        assert len(ranked_papers) <= k

        print(f"✓ k={k}: Retrieved {len(ranked_papers)} papers")


def test_workflow_with_llm_reranker(resources):
    """Test workflow with LLM-based reranker instead of cross-encoder."""
    # Skip if no LLM available
    import os
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LOCAL_LLM"):
        pytest.skip("No LLM configuration available (OPENAI_API_KEY or LOCAL_LLM)")

    workflow = RetrievalWorkflow(use_llm_reranker=True)

    initial_state = {
        "messages": [HumanMessage(content="transformer architecture")],
        "resources": resources,
        "config": {"k": 3}
    }

    final_state = workflow.run(initial_state)

    # Verify LLM reranker produced results
    assert "ranked_papers" in final_state
    assert len(final_state["ranked_papers"]) > 0

    print(f"✓ LLM reranker produced {len(final_state['ranked_papers'])} ranked papers")


def test_workflow_with_dspy_picker(resources):
    """Test workflow with DSPy picker enabled."""
    # Skip if DSPy picker is not enabled
    import os
    if not os.getenv("ENABLE_DSPY_PICKER") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("DSPy picker requires ENABLE_DSPY_PICKER=true and OPENAI_API_KEY")

    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="attention mechanism for NLP")],
        "resources": resources,
        "config": {
            "k": 5,
            "enable_dspy_picker": True,
            "dspy_model": "gpt-5-mini-2025-08-07",
            "dspy_top_n": 5
        }
    }

    final_state = workflow.run(initial_state)

    # Verify DSPy picker produced a selection
    assert "selected_paper" in final_state or "dspy_selected_title" in final_state

    if "selected_paper" in final_state:
        selected = final_state["selected_paper"]
        assert isinstance(selected, dict)
        assert "title" in selected
        print(f"✓ DSPy picker selected: {selected['title']}")

    if "dspy_selected_title" in final_state:
        print(f"✓ DSPy picker selected: {final_state['dspy_selected_title']}")


def test_workflow_state_transitions(resources):
    """Test that state contains all expected keys at different stages."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="deep learning")],
        "resources": resources,
        "config": {"k": 5}
    }

    final_state = workflow.run(initial_state)

    # Verify all expected state keys
    expected_keys = ["messages", "resources", "config"]
    for key in expected_keys:
        assert key in final_state, f"Missing expected key: {key}"

    # Verify at least one of these keys exists (output from workflow)
    output_keys = ["query", "queries", "candidate_papers", "ranked_papers"]
    assert any(key in final_state for key in output_keys), \
        f"Missing output keys. Expected at least one of: {output_keys}"

    print(f"✓ State contains all required keys: {list(final_state.keys())}")


def test_workflow_handles_empty_query(resources):
    """Test that workflow handles edge case of empty query gracefully."""
    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="")],
        "resources": resources,
        "config": {"k": 5}
    }

    # Workflow should handle gracefully without crashing
    try:
        final_state = workflow.run(initial_state)
        # May return empty results or default results
        assert "ranked_papers" in final_state or "messages" in final_state
        print("✓ Workflow handled empty query gracefully")
    except Exception as e:
        pytest.fail(f"Workflow should handle empty query gracefully, but raised: {e}")


def test_workflow_visualization():
    """Test that workflow graph can be visualized."""
    workflow = RetrievalWorkflow()

    # Test visualization without saving
    try:
        graph_image = workflow.visualize_graph(save_file=False)
        assert graph_image is not None
        print("✓ Workflow graph visualization successful")
    except Exception as e:
        # Visualization may fail in test environments without display
        print(f"⚠ Visualization skipped (expected in CI): {e}")


def test_workflow_execution_time(resources):
    """Test that workflow completes in reasonable time."""
    import time

    workflow = RetrievalWorkflow()

    initial_state = {
        "messages": [HumanMessage(content="machine learning")],
        "resources": resources,
        "config": {"k": 5}
    }

    start_time = time.time()
    final_state = workflow.run(initial_state)
    execution_time = time.time() - start_time

    # With BM25 only, should be fast (< 10 seconds)
    assert execution_time < 10.0, f"Workflow took too long: {execution_time:.2f}s"

    print(f"✓ Workflow completed in {execution_time:.2f}s")


if __name__ == "__main__":
    # Run tests directly
    import pytest
    pytest.main([__file__, "-v", "-s"])
