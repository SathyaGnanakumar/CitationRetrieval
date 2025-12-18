"""
Unit tests for the aggregator agent.

These are fast unit tests using mock data. For integration tests with
the actual dataset, see test_aggregator_integration.py.

To run only unit tests:
    pytest tests/test_aggregator.py -v
"""

import pytest
from src.agents.formulators.aggregator import aggregator

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


def test_aggregator_basic():
    """Test basic aggregation of results from multiple retrievers."""
    state = {
        "bm25_results": [
            {"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "bm25"},
            {"id": "paper2", "title": "Paper 2", "score": 0.8, "source": "bm25"},
            {"id": "paper3", "title": "Paper 3", "score": 0.7, "source": "bm25"},
        ],
        "e5_results": [
            {"id": "paper2", "title": "Paper 2", "score": 0.95, "source": "e5"},
            {"id": "paper4", "title": "Paper 4", "score": 0.85, "source": "e5"},
            {"id": "paper1", "title": "Paper 1", "score": 0.75, "source": "e5"},
        ],
        "specter_results": [
            {"id": "paper3", "title": "Paper 3", "score": 0.88, "source": "specter"},
            {"id": "paper5", "title": "Paper 5", "score": 0.82, "source": "specter"},
        ],
        "config": {"aggregation_method": "rrf", "rrf_k": 60},
    }

    result = aggregator(state)

    # Check basic structure
    assert "retriever_results" in result
    assert "candidate_papers" in result
    assert "messages" in result

    # Check retriever results stored correctly
    assert len(result["retriever_results"]) == 3
    assert "bm25" in result["retriever_results"]
    assert "e5" in result["retriever_results"]
    assert "specter" in result["retriever_results"]

    # Check deduplication - should have 5 unique papers
    candidate_papers = result["candidate_papers"]
    assert len(candidate_papers) == 5
    paper_ids = {p["id"] for p in candidate_papers}
    assert paper_ids == {"paper1", "paper2", "paper3", "paper4", "paper5"}

    # Check RRF scores are present
    for paper in candidate_papers:
        assert "rrf_score" in paper
        assert "retriever_count" in paper
        assert "sources" in paper

    # Papers appearing in multiple retrievers should have higher RRF scores
    paper2 = next(p for p in candidate_papers if p["id"] == "paper2")
    paper5 = next(p for p in candidate_papers if p["id"] == "paper5")
    assert paper2["retriever_count"] == 2  # Appears in bm25 and e5
    assert paper5["retriever_count"] == 1  # Only in specter
    # Paper2 should have higher RRF score due to appearing in multiple retrievers
    assert paper2["rrf_score"] > paper5["rrf_score"]


def test_aggregator_simple_method():
    """Test simple aggregation method (max score deduplication)."""
    state = {
        "bm25_results": [
            {"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "bm25"},
            {"id": "paper2", "title": "Paper 2", "score": 0.8, "source": "bm25"},
        ],
        "e5_results": [
            {"id": "paper1", "title": "Paper 1", "score": 0.7, "source": "e5"},
            {"id": "paper3", "title": "Paper 3", "score": 0.85, "source": "e5"},
        ],
        "specter_results": [],
        "config": {"aggregation_method": "simple"},
    }

    result = aggregator(state)

    candidate_papers = result["candidate_papers"]
    assert len(candidate_papers) == 3

    # Check that papers are sorted by score
    scores = [p["score"] for p in candidate_papers]
    assert scores == sorted(scores, reverse=True)


def test_aggregator_no_results():
    """Test aggregator with no retriever results."""
    state = {
        "bm25_results": [],
        "e5_results": [],
        "specter_results": [],
    }

    result = aggregator(state)

    assert result["candidate_papers"] == []
    assert "No results from any retriever" in result["messages"][0].content


def test_aggregator_single_retriever():
    """Test aggregator with results from only one retriever."""
    state = {
        "bm25_results": [
            {"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "bm25"},
            {"id": "paper2", "title": "Paper 2", "score": 0.8, "source": "bm25"},
        ],
        "e5_results": [],
        "specter_results": [],
        "config": {"aggregation_method": "rrf"},
    }

    result = aggregator(state)

    candidate_papers = result["candidate_papers"]
    assert len(candidate_papers) == 2

    # All papers should have retriever_count = 1
    for paper in candidate_papers:
        assert paper["retriever_count"] == 1
        assert paper["sources"] == ["bm25"]


def test_aggregator_missing_state_fields():
    """Test aggregator handles missing state fields gracefully."""
    state = {}

    result = aggregator(state)

    assert result["candidate_papers"] == []
    assert "retriever_results" in result


def test_rrf_score_calculation():
    """Test that RRF scores are calculated correctly."""
    state = {
        "bm25_results": [
            {"id": "paper1", "title": "Paper 1", "score": 1.0, "source": "bm25"},  # rank 1
            {"id": "paper2", "title": "Paper 2", "score": 0.9, "source": "bm25"},  # rank 2
        ],
        "e5_results": [
            {"id": "paper1", "title": "Paper 1", "score": 0.8, "source": "e5"},  # rank 1
        ],
        "specter_results": [],
        "config": {"aggregation_method": "rrf", "rrf_k": 60},
    }

    result = aggregator(state)
    papers = result["candidate_papers"]

    # Paper 1 appears in both retrievers at rank 1
    # RRF score = 1/(60+1) + 1/(60+1) = 2/61
    paper1 = next(p for p in papers if p["id"] == "paper1")
    expected_rrf = 2 / 61
    assert abs(paper1["rrf_score"] - expected_rrf) < 0.0001

    # Paper 2 appears only in bm25 at rank 2
    # RRF score = 1/(60+2) = 1/62
    paper2 = next(p for p in papers if p["id"] == "paper2")
    expected_rrf = 1 / 62
    assert abs(paper2["rrf_score"] - expected_rrf) < 0.0001

    # Paper 1 should rank higher
    assert papers[0]["id"] == "paper1"
    assert papers[1]["id"] == "paper2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
