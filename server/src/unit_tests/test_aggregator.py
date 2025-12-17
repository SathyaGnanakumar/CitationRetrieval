"""Unit tests for the aggregator agent in aggregator.py."""

import pytest
from langchain_core.messages import AIMessage

from src.agents.formulators.aggregator import (
    _normalize_scores,
    _reciprocal_rank_fusion,
    aggregator,
)


class TestNormalizeScores:
    """Tests for the _normalize_scores helper function."""

    def test_empty_results_returns_empty(self):
        """Should return empty list for empty input."""
        result = _normalize_scores([], "bm25")
        assert result == []

    def test_normalizes_scores_to_0_1_range(self):
        """Should normalize scores to [0, 1] range."""
        results = [
            {"id": "1", "score": 10.0},
            {"id": "2", "score": 5.0},
            {"id": "3", "score": 0.0},
        ]
        normalized = _normalize_scores(results, "bm25")

        # Highest score should be 1.0, lowest should be 0.0
        scores = [r["normalized_score"] for r in normalized]
        assert max(scores) == 1.0
        assert min(scores) == 0.0

    def test_preserves_original_score(self):
        """Should preserve original score in separate field."""
        results = [
            {"id": "1", "score": 10.0},
            {"id": "2", "score": 5.0},  # Need multiple different scores for normalization to run
        ]
        normalized = _normalize_scores(results, "bm25")

        assert normalized[0]["original_score"] == 10.0
        assert normalized[1]["original_score"] == 5.0

    def test_handles_single_result(self):
        """Should handle single result (score_range == 0)."""
        results = [{"id": "1", "score": 5.0}]
        normalized = _normalize_scores(results, "bm25")

        # With only one result, score_range is 0, returns unchanged
        assert normalized == results

    def test_handles_identical_scores(self):
        """Should handle all identical scores (score_range == 0)."""
        results = [
            {"id": "1", "score": 5.0},
            {"id": "2", "score": 5.0},
            {"id": "3", "score": 5.0},
        ]
        normalized = _normalize_scores(results, "bm25")

        # All same scores means score_range is 0, returns unchanged
        assert normalized == results

    def test_correct_normalization_formula(self):
        """Should apply correct min-max normalization formula."""
        results = [
            {"id": "1", "score": 100.0},  # max
            {"id": "2", "score": 50.0},   # middle
            {"id": "3", "score": 0.0},    # min
        ]
        normalized = _normalize_scores(results, "bm25")

        assert normalized[0]["normalized_score"] == 1.0    # (100-0)/(100-0) = 1.0
        assert normalized[1]["normalized_score"] == 0.5    # (50-0)/(100-0) = 0.5
        assert normalized[2]["normalized_score"] == 0.0    # (0-0)/(100-0) = 0.0

    def test_preserves_other_fields(self):
        """Should preserve all other fields in results."""
        results = [
            {"id": "1", "score": 10.0, "title": "Paper A", "source": "bm25"},
            {"id": "2", "score": 5.0, "title": "Paper B", "source": "bm25"},
        ]
        normalized = _normalize_scores(results, "bm25")

        assert normalized[0]["title"] == "Paper A"
        assert normalized[0]["source"] == "bm25"
        assert normalized[1]["title"] == "Paper B"


class TestReciprocalRankFusion:
    """Tests for the _reciprocal_rank_fusion function."""

    def test_single_retriever_results(self):
        """Should work with results from a single retriever."""
        paper_id_to_results = {
            "paper1": [{"id": "paper1", "rank": 1, "title": "Paper 1", "source": "bm25"}],
            "paper2": [{"id": "paper2", "rank": 2, "title": "Paper 2", "source": "bm25"}],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        assert len(fused) == 2
        # Paper with rank 1 should have higher RRF score
        assert fused[0]["id"] == "paper1"
        assert fused[1]["id"] == "paper2"

    def test_multiple_retriever_boost(self):
        """Papers appearing in multiple retrievers should get higher scores."""
        paper_id_to_results = {
            # Paper 1 appears in 2 retrievers
            "paper1": [
                {"id": "paper1", "rank": 2, "title": "Paper 1", "source": "bm25"},
                {"id": "paper1", "rank": 2, "title": "Paper 1", "source": "e5"},
            ],
            # Paper 2 appears in only 1 retriever but at rank 1
            "paper2": [
                {"id": "paper2", "rank": 1, "title": "Paper 2", "source": "bm25"},
            ],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        # Paper 1 with 2 retrievers should rank higher than paper 2 with 1
        # RRF(paper1) = 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.0323
        # RRF(paper2) = 1/(60+1) = 1/61 ≈ 0.0164
        assert fused[0]["id"] == "paper1"
        assert fused[1]["id"] == "paper2"

    def test_rrf_score_included(self):
        """Should include rrf_score in results."""
        paper_id_to_results = {
            "paper1": [{"id": "paper1", "rank": 1, "source": "bm25"}],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        assert "rrf_score" in fused[0]
        # RRF score = 1/(60+1) = 1/61
        assert abs(fused[0]["rrf_score"] - 1/61) < 0.0001

    def test_retriever_count_included(self):
        """Should include retriever_count in results."""
        paper_id_to_results = {
            "paper1": [
                {"id": "paper1", "rank": 1, "source": "bm25"},
                {"id": "paper1", "rank": 3, "source": "e5"},
            ],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        assert fused[0]["retriever_count"] == 2

    def test_sources_list_included(self):
        """Should include sources list in results."""
        paper_id_to_results = {
            "paper1": [
                {"id": "paper1", "rank": 1, "source": "bm25"},
                {"id": "paper1", "rank": 3, "source": "e5"},
                {"id": "paper1", "rank": 2, "source": "specter"},
            ],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        assert set(fused[0]["sources"]) == {"bm25", "e5", "specter"}

    def test_custom_k_value(self):
        """Should use custom k value in RRF formula."""
        paper_id_to_results = {
            "paper1": [{"id": "paper1", "rank": 1, "source": "bm25"}],
        }

        # With k=10: score = 1/(10+1) = 1/11
        fused_k10 = _reciprocal_rank_fusion(paper_id_to_results, k=10)
        # With k=100: score = 1/(100+1) = 1/101
        fused_k100 = _reciprocal_rank_fusion(paper_id_to_results, k=100)

        assert abs(fused_k10[0]["rrf_score"] - 1/11) < 0.0001
        assert abs(fused_k100[0]["rrf_score"] - 1/101) < 0.0001

    def test_preserves_paper_metadata(self):
        """Should preserve paper metadata from first occurrence."""
        paper_id_to_results = {
            "paper1": [
                {"id": "paper1", "rank": 1, "title": "Paper Title", "source": "bm25", "year": 2023},
            ],
        }
        fused = _reciprocal_rank_fusion(paper_id_to_results, k=60)

        assert fused[0]["title"] == "Paper Title"
        assert fused[0]["year"] == 2023


class TestAggregatorNoResults:
    """Tests for aggregator with no results from any retriever."""

    def test_all_empty_returns_empty_candidates(self):
        """Should return empty candidate_papers when all retrievers return empty."""
        state = {
            "bm25_results": [],
            "e5_results": [],
            "specter_results": [],
        }
        result = aggregator(state)

        assert result["candidate_papers"] == []
        assert result["retriever_results"]["bm25"] == []
        assert result["retriever_results"]["e5"] == []
        assert result["retriever_results"]["specter"] == []

    def test_missing_keys_treated_as_empty(self):
        """Should treat missing result keys as empty lists."""
        state = {}
        result = aggregator(state)

        assert result["candidate_papers"] == []

    def test_none_values_treated_as_empty(self):
        """Should treat None values as empty lists."""
        state = {
            "bm25_results": None,
            "e5_results": None,
            "specter_results": None,
        }
        result = aggregator(state)

        assert result["candidate_papers"] == []

    def test_returns_ai_message_for_no_results(self):
        """Should return appropriate AIMessage for no results."""
        state = {}
        result = aggregator(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "No results" in result["messages"][0].content


class TestAggregatorRRFMethod:
    """Tests for aggregator using RRF aggregation method."""

    def test_rrf_is_default_method(self):
        """RRF should be the default aggregation method."""
        state = {
            "bm25_results": [{"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"}],
            "e5_results": [],
            "specter_results": [],
        }
        result = aggregator(state)

        assert "RRF" in result["messages"][0].content

    def test_deduplicates_by_paper_id(self):
        """Should deduplicate papers appearing in multiple retrievers."""
        state = {
            "bm25_results": [{"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "bm25"}],
            "e5_results": [{"id": "paper1", "title": "Paper 1", "score": 0.8, "source": "e5"}],
            "specter_results": [{"id": "paper1", "title": "Paper 1", "score": 0.7, "source": "specter"}],
        }
        result = aggregator(state)

        # Should have only 1 unique paper
        assert len(result["candidate_papers"]) == 1
        assert result["candidate_papers"][0]["id"] == "paper1"

    def test_rrf_ranks_multi_retriever_papers_higher(self):
        """Papers in multiple retrievers should rank higher."""
        state = {
            "bm25_results": [
                {"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "bm25"},
                {"id": "paper2", "title": "Paper 2", "score": 0.8, "source": "bm25"},
            ],
            "e5_results": [
                {"id": "paper1", "title": "Paper 1", "score": 0.85, "source": "e5"},
            ],
            "specter_results": [],
        }
        result = aggregator(state)

        # Paper 1 appears in 2 retrievers, paper 2 in only 1
        # Paper 1 should rank first despite paper 2 being first in one retriever
        assert result["candidate_papers"][0]["id"] == "paper1"

    def test_stores_retriever_results(self):
        """Should store raw results from each retriever."""
        bm25_results = [{"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"}]
        e5_results = [{"id": "2", "title": "Paper 2", "score": 0.8, "source": "e5"}]

        state = {
            "bm25_results": bm25_results,
            "e5_results": e5_results,
            "specter_results": [],
        }
        result = aggregator(state)

        assert result["retriever_results"]["bm25"] == bm25_results
        assert result["retriever_results"]["e5"] == e5_results
        assert result["retriever_results"]["specter"] == []

    def test_adds_rank_to_results(self):
        """Should add rank information to each result."""
        state = {
            "bm25_results": [
                {"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"},
                {"id": "2", "title": "Paper 2", "score": 0.8, "source": "bm25"},
            ],
            "e5_results": [],
            "specter_results": [],
        }
        result = aggregator(state)

        # Ranks are added in-place to the input lists
        assert state["bm25_results"][0]["rank"] == 1
        assert state["bm25_results"][1]["rank"] == 2

    def test_custom_rrf_k(self):
        """Should use custom rrf_k from config."""
        state = {
            "bm25_results": [{"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"}],
            "e5_results": [],
            "specter_results": [],
            "config": {"rrf_k": 100},
        }
        result = aggregator(state)

        assert "k=100" in result["messages"][0].content

    def test_returns_ai_message(self):
        """Should return AIMessage with aggregation summary."""
        state = {
            "bm25_results": [{"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"}],
            "e5_results": [{"id": "2", "title": "Paper 2", "score": 0.8, "source": "e5"}],
            "specter_results": [],
        }
        result = aggregator(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "aggregator"
        assert "Aggregated" in result["messages"][0].content


class TestAggregatorSimpleMethod:
    """Tests for aggregator using simple aggregation method."""

    def test_simple_method_selection(self):
        """Should use simple method when specified in config."""
        state = {
            "bm25_results": [{"id": "1", "title": "Paper 1", "score": 0.9, "source": "bm25"}],
            "e5_results": [],
            "specter_results": [],
            "config": {"aggregation_method": "simple"},
        }
        result = aggregator(state)

        assert "simple" in result["messages"][0].content

    def test_simple_keeps_highest_score(self):
        """Simple method should keep highest normalized score per paper."""
        state = {
            "bm25_results": [{"id": "paper1", "title": "Paper 1", "score": 0.5, "source": "bm25"}],
            "e5_results": [{"id": "paper1", "title": "Paper 1", "score": 0.9, "source": "e5"}],
            "specter_results": [],
            "config": {"aggregation_method": "simple"},
        }
        result = aggregator(state)

        assert len(result["candidate_papers"]) == 1

    def test_simple_sorts_by_score_descending(self):
        """Simple method should sort by score descending."""
        state = {
            "bm25_results": [
                {"id": "1", "title": "Low Score", "score": 0.3, "source": "bm25"},
                {"id": "2", "title": "High Score", "score": 0.9, "source": "bm25"},
            ],
            "e5_results": [],
            "specter_results": [],
            "config": {"aggregation_method": "simple"},
        }
        result = aggregator(state)

        # Higher score should be first
        assert result["candidate_papers"][0]["id"] == "2"
        assert result["candidate_papers"][1]["id"] == "1"


class TestAggregatorIntegration:
    """Integration tests for aggregator with realistic data."""

    def test_full_pipeline_with_all_retrievers(self):
        """Should correctly aggregate results from all three retrievers."""
        state = {
            "bm25_results": [
                {"id": "paper1", "title": "Attention Is All You Need", "score": 0.95, "source": "bm25"},
                {"id": "paper2", "title": "BERT", "score": 0.85, "source": "bm25"},
                {"id": "paper3", "title": "GPT-2", "score": 0.75, "source": "bm25"},
            ],
            "e5_results": [
                {"id": "paper1", "title": "Attention Is All You Need", "score": 0.92, "source": "e5"},
                {"id": "paper4", "title": "T5", "score": 0.88, "source": "e5"},
                {"id": "paper2", "title": "BERT", "score": 0.80, "source": "e5"},
            ],
            "specter_results": [
                {"id": "paper1", "title": "Attention Is All You Need", "score": 0.90, "source": "specter"},
                {"id": "paper5", "title": "XLNet", "score": 0.85, "source": "specter"},
            ],
        }
        result = aggregator(state)

        # Should have 5 unique papers
        assert len(result["candidate_papers"]) == 5

        # Paper 1 appears in all 3 retrievers, should be first
        assert result["candidate_papers"][0]["id"] == "paper1"
        assert result["candidate_papers"][0]["retriever_count"] == 3
        assert set(result["candidate_papers"][0]["sources"]) == {"bm25", "e5", "specter"}

    def test_handles_large_result_sets(self):
        """Should handle larger result sets efficiently."""
        # Create 100 papers per retriever
        bm25_results = [
            {"id": f"bm25_{i}", "title": f"Paper {i}", "score": 1.0 - i * 0.01, "source": "bm25"}
            for i in range(100)
        ]
        e5_results = [
            {"id": f"e5_{i}", "title": f"Paper {i}", "score": 1.0 - i * 0.01, "source": "e5"}
            for i in range(100)
        ]
        specter_results = [
            {"id": f"specter_{i}", "title": f"Paper {i}", "score": 1.0 - i * 0.01, "source": "specter"}
            for i in range(100)
        ]

        state = {
            "bm25_results": bm25_results,
            "e5_results": e5_results,
            "specter_results": specter_results,
        }
        result = aggregator(state)

        # Should have 300 unique papers (no overlap in this case)
        assert len(result["candidate_papers"]) == 300
        assert result["retriever_results"]["bm25"] == bm25_results

