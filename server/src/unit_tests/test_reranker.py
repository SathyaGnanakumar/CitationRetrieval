"""Unit tests for the cross-encoder reranker in reranker.py."""

import pytest
from unittest.mock import MagicMock, patch
import sys

from langchain_core.messages import HumanMessage, AIMessage


# Mock FlagEmbedding at module level before any imports
mock_flag_embedding = MagicMock()
sys.modules["FlagEmbedding"] = mock_flag_embedding


class TestGetQuery:
    """Tests for the _get_query helper function."""

    def test_returns_query_from_state_when_present(self):
        """Should return query directly when it exists in state."""
        from src.agents.formulators.reranker import _get_query

        state = {"query": "transformer architecture", "messages": []}
        result = _get_query(state)

        assert result == "transformer architecture"

    def test_strips_whitespace_from_query(self):
        """Should strip leading/trailing whitespace from query."""
        from src.agents.formulators.reranker import _get_query

        state = {"query": "  attention mechanisms  ", "messages": []}
        result = _get_query(state)

        assert result == "attention mechanisms"

    def test_falls_back_to_last_human_message(self):
        """Should use last HumanMessage when query is not present."""
        from src.agents.formulators.reranker import _get_query

        state = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="neural network papers"),
            ]
        }
        result = _get_query(state)

        assert result == "neural network papers"

    def test_skips_ai_messages_in_fallback(self):
        """Should skip AIMessages when looking for fallback query."""
        from src.agents.formulators.reranker import _get_query

        state = {
            "messages": [
                HumanMessage(content="my query"),
                AIMessage(content="AI response"),
            ]
        }
        result = _get_query(state)

        assert result == "my query"

    def test_returns_none_when_no_query_available(self):
        """Should return None when neither query nor HumanMessage exists."""
        from src.agents.formulators.reranker import _get_query

        state = {"messages": [AIMessage(content="only AI")]}
        result = _get_query(state)

        assert result is None

    def test_returns_none_for_empty_query(self):
        """Should return None when query is empty string."""
        from src.agents.formulators.reranker import _get_query

        state = {"query": "", "messages": []}
        result = _get_query(state)

        assert result is None

    def test_returns_none_for_whitespace_only_query(self):
        """Should return None when query is only whitespace."""
        from src.agents.formulators.reranker import _get_query

        state = {"query": "   ", "messages": []}
        result = _get_query(state)

        assert result is None


class TestPairs:
    """Tests for the _pairs helper function."""

    def test_creates_query_title_pairs(self):
        """Should create pairs of query and paper titles."""
        from src.agents.formulators.reranker import _pairs

        query = "attention mechanism"
        papers = [
            {"title": "Attention Is All You Need", "id": "1"},
            {"title": "BERT: Pre-training", "id": "2"},
        ]

        result = _pairs(query, papers)

        assert len(result) == 2
        assert result[0] == ["attention mechanism", "Attention Is All You Need"]
        assert result[1] == ["attention mechanism", "BERT: Pre-training"]

    def test_handles_missing_title(self):
        """Should use empty string for papers without title."""
        from src.agents.formulators.reranker import _pairs

        query = "test"
        papers = [{"id": "1"}, {"title": "Has title", "id": "2"}]

        result = _pairs(query, papers)

        assert result[0] == ["test", ""]
        assert result[1] == ["test", "Has title"]

    def test_handles_none_title(self):
        """Should use empty string when title is None."""
        from src.agents.formulators.reranker import _pairs

        query = "test"
        papers = [{"title": None, "id": "1"}]

        result = _pairs(query, papers)

        assert result[0] == ["test", ""]

    def test_handles_empty_papers_list(self):
        """Should return empty list for empty papers."""
        from src.agents.formulators.reranker import _pairs

        result = _pairs("test", [])

        assert result == []


class TestRerankerMissingQuery:
    """Tests for reranker error handling when query is missing."""

    def test_returns_error_when_query_missing(self):
        """Should return error message when no query available."""
        from src.agents.formulators.reranker import reranker

        state = {
            "messages": [AIMessage(content="no human message")],
            "candidate_papers": [{"title": "Paper A"}],
        }

        result = reranker(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "RERANK_ERROR" in result["messages"][0].content


class TestRerankerEmptyCandidates:
    """Tests for reranker with empty candidate papers."""

    def test_returns_empty_for_no_candidates(self):
        """Should return empty ranked_papers for no candidates."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "test query",
            "candidate_papers": [],
        }

        result = reranker(state)

        assert result["ranked_papers"] == []

    def test_returns_empty_for_none_candidates(self):
        """Should return empty ranked_papers when candidates is None."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "test query",
            "candidate_papers": None,
        }

        result = reranker(state)

        assert result["ranked_papers"] == []


class TestRerankerWithMockedModel:
    """Tests for reranker with mocked FlagReranker model."""

    @pytest.fixture
    def mock_reranker_model(self):
        """Create a mock reranker model."""
        mock_model = MagicMock()
        # compute_score returns a list of scores for each pair
        mock_model.compute_score.return_value = [0.8, 0.5, 0.9]
        return mock_model

    def test_successful_reranking(self, mock_reranker_model):
        """Should rank papers by reranker scores in descending order."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "attention mechanism",
            "candidate_papers": [
                {"title": "Paper A", "id": "1"},
                {"title": "Paper B", "id": "2"},
                {"title": "Paper C", "id": "3"},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        result = reranker(state)

        # Papers should be sorted by score descending: C (0.9), A (0.8), B (0.5)
        assert len(result["ranked_papers"]) == 3
        assert result["ranked_papers"][0]["id"] == "3"  # Paper C with 0.9
        assert result["ranked_papers"][1]["id"] == "1"  # Paper A with 0.8
        assert result["ranked_papers"][2]["id"] == "2"  # Paper B with 0.5

    def test_rerank_scores_added_to_papers(self, mock_reranker_model):
        """Should add rerank_score to each paper."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "test",
            "candidate_papers": [
                {"title": "Paper A"},
                {"title": "Paper B"},
                {"title": "Paper C"},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        result = reranker(state)

        # Each paper should have rerank_score
        for paper in result["ranked_papers"]:
            assert "rerank_score" in paper
            assert isinstance(paper["rerank_score"], float)

    def test_uses_model_from_resources(self, mock_reranker_model):
        """Should use reranker_model from resources if available."""
        from src.agents.formulators.reranker import reranker

        # Update the FlagReranker mock to track if constructor was called
        mock_flag_embedding.FlagReranker.reset_mock()

        state = {
            "query": "test",
            "candidate_papers": [
                {"title": "Paper A"},
                {"title": "Paper B"},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        # Set return value for this test
        mock_reranker_model.compute_score.return_value = [0.7, 0.3]

        result = reranker(state)

        # Should have used the provided model
        mock_reranker_model.compute_score.assert_called_once()
        # FlagReranker constructor should NOT have been called
        mock_flag_embedding.FlagReranker.assert_not_called()

    def test_returns_ai_message_with_results(self, mock_reranker_model):
        """Should return AIMessage with ranked papers content."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "test",
            "candidate_papers": [
                {"title": "Paper A"},
                {"title": "Paper B"},
                {"title": "Paper C"},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        result = reranker(state)

        # Should have messages with AIMessage
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].name == "reranking"

    def test_preserves_original_paper_data(self, mock_reranker_model):
        """Should preserve all original paper fields in ranked results."""
        from src.agents.formulators.reranker import reranker

        state = {
            "query": "test",
            "candidate_papers": [
                {"title": "Paper A", "id": "1", "authors": ["Author 1"], "year": 2023},
                {"title": "Paper B", "id": "2", "authors": ["Author 2"], "year": 2024},
                {"title": "Paper C", "id": "3", "authors": ["Author 3"], "year": 2022},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        result = reranker(state)

        # Check that all original fields are preserved
        for paper in result["ranked_papers"]:
            assert "title" in paper
            assert "id" in paper
            assert "authors" in paper
            assert "year" in paper
            assert "rerank_score" in paper  # Plus the new score

    def test_correct_score_assignment(self, mock_reranker_model):
        """Should correctly assign scores to the right papers."""
        from src.agents.formulators.reranker import reranker

        # Scores correspond to papers in order: A=0.8, B=0.5, C=0.9
        mock_reranker_model.compute_score.return_value = [0.8, 0.5, 0.9]

        state = {
            "query": "test",
            "candidate_papers": [
                {"title": "Paper A", "id": "1"},
                {"title": "Paper B", "id": "2"},
                {"title": "Paper C", "id": "3"},
            ],
            "resources": {"reranker_model": mock_reranker_model},
        }

        result = reranker(state)

        # Find each paper and verify score
        paper_scores = {p["id"]: p["rerank_score"] for p in result["ranked_papers"]}
        assert paper_scores["1"] == 0.8  # Paper A
        assert paper_scores["2"] == 0.5  # Paper B
        assert paper_scores["3"] == 0.9  # Paper C
