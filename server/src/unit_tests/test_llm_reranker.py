"""Unit tests for the LLM reranker in llm_agent.py."""

import pytest
from unittest.mock import MagicMock, patch
import sys


class TestLLMRerankerEmptyCandidates:
    """Tests for llm_reranker with empty candidate papers."""

    def test_empty_candidates_returns_empty_list(self):
        """When candidate_papers is empty, should return empty ranked_papers."""
        from src.agents.formulators.llm_agent import llm_reranker

        # Create a mock state with empty candidates
        mock_state = MagicMock()
        mock_state.query = "test query"
        mock_state.candidate_papers = []

        result = llm_reranker(mock_state)

        assert result == {"ranked_papers": []}

    def test_no_candidates_attribute_returns_empty_list(self):
        """When candidate_papers attribute doesn't exist, should return empty."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_state = MagicMock(spec=[])  # No attributes
        mock_state.query = "test query"
        mock_state.candidate_papers = []

        result = llm_reranker(mock_state)

        assert result == {"ranked_papers": []}


class TestLLMRerankerSuccessfulParsing:
    """Tests for llm_reranker with successful LLM response parsing."""

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_successful_json_array_parsing(self, mock_ollama_class, mock_prompt_class):
        """Should correctly parse JSON array response and rank papers."""
        from src.agents.formulators.llm_agent import llm_reranker

        # Setup mock LLM
        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        # Mock prompt
        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # Mock response with valid JSON array
        mock_response = MagicMock()
        mock_response.content = '[{"index": 2, "score": 0.95}, {"index": 1, "score": 0.80}]'
        mock_llm.invoke.return_value = mock_response

        # Create mock state
        mock_state = MagicMock()
        mock_state.query = "neural network attention mechanisms"
        mock_state.candidate_papers = [
            {"title": "Paper A", "id": "1"},
            {"title": "Paper B", "id": "2"},
        ]

        result = llm_reranker(mock_state, closed_source=False)

        # Paper at index 2 (1-based) should be first with score 0.95
        assert len(result["ranked_papers"]) == 2
        assert result["ranked_papers"][0] == ({"title": "Paper B", "id": "2"}, 0.95)
        assert result["ranked_papers"][1] == ({"title": "Paper A", "id": "1"}, 0.80)

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_json_embedded_in_text(self, mock_ollama_class, mock_prompt_class):
        """Should extract JSON array even when embedded in other text."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # Response with JSON embedded in text
        mock_response = MagicMock()
        mock_response.content = '''Based on my analysis:
        [{"index": 1, "score": 0.90}]
        These papers are highly relevant.'''
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=False)

        assert len(result["ranked_papers"]) == 1
        assert result["ranked_papers"][0][1] == 0.90

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_missing_papers_get_zero_score(self, mock_ollama_class, mock_prompt_class):
        """Papers not ranked by LLM should get score 0.0."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # LLM only ranks paper at index 1
        mock_response = MagicMock()
        mock_response.content = '[{"index": 1, "score": 0.85}]'
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [
            {"title": "Paper A", "id": "1"},
            {"title": "Paper B", "id": "2"},
            {"title": "Paper C", "id": "3"},
        ]

        result = llm_reranker(mock_state, closed_source=False)

        # Should have all 3 papers
        assert len(result["ranked_papers"]) == 3
        # First paper (index 1 in 1-based) is ranked
        assert result["ranked_papers"][0] == ({"title": "Paper A", "id": "1"}, 0.85)
        # Other papers get score 0.0
        scores = [score for _, score in result["ranked_papers"]]
        assert 0.0 in scores

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_duplicate_indices_handled(self, mock_ollama_class, mock_prompt_class):
        """Should skip duplicate paper indices in LLM response."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # LLM returns duplicate indices
        mock_response = MagicMock()
        mock_response.content = '[{"index": 1, "score": 0.95}, {"index": 1, "score": 0.80}]'
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=False)

        # Should only have 1 paper (duplicate skipped)
        assert len(result["ranked_papers"]) == 1
        # First occurrence is used
        assert result["ranked_papers"][0][1] == 0.95

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_out_of_bounds_index_skipped(self, mock_ollama_class, mock_prompt_class):
        """Should skip indices that are out of bounds."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # Index 10 is out of bounds
        mock_response = MagicMock()
        mock_response.content = '[{"index": 10, "score": 0.95}, {"index": 1, "score": 0.80}]'
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=False)

        # Only valid index (1) should be included
        assert len(result["ranked_papers"]) == 1
        assert result["ranked_papers"][0][1] == 0.80


class TestLLMRerankerErrorHandling:
    """Tests for llm_reranker error handling and fallback behavior."""

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_invalid_json_fallback(self, mock_ollama_class, mock_prompt_class):
        """Should fallback to original order with 0.1 scores on JSON error."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # Invalid JSON response
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [
            {"title": "Paper A", "id": "1"},
            {"title": "Paper B", "id": "2"},
        ]

        result = llm_reranker(mock_state, closed_source=False)

        # Fallback: all papers with score 0.1
        assert len(result["ranked_papers"]) == 2
        assert result["ranked_papers"][0] == ({"title": "Paper A", "id": "1"}, 0.1)
        assert result["ranked_papers"][1] == ({"title": "Paper B", "id": "2"}, 0.1)

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_missing_score_key_fallback(self, mock_ollama_class, mock_prompt_class):
        """Should fallback when JSON is valid but missing required keys."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        # JSON missing 'score' key
        mock_response = MagicMock()
        mock_response.content = '[{"index": 1}]'  # Missing score
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=False)

        # Fallback behavior
        assert len(result["ranked_papers"]) == 1
        assert result["ranked_papers"][0][1] == 0.1


class TestLLMRerankerModelSelection:
    """Tests for LLM model selection."""

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOpenAI")
    def test_closed_source_uses_openai(self, mock_openai_class, mock_prompt_class):
        """When closed_source=True, should use OpenAI."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_openai_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        mock_response = MagicMock()
        mock_response.content = '[{"index": 1, "score": 0.9}]'
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=True)

        # OpenAI should have been instantiated
        mock_openai_class.assert_called_once()
        assert len(result["ranked_papers"]) == 1

    @patch("src.agents.formulators.llm_agent.LLMRerankerPrompt")
    @patch("src.agents.formulators.llm_agent.ChatOllama")
    def test_open_source_uses_ollama(self, mock_ollama_class, mock_prompt_class):
        """When closed_source=False, should use Ollama."""
        from src.agents.formulators.llm_agent import llm_reranker

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.get_prompt.return_value = "mocked prompt"
        mock_prompt_class.return_value = mock_prompt_instance

        mock_response = MagicMock()
        mock_response.content = '[{"index": 1, "score": 0.9}]'
        mock_llm.invoke.return_value = mock_response

        mock_state = MagicMock()
        mock_state.query = "test"
        mock_state.candidate_papers = [{"title": "Paper A", "id": "1"}]

        result = llm_reranker(mock_state, closed_source=False)

        # Ollama should have been instantiated
        mock_ollama_class.assert_called_once()
        assert len(result["ranked_papers"]) == 1


class TestLLMRerankerPrompt:
    """Tests for the LLMRerankerPrompt class."""

    def test_prompt_builds_correctly(self):
        """Should build a prompt with query and candidate papers."""
        from src.prompts.llm_reranker import LLMRerankerPrompt

        query = "transformer attention"
        papers = [
            {"title": "Attention Is All You Need", "score": 0.9},
            {"title": "BERT Paper", "score": 0.7},
        ]

        prompt = LLMRerankerPrompt(query=query, candidate_papers=papers)
        result = prompt.get_prompt()

        assert "transformer attention" in result
        assert "Attention Is All You Need" in result
        assert "BERT Paper" in result
        assert "1." in result
        assert "2." in result

    def test_prompt_handles_missing_title(self):
        """Should handle papers without titles gracefully."""
        from src.prompts.llm_reranker import LLMRerankerPrompt

        papers = [{"id": "1"}]  # No title
        prompt = LLMRerankerPrompt(query="test", candidate_papers=papers)
        result = prompt.get_prompt()

        # Should not crash, should have prompt structure
        assert "test" in result
        assert "1." in result

    def test_prompt_handles_non_float_score(self):
        """Should handle non-float scores in papers."""
        from src.prompts.llm_reranker import LLMRerankerPrompt

        papers = [{"title": "Paper A", "score": "not a number"}]
        prompt = LLMRerankerPrompt(query="test", candidate_papers=papers)
        result = prompt.get_prompt()

        # Should not crash, should use 0.0 as fallback
        assert "Paper A" in result
        assert "0.0000" in result
