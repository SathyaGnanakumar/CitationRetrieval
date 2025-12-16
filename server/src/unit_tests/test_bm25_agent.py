"""Unit tests for the BM25 agent."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from src.agents.retrievers.bm25_agent import bm25_agent, _get_queries


class TestGetQueries:
    """Tests for the _get_queries helper function."""
    
    def test_returns_queries_from_list(self):
        """Should return queries when provided as a list."""
        state = {"queries": ["query1", "query2"]}
        result = _get_queries(state)
        assert result == ["query1", "query2"]
    
    def test_filters_empty_strings(self):
        """Should filter out empty strings from queries list."""
        state = {"queries": ["query1", "", "  ", "query2"]}
        result = _get_queries(state)
        assert result == ["query1", "query2"]
    
    def test_returns_empty_list_when_no_queries(self):
        """Should return empty list when no queries provided."""
        state = {}
        result = _get_queries(state)
        assert result == []


class TestBm25AgentNoResources:
    """Tests for BM25 agent behavior without resources (error handling)."""
    
    def test_returns_error_when_no_queries(self):
        """Should return error message when no queries provided."""
        state = {"queries": []}
        result = bm25_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "no queries" in result["messages"][0].content.lower()
    
    def test_returns_error_when_missing_resources(self):
        """Should return error message when bm25 resources are missing."""
        state = {"queries": ["test query"]}
        result = bm25_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "missing bm25 resources" in result["messages"][0].content.lower()


class TestBm25AgentWithMockedResources:
    """Tests for BM25 agent with mocked resources."""
    
    @pytest.fixture
    def mock_bm25_resources(self):
        """Create mock BM25 resources."""
        mock_stemmer = MagicMock()
        mock_index = MagicMock()
        # Mock retrieve to return doc indices and scores
        mock_index.retrieve.return_value = (
            [[0, 1, 2]],  # doc_ids (corpus indices)
            [[0.9, 0.7, 0.5]]  # scores
        )
        
        return {
            "stemmer": mock_stemmer,
            "index": mock_index,
            "titles": ["Paper A", "Paper B", "Paper C"],
            "ids": ["id_a", "id_b", "id_c"],
        }
    
    @pytest.fixture
    def mock_bm25s_module(self):
        """Create a mock bm25s module."""
        mock_bm25s = MagicMock()
        mock_bm25s.tokenize.return_value = ["test", "tokens"]
        return mock_bm25s
    
    def test_returns_dict_with_results(self, mock_bm25s_module, mock_bm25_resources):
        """Should return dict with bm25_results key."""
        with patch.dict(sys.modules, {'bm25s': mock_bm25s_module}):
            # Need to reload the module to use the mocked bm25s
            from src.agents.retrievers import bm25_agent as bm25_mod
            
            state = {
                "queries": ["test query"],
                "resources": {"bm25": mock_bm25_resources}
            }
            
            result = bm25_mod.bm25_agent(state)
            
            assert isinstance(result, dict)
            assert "bm25_results" in result
            assert "messages" in result
    
    def test_results_have_correct_structure(self, mock_bm25s_module, mock_bm25_resources):
        """Each result should have id, title, score, source."""
        with patch.dict(sys.modules, {'bm25s': mock_bm25s_module}):
            from src.agents.retrievers import bm25_agent as bm25_mod
            
            state = {
                "queries": ["test query"],
                "resources": {"bm25": mock_bm25_resources}
            }
            
            result = bm25_mod.bm25_agent(state)
            
            assert isinstance(result["bm25_results"], list)
            for item in result["bm25_results"]:
                assert "id" in item
                assert "title" in item
                assert "score" in item
                assert "source" in item
                assert item["source"] == "bm25"
    
    def test_returns_correct_number_of_results(self, mock_bm25s_module, mock_bm25_resources):
        """Should return the mocked number of results."""
        with patch.dict(sys.modules, {'bm25s': mock_bm25s_module}):
            from src.agents.retrievers import bm25_agent as bm25_mod
            
            state = {
                "queries": ["test query"],
                "resources": {"bm25": mock_bm25_resources}
            }
            
            result = bm25_mod.bm25_agent(state)
            
            # We mocked 3 results
            assert len(result["bm25_results"]) == 3
    
    def test_respects_k_config(self, mock_bm25s_module, mock_bm25_resources):
        """Should pass k value from config to the retriever."""
        with patch.dict(sys.modules, {'bm25s': mock_bm25s_module}):
            from src.agents.retrievers import bm25_agent as bm25_mod
            
            state = {
                "queries": ["test query"],
                "resources": {"bm25": mock_bm25_resources},
                "config": {"k": 10}
            }
            
            result = bm25_mod.bm25_agent(state)
            
            # Verify retrieve was called with k=10
            mock_bm25_resources["index"].retrieve.assert_called_once()
            call_args = mock_bm25_resources["index"].retrieve.call_args
            assert call_args.kwargs.get("k") == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
