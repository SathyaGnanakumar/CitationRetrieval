"""Unit tests for the E5 agent."""

import pytest
from unittest.mock import MagicMock
import torch

from src.agents.retrievers.e5_agent import e5_agent, E5Retriever, _get_queries


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


class TestE5AgentNoResources:
    """Tests for E5 agent behavior without resources (error handling)."""
    
    def test_returns_error_when_no_queries(self):
        """Should return error message when no queries provided."""
        state = {"queries": []}
        result = e5_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "no queries" in result["messages"][0].content.lower()
    
    def test_returns_error_when_missing_resources(self):
        """Should return error message when e5 resources are missing."""
        state = {"queries": ["test query"]}
        result = e5_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "missing e5 resources" in result["messages"][0].content.lower()


class TestE5Retriever:
    """Tests for the E5Retriever class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock SentenceTransformer model."""
        model = MagicMock()
        # Mock encode to return a tensor
        model.encode.return_value = torch.randn(768)  # Single query
        return model
    
    @pytest.fixture
    def mock_corpus_data(self):
        """Create mock corpus data."""
        return {
            "embeddings": torch.randn(5, 768),  # 5 documents
            "ids": ["id1", "id2", "id3", "id4", "id5"],
            "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"]
        }
    
    def test_single_query_returns_list(self, mock_model, mock_corpus_data):
        """single_query should return a list of results."""
        retriever = E5Retriever(mock_model, device="cpu")
        
        # Need to return tensor for encode
        mock_model.encode.return_value = torch.randn(768)
        
        results = retriever.single_query(
            "test query",
            mock_corpus_data["embeddings"],
            mock_corpus_data["ids"],
            mock_corpus_data["titles"],
            k=3
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
    
    def test_single_query_result_structure(self, mock_model, mock_corpus_data):
        """Each result should have id, title, score, source."""
        retriever = E5Retriever(mock_model, device="cpu")
        mock_model.encode.return_value = torch.randn(768)
        
        results = retriever.single_query(
            "test query",
            mock_corpus_data["embeddings"],
            mock_corpus_data["ids"],
            mock_corpus_data["titles"],
            k=3
        )
        
        for result in results:
            assert "id" in result
            assert "title" in result
            assert "score" in result
            assert "source" in result
            assert result["source"] == "e5"
    
    def test_batch_query_returns_list_of_lists(self, mock_model, mock_corpus_data):
        """batch_query should return a list of result lists."""
        retriever = E5Retriever(mock_model, device="cpu")
        
        # Mock encode to return tensor for batch
        mock_model.encode.return_value = torch.randn(2, 768)  # 2 queries
        
        results = retriever.batch_query(
            ["query1", "query2"],
            mock_corpus_data["embeddings"],
            mock_corpus_data["ids"],
            mock_corpus_data["titles"],
            k=3
        )
        
        assert isinstance(results, list)
        assert len(results) == 2  # One list per query
        for query_results in results:
            assert isinstance(query_results, list)
            assert len(query_results) == 3


class TestE5AgentWithMockedResources:
    """Tests for E5 agent with mocked resources."""
    
    @pytest.fixture
    def mock_e5_resources(self):
        """Create mock E5 resources."""
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.randn(768)
        
        return {
            "model": mock_model,
            "corpus_embeddings": torch.randn(5, 768),
            "ids": ["id1", "id2", "id3", "id4", "id5"],
            "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "device": "cpu"
        }
    
    def test_returns_dict_with_results(self, mock_e5_resources):
        """Should return dict with e5_results key."""
        state = {
            "queries": ["test query"],
            "resources": {"e5": mock_e5_resources}
        }
        
        result = e5_agent(state)
        
        assert isinstance(result, dict)
        assert "e5_results" in result
        assert "messages" in result
    
    def test_results_have_correct_structure(self, mock_e5_resources):
        """Each result should have id, title, score, source."""
        state = {
            "queries": ["test query"],
            "resources": {"e5": mock_e5_resources}
        }
        
        result = e5_agent(state)
        
        assert isinstance(result["e5_results"], list)
        for item in result["e5_results"]:
            assert "id" in item
            assert "title" in item
            assert "score" in item
            assert "source" in item
            assert item["source"] == "e5"
    
    def test_handles_multiple_queries(self, mock_e5_resources):
        """Should handle multiple queries using batch_query."""
        # Update mock for batch encoding
        mock_e5_resources["model"].encode.return_value = torch.randn(2, 768)
        
        state = {
            "queries": ["query1", "query2"],
            "resources": {"e5": mock_e5_resources}
        }
        
        result = e5_agent(state)
        
        assert isinstance(result, dict)
        assert "e5_results" in result
    
    def test_respects_k_config(self, mock_e5_resources):
        """Should respect k value from config."""
        state = {
            "queries": ["test query"],
            "resources": {"e5": mock_e5_resources},
            "config": {"k": 3}
        }
        
        result = e5_agent(state)
        
        # With k=3 and 5 docs, should get 3 results
        assert len(result["e5_results"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
