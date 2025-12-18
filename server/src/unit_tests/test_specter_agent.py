"""Unit tests for the SPECTER agent."""

import pytest
from unittest.mock import MagicMock
import torch

from src.agents.retrievers.specter_agent import specter_agent, SPECTERRetriever, _get_queries


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


class TestSpecterAgentNoResources:
    """Tests for SPECTER agent behavior without resources (error handling)."""
    
    def test_returns_error_when_no_queries(self):
        """Should return error message when no queries provided."""
        state = {"queries": []}
        result = specter_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "no queries" in result["messages"][0].content.lower()
    
    def test_returns_error_when_missing_resources(self):
        """Should return error message when specter resources are missing."""
        state = {"queries": ["test query"]}
        result = specter_agent(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "missing specter resources" in result["messages"][0].content.lower()


class TestSPECTERRetriever:
    """Tests for the SPECTERRetriever class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock SPECTER model."""
        model = MagicMock()
        # Mock the forward call to return an object with last_hidden_state
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)  # (batch, seq_len, hidden)
        model.return_value = mock_output
        model.to = MagicMock(return_value=model)
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Return mock tensor inputs
        mock_inputs = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }
        tokenizer.return_value = MagicMock(**mock_inputs)
        tokenizer.return_value.to = MagicMock(return_value=tokenizer.return_value)
        return tokenizer
    
    @pytest.fixture
    def mock_corpus_data(self):
        """Create mock corpus data."""
        return {
            "embeddings": torch.randn(5, 768),  # 5 documents
            "ids": ["id1", "id2", "id3", "id4", "id5"],
            "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"]
        }
    
    def test_single_query_returns_list(self, mock_model, mock_tokenizer, mock_corpus_data):
        """single_query should return a list of results."""
        retriever = SPECTERRetriever(mock_model, mock_tokenizer, device="cpu")
        
        results = retriever.single_query(
            "test query",
            mock_corpus_data["embeddings"],
            mock_corpus_data["ids"],
            mock_corpus_data["titles"],
            k=3
        )
        
        assert isinstance(results, list)
        assert len(results) == 3
    
    def test_single_query_result_structure(self, mock_model, mock_tokenizer, mock_corpus_data):
        """Each result should have id, title, score, source."""
        retriever = SPECTERRetriever(mock_model, mock_tokenizer, device="cpu")
        
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
            assert result["source"] == "specter"
    
    def test_batch_query_returns_list_of_lists(self, mock_model, mock_tokenizer, mock_corpus_data):
        """batch_query should return a list of result lists."""
        # Update mock output for batch
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)  # 2 queries
        mock_model.return_value = mock_output
        
        retriever = SPECTERRetriever(mock_model, mock_tokenizer, device="cpu")
        
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


class TestSpecterAgentWithMockedResources:
    """Tests for SPECTER agent with mocked resources."""
    
    @pytest.fixture
    def mock_specter_resources(self):
        """Create mock SPECTER resources."""
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_output
        mock_model.to = MagicMock(return_value=mock_model)
        
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs
        
        return {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "corpus_embeddings": torch.randn(5, 768),
            "ids": ["id1", "id2", "id3", "id4", "id5"],
            "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "device": "cpu"
        }
    
    def test_returns_dict_with_results(self, mock_specter_resources):
        """Should return dict with specter_results key."""
        state = {
            "queries": ["test query"],
            "resources": {"specter": mock_specter_resources}
        }
        
        result = specter_agent(state)
        
        assert isinstance(result, dict)
        assert "specter_results" in result
        assert "messages" in result
    
    def test_results_have_correct_structure(self, mock_specter_resources):
        """Each result should have id, title, score, source."""
        state = {
            "queries": ["test query"],
            "resources": {"specter": mock_specter_resources}
        }
        
        result = specter_agent(state)
        
        assert isinstance(result["specter_results"], list)
        for item in result["specter_results"]:
            assert "id" in item
            assert "title" in item
            assert "score" in item
            assert "source" in item
            assert item["source"] == "specter"
    
    def test_handles_multiple_queries(self, mock_specter_resources):
        """Should handle multiple queries using batch_query."""
        # Update mock for batch processing
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_specter_resources["model"].return_value = mock_output
        
        state = {
            "queries": ["query1", "query2"],
            "resources": {"specter": mock_specter_resources}
        }
        
        result = specter_agent(state)
        
        assert isinstance(result, dict)
        assert "specter_results" in result
    
    def test_respects_k_config(self, mock_specter_resources):
        """Should respect k value from config."""
        state = {
            "queries": ["test query"],
            "resources": {"specter": mock_specter_resources},
            "config": {"k": 3}
        }
        
        result = specter_agent(state)
        
        # With k=3 and 5 docs, should get 3 results
        assert len(result["specter_results"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
