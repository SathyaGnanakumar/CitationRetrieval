"""Unit tests for DSPy modules."""

import pytest
from unittest.mock import MagicMock, patch
import dspy

from src.agents.formulators.dspy_prompt_generator.modules import (
    SimpleCitationRetriever,
    QueryThenRetrieve,
    RerankAndSelect,
    VerifyAndSelect,
    EnsembleRetriever,
    get_module,
)


class TestGetModule:
    """Tests for the get_module factory function."""
    
    def test_get_simple_module(self):
        """Should return SimpleCitationRetriever for 'simple'."""
        module = get_module("simple")
        assert isinstance(module, SimpleCitationRetriever)
    
    def test_get_query_module(self):
        """Should return QueryThenRetrieve for 'query'."""
        module = get_module("query")
        assert isinstance(module, QueryThenRetrieve)
    
    def test_get_rerank_module(self):
        """Should return RerankAndSelect for 'rerank'."""
        module = get_module("rerank")
        assert isinstance(module, RerankAndSelect)
    
    def test_get_verify_module(self):
        """Should return VerifyAndSelect for 'verify'."""
        module = get_module("verify")
        assert isinstance(module, VerifyAndSelect)
    
    def test_get_ensemble_module(self):
        """Should return EnsembleRetriever for 'ensemble'."""
        module = get_module("ensemble")
        assert isinstance(module, EnsembleRetriever)
    
    def test_unknown_module_raises_error(self):
        """Should raise ValueError for unknown module name."""
        with pytest.raises(ValueError) as exc_info:
            get_module("nonexistent")
        
        assert "Unknown module" in str(exc_info.value)


class TestSimpleCitationRetrieverFormatting:
    """Tests for SimpleCitationRetriever's candidate formatting."""
    
    @pytest.fixture
    def retriever(self):
        """Create a SimpleCitationRetriever instance."""
        return SimpleCitationRetriever()
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate papers."""
        return [
            {"title": "Paper One", "abstract": "Abstract for paper one."},
            {"title": "Paper Two", "abstract": "Abstract for paper two."},
            {"title": "Paper Three", "abstract": ""},  # No abstract
        ]
    
    def test_format_candidates_includes_all_titles(self, retriever, sample_candidates):
        """Formatted string should include all titles."""
        result = retriever._format_candidates(sample_candidates)
        
        assert "Paper One" in result
        assert "Paper Two" in result
        assert "Paper Three" in result
    
    def test_format_candidates_numbers_papers(self, retriever, sample_candidates):
        """Papers should be numbered starting from 1."""
        result = retriever._format_candidates(sample_candidates)
        
        assert "1. Title:" in result
        assert "2. Title:" in result
        assert "3. Title:" in result
    
    def test_format_candidates_includes_abstracts(self, retriever, sample_candidates):
        """Formatted string should include abstracts."""
        result = retriever._format_candidates(sample_candidates)
        
        assert "Abstract for paper one" in result
        assert "Abstract for paper two" in result
    
    def test_format_candidates_truncates_long_abstracts(self, retriever):
        """Long abstracts should be truncated to 300 chars."""
        candidates = [
            {"title": "Long Paper", "abstract": "A" * 500}
        ]
        
        result = retriever._format_candidates(candidates)
        
        # Should have truncated abstract with "..."
        assert "A" * 300 in result
        assert "A" * 400 not in result
    
    def test_format_candidates_handles_missing_title(self, retriever):
        """Should handle missing title gracefully."""
        candidates = [
            {"abstract": "Abstract only"}  # No title
        ]
        
        result = retriever._format_candidates(candidates)
        
        assert "Unknown" in result
    
    def test_format_candidates_handles_empty_list(self, retriever):
        """Should handle empty candidate list."""
        result = retriever._format_candidates([])
        
        assert result == ""


class TestQueryThenRetrieveFormatting:
    """Tests for QueryThenRetrieve's candidate formatting."""
    
    @pytest.fixture
    def retriever(self):
        """Create a QueryThenRetrieve instance."""
        return QueryThenRetrieve()
    
    def test_format_candidates_same_as_simple(self, retriever):
        """Format should be consistent with SimpleCitationRetriever."""
        candidates = [
            {"title": "Test Paper", "abstract": "Test abstract."}
        ]
        
        result = retriever._format_candidates(candidates)
        
        assert "1. Title: Test Paper" in result
        assert "Abstract: Test abstract." in result


class TestRerankAndSelectFormatting:
    """Tests for RerankAndSelect's candidate formatting."""
    
    @pytest.fixture
    def retriever(self):
        """Create a RerankAndSelect instance."""
        return RerankAndSelect()
    
    def test_format_candidates_includes_all(self, retriever):
        """Should format all candidates."""
        candidates = [
            {"title": "Paper A", "abstract": "Abstract A"},
            {"title": "Paper B", "abstract": "Abstract B"},
        ]
        
        result = retriever._format_candidates(candidates)
        
        assert "Paper A" in result
        assert "Paper B" in result


class TestVerifyAndSelectConfig:
    """Tests for VerifyAndSelect configuration."""
    
    def test_default_max_candidates(self):
        """Default max_candidates should be 5."""
        retriever = VerifyAndSelect()
        assert retriever.max_candidates == 5
    
    def test_custom_max_candidates(self):
        """Should accept custom max_candidates."""
        retriever = VerifyAndSelect(max_candidates=10)
        assert retriever.max_candidates == 10


class TestModuleInheritance:
    """Tests to verify modules inherit from dspy.Module."""
    
    def test_simple_is_dspy_module(self):
        """SimpleCitationRetriever should be a dspy.Module."""
        retriever = SimpleCitationRetriever()
        assert isinstance(retriever, dspy.Module)
    
    def test_query_is_dspy_module(self):
        """QueryThenRetrieve should be a dspy.Module."""
        retriever = QueryThenRetrieve()
        assert isinstance(retriever, dspy.Module)
    
    def test_rerank_is_dspy_module(self):
        """RerankAndSelect should be a dspy.Module."""
        retriever = RerankAndSelect()
        assert isinstance(retriever, dspy.Module)
    
    def test_verify_is_dspy_module(self):
        """VerifyAndSelect should be a dspy.Module."""
        retriever = VerifyAndSelect()
        assert isinstance(retriever, dspy.Module)
    
    def test_ensemble_is_dspy_module(self):
        """EnsembleRetriever should be a dspy.Module."""
        retriever = EnsembleRetriever()
        assert isinstance(retriever, dspy.Module)


class TestModuleHasForwardMethod:
    """Tests to verify modules have forward method."""
    
    def test_simple_has_forward(self):
        """SimpleCitationRetriever should have forward method."""
        retriever = SimpleCitationRetriever()
        assert hasattr(retriever, "forward")
        assert callable(retriever.forward)
    
    def test_query_has_forward(self):
        """QueryThenRetrieve should have forward method."""
        retriever = QueryThenRetrieve()
        assert hasattr(retriever, "forward")
        assert callable(retriever.forward)
    
    def test_rerank_has_forward(self):
        """RerankAndSelect should have forward method."""
        retriever = RerankAndSelect()
        assert hasattr(retriever, "forward")
        assert callable(retriever.forward)
    
    def test_verify_has_forward(self):
        """VerifyAndSelect should have forward method."""
        retriever = VerifyAndSelect()
        assert hasattr(retriever, "forward")
        assert callable(retriever.forward)
    
    def test_ensemble_has_forward(self):
        """EnsembleRetriever should have forward method."""
        retriever = EnsembleRetriever()
        assert hasattr(retriever, "forward")
        assert callable(retriever.forward)


class TestEnsembleRetrieverComponents:
    """Tests for EnsembleRetriever's component modules."""
    
    def test_has_simple_component(self):
        """Should have SimpleCitationRetriever component."""
        ensemble = EnsembleRetriever()
        assert hasattr(ensemble, "simple")
        assert isinstance(ensemble.simple, SimpleCitationRetriever)
    
    def test_has_query_component(self):
        """Should have QueryThenRetrieve component."""
        ensemble = EnsembleRetriever()
        assert hasattr(ensemble, "query_then_retrieve")
        assert isinstance(ensemble.query_then_retrieve, QueryThenRetrieve)
    
    def test_has_rerank_component(self):
        """Should have RerankAndSelect component."""
        ensemble = EnsembleRetriever()
        assert hasattr(ensemble, "rerank")
        assert isinstance(ensemble.rerank, RerankAndSelect)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

