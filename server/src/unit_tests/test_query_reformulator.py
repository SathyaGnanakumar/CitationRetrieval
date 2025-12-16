"""Unit tests for the query reformulator."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.formulators.query_reformulator import (
    query_reformulator,
    extract_keywords,
    expand_keywords,
    academic_style_rewrite,
    ACADEMIC_EXPANSIONS,
)


class TestExtractKeywords:
    """Tests for the extract_keywords function."""
    
    def test_extracts_words_longer_than_3_chars(self):
        """Should only extract words with more than 3 characters."""
        query = "the transformer model is efficient"
        keywords = extract_keywords(query)
        
        # "the" and "is" should be excluded (too short)
        assert "the" not in keywords
        assert "transformer" in keywords
        assert "model" in keywords
        assert "efficient" in keywords
    
    def test_converts_to_lowercase(self):
        """Should convert all keywords to lowercase."""
        query = "TRANSFORMER Model RETRIEVAL"
        keywords = extract_keywords(query)
        
        assert "transformer" in keywords
        assert "model" in keywords
        assert "retrieval" in keywords
        assert "TRANSFORMER" not in keywords
    
    def test_handles_empty_query(self):
        """Should handle empty query."""
        keywords = extract_keywords("")
        assert keywords == []


class TestExpandKeywords:
    """Tests for the expand_keywords function."""
    
    def test_expands_known_keywords(self):
        """Should expand keywords that exist in ACADEMIC_EXPANSIONS."""
        keywords = ["transformer", "model"]
        expansions = expand_keywords(keywords)
        
        # Should include expansions for 'transformer'
        assert "self-attention" in expansions
        assert "encoder-decoder" in expansions
        # Should include expansions for 'model'
        assert "architecture" in expansions
        assert "neural network" in expansions
    
    def test_ignores_unknown_keywords(self):
        """Should not add anything for unknown keywords."""
        keywords = ["unknown", "randomword"]
        expansions = expand_keywords(keywords)
        
        assert expansions == []
    
    def test_handles_empty_list(self):
        """Should handle empty keyword list."""
        expansions = expand_keywords([])
        assert expansions == []


class TestAcademicStyleRewrite:
    """Tests for the academic_style_rewrite function."""
    
    def test_produces_formatted_string(self):
        """Should produce a formatted academic-style query."""
        query = "transformer model"
        keywords = ["transformer", "model"]
        expansions = ["self-attention", "encoder-decoder", "architecture", "neural network"]
        
        result = academic_style_rewrite(query, keywords, expansions)
        
        assert "paper discussing" in result
        assert "citation retrieval" in result
        assert "transformer" in result
        assert "model" in result
    
    def test_includes_first_three_expansions(self):
        """Should include at most first 3 expansions."""
        query = "test"
        keywords = ["transformer"]
        expansions = ["exp1", "exp2", "exp3", "exp4", "exp5"]
        
        result = academic_style_rewrite(query, keywords, expansions)
        
        assert "exp1" in result
        assert "exp2" in result
        assert "exp3" in result
        # exp4 and exp5 should not be included (only first 3)
        assert "exp4" not in result
        assert "exp5" not in result


class TestQueryReformulator:
    """Tests for the main query_reformulator function."""
    
    def test_returns_dict(self):
        """Should return a dictionary."""
        state = {"messages": [HumanMessage(content="transformer retrieval")]}
        result = query_reformulator(state)
        
        assert isinstance(result, dict)
    
    def test_includes_query_key(self):
        """Should include 'query' key with original query."""
        state = {"messages": [HumanMessage(content="transformer retrieval")]}
        result = query_reformulator(state)
        
        assert "query" in result
        assert result["query"] == "transformer retrieval"
    
    def test_includes_queries_key(self):
        """Should include 'queries' key with list of expanded queries."""
        state = {"messages": [HumanMessage(content="transformer retrieval")]}
        result = query_reformulator(state)
        
        assert "queries" in result
        assert isinstance(result["queries"], list)
        assert len(result["queries"]) == 4  # Original + 3 variations
    
    def test_includes_messages_key(self):
        """Should include 'messages' key for workflow compatibility."""
        state = {"messages": [HumanMessage(content="transformer retrieval")]}
        result = query_reformulator(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
    
    def test_original_query_in_queries(self):
        """Original query should be the first in queries list."""
        state = {"messages": [HumanMessage(content="transformer retrieval")]}
        result = query_reformulator(state)
        
        assert result["queries"][0] == "transformer retrieval"
    
    def test_handles_no_human_message(self):
        """Should handle state with no human message."""
        state = {"messages": [AIMessage(content="some ai response")]}
        result = query_reformulator(state)
        
        assert isinstance(result, dict)
        assert "messages" in result
        assert "[]" in result["messages"][0].content
    
    def test_uses_last_human_message(self):
        """Should use the last human message in the list."""
        state = {
            "messages": [
                HumanMessage(content="first query"),
                AIMessage(content="response"),
                HumanMessage(content="second query"),
            ]
        }
        result = query_reformulator(state)
        
        assert result["query"] == "second query"
    
    def test_expanded_queries_include_expansions(self):
        """Expanded queries should include academic expansions for relevant keywords."""
        state = {"messages": [HumanMessage(content="transformer model retrieval")]}
        result = query_reformulator(state)
        
        # Join all queries to check for expansion terms
        all_queries_text = " ".join(result["queries"])
        
        # Should include some expanded terms
        assert "self-attention" in all_queries_text or "attention mechanism" in all_queries_text
    
    def test_strips_whitespace_from_query(self):
        """Should strip whitespace from the input query."""
        state = {"messages": [HumanMessage(content="  transformer model  ")]}
        result = query_reformulator(state)
        
        assert result["query"] == "transformer model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
