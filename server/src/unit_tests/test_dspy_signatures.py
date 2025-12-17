"""Unit tests for DSPy signatures."""

import pytest
import dspy

from src.agents.formulators.dspy_prompt_generator.signatures import (
    CitationRetrieval,
    QueryGeneration,
    CitationReranking,
    CitationVerification,
)


class TestCitationRetrievalSignature:
    """Tests for the CitationRetrieval signature."""
    
    def test_is_dspy_signature(self):
        """Should be a valid dspy.Signature."""
        assert issubclass(CitationRetrieval, dspy.Signature)
    
    def test_has_citation_context_input(self):
        """Should have citation_context as input field."""
        fields = CitationRetrieval.model_fields
        assert "citation_context" in fields
    
    def test_has_candidate_papers_input(self):
        """Should have candidate_papers as input field."""
        fields = CitationRetrieval.model_fields
        assert "candidate_papers" in fields
    
    def test_has_reasoning_output(self):
        """Should have reasoning as output field."""
        fields = CitationRetrieval.model_fields
        assert "reasoning" in fields
    
    def test_has_selected_title_output(self):
        """Should have selected_title as output field."""
        fields = CitationRetrieval.model_fields
        assert "selected_title" in fields


class TestQueryGenerationSignature:
    """Tests for the QueryGeneration signature."""
    
    def test_is_dspy_signature(self):
        """Should be a valid dspy.Signature."""
        assert issubclass(QueryGeneration, dspy.Signature)
    
    def test_has_citation_context_input(self):
        """Should have citation_context as input field."""
        fields = QueryGeneration.model_fields
        assert "citation_context" in fields
    
    def test_has_reasoning_output(self):
        """Should have reasoning as output field."""
        fields = QueryGeneration.model_fields
        assert "reasoning" in fields
    
    def test_has_search_query_output(self):
        """Should have search_query as output field."""
        fields = QueryGeneration.model_fields
        assert "search_query" in fields


class TestCitationRerankingSignature:
    """Tests for the CitationReranking signature."""
    
    def test_is_dspy_signature(self):
        """Should be a valid dspy.Signature."""
        assert issubclass(CitationReranking, dspy.Signature)
    
    def test_has_citation_context_input(self):
        """Should have citation_context as input field."""
        fields = CitationReranking.model_fields
        assert "citation_context" in fields
    
    def test_has_candidate_papers_input(self):
        """Should have candidate_papers as input field."""
        fields = CitationReranking.model_fields
        assert "candidate_papers" in fields
    
    def test_has_reasoning_output(self):
        """Should have reasoning as output field."""
        fields = CitationReranking.model_fields
        assert "reasoning" in fields
    
    def test_has_ranked_titles_output(self):
        """Should have ranked_titles as output field."""
        fields = CitationReranking.model_fields
        assert "ranked_titles" in fields


class TestCitationVerificationSignature:
    """Tests for the CitationVerification signature."""
    
    def test_is_dspy_signature(self):
        """Should be a valid dspy.Signature."""
        assert issubclass(CitationVerification, dspy.Signature)
    
    def test_has_citation_context_input(self):
        """Should have citation_context as input field."""
        fields = CitationVerification.model_fields
        assert "citation_context" in fields
    
    def test_has_candidate_title_input(self):
        """Should have candidate_title as input field."""
        fields = CitationVerification.model_fields
        assert "candidate_title" in fields
    
    def test_has_candidate_abstract_input(self):
        """Should have candidate_abstract as input field."""
        fields = CitationVerification.model_fields
        assert "candidate_abstract" in fields
    
    def test_has_reasoning_output(self):
        """Should have reasoning as output field."""
        fields = CitationVerification.model_fields
        assert "reasoning" in fields
    
    def test_has_is_match_output(self):
        """Should have is_match as output field."""
        fields = CitationVerification.model_fields
        assert "is_match" in fields
    
    def test_has_confidence_output(self):
        """Should have confidence as output field."""
        fields = CitationVerification.model_fields
        assert "confidence" in fields


class TestSignatureFieldCounts:
    """Tests for expected field counts in signatures."""
    
    def test_citation_retrieval_field_count(self):
        """CitationRetrieval should have 4 fields (2 in, 2 out)."""
        fields = CitationRetrieval.model_fields
        # citation_context, candidate_papers, reasoning, selected_title
        assert len(fields) == 4
    
    def test_query_generation_field_count(self):
        """QueryGeneration should have 3 fields (1 in, 2 out)."""
        fields = QueryGeneration.model_fields
        # citation_context, reasoning, search_query
        assert len(fields) == 3
    
    def test_citation_reranking_field_count(self):
        """CitationReranking should have 4 fields (2 in, 2 out)."""
        fields = CitationReranking.model_fields
        # citation_context, candidate_papers, reasoning, ranked_titles
        assert len(fields) == 4
    
    def test_citation_verification_field_count(self):
        """CitationVerification should have 6 fields (3 in, 3 out)."""
        fields = CitationVerification.model_fields
        # citation_context, candidate_title, candidate_abstract, reasoning, is_match, confidence
        assert len(fields) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

