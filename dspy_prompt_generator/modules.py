"""
DSPy Modules for Citation Retrieval.

Composable modules that implement citation retrieval pipelines.
"""

import dspy
from typing import List, Dict, Optional
import json

from .signatures import (
    CitationRetrieval,
    QueryGeneration,
    CitationReranking,
    CitationVerification,
)


class SimpleCitationRetriever(dspy.Module):
    """
    Simple citation retrieval module.
    
    Given a citation context and candidate papers, directly selects the cited paper.
    Uses Chain-of-Thought reasoning for better accuracy.
    """
    
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ChainOfThought(CitationRetrieval)
    
    def forward(
        self,
        citation_context: str,
        candidates: List[Dict[str, str]]
    ) -> dspy.Prediction:
        """
        Retrieve the cited paper from candidates.
        
        Args:
            citation_context: Text with [CITATION] marker
            candidates: List of {title, abstract} dicts
            
        Returns:
            Prediction with selected_title and reasoning
        """
        # Format candidates as string for LLM
        candidate_str = self._format_candidates(candidates)
        
        result = self.retrieve(
            citation_context=citation_context,
            candidate_papers=candidate_str
        )
        
        return result
    
    def _format_candidates(self, candidates: List[Dict[str, str]]) -> str:
        """Format candidates as numbered list."""
        lines = []
        for i, c in enumerate(candidates, 1):
            title = c.get("title", "Unknown")
            abstract = c.get("abstract", "")[:300]  # Truncate long abstracts
            lines.append(f"{i}. Title: {title}")
            if abstract:
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
        return "\n".join(lines)


class QueryThenRetrieve(dspy.Module):
    """
    Two-stage retrieval: generate query, then select from candidates.
    
    Stage 1: Generate a search query from the citation context
    Stage 2: Use the query understanding to select the best candidate
    """
    
    def __init__(self):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(QueryGeneration)
        self.retrieve = dspy.ChainOfThought(CitationRetrieval)
    
    def forward(
        self,
        citation_context: str,
        candidates: List[Dict[str, str]]
    ) -> dspy.Prediction:
        """
        Generate query then retrieve.
        
        Args:
            citation_context: Text with [CITATION] marker
            candidates: List of {title, abstract} dicts
            
        Returns:
            Prediction with selected_title, reasoning, and search_query
        """
        # Stage 1: Generate query to understand what we're looking for
        query_result = self.generate_query(citation_context=citation_context)
        
        # Stage 2: Select best candidate
        candidate_str = self._format_candidates(candidates)
        
        # Enhance context with query understanding
        enhanced_context = f"""Citation Context: {citation_context}

Search Understanding: {query_result.reasoning}
Likely looking for: {query_result.search_query}"""
        
        result = self.retrieve(
            citation_context=enhanced_context,
            candidate_papers=candidate_str
        )
        
        # Combine results
        result.search_query = query_result.search_query
        
        return result
    
    def _format_candidates(self, candidates: List[Dict[str, str]]) -> str:
        """Format candidates as numbered list."""
        lines = []
        for i, c in enumerate(candidates, 1):
            title = c.get("title", "Unknown")
            abstract = c.get("abstract", "")[:300]
            lines.append(f"{i}. Title: {title}")
            if abstract:
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
        return "\n".join(lines)


class RerankAndSelect(dspy.Module):
    """
    Reranking-based retrieval module.
    
    First reranks all candidates, then selects the top-ranked one.
    Better for handling many candidates.
    """
    
    def __init__(self):
        super().__init__()
        self.rerank = dspy.ChainOfThought(CitationReranking)
    
    def forward(
        self,
        citation_context: str,
        candidates: List[Dict[str, str]]
    ) -> dspy.Prediction:
        """
        Rerank candidates and select best.
        
        Args:
            citation_context: Text with [CITATION] marker
            candidates: List of {title, abstract} dicts
            
        Returns:
            Prediction with selected_title, ranked_titles, and reasoning
        """
        candidate_str = self._format_candidates(candidates)
        
        result = self.rerank(
            citation_context=citation_context,
            candidate_papers=candidate_str
        )
        
        # Extract top-ranked title
        ranked = result.ranked_titles.split(",")
        if ranked:
            result.selected_title = ranked[0].strip()
        else:
            result.selected_title = ""
        
        return result
    
    def _format_candidates(self, candidates: List[Dict[str, str]]) -> str:
        """Format candidates as numbered list."""
        lines = []
        for i, c in enumerate(candidates, 1):
            title = c.get("title", "Unknown")
            abstract = c.get("abstract", "")[:300]
            lines.append(f"{i}. Title: {title}")
            if abstract:
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
        return "\n".join(lines)


class VerifyAndSelect(dspy.Module):
    """
    Verification-based retrieval module.
    
    Verifies each candidate individually and selects the best match.
    More thorough but slower (makes N calls for N candidates).
    """
    
    def __init__(self, max_candidates: int = 5):
        super().__init__()
        self.verify = dspy.ChainOfThought(CitationVerification)
        self.max_candidates = max_candidates
    
    def forward(
        self,
        citation_context: str,
        candidates: List[Dict[str, str]]
    ) -> dspy.Prediction:
        """
        Verify each candidate and select best match.
        
        Args:
            citation_context: Text with [CITATION] marker
            candidates: List of {title, abstract} dicts
            
        Returns:
            Prediction with selected_title and verification details
        """
        # Limit candidates for efficiency
        candidates = candidates[:self.max_candidates]
        
        best_match = None
        best_confidence = None
        verifications = []
        
        for candidate in candidates:
            result = self.verify(
                citation_context=citation_context,
                candidate_title=candidate.get("title", ""),
                candidate_abstract=candidate.get("abstract", "")
            )
            
            verifications.append({
                "title": candidate.get("title", ""),
                "is_match": result.is_match,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            })
            
            # Track best match
            if result.is_match:
                conf_rank = {"high": 3, "medium": 2, "low": 1}.get(
                    result.confidence.lower(), 0
                )
                if best_confidence is None or conf_rank > best_confidence:
                    best_match = candidate.get("title", "")
                    best_confidence = conf_rank
        
        return dspy.Prediction(
            selected_title=best_match or candidates[0].get("title", ""),
            verifications=verifications,
            reasoning=f"Verified {len(candidates)} candidates, best match: {best_match}"
        )


class EnsembleRetriever(dspy.Module):
    """
    Ensemble module combining multiple retrieval strategies.
    
    Runs multiple retrievers and combines their outputs via voting.
    """
    
    def __init__(self):
        super().__init__()
        self.simple = SimpleCitationRetriever()
        self.query_then_retrieve = QueryThenRetrieve()
        self.rerank = RerankAndSelect()
    
    def forward(
        self,
        citation_context: str,
        candidates: List[Dict[str, str]]
    ) -> dspy.Prediction:
        """
        Run ensemble and combine results.
        
        Args:
            citation_context: Text with [CITATION] marker
            candidates: List of {title, abstract} dicts
            
        Returns:
            Prediction with selected_title and votes from each method
        """
        # Get predictions from each method
        simple_result = self.simple(citation_context, candidates)
        query_result = self.query_then_retrieve(citation_context, candidates)
        rerank_result = self.rerank(citation_context, candidates)
        
        # Voting
        votes = {}
        for result in [simple_result, query_result, rerank_result]:
            title = result.selected_title
            votes[title] = votes.get(title, 0) + 1
        
        # Select by majority vote
        selected = max(votes.keys(), key=lambda k: votes[k])
        
        return dspy.Prediction(
            selected_title=selected,
            votes=votes,
            simple_choice=simple_result.selected_title,
            query_choice=query_result.selected_title,
            rerank_choice=rerank_result.selected_title,
            reasoning=f"Ensemble vote: {votes}"
        )


# Convenience function to get a module by name
def get_module(name: str) -> dspy.Module:
    """
    Get a citation retrieval module by name.
    
    Args:
        name: One of 'simple', 'query', 'rerank', 'verify', 'ensemble'
        
    Returns:
        DSPy Module instance
    """
    modules = {
        "simple": SimpleCitationRetriever,
        "query": QueryThenRetrieve,
        "rerank": RerankAndSelect,
        "verify": VerifyAndSelect,
        "ensemble": EnsembleRetriever,
    }
    
    if name not in modules:
        raise ValueError(f"Unknown module: {name}. Choose from {list(modules.keys())}")
    
    return modules[name]()


