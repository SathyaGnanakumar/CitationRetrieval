"""
DSPy Signatures for Citation Retrieval.

Defines the input/output specifications for citation retrieval tasks.
"""

import dspy
from typing import List


class CitationRetrieval(dspy.Signature):
    """
    You are an expert citation retrieval system as described in the paper "Multi-Agent System for Reliable Citation Retrieval".
    Your goal is to autonomously retrieve, verify, and recommend academic references given a query or document excerpt.
    
    Task:
    Given a citation context from a scientific paper (where a citation is missing), identify the correct paper from a list of candidates.
    Analyze the context to understand the specific claim, method, or result being cited.
    Then, evaluate each candidate paper to see if it matches the context.
    Finally, select the best matching paper.
    """
    
    citation_context: str = dspy.InputField(
        desc="Text excerpt from a paper containing [CITATION] marker where a reference was removed"
    )
    candidate_papers: str = dspy.InputField(
        desc="List of candidate papers with titles and abstracts to choose from"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Step-by-step analysis of why a particular paper is the cited reference"
    )
    selected_title: str = dspy.OutputField(
        desc="The exact title of the paper that is being cited"
    )


class QueryGeneration(dspy.Signature):
    """Generate an effective search query from a citation context."""
    
    citation_context: str = dspy.InputField(
        desc="Text excerpt containing [CITATION] marker"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Analysis of what the citation is likely referring to"
    )
    search_query: str = dspy.OutputField(
        desc="Effective search query to find the cited paper"
    )


class CitationReranking(dspy.Signature):
    """Rerank candidate papers based on relevance to citation context."""
    
    citation_context: str = dspy.InputField(
        desc="Text excerpt containing [CITATION] marker"
    )
    candidate_papers: str = dspy.InputField(
        desc="List of candidate papers with titles and abstracts"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Analysis of how each candidate relates to the citation context"
    )
    ranked_titles: str = dspy.OutputField(
        desc="Comma-separated list of paper titles ranked by relevance (most relevant first)"
    )


class CitationVerification(dspy.Signature):
    """Verify if a candidate paper matches the citation context."""
    
    citation_context: str = dspy.InputField(
        desc="Text excerpt containing [CITATION] marker"
    )
    candidate_title: str = dspy.InputField(
        desc="Title of the candidate paper"
    )
    candidate_abstract: str = dspy.InputField(
        desc="Abstract of the candidate paper"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Analysis of whether this paper matches the citation"
    )
    is_match: bool = dspy.OutputField(
        desc="True if this paper is likely the cited reference, False otherwise"
    )
    confidence: str = dspy.OutputField(
        desc="Confidence level: high, medium, or low"
    )


