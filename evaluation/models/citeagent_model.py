"""
CiteAgent Model - Local corpus-based implementation

Simplified CiteAgent that works with local corpus only - NO API KEYS NEEDED
"""

from typing import List, Dict, Any
from difflib import SequenceMatcher
import random
from .base_model import BaseRetrievalModel


class CiteAgentLocal(BaseRetrievalModel):
    """
    Simplified CiteAgent that ranks papers from local corpus.
    No API keys or external searches needed.
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: If True, would use LLM for ranking (not implemented)
                    If False, uses similarity matching
        """
        super().__init__("CiteAgent-Local")
        self.use_llm = use_llm
        
    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve papers from local corpus based on query similarity.
        
        Args:
            query: Citation context
            corpus: List of candidate papers
            k: Number to retrieve
            
        Returns:
            Top-k ranked papers
        """
        if not corpus:
            return []
        
        # Score each paper in corpus based on similarity to query
        scored_papers = []
        
        for paper in corpus:
            # Combine title and abstract for matching
            paper_text = paper.get('title', '') + ' ' + paper.get('abstract', '')[:500]
            
            # Calculate similarity score
            score = self._calculate_relevance(query, paper_text)
            
            scored_paper = paper.copy()
            scored_paper['score'] = score
            scored_papers.append(scored_paper)
        
        # Sort by score
        scored_papers.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks and return top k
        results = []
        for i, paper in enumerate(scored_papers[:k]):
            paper['rank'] = i + 1
            results.append(paper)
            
        return results
    
    def _calculate_relevance(self, query: str, paper_text: str) -> float:
        """
        Calculate relevance score between query and paper.
        
        This is a simplified version. The real CiteAgent would use
        LLM + Semantic Scholar search.
        """
        # Convert to lowercase for matching
        query_lower = query.lower()
        paper_lower = paper_text.lower()
        
        # Method 1: Direct sequence matching
        sequence_score = SequenceMatcher(None, query_lower, paper_lower).ratio()
        
        # Method 2: Keyword overlap
        query_words = set(query_lower.split())
        paper_words = set(paper_lower.split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        query_words = query_words - stopwords
        paper_words = paper_words - stopwords
        
        if query_words:
            overlap_score = len(query_words & paper_words) / len(query_words)
        else:
            overlap_score = 0
        
        # Method 3: Check for important citation keywords
        citation_keywords = ['propose', 'method', 'approach', 'model', 'algorithm', 'technique', 'framework', 'system']
        keyword_score = sum(1 for kw in citation_keywords if kw in paper_lower) / len(citation_keywords)
        
        # Combine scores
        final_score = (sequence_score * 0.3 + overlap_score * 0.5 + keyword_score * 0.2)
        
        return final_score
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'model_name': 'CiteAgent-Local',
            'use_llm': self.use_llm,
            'method': 'local_corpus_ranking'
        }