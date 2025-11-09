# """CiteAgent model wrapper for evaluation framework"""

# import sys
# import os
# from pathlib import Path
# from typing import List, Dict, Any
# import pandas as pd

# # Add CiteAgent source to path - make sure this path is correct relative to where this file is
# citeagent_path = Path(__file__).parent.parent.parent / "citeagent_src"
# sys.path.insert(0, str(citeagent_path))

# # Now import from the citeagent_src directory
# try:
#     from retriever.agent import LLMSelfAskAgentPydantic, Output
#     from retriever.search_provider import SemanticScholarSearchProvider
#     from utils.str_matcher import find_match_psr, SM
#     from utils.data_model import PaperSearchResult
# except ImportError as e:
#     print(f"Error importing CiteAgent modules: {e}")
#     print(f"Looking for modules in: {citeagent_path}")
#     print(f"Directory contents: {os.listdir(citeagent_path) if citeagent_path.exists() else 'Directory not found'}")
#     raise

# # Import the base model from the current package
# from .base_model import BaseRetrievalModel


# class CiteAgentModel(BaseRetrievalModel):
#     """
#     CiteAgent retrieval model wrapper.
    
#     Adapts CiteAgent to work with the evaluation framework.
#     """
    
#     def __init__(
#         self,
#         model_name: str = "gpt-4o",
#         temperature: float = 0.95,
#         search_limit: int = 10,
#         max_actions: int = 15,
#         prompt_name: str = "few_shot_search",
#         only_open_access: bool = False
#     ):
#         """
#         Initialize CiteAgent.
        
#         Args:
#             model_name: LLM model name (gpt-4o, claude-3-opus, etc.)
#             temperature: LLM temperature
#             search_limit: Number of papers to retrieve per search
#             max_actions: Maximum agent actions
#             prompt_name: Prompt template to use
#             only_open_access: Whether to filter for open access papers
#         """
#         super().__init__(f"CiteAgent-{model_name}")
        
#         try:
#             # Initialize the agent
#             self.agent = LLMSelfAskAgentPydantic(
#                 model_name=model_name,
#                 temperature=temperature,
#                 search_limit=search_limit,
#                 only_open_access=only_open_access,
#                 prompt_name=prompt_name,
#                 pydantic_object=Output,
#                 use_web_search=False  # Add this if needed
#             )
            
#             self.max_actions = max_actions
#             self.threshold = 0.8
#             self.model_config = {
#                 'model_name': model_name,
#                 'temperature': temperature,
#                 'search_limit': search_limit,
#                 'max_actions': max_actions,
#                 'prompt_name': prompt_name,
#             }
#         except Exception as e:
#             print(f"Error initializing CiteAgent: {e}")
#             raise
        
#     def retrieve(
#         self,
#         query: str,
#         corpus: List[Dict],
#         k: int = 10
#     ) -> List[Dict]:
#         """
#         Retrieve papers using CiteAgent.
        
#         Args:
#             query: Citation context
#             corpus: List of candidate papers (with title, abstract, etc.)
#             k: Number of papers to retrieve
            
#         Returns:
#             Top-k ranked papers from corpus
#         """
#         # Reset agent
#         self.agent.reset([])
        
#         # Run CiteAgent on the query
#         # Note: CiteAgent expects year, we'll use 2024 as default
#         try:
#             # CiteAgent returns a single selected paper
#             selected_paper = self.agent(
#                 excerpt=query,
#                 year="2024",
#                 max_actions=self.max_actions
#             )
            
#             # Get all papers from agent's search buffer
#             all_searched = []
#             for buffer in self.agent.paper_buffer:
#                 all_searched.extend(buffer)
            
#             # Match against corpus
#             results = []
#             seen_titles = set()
            
#             # First add the selected paper if it matches something in corpus
#             if selected_paper and hasattr(selected_paper, 'title') and selected_paper.title:
#                 for item in corpus:
#                     if item.get('title') and item['title'] not in seen_titles:
#                         similarity = self._compute_similarity(
#                             selected_paper.title, 
#                             item['title']
#                         )
#                         if similarity > self.threshold:
#                             result = item.copy()
#                             result['score'] = 1.0  # Highest score for selected
#                             result['rank'] = 1
#                             results.append(result)
#                             seen_titles.add(item['title'])
#                             break
            
#             # Then add other papers from search buffer
#             for paper in all_searched:
#                 if len(results) >= k:
#                     break
                    
#                 if hasattr(paper, 'title') and paper.title:
#                     for item in corpus:
#                         if item.get('title') and item['title'] not in seen_titles:
#                             similarity = self._compute_similarity(
#                                 paper.title,
#                                 item['title']
#                             )
#                             if similarity > 0.7:  # Lower threshold for buffer
#                                 result = item.copy()
#                                 result['score'] = similarity
#                                 result['rank'] = len(results) + 1
#                                 results.append(result)
#                                 seen_titles.add(item['title'])
#                                 break
            
#             # If we don't have enough results, add random papers from corpus
#             if len(results) < k:
#                 for item in corpus:
#                     if item.get('title') and item['title'] not in seen_titles:
#                         result = item.copy()
#                         result['score'] = 0.0
#                         result['rank'] = len(results) + 1
#                         results.append(result)
#                         seen_titles.add(item['title'])
#                         if len(results) >= k:
#                             break
                            
#         except Exception as e:
#             print(f"CiteAgent error: {e}")
#             import traceback
#             traceback.print_exc()
#             # Fallback: return first k items from corpus
#             results = []
#             for i, item in enumerate(corpus[:k]):
#                 result = item.copy()
#                 result['score'] = 0.0
#                 result['rank'] = i + 1
#                 results.append(result)
                
#         return results[:k]
    
#     def _compute_similarity(self, title1: str, title2: str) -> float:
#         """Compute title similarity using CiteAgent's string matcher"""
#         try:
#             # Use the SM from str_matcher that we imported
#             return SM(a=title1.lower(), b=title2.lower()).ratio()
#         except:
#             # Fallback to simple comparison if SM not available
#             return 1.0 if title1.lower() == title2.lower() else 0.0
    
#     def get_config(self) -> Dict[str, Any]:
#         """Return model configuration"""
#         return self.model_config
"""
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