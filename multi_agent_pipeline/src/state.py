from typing import List, Dict, Any
from langgraph.graph import MessagesState

class PipelineState(MessagesState):
    query: str
    expanded_queries: List[str] = []

    # retrieval cluster candidates stored here
    bm25_results: List[Dict[str, Any]] = []
    e5_results: List[Dict[str, Any]] = []
    specter_results: List[Dict[str, Any]] = []
    citeagent_results: List[Dict[str, Any]] = []

    # combined candidates
    candidate_papers: List[Dict[str, Any]] = []

    # downstream processing
    ranked_papers: List[Dict[str, Any]] = []
    verified_paper: Dict[str, Any] = {}
