from typing import Any, Dict, List
from langgraph.graph import MessagesState


class RetrievalState(MessagesState):
    """
    Dataset-agnostic state schema for the retrieval workflow.

    Notes:
    - LangGraph state is still a dict at runtime; this is a schema/type contract.
    - Keep `messages` from MessagesState for LangGraph Studio visibility, but prefer
      structured keys below for logic.
    """

    # Query inputs
    query: str
    queries: List[str]

    # Dependency injection / runtime knobs
    resources: Dict[str, Any]  # indexes, embedding matrices, external clients, etc.
    config: Dict[str, Any]  # k, model names, thresholds, etc.

    # Individual retriever outputs
    bm25_results: List[Dict[str, Any]]
    e5_results: List[Dict[str, Any]]
    specter_results: List[Dict[str, Any]]

    # Retriever outputs (per retriever) - aggregated dict
    retriever_results: Dict[str, List[Dict[str, Any]]]

    # Merged candidates + downstream outputs
    candidate_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]


# Backwards-compat alias (older code referenced PipelineState)
PipelineState = RetrievalState
