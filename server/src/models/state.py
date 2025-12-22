from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RetrievalState(TypedDict):
    """
    Dataset-agnostic state schema for the retrieval workflow.

    Notes:
    - LangGraph state is still a dict at runtime; this is a schema/type contract.
    - messages uses add_messages reducer for LangGraph Studio visibility
    - All other fields use default TypedDict behavior (replace on update)
    """

    # LangGraph messages (for Studio visibility)
    messages: Annotated[list, add_messages]

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
