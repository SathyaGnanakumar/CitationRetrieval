from typing import Any, Dict, List, Annotated
from langgraph.graph import MessagesState
from operator import add


def merge_dicts(left: Dict | None, right: Dict | None) -> Dict:
    """Merge two dicts, with right taking precedence."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


class RetrievalState(MessagesState):
    """
    Dataset-agnostic state schema for the retrieval workflow.

    Notes:
    - LangGraph state is still a dict at runtime; this is a schema/type contract.
    - Keep `messages` from MessagesState for LangGraph Studio visibility, but prefer
      structured keys below for logic.
    - All fields use reducers to properly merge state across nodes
    """

    # Query inputs (replace strategy - last writer wins)
    query: str
    queries: Annotated[List[str], add]  # Append queries from reformulator

    # Dependency injection / runtime knobs (merge strategy)
    resources: Annotated[Dict[str, Any], merge_dicts]
    config: Annotated[Dict[str, Any], merge_dicts]

    # Individual retriever outputs (replace strategy - each retriever writes once)
    bm25_results: List[Dict[str, Any]]
    e5_results: List[Dict[str, Any]]
    specter_results: List[Dict[str, Any]]

    # Retriever outputs (per retriever) - aggregated dict (merge strategy)
    retriever_results: Annotated[Dict[str, List[Dict[str, Any]]], merge_dicts]

    # Merged candidates + downstream outputs (replace strategy - aggregator/reranker write once)
    candidate_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]

    # Self-evolution fields (replace/merge as needed)
    module_version: Annotated[Dict[str, int], merge_dicts]
    evaluation_metadata: Annotated[Dict[str, Any], merge_dicts]
    optimization_enabled: bool
    dspy_prediction: Any
    dspy_candidates: List[Dict[str, Any]]
    selected_paper: Dict[str, Any]


# Backwards-compat alias (older code referenced PipelineState)
PipelineState = RetrievalState
