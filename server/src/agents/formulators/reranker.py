from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None
from langchain_core.messages import AIMessage, HumanMessage


def _get_query(state: Dict[str, Any]) -> Optional[str]:
    q = state.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()

    # Fallback: last human message
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return None


def _pairs(query: str, candidate_papers: List[Dict[str, Any]]) -> List[List[str]]:
    return [[query, (paper.get("title") or "")] for paper in candidate_papers]


def reranker(state: Dict[str, Any], model_name: str = "BAAI/bge-reranker-v2-m3"):
    """
    Cross-encoder reranker.

    Expects:
    - state["candidate_papers"]: list[dict] with at least `title`
    - state["query"] (optional; falls back to last HumanMessage)
    Optionally uses:
    - state["resources"]["reranker_model"] to reuse a loaded model
    """

    query = _get_query(state)
    candidates = state.get("candidate_papers") or []
    if not query:
        return {"messages": [AIMessage(name="reranking", content="RERANK_ERROR: missing query")]}
    if not isinstance(candidates, list) or not candidates:
        return {"ranked_papers": [], "messages": [AIMessage(name="reranking", content="[]")]}

    resources = state.get("resources", {}) or {}
    reranker_model = resources.get("reranker_model")
    if reranker_model is None:
        if FlagReranker is None:
            ranked_items = []
            for paper in sorted(candidates, key=lambda p: float(p.get("score") or 0.0), reverse=True):
                item = dict(paper)
                item["rerank_score"] = float(item.get("score") or 0.0)
                ranked_items.append(item)
            return {
                "ranked_papers": ranked_items,
                "messages": [AIMessage(name="reranking", content=str(ranked_items))],
            }
        reranker_model = FlagReranker(model_name, use_fp16=True)

    pairs = _pairs(query, candidates)
    scores = reranker_model.compute_score(pairs, normalize=True)

    ranked_items = []
    for paper, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
        item = dict(paper)
        item["rerank_score"] = float(score)
        ranked_items.append(item)

    return {
        "ranked_papers": ranked_items,
        "messages": [AIMessage(name="reranking", content=str(ranked_items))],
    }
