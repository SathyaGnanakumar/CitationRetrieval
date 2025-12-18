from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from FlagEmbedding import FlagReranker
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


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
    logger.info("🎯 Reranker starting...")

    query = _get_query(state)
    candidates = state.get("candidate_papers") or []

    if not query:
        logger.error("❌ Reranker: missing query")
        return {"messages": [AIMessage(name="reranking", content="RERANK_ERROR: missing query")]}

    if not isinstance(candidates, list) or not candidates:
        logger.warning("⚠️  No candidate papers to rerank")
        return {"ranked_papers": [], "messages": [AIMessage(name="reranking", content="[]")]}

    logger.info(f"Reranking {len(candidates)} candidate papers...")
    logger.debug(f"Query: {query[:100]}...")

    resources = state.get("resources", {}) or {}
    if "reranker_model" not in resources:
        logger.info(
            f"📥 Loading reranker model: {model_name} (this may take a few minutes on first run)"
        )

    reranker_model = resources.get("reranker_model") or FlagReranker(model_name, use_fp16=True)
    # Cache the model back into resources so subsequent queries reuse it
    resources["reranker_model"] = reranker_model

    logger.debug("Computing reranking scores...")
    pairs = _pairs(query, candidates)
    scores = reranker_model.compute_score(pairs, normalize=True)

    ranked_items = []
    for paper, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
        item = dict(paper)
        item["rerank_score"] = float(score)
        ranked_items.append(item)

    if ranked_items:
        top_score = ranked_items[0]["rerank_score"]
        bottom_score = ranked_items[-1]["rerank_score"]
        logger.info(
            f"✅ Reranking complete: {len(ranked_items)} papers (scores: {top_score:.3f} to {bottom_score:.3f})"
        )

    return {
        "ranked_papers": ranked_items,
        "messages": [AIMessage(name="reranking", content=str(ranked_items))],
    }
