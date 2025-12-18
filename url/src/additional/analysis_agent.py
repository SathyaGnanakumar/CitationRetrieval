from __future__ import annotations

import ast
from typing import Any, Dict, List

from langchain_core.messages import AIMessage


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def analysis_agent(state: Dict[str, Any]):
    """
    Merge retriever outputs into a single, deduplicated candidate list.

    Preferred inputs:
    - state["bm25_results"], state["e5_results"], state["specter_results"]

    Backwards-compat:
    - If structured keys are missing, fall back to parsing retriever messages.
    """

    retrieval_results: List[Dict[str, Any]] = []

    # Preferred: structured retriever outputs
    for key in ("bm25_results", "e5_results", "specter_results", "citeagent_results"):
        for item in _safe_list(state.get(key)):
            if isinstance(item, dict):
                retrieval_results.append(item)

    # Fallback: parse messages (older flow)
    if not retrieval_results:
        for m in state.get("messages", []):
            if getattr(m, "type", None) == "ai" and getattr(m, "name", None) in [
                "bm25",
                "e5",
                "specter",
                "citeagent",
            ]:
                try:
                    payload = ast.literal_eval(m.content)
                except Exception:
                    continue

                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            retrieval_results.append(item)
                        elif isinstance(item, str):
                            retrieval_results.append({"title": item, "score": None})
                elif isinstance(payload, dict):
                    retrieval_results.append(payload)

    # Deduplicate by title (keep the best scored item)
    by_title: Dict[str, Dict[str, Any]] = {}

    for item in retrieval_results:
        title = (item.get("title") or "").strip()
        if not title:
            continue

        if title not in by_title:
            by_title[title] = item
            continue

        old = by_title[title]
        old_score = old.get("score")
        new_score = item.get("score")

        if old_score is None and isinstance(new_score, (int, float)):
            by_title[title] = item
        elif (
            isinstance(new_score, (int, float))
            and isinstance(old_score, (int, float))
            and new_score > old_score
        ):
            by_title[title] = item

    deduped = list(by_title.values())

    return {
        "candidate_papers": deduped,
        "messages": [AIMessage(name="analysis", content=str(deduped))],
    }
