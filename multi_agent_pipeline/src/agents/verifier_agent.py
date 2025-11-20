# src/agents/verifier_agent.py

import ast
import json
from langchain_core.messages import AIMessage
from src.services.semantic_scholar_verifier import SemanticScholarCitationVerifier

verifier = SemanticScholarCitationVerifier()


def _parse_candidates(raw_msg):
    """
    Normalize analysis output to a list of {title, score}.
    Accepts str(repr(list)), JSON string, or list objects.
    """
    payload = raw_msg
    if isinstance(raw_msg, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(raw_msg)
                break
            except Exception:
                continue

    if not isinstance(payload, list):
        return []

    candidates = []
    for item in payload:
        title = None
        score = None
        if isinstance(item, dict):
            title = item.get("title")
            score = item.get("score")
        elif isinstance(item, (list, tuple)) and item:
            title = item[0]
            score = item[1] if len(item) > 1 else None
        elif isinstance(item, str):
            title = item

        if title:
            candidates.append({"title": str(title), "score": score})

    return candidates


def verifier_agent(state):
    """
    Takes analysis agent's ranked papers, verifies each via Semantic Scholar.
    """
    msg = state["messages"][-1].content
    candidates = _parse_candidates(msg)

    if not candidates:
        return {"messages": [AIMessage(content="[]", name="verifier")]}

    verified_results = []

    for item in candidates:
        title = item["title"]
        score = item.get("score", 0.0)

        try:
            outcome = verifier.verify(title)
            verified_results.append({
                "title": title,
                "score": score,
                "matched": outcome.matched,
                "best_title": outcome.best_paper.title if outcome.best_paper else None,
                "confidence": outcome.metrics.get("score", 0.0),
                "reason": outcome.reason,
                "candidates_considered": outcome.candidates_considered,
                "comparisons": outcome.comparisons or [],
            })
        except Exception as e:
            verified_results.append({
                "title": title,
                "score": score,
                "matched": False,
                "best_title": None,
                "confidence": 0.0,
                "reason": f"verification_error: {e}",
                "candidates_considered": 0,
                "comparisons": [],
            })

    return {
        "messages": [
            AIMessage(content=json.dumps(verified_results), name="verifier")
        ]
    }
