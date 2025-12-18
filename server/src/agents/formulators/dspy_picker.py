from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage


def _get_query(state: Dict[str, Any]) -> Optional[str]:
    q = state.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()

    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def dspy_picker(state: Dict[str, Any]):
    cfg = state.get("config", {}) or {}
    enabled = _truthy(cfg.get("enable_dspy_picker")) or _truthy(os.getenv("ENABLE_DSPY_PICKER"))
    if not enabled:
        return {}

    ranked = state.get("ranked_papers") or []
    if not isinstance(ranked, list) or not ranked:
        return {
            "messages": [AIMessage(name="dspy_picker", content="DSPY_PICKER: no ranked_papers")]
        }

    query = _get_query(state)
    if not query:
        return {
            "messages": [AIMessage(name="dspy_picker", content="DSPY_PICKER_ERROR: missing query")]
        }

    top_n = int(cfg.get("dspy_top_n") or os.getenv("DSPY_TOP_N") or 10)
    candidates: List[Dict[str, Any]] = []
    for item in ranked[: max(1, top_n)]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        abstract = item.get("abstract")
        if not isinstance(abstract, str):
            abstract = ""
        c = dict(item)
        c["title"] = title
        c["abstract"] = abstract
        candidates.append(c)

    if not candidates:
        return {
            "messages": [AIMessage(name="dspy_picker", content="DSPY_PICKER: no usable candidates")]
        }

    try:
        import dspy
        from src.agents.formulators.dspy_prompt_generator.modules import get_module

        model = str(cfg.get("dspy_model") or os.getenv("DSPY_MODEL") or "gpt-4o-mini")
        temperature = float(cfg.get("dspy_temperature") or os.getenv("DSPY_TEMPERATURE") or 0.0)
        max_tokens = int(cfg.get("dspy_max_tokens") or os.getenv("DSPY_MAX_TOKENS") or 800)

        resources = state.get("resources", {}) or {}
        lm = resources.get("dspy_lm")
        if lm is None:
            lm = dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens)

        dspy.configure(lm=lm)

        module_name = str(cfg.get("dspy_module") or os.getenv("DSPY_MODULE") or "simple")
        module = get_module(module_name)

        pred = module(citation_context=query, candidates=candidates)

        selected_title = getattr(pred, "selected_title", None)
        reasoning = getattr(pred, "reasoning", None)

    except Exception as e:
        return {"messages": [AIMessage(name="dspy_picker", content=f"DSPY_PICKER_ERROR: {e}")]}

    selected_paper = candidates[0]
    if isinstance(selected_title, str) and selected_title.strip():
        target = selected_title.strip().lower()
        for c in candidates:
            t = c.get("title")
            if isinstance(t, str) and t.strip().lower() == target:
                selected_paper = c
                break

    return {
        "selected_paper": selected_paper,
        "dspy_selected_title": selected_title or "",
        "dspy_reasoning": reasoning or "",
        "messages": [
            AIMessage(
                name="dspy_picker",
                content=str(
                    {"selected_title": selected_title, "selected_id": selected_paper.get("id")}
                ),
            )
        ],
    }
