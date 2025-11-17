# src/agents/analysis_agent.py

from langchain_core.messages import AIMessage
import ast

def analysis_agent(state):
    messages = state["messages"]

    retrieval_results = []

    # Extract all retrieval agent messages
    for m in messages:
        if m.type == "ai" and m.name in ["bm25", "e5", "specter", "citeagent"]:
            try:
                payload = ast.literal_eval(m.content)
            except:
                continue

            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        retrieval_results.append(item)
                    elif isinstance(item, str):
                        retrieval_results.append({"title": item, "score": None})
            elif isinstance(payload, dict):
                retrieval_results.append(payload)

    # -----------------------------------------
    #  Deduplicate by title (keep best version)
    # -----------------------------------------
    by_title = {}

    for item in retrieval_results:
        title = item.get("title", "").strip()
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
        elif isinstance(new_score, (int, float)) and old_score is not None and new_score > old_score:
            by_title[title] = item

    deduped = list(by_title.values())

    return {
        "messages": [
            AIMessage(
                name="analysis",
                content=str(deduped)
            )
        ]
    }
