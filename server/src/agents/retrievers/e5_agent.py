import ast
from typing import Any, Dict, List

from langchain_core.messages import AIMessage


def _get_queries(state: Dict[str, Any]) -> List[str]:
    queries = state.get("queries")
    if isinstance(queries, list) and queries:
        return [str(q) for q in queries if str(q).strip()]

    try:
        msg = state["messages"][-1].content
        parsed = ast.literal_eval(msg) if isinstance(msg, str) else msg
    except Exception:
        return []

    if isinstance(parsed, dict):
        qs = parsed.get("queries", [])
        return [str(q) for q in qs if str(q).strip()] if isinstance(qs, list) else []
    if isinstance(parsed, list):
        return [str(q) for q in parsed if str(q).strip()]
    return []


def e5_agent(state: Dict[str, Any]):
    """
    E5 dense retrieval agent (dataset-agnostic).

    Expects:
    - `state["queries"]`: list[str]
    - `state["resources"]["e5"]`: built via src.resources.builders.build_e5_resources()
    """

    queries = _get_queries(state)
    if not queries:
        return {"messages": [AIMessage(content="E5 received no queries", name="e5")]}

    resources = state.get("resources", {}) or {}
    e5_res = resources.get("e5")
    if not e5_res:
        return {"messages": [AIMessage(content="E5_ERROR: missing e5 resources", name="e5")]}

    import torch
    from sentence_transformers import util

    k = int((state.get("config", {}) or {}).get("k", 5))

    model = e5_res["model"]
    corpus_embeddings = e5_res["corpus_embeddings"]

    query_embedding = model.encode(
        queries[0],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = torch.topk(scores, k=min(k, scores.shape[0]))

    titles = e5_res["titles"]
    ids = e5_res["ids"]

    results = []
    for score, idx in zip(top_k.values, top_k.indices):
        i = int(idx)
        results.append(
            {
                "id": ids[i],
                "title": titles[i],
                "score": float(score),
                "source": "e5",
            }
        )

    return {
        "e5_results": results,
        "messages": [AIMessage(content=str(results), name="e5")],
    }
