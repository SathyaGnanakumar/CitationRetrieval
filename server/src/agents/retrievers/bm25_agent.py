import ast
from typing import Any, Dict, List

from langchain_core.messages import AIMessage


def _get_queries(state: Dict[str, Any]) -> List[str]:
    queries = state.get("queries")
    if isinstance(queries, list) and queries:
        return [str(q) for q in queries if str(q).strip()]

    # Backwards-compat: reformulator used to stuff a stringified list into messages
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


def bm25_agent(state: Dict[str, Any]):
    """
    BM25 retrieval agent (dataset-agnostic).

    Expects:
    - `state["queries"]`: list[str]
    - `state["resources"]["bm25"]`: built via src.resources.builders.build_bm25_resources()
    """

    queries = _get_queries(state)
    if not queries:
        return {"messages": [AIMessage(content="BM25 received no queries", name="bm25")]}

    resources = state.get("resources", {}) or {}
    bm25_res = resources.get("bm25")
    if not bm25_res:
        return {"messages": [AIMessage(content="BM25_ERROR: missing bm25 resources", name="bm25")]}

    import bm25s

    k = int((state.get("config", {}) or {}).get("k", 5))
    q0 = queries[0]

    stemmer = bm25_res["stemmer"]
    query_tokens = bm25s.tokenize(q0, stopwords="en", stemmer=stemmer)

    doc_ids, scores = bm25_res["index"].retrieve(query_tokens, k=k)
    doc_ids = doc_ids[0]
    scores = scores[0]

    titles = bm25_res["titles"]
    ids = bm25_res["ids"]

    results = []
    for rank_idx, corpus_idx in enumerate(doc_ids):
        results.append(
            {
                "id": ids[corpus_idx],
                "title": titles[corpus_idx],
                "score": float(scores[rank_idx]),
                "source": "bm25",
            }
        )

    return {
        "bm25_results": results,
        "messages": [AIMessage(content=str(results), name="bm25")],
    }
