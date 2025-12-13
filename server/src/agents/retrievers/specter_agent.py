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


def specter_agent(state: Dict[str, Any]):
    """
    SPECTER2 dense retrieval agent (dataset-agnostic).

    Expects:
    - `state["queries"]`: list[str]
    - `state["resources"]["specter"]`: built via src.resources.builders.build_specter_resources()
    """

    queries = _get_queries(state)
    if not queries:
        return {"messages": [AIMessage(content="SPECTER received no queries", name="specter")]}

    resources = state.get("resources", {}) or {}
    sp_res = resources.get("specter")
    if not sp_res:
        return {
            "messages": [
                AIMessage(content="SPECTER_ERROR: missing specter resources", name="specter")
            ]
        }

    import torch

    k = int((state.get("config", {}) or {}).get("k", 5))

    tokenizer = sp_res["tokenizer"]
    model = sp_res["model"]
    device = sp_res.get("device", "cpu")

    # Encode query on GPU if available
    if device == "cuda":
        model.to("cuda")

    inputs = tokenizer(
        [queries[0]], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    if device == "cuda":
        inputs = inputs.to("cuda")

    with torch.no_grad():
        q_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu()

    model.to("cpu")

    corpus_embeddings = sp_res["corpus_embeddings"]
    scores = torch.cosine_similarity(q_emb.expand_as(corpus_embeddings), corpus_embeddings, dim=1)
    top_indices = torch.topk(scores, k=min(k, scores.shape[0])).indices.tolist()

    titles = sp_res["titles"]
    ids = sp_res["ids"]

    results = []
    for i in top_indices:
        results.append(
            {
                "id": ids[i],
                "title": titles[i],
                "score": float(scores[i]),
                "source": "specter",
            }
        )

    return {
        "specter_results": results,
        "messages": [AIMessage(content=str(results), name="specter")],
    }
