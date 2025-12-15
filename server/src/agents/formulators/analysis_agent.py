from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _build_id_to_meta(resources: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    id_to_meta: Dict[str, Dict[str, Any]] = {}

    corpus = resources.get("corpus")
    if isinstance(corpus, list):
        for d in corpus:
            if not isinstance(d, dict):
                continue
            doc_id = _safe_str(d.get("id")).strip()
            if doc_id:
                id_to_meta[doc_id] = d

    for key in ("bm25", "e5", "specter"):
        res = resources.get(key)
        if not isinstance(res, dict):
            continue

        ids = res.get("ids")
        titles = res.get("titles")
        texts = res.get("texts")

        if not isinstance(ids, list) or not isinstance(titles, list):
            continue

        for i, doc_id in enumerate(ids):
            did = _safe_str(doc_id).strip()
            if not did:
                continue

            meta = id_to_meta.get(did)
            if meta is None:
                meta = {"id": did}
                id_to_meta[did] = meta

            if i < len(titles) and not meta.get("title"):
                meta["title"] = titles[i]

            if isinstance(texts, list) and i < len(texts) and not meta.get("text"):
                meta["text"] = texts[i]

    return id_to_meta


def analysis_agent(state: Dict[str, Any]):
    resources = state.get("resources", {}) or {}
    id_to_meta = _build_id_to_meta(resources)

    all_results: List[Dict[str, Any]] = []
    for key in ("bm25_results", "e5_results", "specter_results"):
        res = state.get(key)
        if isinstance(res, list):
            all_results.extend([r for r in res if isinstance(r, dict)])

    retriever_results = state.get("retriever_results")
    if isinstance(retriever_results, dict):
        for v in retriever_results.values():
            if isinstance(v, list):
                all_results.extend([r for r in v if isinstance(r, dict)])

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for r in all_results:
        paper_id = _safe_str(r.get("id")).strip()
        title = _safe_str(r.get("title")).strip()
        dedupe_key = paper_id or title.lower()
        if not dedupe_key:
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        item = dict(r)

        meta = id_to_meta.get(paper_id) if paper_id else None
        if isinstance(meta, dict):
            abstract = meta.get("abstract") or meta.get("text") or ""
            if abstract and not item.get("abstract"):
                item["abstract"] = abstract

        merged.append(item)

    return {
        "candidate_papers": merged,
        "messages": [AIMessage(name="analysis", content=str(merged))],
    }
