from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def clean_text(text: str) -> str:
    """
    Light cleaning for LaTeX-ish artifacts and whitespace.

    Keep it conservative; retrieval models can handle some noise.
    """

    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(dataset_path: str | Path) -> List[Dict[str, Any]]:
    """Load the ScholarCopilot JSON dataset from disk."""

    p = Path(dataset_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_citations(dataset: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    Yield citation dicts from each paper's `bib_info`.

    ScholarCopilot format appears to be:
      paper["bib_info"] = { tag: [ {citation fields...}, ... ], ... }
    """

    for paper in dataset:
        bib_info = paper.get("bib_info", {}) or {}
        if not isinstance(bib_info, dict):
            continue
        for entries in bib_info.values():
            if not isinstance(entries, list):
                continue
            for citation in entries:
                if isinstance(citation, dict):
                    yield citation


def build_citation_corpus(
    dataset: List[Dict[str, Any]],
    *,
    clean: bool = True,
    require_title: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a citation corpus from `bib_info`, deduplicated by normalized title.

    Returns documents with a shared schema:
      { "id": str, "title": str, "abstract": str, "text": str }
    """

    unique_by_title: Dict[str, Dict[str, Any]] = {}

    for citation in iter_citations(dataset):
        raw_title = (citation.get("title") or "").strip()
        if require_title and not raw_title:
            continue

        title = clean_text(raw_title) if clean else raw_title
        abstract = citation.get("abstract") or ""
        abstract = clean_text(abstract) if (clean and abstract) else abstract

        normalized_title = _normalize_title(title)
        if not normalized_title:
            continue

        if normalized_title in unique_by_title:
            continue

        citation_key: Optional[str] = citation.get("citation_key") or citation.get("paper_id")
        # Fall back to normalized title if dataset doesn't provide an id
        doc_id = str(citation_key) if citation_key else normalized_title

        unique_by_title[normalized_title] = {
            "id": doc_id,
            "title": title,
            "abstract": abstract,
            "text": f"{title}. {abstract}".strip(),
        }

    return list(unique_by_title.values())
