"""
Semantic Scholar Citation Verification Service
----------------------------------------------

This module provides:
  - SemanticScholarPaper dataclass
  - SemanticScholarLookupService for API calls
  - SemanticScholarCitationVerifier for matching citations to actual papers

It is a cleaned, self-contained version suitable for use inside your multi-agent pipeline.
"""

from __future__ import annotations

import re
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Sequence
from difflib import SequenceMatcher


# ================================================================
# 📌 1. DATA STRUCTURES
# ================================================================

@dataclass
class SemanticScholarPaper:
    """Structured paper returned from Semantic Scholar."""
    paper_id: str
    title: str
    abstract: str
    year: Optional[int]
    venue: Optional[str]
    url: Optional[str]
    authors: List[str]
    external_ids: Dict[str, str]
    publication_types: List[str]


# ================================================================
# 📌 2. LOOKUP SERVICE (API WRAPPER)
# ================================================================

class SemanticScholarLookupService:
    """
    Wrapper around Semantic Scholar API.

    Only uses the public `/paper/search` endpoint.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,venue,authors,url,externalIds,publicationTypes"

    def lookup(self, query: str, year: Optional[int] = None) -> List[SemanticScholarPaper]:
        params = {
            "query": query,
            "fields": self.FIELDS,
            "limit": 10,
        }
        if year:
            params["year"] = year

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[SemanticScholarLookupService] ERROR: {e}")
            return []

        data = response.json()
        papers = []

        for item in data.get("data", []):
            authors = [a.get("name", "") for a in item.get("authors", [])]
            papers.append(
                SemanticScholarPaper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    abstract=item.get("abstract", ""),
                    year=item.get("year"),
                    venue=item.get("venue"),
                    url=item.get("url"),
                    authors=authors,
                    external_ids=item.get("externalIds", {}),
                    publication_types=item.get("publicationTypes", []),
                )
            )

        return papers


# ================================================================
# 📌 3. VERIFICATION OUTCOME STRUCTURE
# ================================================================

@dataclass
class CitationVerificationOutcome:
    matched: bool
    citation: str
    year_filter: Optional[int]
    best_paper: Optional[SemanticScholarPaper]
    metrics: Dict[str, float]
    candidates_considered: int
    reason: Optional[str] = None
    comparisons: List[Tuple[str, Dict[str, float]]] = None

    def to_payload(self):
        """Return a serializable dictionary."""
        return {
            "matched": self.matched,
            "citation": self.citation,
            "year_filter": self.year_filter,
            "metrics": self.metrics,
            "candidates_considered": self.candidates_considered,
            "reason": self.reason,
            "comparisons": self.comparisons or [],
            "best_paper": self._paper_dict(self.best_paper),
        }

    @staticmethod
    def _paper_dict(paper):
        if not paper:
            return None
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "venue": paper.venue,
            "url": paper.url,
            "authors": paper.authors,
            "external_ids": paper.external_ids,
            "publication_types": paper.publication_types,
        }


# ================================================================
# 📌 4. VERIFIER CLASS
# ================================================================

class SemanticScholarCitationVerifier:
    """
    Validates whether a citation corresponds to a real paper.
    """

    def __init__(
        self,
        lookup_service: Optional[SemanticScholarLookupService] = None,
        title_threshold: float = 0.85,
        score_threshold: float = 0.72,
        author_overlap_threshold: float = 0.4,
    ):
        self.lookup_service = lookup_service or SemanticScholarLookupService()
        self.title_threshold = title_threshold
        self.score_threshold = score_threshold
        self.author_overlap_threshold = author_overlap_threshold

    # ------------------------------------------------------------
    # 🔹 Extract year from text
    # ------------------------------------------------------------

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        m = re.search(r"\b(19|20)\d{2}\b", text)
        return int(m.group(0)) if m else None

    # ------------------------------------------------------------
    # 🔹 Extract title hint (quotes, braces, fallback segments)
    # ------------------------------------------------------------

    @staticmethod
    def _extract_title_hint(text: str) -> str:
        for pat in (r"“([^”]+)”", r'"([^"]+)"'):
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()

        m = re.search(r"{([^}]+)}", text)
        if m:
            return m.group(1).strip()

        # fallback: best segment
        cleaned = re.sub(r"\b(19|20)\d{2}\b", "", text)
        segments = [s.strip(" ,;:-") for s in cleaned.split(".") if s.strip()]
        if segments:
            return max(segments, key=lambda seg: sum(ch.isalpha() for ch in seg))

        return text.strip()

    # ------------------------------------------------------------
    # 🔹 Extract token set for scoring
    # ------------------------------------------------------------

    @staticmethod
    def _extract_token_set(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"[A-Za-z'\-]+", text)]

    # ------------------------------------------------------------
    # 🔹 Compute author overlap
    # ------------------------------------------------------------

    @staticmethod
    def _author_overlap(citation_tokens, authors) -> float:
        if not authors:
            return 0.0

        token_set = {t for t in citation_tokens if len(t) > 2}
        if not token_set:
            return 0.0

        last_names = {
            parts[-1].lower()
            for a in authors
            if (parts := re.findall(r"[A-Za-z'\-]+", a))
        }

        if not last_names:
            return 0.0

        return len(token_set & last_names) / len(last_names)

    # ------------------------------------------------------------
    # 🔹 Score a single paper
    # ------------------------------------------------------------

    def _score_paper(self, query_title: str, citation_tokens, paper):
        title = (paper.title or "").lower()
        q = query_title.lower()

        title_ratio = SequenceMatcher(None, q, title).ratio() if title else 0.0
        author_overlap = self._author_overlap(citation_tokens, paper.authors)
        score = 0.75 * title_ratio + 0.25 * author_overlap

        return {
            "score": score,
            "title_ratio": title_ratio,
            "author_overlap": author_overlap,
            "title_exact": (q == title),
        }

    # ------------------------------------------------------------
    # 🔹 Select best match from list of candidates
    # ------------------------------------------------------------

    def _select_best_match(self, citation: str, query_title: str, papers):
        citation_tokens = self._extract_token_set(citation)

        best_paper = None
        best_metrics = {"score": 0, "title_ratio": 0, "author_overlap": 0, "title_exact": False}
        comparisons = []

        for paper in papers:
            metrics = self._score_paper(query_title, citation_tokens, paper)
            comparisons.append((paper.title, metrics))

            if metrics["score"] > best_metrics["score"]:
                best_metrics = metrics
                best_paper = paper

        return best_paper, best_metrics, comparisons

    # ------------------------------------------------------------
    # 🔹 Public API: verify citation string
    # ------------------------------------------------------------

    def verify(self, citation: str) -> CitationVerificationOutcome:
        citation = citation.strip()
        if not citation:
            raise ValueError("Citation cannot be empty")

        year_hint = self._extract_year(citation)
        title_hint = self._extract_title_hint(citation)

        papers = self.lookup_service.lookup(title_hint, year=year_hint)

        best_paper, metrics, comparisons = self._select_best_match(
            citation, title_hint, papers
        )

        matched = (
            best_paper
            and (
                metrics["title_ratio"] >= self.title_threshold
                or metrics["score"] >= self.score_threshold
                or metrics["author_overlap"] >= self.author_overlap_threshold
            )
        )

        reason = None
        if not papers:
            reason = "Semantic Scholar returned no papers"
        elif not matched:
            reason = "No candidate met similarity thresholds"

        return CitationVerificationOutcome(
            matched=bool(matched),
            citation=citation,
            year_filter=year_hint,
            best_paper=best_paper,
            metrics=metrics,
            candidates_considered=len(papers),
            comparisons=comparisons,
            reason=reason,
        )
