"""Agent that verifies free-form citations against Semantic Scholar."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from ..services.semantic_scholar import (
    SemanticScholarLookupService,
    SemanticScholarPaper,
)


@dataclass(slots=True)
class CitationVerificationOutcome:
    """Structured response returned by SemanticScholarCitationVerifier."""

    matched: bool
    citation: str
    year_filter: Optional[int]
    best_paper: Optional[SemanticScholarPaper]
    metrics: Dict[str, float]
    candidates_considered: int
    reason: Optional[str] = None
    comparisons: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "matched": self.matched,
            "citation": self.citation,
            "year_filter": self.year_filter,
            "metrics": self.metrics,
            "candidates_considered": self.candidates_considered,
            "comparisons": self.comparisons,
        }
        if self.best_paper:
            payload["best_paper"] = _format_paper(self.best_paper)
        else:
            payload["best_paper"] = None
        if self.reason:
            payload["reason"] = self.reason
        return payload


class SemanticScholarCitationVerifier:
    """Validate whether a free-form citation exists on Semantic Scholar."""

    def __init__(
        self,
        lookup_service: Optional[SemanticScholarLookupService] = None,
        title_threshold: float = 0.85,
        score_threshold: float = 0.72,
        author_overlap_threshold: float = 0.4,
    ) -> None:
        self.lookup_service = lookup_service or SemanticScholarLookupService()
        self.title_threshold = title_threshold
        self.score_threshold = score_threshold
        self.author_overlap_threshold = author_overlap_threshold

    def verify(self, citation: str) -> CitationVerificationOutcome:
        if not citation or not citation.strip():
            raise ValueError("citation must be a non-empty string")

        normalized_citation = citation.strip()
        year_hint = self._extract_year(normalized_citation)
        query_title = self._extract_title_hint(normalized_citation)
        papers = self.lookup_service.lookup(query_title, year=year_hint)

        best_paper, metrics, comparisons = self._select_best_match(
            normalized_citation, query_title, papers
        )

        matched = bool(best_paper) and (
            metrics.get("title_ratio", 0.0) >= self.title_threshold
            or metrics.get("score", 0.0) >= self.score_threshold
            or metrics.get("author_overlap", 0.0) >= self.author_overlap_threshold
        )

        if not papers:
            reason = "Semantic Scholar returned no papers for the supplied query"
        elif not matched:
            reason = "No candidate met title, score, or author-overlap thresholds"
        else:
            reason = None

        return CitationVerificationOutcome(
            matched=matched,
            citation=normalized_citation,
            year_filter=year_hint,
            best_paper=best_paper,
            metrics=metrics,
            comparisons=comparisons,
            candidates_considered=len(papers),
            reason=reason,
        )

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return int(match.group(0))
        return None

    @staticmethod
    def _extract_title_hint(text: str) -> str:
        # Prefer smart quotes
        for pattern in (r"“([^”]+)”", r'"([^"]+)'):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        # Curly braces (common in BibTeX)
        match = re.search(r"{([^}]+)}", text)
        if match:
            return match.group(1).strip()
        # Fallback: drop years and select the most descriptive segment
        cleaned = re.sub(r"\b(19|20)\d{2}\b", "", text)
        segments = [
            segment.strip(" ,;:-")
            for segment in re.split(r"\.", cleaned)
            if segment and segment.strip()
        ]
        if segments:
            best = max(
                segments,
                key=lambda seg: sum(1 for ch in seg if ch.isalpha()),
            )
            if best:
                return best
        return text.strip()

    def _select_best_match(
        self,
        citation: str,
        query_title: str,
        papers: Sequence[SemanticScholarPaper],
    ) -> Tuple[Optional[SemanticScholarPaper], Dict[str, float], List[Tuple[str, Dict[str, float]]]]:
        comparisons: List[Tuple[str, Dict[str, float]]] = []
        best_paper: Optional[SemanticScholarPaper] = None
        best_metrics: Dict[str, float] = {
            "score": 0.0,
            "title_ratio": 0.0,
            "author_overlap": 0.0,
            "title_exact": False,
        }

        citation_tokens = self._extract_token_set(citation)

        for paper in papers:
            metrics = self._score_paper(query_title, citation_tokens, paper)
            comparisons.append((paper.title, metrics))
            if metrics["score"] > best_metrics["score"]:
                best_metrics = metrics
                best_paper = paper

        return best_paper, best_metrics, comparisons

    def _score_paper(
        self,
        query_title: str,
        citation_tokens: Sequence[str],
        paper: SemanticScholarPaper,
    ) -> Dict[str, float]:
        paper_title = (paper.title or "").strip().lower()
        query_norm = query_title.lower()
        title_ratio = SequenceMatcher(None, query_norm, paper_title).ratio() if paper_title else 0.0
        title_exact = bool(query_norm) and query_norm == paper_title

        author_overlap = self._author_overlap(citation_tokens, paper.authors)
        score = 0.75 * title_ratio + 0.25 * author_overlap

        return {
            "score": score,
            "title_ratio": title_ratio,
            "author_overlap": author_overlap,
            "title_exact": title_exact,
        }

    @staticmethod
    def _extract_token_set(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z'\-]+", text)]

    @staticmethod
    def _author_overlap(citation_tokens: Sequence[str], paper_authors: Sequence[str]) -> float:
        if not citation_tokens or not paper_authors:
            return 0.0
        citation_set = {token for token in citation_tokens if len(token) > 2}
        if not citation_set:
            return 0.0
        author_last_names = {
            parts[-1].lower()
            for author in paper_authors
            if (parts := [chunk for chunk in re.findall(r"[A-Za-z'\-]+", author) if chunk])
        }
        if not author_last_names:
            return 0.0
        hits = citation_set & author_last_names
        return len(hits) / len(author_last_names)


def _format_paper(paper: SemanticScholarPaper) -> Dict[str, object]:
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