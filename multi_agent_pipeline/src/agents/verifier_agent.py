"""Verification agent that scores candidate papers using recognized entities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import AgentContext, CandidatePaper, RecognizedEntity, VerificationResult
from ..services.semantic_scholar import (
    SemanticScholarLookupService,
    SemanticScholarPaper,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    candidate: CandidatePaper
    score: float
    matched_entities: List[Tuple[RecognizedEntity, str]]


class VerificationAgent:
    """Selects and validates the best candidate citation for a context."""

    def __init__(
        self,
        title_weight: float = 1.0,
        abstract_weight: float = 0.6,
        retrieval_weight: float = 0.2,
        fuzzy_weight: float = 0.4,
        min_entity_confidence: float = 0.3,
        fuzzy_threshold: float = 0.8,
        min_score: float = 0.0,
        lookup_service: Optional[SemanticScholarLookupService] = None,
        enable_external_lookup: bool = False,
        external_title_threshold: float = 0.85,
        external_author_overlap_threshold: float = 0.4,
        external_title_weight: float = 0.8,
        external_author_weight: float = 0.2,
    ) -> None:
        self.title_weight = title_weight
        self.abstract_weight = abstract_weight
        self.retrieval_weight = retrieval_weight
        self.fuzzy_weight = fuzzy_weight
        self.min_entity_confidence = min_entity_confidence
        self.fuzzy_threshold = fuzzy_threshold
        self.min_score = min_score
        if lookup_service is not None:
            self.lookup_service = lookup_service
            self.enable_external_lookup = True
        else:
            self.enable_external_lookup = enable_external_lookup
            self.lookup_service = (
                SemanticScholarLookupService() if self.enable_external_lookup else None
            )
        self.external_title_threshold = external_title_threshold
        self.external_author_overlap_threshold = external_author_overlap_threshold
        self.external_title_weight = external_title_weight
        self.external_author_weight = external_author_weight

    def __call__(self, context: AgentContext) -> VerificationResult:
        result = self.verify(context)
        context.state.setdefault("verification", {}).update(
            {
                "selected_title": result.selected_candidate.title if result.selected_candidate else None,
                "score": result.metadata.get("score"),
                "justification": result.justification,
            }
        )
        return result

    def verify(self, context: AgentContext) -> VerificationResult:
        candidates = list(context.retrieved_candidates)
        entities = list(context.recognized_entities)

        if not candidates:
            LOGGER.warning("VerificationAgent called with no candidates")
            return VerificationResult(
                selected_candidate=None,
                is_verified=False,
                justification="No candidate citations supplied",
                supporting_entities=entities,
                metadata={"scores": []},
            )

        candidate_scores = [self.score_candidate(candidate, entities) for candidate in candidates]
        candidate_scores = [item for item in candidate_scores if item.score >= self.min_score]

        if not candidate_scores:
            return VerificationResult(
                selected_candidate=None,
                is_verified=False,
                justification="No candidate achieved the minimum verification score",
                supporting_entities=entities,
                metadata={"scores": []},
            )

        candidate_scores.sort(key=lambda x: x.score, reverse=True)
        best = candidate_scores[0]

        justification = self._build_justification(best, entities)
        external_result = self._external_verify(best.candidate, context)
        is_verified = external_result.get("matched", True) if external_result else True
        metadata = {
            "score": best.score,
            "scores": [
                {
                    "title": cs.candidate.title,
                    "score": cs.score,
                    "matched_entities": [
                        {
                            "entity": entity.text,
                            "entity_type": entity.entity_type,
                            "location": location,
                            "confidence": entity.confidence,
                        }
                        for entity, location in cs.matched_entities
                    ],
                }
                for cs in candidate_scores
            ],
        }
        if external_result:
            metadata["external_verification"] = external_result

        return VerificationResult(
            selected_candidate=best.candidate,
            is_verified=is_verified,
            justification=justification,
            supporting_entities=[entity for entity, _ in best.matched_entities],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def score_candidate(
        self, candidate: CandidatePaper, entities: Sequence[RecognizedEntity]
    ) -> CandidateScore:
        title = candidate.title.lower()
        abstract = candidate.abstract.lower()
        base_score = float(candidate.score) if candidate.score is not None else 0.0
        score = base_score * self.retrieval_weight
        matched: List[Tuple[RecognizedEntity, str]] = []

        for entity in entities:
            if entity.confidence < self.min_entity_confidence:
                continue

            entity_text = entity.text.strip().lower()
            if not entity_text:
                continue

            if entity_text in title:
                score += self.title_weight * entity.confidence
                matched.append((entity, "title"))
                continue

            if entity_text and entity_text in abstract:
                score += self.abstract_weight * entity.confidence
                matched.append((entity, "abstract"))
                continue

            # Fuzzy match fallback
            best_ratio, location = self._best_fuzzy_match(entity_text, title, abstract)
            if best_ratio >= self.fuzzy_threshold:
                score += self.fuzzy_weight * entity.confidence * best_ratio
                matched.append((entity, location))

        return CandidateScore(candidate=candidate, score=score, matched_entities=matched)

    def _best_fuzzy_match(
        self, entity_text: str, title: str, abstract: str
    ) -> Tuple[float, str]:
        ratio_title = SequenceMatcher(None, entity_text, title).ratio() if title else 0.0
        ratio_abstract = SequenceMatcher(None, entity_text, abstract).ratio() if abstract else 0.0

        if ratio_title >= ratio_abstract:
            return ratio_title, "title_fuzzy"
        return ratio_abstract, "abstract_fuzzy"

    def _build_justification(
        self, best: CandidateScore, entities: Sequence[RecognizedEntity]
    ) -> str:
        if not best.matched_entities:
            return (
                f"Selected candidate '{best.candidate.title}' based on retrieval score only. "
                "No direct entity matches were found."
            )

        parts = [f"Selected candidate '{best.candidate.title}' with score {best.score:.2f}."]
        grouped: Dict[str, List[str]] = {}
        for entity, location in best.matched_entities:
            grouped.setdefault(location, []).append(f"{entity.text} ({entity.entity_type})")

        for location, labels in grouped.items():
            parts.append(f"Matched entities in {location}: {', '.join(labels)}")

        unmatched = {
            entity.text
            for entity in entities
            if all(entity is not matched_entity for matched_entity, _ in best.matched_entities)
        }
        if unmatched:
            parts.append("Unmatched entities: " + ", ".join(sorted(unmatched)))

        return " \n".join(parts)

    # ------------------------------------------------------------------
    # External verification helpers
    # ------------------------------------------------------------------
    def _external_verify(self, candidate: CandidatePaper, context: AgentContext) -> Optional[Dict[str, Any]]:
        if not self.lookup_service:
            return None

        year = self._candidate_year(candidate, context)
        try:
            papers = self.lookup_service.lookup(candidate.title, year=year)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Semantic Scholar lookup failed: %s", exc)
            return {"matched": False, "error": str(exc), "results": []}

        if not papers:
            return {"matched": False, "results": []}

        best_paper, scores = self._select_best_external_match(candidate, papers)
        matched = False
        combined_score = 0.0
        if best_paper is not None:
            title_ratio = scores[best_paper]["title_ratio"]
            author_overlap = scores[best_paper]["author_overlap"]
            combined_score = (
                self.external_title_weight * title_ratio
                + self.external_author_weight * author_overlap
            )
            matched = (
                title_ratio >= self.external_title_threshold
                or author_overlap >= self.external_author_overlap_threshold
            )

        return {
            "matched": matched,
            "year_filter": year,
            "combined_score": combined_score,
            "best_match": self._format_external_paper(best_paper) if best_paper else None,
            "num_results": len(papers),
        }

    def _select_best_external_match(
        self, candidate: CandidatePaper, papers: Sequence[SemanticScholarPaper]
    ) -> Tuple[Optional[SemanticScholarPaper], Dict[SemanticScholarPaper, Dict[str, float]]]:
        candidate_title = (candidate.title or "").lower().strip()
        candidate_authors = self._candidate_authors(candidate)
        scores: Dict[SemanticScholarPaper, Dict[str, float]] = {}
        best_paper: Optional[SemanticScholarPaper] = None
        best_score = -1.0

        for paper in papers:
            title_ratio = SequenceMatcher(None, candidate_title, (paper.title or "").lower()).ratio()
            author_overlap = self._author_overlap(candidate_authors, paper.authors)
            aggregate = (
                self.external_title_weight * title_ratio
                + self.external_author_weight * author_overlap
            )
            scores[paper] = {
                "title_ratio": title_ratio,
                "author_overlap": author_overlap,
                "aggregate": aggregate,
            }
            if aggregate > best_score:
                best_score = aggregate
                best_paper = paper

        return best_paper, scores

    def _candidate_year(self, candidate: CandidatePaper, context: AgentContext) -> Optional[int]:
        for source in (candidate.metadata, context.paper_metadata):
            if not source:
                continue
            year = source.get("year")
            if isinstance(year, int):
                return year
            if isinstance(year, str) and year.isdigit():
                return int(year)
        return None

    def _candidate_authors(self, candidate: CandidatePaper) -> List[str]:
        authors = candidate.metadata.get("authors") if candidate.metadata else None
        if isinstance(authors, list):
            return [str(author).lower().strip() for author in authors if author]
        if isinstance(authors, str):
            return [author.lower().strip() for author in authors.split(",") if author]
        return []

    def _author_overlap(self, candidate_authors: Sequence[str], paper_authors: Sequence[str]) -> float:
        if not candidate_authors or not paper_authors:
            return 0.0
        candidate_set = {author.lower().strip() for author in candidate_authors if author}
        paper_set = {author.lower().strip() for author in paper_authors if author}
        if not candidate_set or not paper_set:
            return 0.0
        overlap = candidate_set & paper_set
        return len(overlap) / max(len(candidate_set), len(paper_set))

    def _format_external_paper(self, paper: SemanticScholarPaper) -> Dict[str, Any]:
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