"""Data structures shared across the multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class TextSpan:
    """Character span representing where an entity was identified."""

    start: int
    end: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass(slots=True)
class RecognizedEntity:
    """Entity extracted from the citation context."""

    text: str
    entity_type: str
    confidence: float
    span: Optional[TextSpan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidatePaper:
    """Candidate citation returned by a retrieval agent."""

    title: str
    abstract: str = ""
    paper_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        if self.abstract:
            return f"{self.title} {self.abstract}".strip()
        return self.title


@dataclass(slots=True)
class AgentContext:
    """Context shared across agents in the pipeline."""

    citation_context: str
    paper_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieved_candidates: Sequence[CandidatePaper] = field(default_factory=tuple)
    recognized_entities: List[RecognizedEntity] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    def with_entities(self, entities: Sequence[RecognizedEntity]) -> "AgentContext":
        self.recognized_entities = list(entities)
        return self

    def with_candidates(self, candidates: Sequence[CandidatePaper]) -> "AgentContext":
        self.retrieved_candidates = tuple(candidates)
        return self


@dataclass(slots=True)
class VerificationResult:
    """Outcome produced by the verification agent."""

    selected_candidate: Optional[CandidatePaper]
    is_verified: bool
    justification: str
    supporting_entities: List[RecognizedEntity] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
