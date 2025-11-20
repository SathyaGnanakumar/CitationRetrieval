"""Named-entity recognition agent for citation contexts."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from transformers import pipeline

from ..types import AgentContext, RecognizedEntity, TextSpan

LOGGER = logging.getLogger(__name__)


class EntityRecognitionAgent:
    """Extracts entities from citation contexts using a Hugging Face pipeline."""

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        aggregation_strategy: str = "simple",
        min_confidence: float = 0.3,
        device: Optional[str] = None,
        pipeline_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            model_name: Hugging Face model to load for token classification.
            aggregation_strategy: Strategy passed to the transformers pipeline.
            min_confidence: Minimum probability for returning an entity.
            device: Device identifier (e.g., "cuda:0", "cpu", integer GPU id).
            pipeline_kwargs: Additional keyword arguments forwarded to ``pipeline``.
        """
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.min_confidence = min_confidence
        self.device = device
        self.pipeline_kwargs = pipeline_kwargs or {}
        self._pipeline = None

    @property
    def hf_pipeline(self):
        """Lazily initialize and cache the Hugging Face pipeline."""
        if self._pipeline is None:
            LOGGER.info("Loading NER model %s", self.model_name)
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation_strategy,
                device=self.device,
                **self.pipeline_kwargs,
            )
        return self._pipeline

    def extract_entities(self, text: str) -> List[RecognizedEntity]:
        """Run NER on the provided text and return structured entities."""
        if not text.strip():
            return []

        outputs = self.hf_pipeline(text)
        entities: List[RecognizedEntity] = []
        for item in outputs:
            score = float(item.get("score", 0.0))
            if score < self.min_confidence:
                continue

            entity = RecognizedEntity(
                text=item.get("word", ""),
                entity_type=item.get("entity_group", "UNKNOWN"),
                confidence=score,
                span=TextSpan(start=int(item.get("start", -1)), end=int(item.get("end", -1))),
                metadata={"raw": item},
            )
            entities.append(entity)

        return entities

    def __call__(self, context: AgentContext) -> AgentContext:
        """Attach recognized entities to the supplied agent context."""
        entities = self.extract_entities(context.citation_context)
        context.with_entities(entities)
        LOGGER.debug("Recognized %d entities", len(entities))
        return context

    def batch_extract(self, texts: Sequence[str]) -> List[List[RecognizedEntity]]:
        """Run NER on a batch of texts."""
        if not texts:
            return []

        outputs = self.hf_pipeline(list(texts))
        batch_entities: List[List[RecognizedEntity]] = []
        for text_outputs in outputs:
            entities: List[RecognizedEntity] = []
            for item in text_outputs:
                score = float(item.get("score", 0.0))
                if score < self.min_confidence:
                    continue
                entities.append(
                    RecognizedEntity(
                        text=item.get("word", ""),
                        entity_type=item.get("entity_group", "UNKNOWN"),
                        confidence=score,
                        span=TextSpan(start=int(item.get("start", -1)), end=int(item.get("end", -1))),
                        metadata={"raw": item},
                    )
                )
            batch_entities.append(entities)

        return batch_entities