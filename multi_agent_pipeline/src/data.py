"""Utilities for accessing ScholarCopilot data for agent training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from evaluation.data_loader import CitationDataLoader, CitationExample
from .types import AgentContext, CandidatePaper


@dataclass(slots=True)
class TrainingExample:
    """Container used when training reinforcement or supervised agents."""

    query_id: str
    context: AgentContext
    true_title: str
    true_abstract: Optional[str]


class ScholarCopilotDataset:
    """Loads citation contexts for training the multi-agent pipeline."""

    def __init__(
        self,
        data_path: str = "datasets/scholar_copilot_eval_data_1k.json",
        refresh_cache: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.refresh_cache = refresh_cache
        self._loader: Optional[CitationDataLoader] = None
        self._examples: Optional[List[CitationExample]] = None

    @property
    def loader(self) -> CitationDataLoader:
        if self._loader is None or self.refresh_cache:
            self._loader = CitationDataLoader(str(self.data_path))
            self._loader.load_data()
        return self._loader

    def examples(self) -> List[TrainingExample]:
        if self._examples is None or self.refresh_cache:
            citation_examples = self.loader.extract_examples()
            self._examples = [self._to_training_example(example) for example in citation_examples]
        return self._examples

    def iter_examples(self) -> Iterator[TrainingExample]:
        for example in self.examples():
            yield example

    def sample(self, limit: Optional[int] = None) -> List[TrainingExample]:
        records = self.examples()
        if limit is None or limit >= len(records):
            return records
        return records[:limit]

    def _to_training_example(self, example: CitationExample) -> TrainingExample:
        candidates = [
            CandidatePaper(
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                metadata={"citation_key": item.get("citation_key"), "text": item.get("text", "")},
            )
            for item in example.corpus_entries
        ]
        context = AgentContext(
            citation_context=example.citation_context,
            retrieved_candidates=candidates,
            paper_metadata={"paper_id": example.paper_id},
        )
        return TrainingExample(
            query_id=example.query_id,
            context=context,
            true_title=example.true_title,
            true_abstract=example.true_abstract,
        )


def load_training_data(
    data_path: str = "datasets/scholar_copilot_eval_data_1k.json",
    limit: Optional[int] = None,
) -> List[TrainingExample]:
    """Convenience helper returning a list of training examples."""

    dataset = ScholarCopilotDataset(data_path=data_path)
    examples = dataset.examples()
    if limit is not None:
        return examples[:limit]
    return examples
