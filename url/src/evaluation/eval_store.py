"""
Evaluation storage system for self-evolving DSPy agents.

Stores query evaluations, metrics, and provides DSPy training data conversion.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import dspy


def is_evolution_enabled() -> bool:
    """Check if DSPy evolution is enabled via environment variable."""
    return os.getenv("ENABLE_DSPY_EVOLUTION", "false").lower() in {"true", "1", "yes", "on"}


@dataclass
class QueryEvaluation:
    """Individual query evaluation result with rich feedback."""

    query: str
    paper_id: str
    retrieved_ids: List[str]
    relevant_ids: List[str]  # Stored as list for JSON serialization
    metrics: Dict[str, float]

    # For DSPy optimization
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    score: float

    # Metadata
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format for training."""
        return dspy.Example(
            citation_context=self.query,
            retrieved_papers=self.outputs.get("ranked_papers", []),
            ground_truth_ids=self.relevant_ids,
            score=self.score,
            paper_id=self.paper_id,
            metrics=self.metrics,
        ).with_inputs("citation_context")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryEvaluation":
        """Create from dictionary."""
        return cls(**data)


class EvaluationStore:
    """
    Stores evaluation results with file persistence.

    Provides methods for filtering, exporting, and converting to DSPy training format.
    """

    def __init__(self, storage_dir: str = "./data/evaluations"):
        """
        Initialize evaluation store.

        Args:
            storage_dir: Directory to store evaluation data
        """
        self.storage_dir = Path(storage_dir)
        self.evaluations: List[QueryEvaluation] = []
        self._enabled = is_evolution_enabled()

        if self._enabled:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def _load_existing(self):
        """Load existing evaluations from disk."""
        summary_file = self.storage_dir / "summary.json"
        if summary_file.exists():
            try:
                content = summary_file.read_text().strip()
                if not content:
                    return  # Empty file, nothing to load
                data = json.loads(content)
                # Only load fields we need (skip inputs/outputs which may be missing)
                for item in data:
                    try:
                        ev = QueryEvaluation(
                            query=item.get("query", ""),
                            paper_id=item.get("paper_id", ""),
                            retrieved_ids=item.get("retrieved_ids", []),
                            relevant_ids=item.get("relevant_ids", []),
                            metrics=item.get("metrics", {}),
                            inputs=item.get("inputs", {}),
                            outputs=item.get("outputs", {}),
                            score=item.get("score", 0.0),
                            timestamp=item.get("timestamp"),
                        )
                        self.evaluations.append(ev)
                    except Exception:
                        pass  # Skip malformed entries
            except Exception:
                pass  # Silently skip load errors

    def add_evaluation(self, evaluation: QueryEvaluation) -> None:
        """
        Add an evaluation to the store.

        Args:
            evaluation: QueryEvaluation instance
        """
        if not self._enabled:
            return  # No-op when evolution is disabled

        self.evaluations.append(evaluation)

        # Save batch file - build dict manually to avoid asdict() pickling issues
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_file = self.storage_dir / f"batch_{timestamp}.json"

        try:
            data = {
                "query": evaluation.query,
                "paper_id": evaluation.paper_id,
                "retrieved_ids": evaluation.retrieved_ids,
                "relevant_ids": evaluation.relevant_ids,
                "metrics": evaluation.metrics,
                "score": evaluation.score,
                "timestamp": evaluation.timestamp,
            }
            with open(batch_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass  # Silently skip if still not serializable

        # Update summary file
        self._save_summary()

    def _save_summary(self):
        """Save all evaluations to summary file."""
        summary_file = self.storage_dir / "summary.json"
        try:
            sanitized = []
            for ev in self.evaluations:
                data = {
                    "query": ev.query,
                    "paper_id": ev.paper_id,
                    "retrieved_ids": ev.retrieved_ids,
                    "relevant_ids": ev.relevant_ids,
                    "metrics": ev.metrics,
                    "score": ev.score,
                    "timestamp": ev.timestamp,
                }
                sanitized.append(data)
            with open(summary_file, "w") as f:
                json.dump(sanitized, f, indent=2, default=str)
        except Exception:
            pass  # Silently skip if still not serializable

    def get_failures(self, threshold: float = 0.3) -> List[QueryEvaluation]:
        """
        Get queries that performed poorly.

        Args:
            threshold: Maximum score to consider as failure

        Returns:
            List of failed query evaluations
        """
        return [e for e in self.evaluations if e.score < threshold]

    def get_successes(self, threshold: float = 0.7) -> List[QueryEvaluation]:
        """
        Get queries that performed well.

        Args:
            threshold: Minimum score to consider as success

        Returns:
            List of successful query evaluations
        """
        return [e for e in self.evaluations if e.score >= threshold]

    def to_dspy_trainset(self, min_score: Optional[float] = None) -> List[dspy.Example]:
        """
        Convert stored evaluations to DSPy training data.

        Args:
            min_score: Minimum score threshold to include in training set

        Returns:
            List of DSPy Examples
        """
        if not self._enabled:
            return []

        evals = (
            self.evaluations
            if min_score is None
            else [e for e in self.evaluations if e.score >= min_score]
        )
        return [e.to_dspy_example() for e in evals]

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of stored evaluations."""
        if not self.evaluations:
            return {"count": 0}

        scores = [e.score for e in self.evaluations]
        return {
            "count": len(self.evaluations),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "failures": len(self.get_failures()),
            "successes": len(self.get_successes()),
        }

    def clear(self):
        """Clear all evaluations (use with caution)."""
        self.evaluations = []
        self._save_summary()
