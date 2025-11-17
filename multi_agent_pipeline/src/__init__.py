"""Core modules for the multi-agent citation retrieval pipeline."""

from .types import AgentContext, CandidatePaper, RecognizedEntity, VerificationResult
from .data import ScholarCopilotDataset, TrainingExample, load_training_data

__all__ = [
    "AgentContext",
    "CandidatePaper",
    "RecognizedEntity",
    "VerificationResult",
    "ScholarCopilotDataset",
    "TrainingExample",
    "load_training_data",
]
