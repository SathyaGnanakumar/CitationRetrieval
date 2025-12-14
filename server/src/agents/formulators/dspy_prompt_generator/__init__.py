"""
DSPy Prompt Generator for Citation Retrieval.

This module provides tools for:
- Preparing citation data with positives/negatives
- DSPy signatures and modules for citation retrieval
- Training and optimization pipelines

Usage:
    # Prepare data
    from dspy_prompt_generator import CitationDataPrep
    prep = CitationDataPrep("datasets/scholar_copilot_eval_data_1k.json")
    prep.extract_examples()
    prep.save_examples("dspy_prompt_generator/data")
    
    # Train/optimize
    from dspy_prompt_generator import CitationTrainer, TrainingConfig
    config = TrainingConfig(model="gpt-4o-mini")
    trainer = CitationTrainer(config)
    trainer.setup_lm()
    trainer.load_data()
    trainer.optimize(module_name="simple")
    
    # Use optimized module
    result = trainer.optimized_module(
        citation_context="...[CITATION]...",
        candidates=[{"title": "...", "abstract": "..."}]
    )
"""

from .data_prep import CitationDataPrep, CitationTrainingExample
from .signatures import (
    CitationRetrieval,
    QueryGeneration,
    CitationReranking,
    CitationVerification,
)
from .modules import (
    SimpleCitationRetriever,
    QueryThenRetrieve,
    RerankAndSelect,
    VerifyAndSelect,
    EnsembleRetriever,
    get_module,
)
from .trainer import (
    CitationTrainer,
    TrainingConfig,
    load_examples_from_json,
    exact_match_metric,
    fuzzy_match_metric,
    contains_match_metric,
)

__all__ = [
    # Data prep
    "CitationDataPrep",
    "CitationTrainingExample",
    # Signatures
    "CitationRetrieval",
    "QueryGeneration",
    "CitationReranking",
    "CitationVerification",
    # Modules
    "SimpleCitationRetriever",
    "QueryThenRetrieve",
    "RerankAndSelect",
    "VerifyAndSelect",
    "EnsembleRetriever",
    "get_module",
    # Training
    "CitationTrainer",
    "TrainingConfig",
    "load_examples_from_json",
    "exact_match_metric",
    "fuzzy_match_metric",
    "contains_match_metric",
]
