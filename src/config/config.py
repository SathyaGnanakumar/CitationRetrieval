"""Configuration class for Citation Retrieval system."""

import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Configuration parameters for the citation retrieval pipeline."""

    # File paths
    data_file: str = "/content/drive/MyDrive/scholar_copilot_eval_data_1k.json"
    results_dir: str = "/Users/ishaankalra/Dev/CMSC473/CitationRetrieval/results/"

    # Output file paths
    @property
    def bm25_output(self) -> str:
        return self.results_dir + "bm25_results.csv"

    @property
    def dense_output(self) -> str:
        return self.results_dir + "dense_results_e5.csv"

    @property
    def hybrid_output(self) -> str:
        return self.results_dir + "hybrid_results.csv"

    @property
    def rerank_output(self) -> str:
        return self.results_dir + "rerank_results.csv"

    # Retrieval parameters
    top_k_bm25: int = 50  # BM25 candidates per query
    top_k_dense: int = 50  # Dense candidates per query
    top_k_final: int = 10  # Final results per query to save/evaluate

    # Fusion parameters
    fusion_bm25_weight: float = 0.6  # weight for BM25 in hybrid fusion
    fusion_dense_weight: float = 0.4  # weight for Dense in hybrid fusion

    # Reranker parameters
    use_reranker: bool = True  # set False to skip reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_rerank: int = 50  # how many hybrid candidates to rerank

    # Dense retrieval model
    dense_model_name: str = "intfloat/e5-large-v2"

    # Device and batch processing
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32

    # BM25 parameters
    bm25_language: str = "english"
    bm25_stopwords: str = "en"
