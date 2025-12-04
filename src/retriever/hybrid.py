"""Hybrid retrieval combining BM25 and dense retrieval."""

from typing import List, Tuple, Dict
import numpy as np
from src.utils.text_processing import per_query_minmax


class HybridRetriever:
    """Hybrid fusion of BM25 and dense retrieval scores."""

    def __init__(self, bm25_weight: float = 0.6, dense_weight: float = 0.4):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores in fusion
            dense_weight: Weight for dense scores in fusion
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def fuse(
        self,
        bm25_doc_ids: np.ndarray,
        bm25_scores: np.ndarray,
        dense_doc_ids: List[int],
        dense_scores: List[float],
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """
        Fuse BM25 and dense retrieval results.

        Args:
            bm25_doc_ids: Document IDs from BM25
            bm25_scores: Scores from BM25
            dense_doc_ids: Document IDs from dense retrieval
            dense_scores: Scores from dense retrieval
            top_k: Number of top results to return

        Returns:
            List of (doc_id, fused_score) tuples, sorted by score
        """
        # Get union of candidates
        cand_ids = set(bm25_doc_ids) | set(dense_doc_ids)

        # Build score maps
        bm25_map = {int(d): float(s) for d, s in zip(bm25_doc_ids, bm25_scores)}
        dense_map = {int(d): float(s) for d, s in zip(dense_doc_ids, dense_scores)}

        # Normalize scores per query
        bm25_norm = per_query_minmax(bm25_map)
        dense_norm = per_query_minmax(dense_map)

        # Fuse scores
        fused = []
        for did in cand_ids:
            s_b = bm25_norm.get(did, 0.0)
            s_d = dense_norm.get(did, 0.0)
            score = self.bm25_weight * s_b + self.dense_weight * s_d
            fused.append((did, score))

        # Sort by score descending
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def retrieve_batch(
        self,
        bm25_results: List[Tuple[np.ndarray, np.ndarray]],
        dense_results: List[Tuple[List[int], List[float]]],
        top_k: int = 50,
    ) -> List[List[Tuple[int, float]]]:
        """
        Fuse results for a batch of queries.

        Args:
            bm25_results: List of (doc_ids, scores) from BM25 for each query
            dense_results: List of (doc_ids, scores) from dense retrieval for each query
            top_k: Number of top results per query

        Returns:
            List of fused results for each query
        """
        fused_results = []
        for (bm25_ids, bm25_scores), (dense_ids, dense_scores) in zip(bm25_results, dense_results):
            fused = self.fuse(bm25_ids, bm25_scores, dense_ids, dense_scores, top_k)
            fused_results.append(fused)
        return fused_results
