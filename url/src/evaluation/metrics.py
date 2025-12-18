"""Evaluation metrics for retrieval systems."""

import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Cutoff position

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    # Get top-k retrieved IDs
    top_k_ids = set(retrieved_ids[:k])

    # Count how many relevant docs were retrieved in top-k
    hits = len(top_k_ids & relevant_ids)

    # Recall = hits / total_relevant
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank of first relevant document (or 0 if no relevant doc found)

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: Set of relevant document IDs (ground truth)

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved_papers: List[Dict[str, Any]],
    relevant_ids: Set[str],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate retrieval results with multiple metrics.

    Args:
        retrieved_papers: List of retrieved papers (with 'id' field)
        relevant_ids: Set of relevant paper IDs (ground truth)
        k_values: List of k values for Recall@K

    Returns:
        Dictionary with metric scores
    """
    retrieved_ids = [str(paper.get("id", "")) for paper in retrieved_papers]

    metrics = {}

    # Calculate Recall@K for each k
    for k in k_values:
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"R@{k}"] = r_at_k

    # Calculate MRR
    mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    metrics["MRR"] = mrr

    # Count how many relevant docs were found at all
    retrieved_set = set(retrieved_ids)
    hits = len(retrieved_set & relevant_ids)
    metrics["hits"] = hits
    metrics["total_relevant"] = len(relevant_ids)

    return metrics


def print_evaluation_results(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print evaluation metrics."""
    if prefix:
        logger.info(f"\n{prefix}")

    logger.info("📊 Evaluation Metrics:")
    logger.info(f"  • Recall@5:  {metrics.get('R@5', 0.0):.4f}")
    logger.info(f"  • Recall@10: {metrics.get('R@10', 0.0):.4f}")
    logger.info(f"  • Recall@20: {metrics.get('R@20', 0.0):.4f}")
    logger.info(f"  • MRR:       {metrics.get('MRR', 0.0):.4f}")
    logger.info(f"  • Hits:      {metrics.get('hits', 0)}/{metrics.get('total_relevant', 0)}")
