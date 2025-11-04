"""
Metrics calculation for citation retrieval evaluation.

Supports: Recall@k, Precision@k, MRR, Exact Match, nDCG
"""

import numpy as np
from typing import List, Dict, Optional, Set
import difflib
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for all retrieval metrics"""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    mrr: float
    exact_match_rate: float
    avg_rank: float
    num_queries: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "recall@20": round(self.recall_at_20, 4),
            "precision@1": round(self.precision_at_1, 4),
            "precision@5": round(self.precision_at_5, 4),
            "precision@10": round(self.precision_at_10, 4),
            "mrr": round(self.mrr, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "avg_rank": round(self.avg_rank, 2),
            "num_queries": self.num_queries
        }

    def __str__(self) -> str:
        """Pretty print metrics"""
        return f"""
Citation Retrieval Metrics ({self.num_queries} queries):
{'='*50}
Recall@1:     {self.recall_at_1:.4f}
Recall@5:     {self.recall_at_5:.4f}
Recall@10:    {self.recall_at_10:.4f}
Recall@20:    {self.recall_at_20:.4f}
{'='*50}
Precision@1:  {self.precision_at_1:.4f}
Precision@5:  {self.precision_at_5:.4f}
Precision@10: {self.precision_at_10:.4f}
{'='*50}
MRR:          {self.mrr:.4f}
Exact Match:  {self.exact_match_rate:.4f}
Avg Rank:     {self.avg_rank:.2f}
{'='*50}
"""


class MetricsCalculator:
    """Calculate retrieval metrics for citation prediction"""

    def __init__(self, fuzzy_match_threshold: float = 0.85):
        """
        Args:
            fuzzy_match_threshold: Threshold for fuzzy title matching (0-1)
        """
        self.fuzzy_threshold = fuzzy_match_threshold

    def fuzzy_match(self, true_title: str, candidate_title: str) -> bool:
        """
        Check if two titles match using fuzzy string matching.

        Args:
            true_title: Ground truth title
            candidate_title: Retrieved candidate title

        Returns:
            True if titles match above threshold
        """
        if not true_title or not candidate_title:
            return False

        ratio = difflib.SequenceMatcher(
            None,
            true_title.lower().strip(),
            candidate_title.lower().strip()
        ).ratio()

        return ratio >= self.fuzzy_threshold

    def find_match_positions(
        self,
        true_titles: List[str],
        retrieved_titles: List[str]
    ) -> List[int]:
        """
        Find positions of ground truth titles in retrieved list.

        Args:
            true_titles: List of ground truth citation titles
            retrieved_titles: List of retrieved titles (ordered by relevance)

        Returns:
            List of positions (1-indexed) where matches found, -1 if not found
        """
        positions = []

        for true_title in true_titles:
            found = False
            for rank, retrieved_title in enumerate(retrieved_titles, start=1):
                if self.fuzzy_match(true_title, retrieved_title):
                    positions.append(rank)
                    found = True
                    break

            if not found:
                positions.append(-1)

        return positions

    def calculate_recall_at_k(self, positions: List[int], k: int) -> float:
        """
        Calculate Recall@k.

        Args:
            positions: List of match positions (-1 if not found)
            k: Cut-off rank

        Returns:
            Recall@k score
        """
        if not positions:
            return 0.0

        hits = sum(1 for pos in positions if 0 < pos <= k)
        return hits / len(positions)

    def calculate_precision_at_k(
        self,
        positions: List[int],
        k: int,
        num_relevant: int = 1
    ) -> float:
        """
        Calculate Precision@k (assuming typically 1 relevant doc per query).

        Args:
            positions: List of match positions
            k: Cut-off rank
            num_relevant: Number of relevant documents (typically 1 for citations)

        Returns:
            Precision@k score
        """
        if not positions:
            return 0.0

        hits = sum(1 for pos in positions if 0 < pos <= k)
        return hits / (k * len(positions))

    def calculate_mrr(self, positions: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            positions: List of match positions

        Returns:
            MRR score
        """
        if not positions:
            return 0.0

        reciprocal_ranks = [1/pos if pos > 0 else 0 for pos in positions]
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def calculate_metrics(
        self,
        predictions: List[Dict],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> RetrievalMetrics:
        """
        Calculate all metrics for a set of predictions.

        Args:
            predictions: List of prediction dicts with keys:
                - 'true_titles': List[str] - ground truth titles
                - 'retrieved_titles': List[str] - retrieved titles (ranked)
            k_values: List of k values for Recall@k and Precision@k

        Returns:
            RetrievalMetrics object with all computed metrics
        """
        all_positions = []

        # Calculate positions for each query
        for pred in predictions:
            true_titles = pred.get('true_titles', [])
            if isinstance(true_titles, str):
                true_titles = [true_titles]

            retrieved_titles = pred.get('retrieved_titles', [])
            positions = self.find_match_positions(true_titles, retrieved_titles)
            all_positions.extend(positions)

        # Calculate metrics
        recall_1 = self.calculate_recall_at_k(all_positions, 1)
        recall_5 = self.calculate_recall_at_k(all_positions, 5)
        recall_10 = self.calculate_recall_at_k(all_positions, 10)
        recall_20 = self.calculate_recall_at_k(all_positions, 20)

        precision_1 = self.calculate_precision_at_k(all_positions, 1)
        precision_5 = self.calculate_precision_at_k(all_positions, 5)
        precision_10 = self.calculate_precision_at_k(all_positions, 10)

        mrr = self.calculate_mrr(all_positions)
        exact_match_rate = self.calculate_recall_at_k(all_positions, 1)

        # Average rank (only for found items)
        found_positions = [p for p in all_positions if p > 0]
        avg_rank = np.mean(found_positions) if found_positions else float('inf')

        return RetrievalMetrics(
            recall_at_1=recall_1,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            recall_at_20=recall_20,
            precision_at_1=precision_1,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            mrr=mrr,
            exact_match_rate=exact_match_rate,
            avg_rank=avg_rank,
            num_queries=len(all_positions)
        )

    def calculate_per_query_metrics(
        self,
        true_titles: List[str],
        retrieved_titles: List[str],
        k: int = 10
    ) -> Dict:
        """
        Calculate metrics for a single query.

        Args:
            true_titles: Ground truth titles
            retrieved_titles: Retrieved titles
            k: Cut-off for metrics

        Returns:
            Dictionary of metrics for this query
        """
        positions = self.find_match_positions(true_titles, retrieved_titles)

        return {
            'positions': positions,
            'recall@1': self.calculate_recall_at_k(positions, 1),
            'recall@5': self.calculate_recall_at_k(positions, 5),
            'recall@10': self.calculate_recall_at_k(positions, 10),
            'mrr': self.calculate_mrr(positions),
            'hit': any(p > 0 for p in positions),
            'best_rank': min([p for p in positions if p > 0], default=-1)
        }
