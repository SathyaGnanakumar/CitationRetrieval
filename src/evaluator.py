"""Evaluation metrics for citation retrieval."""

import pandas as pd
from typing import Tuple, Dict


class CitationEvaluator:
    """Evaluates citation retrieval results with standard metrics."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate(self, df: pd.DataFrame, k: int = 5) -> Tuple[float, float, float]:
        """
        Compute evaluation metrics for a results DataFrame.

        Args:
            df: DataFrame with columns: query_id, rank, hit
            k: Top-k value for Recall@k

        Returns:
            Tuple of (recall@k, MRR, top-1 accuracy)
        """
        if df.empty:
            return 0.0, 0.0, 0.0

        grouped = df.groupby("query_id", sort=False)

        # Recall@k: fraction of queries with at least one hit in top-k
        recall_k = grouped.apply(
            lambda g: int(g.sort_values("rank").head(k)["hit"].max() > 0)
        ).mean()

        # MRR: Mean Reciprocal Rank
        def _rr(g):
            g = g.sort_values("rank")
            hits = g[g["hit"] == 1]
            return 1.0 / hits.iloc[0]["rank"] if not hits.empty else 0.0

        mrr = grouped.apply(_rr).mean()

        # Top-1 accuracy: fraction of queries with hit at rank 1
        top1 = (df[df["rank"] == 1]["hit"].sum()) / df["query_id"].nunique()

        return float(recall_k), float(mrr), float(top1)

    def compute_all_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all standard metrics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with all metrics
        """
        recall_5, mrr, top1 = self.evaluate(df, k=5)
        recall_10, _, _ = self.evaluate(df, k=10)

        return {
            "recall@5": recall_5,
            "recall@10": recall_10,
            "mrr": mrr,
            "top1": top1,
        }
