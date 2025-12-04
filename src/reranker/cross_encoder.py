"""Cross-encoder reranking implementation."""

from sentence_transformers import CrossEncoder
from typing import List, Tuple


class CrossEncoderReranker:
    """Reranks candidates using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self) -> None:
        """Load the cross-encoder model."""
        print(f"🤝 Loading reranker: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank candidates for a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (candidate_index, score) tuples, sorted by score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Build query-candidate pairs
        pairs = [(query, cand) for cand in candidates]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Create (index, score) pairs and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def rerank_with_ids(
        self,
        query: str,
        candidate_ids: List[int],
        candidate_texts: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Rerank candidates and return their IDs.

        Args:
            query: Query text
            candidate_ids: List of candidate document IDs
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Build query-candidate pairs
        pairs = [(query, cand) for cand in candidate_texts]

        # Get reranking scores
        scores = self.model.predict(pairs)

        # Create (doc_id, score) pairs and sort
        id_scores = [(candidate_ids[i], float(score)) for i, score in enumerate(scores)]
        id_scores.sort(key=lambda x: x[1], reverse=True)

        return id_scores[:top_k]
