"""E5-Large-v2 dense retrieval implementation."""

import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple


class E5Retriever:
    """Dense retrieval using E5-Large-v2 model."""

    def __init__(self, model_name: str = "intfloat/e5-large-v2", device: str = "cpu", batch_size: int = 32):
        """
        Initialize E5 dense retriever.

        Args:
            model_name: Name of the E5 model
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.corpus_emb = None
        self.query_emb = None

    def load_model(self) -> None:
        """Load the E5 model."""
        print(f">à Loading {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_corpus(self, corpus_texts: List[str]) -> torch.Tensor:
        """
        Encode corpus texts with E5 in passage mode.

        Args:
            corpus_texts: List of document texts

        Returns:
            Tensor of corpus embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(">à Encoding corpus with E5 (passage mode)...")
        corpus_passages = [f"passage: {t}" for t in corpus_texts]
        self.corpus_emb = self.model.encode(
            corpus_passages,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return self.corpus_emb

    def encode_queries(self, query_texts: List[str]) -> torch.Tensor:
        """
        Encode query texts with E5 in query mode.

        Args:
            query_texts: List of query texts

        Returns:
            Tensor of query embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print(">à Encoding queries with E5 (query mode)...")
        query_inputs = [f"query: {q}" for q in query_texts]
        self.query_emb = self.model.encode(
            query_inputs,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return self.query_emb

    def retrieve(self, query_emb: torch.Tensor, k: int = 50) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k documents for a query embedding.

        Args:
            query_emb: Query embedding tensor
            k: Number of top documents to retrieve

        Returns:
            Tuple of (doc_ids, scores) as lists
        """
        if self.corpus_emb is None:
            raise ValueError("Corpus not encoded. Call encode_corpus() first.")

        sims = util.cos_sim(query_emb, self.corpus_emb)[0]
        vals, idxs = torch.topk(sims, k=min(k, sims.shape[0]))
        doc_ids = idxs.tolist()
        scores = vals.tolist()
        return doc_ids, scores

    def get_query_embedding(self, index: int) -> torch.Tensor:
        """Get a specific query embedding by index."""
        if self.query_emb is None:
            raise ValueError("Queries not encoded. Call encode_queries() first.")
        return self.query_emb[index]
