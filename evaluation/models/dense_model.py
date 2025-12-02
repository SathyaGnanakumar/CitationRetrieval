"""Dense retrieval model wrapper for embedding-based models"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .base_model import BaseRetrievalModel


class DenseRetrievalModel(BaseRetrievalModel):
    """
    Dense retrieval using sentence transformers.

    Supports models like:
    - SPECTER2: 'allenai/specter2'
    - E5-Large: 'intfloat/e5-large-v2'
    - Other sentence-transformers models
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize_embeddings: Whether to L2 normalize embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔧 Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_length
        print(f"✅ Model loaded successfully")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings [n_texts, embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus.

        Args:
            query_embedding: Query embedding [embedding_dim]
            corpus_embeddings: Corpus embeddings [n_docs, embedding_dim]

        Returns:
            Similarity scores [n_docs]
        """
        # Cosine similarity (assumes normalized embeddings)
        if self.normalize_embeddings:
            scores = corpus_embeddings @ query_embedding
        else:
            # Compute cosine similarity manually
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            corpus_norms = corpus_embeddings / np.linalg.norm(
                corpus_embeddings, axis=1, keepdims=True
            )
            scores = corpus_norms @ query_norm

        return scores

    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve top-k documents using dense embeddings.

        Args:
            query: Citation context
            corpus: List of candidate papers
            k: Number to retrieve

        Returns:
            Top-k ranked papers
        """
        if not corpus:
            return []

        # Extract texts
        corpus_texts = [item.get('text', '') for item in corpus]

        # Encode query and corpus
        query_embedding = self.encode([query])[0]
        corpus_embeddings = self.encode(corpus_texts)

        # Compute similarities
        scores = self.compute_similarity(query_embedding, corpus_embeddings)

        # Get top-k
        top_k = min(k, len(corpus))
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            result = corpus[idx].copy()
            result['score'] = float(scores[idx])
            result['rank'] = rank
            results.append(result)

        return results

    def precompute_corpus_embeddings(self, corpus: List[Dict]) -> np.ndarray:
        """
        Pre-compute and cache corpus embeddings.
        
        Args:
            corpus: List of corpus entries
            
        Returns:
            Numpy array of embeddings
        """
        corpus_texts = [item.get('text', '') for item in corpus]
        return self.encode(corpus_texts)

    def retrieve_batch(
        self,
        queries: List[str],
        corpus: List[Dict],
        k: int = 10
    ) -> List[List[Dict]]:
        """
        Retrieve top-k documents for multiple queries using dense embeddings.
        
        Args:
            queries: List of citation contexts
            corpus: List of candidate papers
            k: Number to retrieve
            
        Returns:
            List of lists of top-k ranked papers
        """
        if not corpus:
            return [[] for _ in queries]

        # 1. Encode queries
        query_embeddings = self.encode(queries)
        
        # 2. Get corpus embeddings (compute if not cached or size mismatch)
        # Note: In a real system we'd have a more robust caching mechanism.
        # Here we assume if corpus object is same, we can reuse (but we can't easily check identity).
        # So we'll just recompute for now unless we add a specific cache method.
        # Ideally, the evaluator calls precompute_corpus_embeddings explicitly.
        # For now, let's just compute them here.
        corpus_texts = [item.get('text', '') for item in corpus]
        corpus_embeddings = self.encode(corpus_texts)

        # 3. Compute similarities (Matrix Multiplication)
        # query_embeddings: [num_queries, dim]
        # corpus_embeddings: [num_docs, dim]
        # scores: [num_queries, num_docs]
        
        if self.normalize_embeddings:
            scores = query_embeddings @ corpus_embeddings.T
        else:
            # Manual cosine similarity
            q_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            c_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            scores = q_norm @ c_norm.T

        # 4. Get top-k for each query
        top_k = min(k, len(corpus))
        # argsort gives indices of sorted elements. We want descending order.
        # [:, ::-1] reverses the columns
        top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        
        # 5. Build results
        batch_results = []
        for i, row_indices in enumerate(top_indices):
            query_results = []
            for rank, idx in enumerate(row_indices, start=1):
                result = corpus[idx].copy()
                result['score'] = float(scores[i, idx])
                result['rank'] = rank
                query_results.append(result)
            batch_results.append(query_results)

        return batch_results

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'normalize_embeddings': self.normalize_embeddings,
            'device': self.device
        }


# Convenience constructors
def get_specter2_model(**kwargs) -> DenseRetrievalModel:
    """Get SPECTER2 model"""
    return DenseRetrievalModel(
        model_name="allenai/specter2",
        **kwargs
    )


def get_e5_large_model(**kwargs) -> DenseRetrievalModel:
    """Get E5-Large model"""
    return DenseRetrievalModel(
        model_name="intfloat/e5-large-v2",
        **kwargs
    )


def get_e5_base_model(**kwargs) -> DenseRetrievalModel:
    """Get E5-Base model (faster, lower quality)"""
    return DenseRetrievalModel(
        model_name="intfloat/e5-base-v2",
        **kwargs
    )
