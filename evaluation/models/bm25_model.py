"""BM25 baseline model wrapper"""

from typing import List, Dict, Any
import bm25s
import Stemmer

from .base_model import BaseRetrievalModel


class BM25Model(BaseRetrievalModel):
    """
    BM25 sparse retrieval baseline.

    Uses bm25s library with English stemming and stopword removal.
    """

    def __init__(
        self,
        use_stemming: bool = True,
        use_stopwords: bool = True,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Args:
            use_stemming: Whether to use Porter stemmer
            use_stopwords: Whether to remove English stopwords
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        super().__init__("BM25")
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.k1 = k1
        self.b = b

        # Initialize stemmer
        self.stemmer = Stemmer.Stemmer("english") if use_stemming else None

    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve top-k documents using BM25.

        Args:
            query: Citation context
            corpus: List of candidate papers
            k: Number to retrieve

        Returns:
            Top-k ranked papers
        """
        if not corpus:
            return []

        # Extract texts from corpus
        corpus_texts = [item.get('text', '') for item in corpus]

        # Tokenize corpus
        stopwords = "en" if self.use_stopwords else None
        corpus_tokens = bm25s.tokenize(
            corpus_texts,
            stopwords=stopwords,
            stemmer=self.stemmer
        )

        # Build BM25 index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # Tokenize query
        query_tokens = bm25s.tokenize(
            query,
            stopwords=stopwords,
            stemmer=self.stemmer
        )

        # Retrieve
        doc_ids, scores = retriever.retrieve(query_tokens, k=min(k, len(corpus)))

        # Extract results (bm25s returns 2D arrays)
        doc_ids = doc_ids[0] if len(doc_ids.shape) > 1 else doc_ids
        scores = scores[0] if len(scores.shape) > 1 else scores

        # Build results
        results = []
        for doc_id, score in zip(doc_ids, scores):
            result = corpus[doc_id].copy()
            result['score'] = float(score)
            result['rank'] = len(results) + 1
            results.append(result)

        return results

    def get_config(self) -> Dict[str, Any]:
        """Return BM25 configuration"""
        return {
            'model_name': self.model_name,
            'use_stemming': self.use_stemming,
            'use_stopwords': self.use_stopwords,
            'k1': self.k1,
            'b': self.b
        }
