"""BM25 sparse retrieval implementation."""

import bm25s
import Stemmer
from typing import List, Tuple
import numpy as np


class BM25Retriever:
    """BM25 sparse retrieval using bm25s library."""

    def __init__(self, language: str = "english", stopwords: str = "en"):
        """
        Initialize BM25 retriever.

        Args:
            language: Language for stemmer
            stopwords: Stopwords language code
        """
        self.language = language
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer(language)
        self.retriever = bm25s.BM25()
        self.corpus_tokens = None
        self.is_indexed = False

    def index(self, corpus_texts: List[str]) -> None:
        """
        Build BM25 index from corpus texts.

        Args:
            corpus_texts: List of document texts to index
        """
        print("⚙️  Building BM25 index...")
        self.corpus_tokens = bm25s.tokenize(
            corpus_texts, stopwords=self.stopwords, stemmer=self.stemmer
        )
        self.retriever.index(self.corpus_tokens)
        self.is_indexed = True
        print("✅ BM25 index ready")

    def retrieve(
        self, query: str, k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text
            k: Number of top documents to retrieve

        Returns:
            Tuple of (doc_ids, scores) as numpy arrays
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call index() first.")

        q_tokens = bm25s.tokenize(query, stopwords=self.stopwords, stemmer=self.stemmer)
        doc_ids, scores = self.retriever.retrieve(q_tokens, k=k)
        return doc_ids[0], scores[0]
