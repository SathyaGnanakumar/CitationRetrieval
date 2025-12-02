"""BM25 baseline model wrapper"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

try:  # Optional dependency – Stemmer (PyStemmer)
    import Stemmer  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    Stemmer = None  # type: ignore

from .base_model import BaseRetrievalModel


# Lightweight English stopword list keeps implementation self-contained.
_DEFAULT_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class _BM25Index:
    """Minimal BM25 index independent of numba/llvmlite."""

    def __init__(
        self,
        tokenized_corpus: List[List[str]],
        *,
        k1: float,
        b: float,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._f = [Counter(doc) for doc in tokenized_corpus]
        self._doc_len = [len(doc) for doc in tokenized_corpus]
        self._num_docs = len(tokenized_corpus)
        self._avgdl = sum(self._doc_len) / self._num_docs if self._num_docs else 0.0

        # Document frequencies per term
        df = Counter()
        for freq in self._f:
            df.update(freq.keys())

        self._idf = {
            term: math.log(1 + (self._num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: Iterable[str]) -> List[float]:
        scores: List[float] = []
        for freq, doc_len in zip(self._f, self._doc_len):
            score = 0.0
            for term in query_tokens:
                term_freq = freq.get(term)
                if not term_freq:
                    continue
                idf = self._idf.get(term)
                if idf is None:
                    continue
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_len / (self._avgdl or 1.0))
                score += idf * (term_freq * (self.k1 + 1) / denom)
            scores.append(score)
        return scores


class BM25Model(BaseRetrievalModel):
    """
    BM25 sparse retrieval baseline.

    Implements a lightweight BM25 scorer entirely in Python to avoid the
    bm25s/numba dependency chain, which is problematic on Windows.
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
            use_stemming: Whether to use Porter stemming (requires PyStemmer)
            use_stopwords: Whether to remove English stopwords
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        super().__init__("BM25")
        self.use_stopwords = use_stopwords
        self.stopwords = _DEFAULT_STOPWORDS if use_stopwords else set()

        self.use_stemming = use_stemming and Stemmer is not None
        self._stemmer: Optional[object] = None
        if self.use_stemming:
            # Stemmer.Stemmer returns callable stemWords(list[str]) -> list[str]
            self._stemmer = Stemmer.Stemmer("english")  # type: ignore[arg-type]

        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if self.stopwords:
            tokens = [tok for tok in tokens if tok not in self.stopwords]
        if self._stemmer is not None:
            tokens = self._stemmer.stemWords(tokens)  # type: ignore[attr-defined]
        return tokens

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

        tokenized_corpus = [self._tokenize(item.get("text", "")) for item in corpus]
        index = _BM25Index(tokenized_corpus, k1=self.k1, b=self.b)

        query_tokens = self._tokenize(query)
        scores = index.score(query_tokens)

        top_k = min(k, len(corpus))
        ranked_indices = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            result = corpus[idx].copy()
            result["score"] = float(scores[idx])
            result["rank"] = rank
            results.append(result)

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        corpus: List[Dict],
        k: int = 10
    ) -> List[List[Dict]]:
        """
        Retrieve top-k documents for multiple queries using BM25.
        Optimized to build index only once per batch.
        
        Args:
            queries: List of citation contexts
            corpus: List of candidate papers
            k: Number to retrieve
            
        Returns:
            List of lists of top-k ranked papers
        """
        if not corpus:
            return [[] for _ in queries]

        # 1. Build index ONCE
        tokenized_corpus = [self._tokenize(item.get("text", "")) for item in corpus]
        index = _BM25Index(tokenized_corpus, k1=self.k1, b=self.b)

        # 2. Process all queries
        batch_results = []
        top_k = min(k, len(corpus))
        
        for query in queries:
            query_tokens = self._tokenize(query)
            scores = index.score(query_tokens)
            
            ranked_indices = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            results = []
            for rank, idx in enumerate(ranked_indices, start=1):
                result = corpus[idx].copy()
                result["score"] = float(scores[idx])
                result["rank"] = rank
                results.append(result)
            batch_results.append(results)

        return batch_results

    def get_config(self) -> Dict[str, Any]:
        """Return BM25 configuration"""
        return {
            "model_name": self.model_name,
            "use_stemming": self.use_stemming,
            "use_stopwords": self.use_stopwords,
            "k1": self.k1,
            "b": self.b,
        }
