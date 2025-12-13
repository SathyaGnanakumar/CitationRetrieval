"""Common preprocessing functions for testing datasets."""

import json
import re
from typing import Dict, List, Any


def clean_text(text: str) -> str:
    """Remove LaTeX commands and extra whitespace.

    Args:
        text: Input text that may contain LaTeX commands

    Returns:
        Cleaned text with LaTeX commands removed and normalized whitespace
    """
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_from_paper(paper_text: str, max_length: int = 200) -> str:
    """Extract a query from the paper text (first few sentences).

    Args:
        paper_text: Full text of the paper
        max_length: Maximum length of the extracted query

    Returns:
        Cleaned query text extracted from the beginning of the paper
    """
    sentences = paper_text.split(". ")[:2]
    query = ". ".join(sentences)
    if len(query) > max_length:
        query = query[:max_length]
    return clean_text(query)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        List of paper dictionaries from the dataset
    """
    with open(dataset_path) as f:
        data = json.load(f)
    return data


def run_retriever(
    retriever_name: str, query: str, dataset_path: str = None
) -> List[Dict[str, Any]]:
    """Run the specified retriever and return candidate papers.

    Args:
        retriever_name: Name of the retriever to use ("bm25", "e5", or "specter")
        query: Query string to search for
        dataset_path: Optional path to dataset (currently not used but kept for API compatibility)

    Returns:
        List of candidate papers with 'title' and 'score' keys, sorted by score (descending)

    Raises:
        ValueError: If retriever_name is not one of the supported retrievers
    """
    if retriever_name == "bm25":
        from src.agents.retrievers.bm25_agent import bm25, corpus_titles, corpus_texts, stemmer
        import bm25s

        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
        doc_ids, scores = bm25.retrieve(query_tokens, k=10)
        doc_ids = doc_ids[0]
        scores = scores[0]

        results = [
            {"title": corpus_titles[i], "score": float(scores[idx])}
            for idx, i in enumerate(doc_ids)
        ]

    elif retriever_name == "e5":
        from src.agents.retrievers.e5_agent import model, corpus_embeddings, corpus_titles
        from sentence_transformers import util
        import torch

        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_10 = torch.topk(scores, k=10)

        results = [
            {"title": corpus_titles[idx], "score": float(score)}
            for score, idx in zip(top_10.values, top_10.indices)
        ]

    elif retriever_name == "specter":
        from src.agents.retrievers.specter_agent import (
            model,
            tokenizer,
            corpus_embeddings,
            corpus_titles,
            device,
        )
        import torch

        if device == "cuda":
            model.to("cuda")

        inputs = tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        if device == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            q_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu()

        model.to("cpu")

        scores = torch.cosine_similarity(
            q_emb.expand_as(corpus_embeddings), corpus_embeddings, dim=1
        )
        top_indices = torch.topk(scores, k=10).indices.tolist()

        results = [{"title": corpus_titles[i], "score": float(scores[i])} for i in top_indices]

    else:
        raise ValueError(f"Unknown retriever: {retriever_name}. Choose from: bm25, e5, specter")

    return results
