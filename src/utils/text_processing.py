"""Text processing utilities for citation retrieval."""

import re
import difflib
import numpy as np
from typing import List, Dict


# Regex patterns
_title_braces = re.compile(r"[{}]")
_ws_multi = re.compile(r"\s+")
CITE_TAG_RE = re.compile(r"<\|cite_(\d+)\|>")


def normalize_title(t: str) -> str:
    """
    Normalize a title by removing braces, converting to lowercase, and normalizing whitespace.

    Args:
        t: Title string to normalize

    Returns:
        Normalized title string
    """
    if t is None:
        return ""
    t = _title_braces.sub("", t)
    t = t.strip().lower()
    t = _ws_multi.sub(" ", t)
    return t


def clean_text(text: str) -> str:
    """
    Clean text by removing LaTeX macros and normalizing whitespace.

    Args:
        text: Text to clean

    Returns:
        Cleaned text string
    """
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)  # remove LaTeX macros
    text = _ws_multi.sub(" ", text).strip()
    return text


def extract_contexts(paper_text: str, context_window: int = 400) -> List[tuple]:
    """
    Extract citation contexts from paper text.

    Args:
        paper_text: Full paper text with citation tags
        context_window: Number of characters before and after citation tag

    Returns:
        List of tuples (citation_tag, context_text)
    """
    contexts = []
    for m in CITE_TAG_RE.finditer(paper_text):
        start, end = m.span()
        ctx = paper_text[max(0, start - context_window) : min(len(paper_text), end + context_window)]
        contexts.append((m.group(0), clean_text(ctx)))
    return contexts


def fuzzy_hit(true_title: str, retrieved_titles: List[str], threshold: float = 0.90) -> bool:
    """
    Check if the true title fuzzy-matches any retrieved titles.

    Args:
        true_title: Ground truth title
        retrieved_titles: List of retrieved candidate titles
        threshold: Similarity threshold (0-1)

    Returns:
        True if a match is found, False otherwise
    """
    t = normalize_title(true_title)
    for cand in retrieved_titles:
        if t == normalize_title(cand):
            return True
        if difflib.SequenceMatcher(None, t, normalize_title(cand)).ratio() >= threshold:
            return True
    return False


def per_query_minmax(scores_dict: Dict[int, float]) -> Dict[int, float]:
    """
    Apply per-query min-max normalization to scores.

    Args:
        scores_dict: Dictionary mapping doc IDs to scores

    Returns:
        Dictionary with normalized scores [0, 1]
    """
    vals = np.array(list(scores_dict.values()), dtype=float)
    if len(vals) == 0:
        return {}
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax - vmin < 1e-12:
        return {k: 0.0 for k in scores_dict}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in scores_dict.items()}
