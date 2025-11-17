"""CLI to verify a free-form citation against a Google Scholar profile."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .services.google_scholar import GoogleScholarLookupService, GoogleScholarPublication

LOGGER = logging.getLogger("multi_agent_pipeline.verify_google_scholar")


def normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w'-]+", text)}


def author_last_names(authors: Sequence[str]) -> set[str]:
    last_names: set[str] = set()
    for author in authors:
        pieces = [piece for piece in re.split(r"\s+", author.strip()) if piece]
        if pieces:
            last_names.add(pieces[-1].lower())
    return last_names


def score_publication(citation: str, publication: GoogleScholarPublication) -> Dict[str, Any]:
    norm_citation = normalize(citation)
    norm_title = normalize(publication.title)
    if not norm_title:
        return {"score": 0.0, "title_match": False, "author_overlap": 0.0}

    ratio = SequenceMatcher(None, norm_title, norm_citation).ratio()
    title_match = norm_title in norm_citation or ratio >= 0.75

    citation_tokens = extract_tokens(norm_citation)
    author_tokens = author_last_names(publication.authors)
    if not author_tokens or not citation_tokens:
        author_overlap = 0.0
    else:
        hits = author_tokens & citation_tokens
        author_overlap = len(hits) / len(author_tokens)

    score = 0.7 * ratio + 0.3 * author_overlap

    return {
        "score": score,
        "title_match": title_match,
        "title_ratio": ratio,
        "author_overlap": author_overlap,
    }


def find_best_match(
    citation: str, publications: Iterable[GoogleScholarPublication]
) -> Tuple[Optional[GoogleScholarPublication], Dict[str, Any]]:
    best_pub: Optional[GoogleScholarPublication] = None
    best_metrics: Dict[str, Any] = {"score": 0.0, "title_match": False, "title_ratio": 0.0, "author_overlap": 0.0}

    for publication in publications:
        metrics = score_publication(citation, publication)
        if metrics["score"] > best_metrics["score"]:
            best_pub = publication
            best_metrics = metrics

    return best_pub, best_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify whether a citation appears on a Google Scholar profile",
    )
    parser.add_argument("citation", help="Free-form citation text to verify")
    parser.add_argument(
        "--scholar-id",
        required=True,
        help="Google Scholar profile identifier (value of the `user` query parameter)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/googlescholar.php",
        help="Base URL of the googlescholar-api endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Minimum aggregate score to consider the citation verified",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON output",
    )
    return parser


def build_payload(
    citation: str,
    scholar_id: str,
    publications: Sequence[GoogleScholarPublication],
    best_pub: Optional[GoogleScholarPublication],
    metrics: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    matched = bool(best_pub) and (
        metrics.get("title_match")
        or metrics.get("score", 0.0) >= threshold
    )

    payload: Dict[str, Any] = {
        "scholar_id": scholar_id,
        "citation": citation,
        "matched": matched,
        "metrics": metrics,
        "threshold": threshold,
        "publications_considered": len(publications),
    }

    if best_pub:
        payload["best_publication"] = asdict(best_pub)
    else:
        payload["best_publication"] = None

    if not publications:
        payload["reason"] = "Google Scholar returned no publications for the provided profile"
    elif not matched:
        payload["reason"] = "No publication met the title or score threshold"

    return payload


def run(args: argparse.Namespace) -> None:
    service = GoogleScholarLookupService(
        base_url=args.base_url,
        timeout=args.timeout,
    )
    publications = service.lookup(args.scholar_id)

    best_pub, metrics = find_best_match(args.citation, publications)
    payload = build_payload(
        citation=args.citation,
        scholar_id=args.scholar_id,
        publications=publications,
        best_pub=best_pub,
        metrics=metrics,
        threshold=args.score_threshold,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved verification output to {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
