"""Semantic Scholar lookup service for citation verification."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_DEFAULT_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "venue",
        "year",
        "externalIds",
        "url",
        "authors.name",
        "publicationTypes",
    ]
)


@dataclass(slots=True)
class SemanticScholarPaper:
    """Normalized representation of a paper returned by Semantic Scholar."""

    paper_id: str
    title: str
    abstract: Optional[str]
    venue: Optional[str]
    year: Optional[int]
    external_ids: Dict[str, str]
    url: Optional[str]
    authors: List[str]
    publication_types: List[str]

    def __hash__(self) -> int:
        """Allow use in dictionaries/sets by hashing on the stable paper ID."""
        return hash(self.paper_id)


class SemanticScholarLookupService:
    """Thin client for Semantic Scholar's paper search endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_results: int = 5,
        backoff_seconds: float = 2.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY")
        if not self.api_key:
            LOGGER.warning(
                "S2_API_KEY not configured; Semantic Scholar lookups may be rate limited."
            )
        self.timeout = timeout
        self.max_results = max_results
        self.backoff_seconds = backoff_seconds
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def lookup(self, title: str, year: Optional[int] = None) -> List[SemanticScholarPaper]:
        """Lookup a paper by title (and optional year) on Semantic Scholar."""
        if not title.strip():
            return []

        params = {
            "query": title,
            "limit": str(self.max_results),
            "fields": _DEFAULT_FIELDS,
        }
        if year:
            params["year"] = str(year)

        cache_key = self._cache_key(title, year)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        headers = {"User-Agent": "CitationRetrieval/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    _API_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                papers = self._parse_response(data)
                self._cache[cache_key] = papers
                return papers
            except requests.HTTPError as err:
                status = err.response.status_code if err.response else None
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    sleep_for = self.backoff_seconds * attempt
                    LOGGER.warning(
                        "Semantic Scholar request failed with %s. Retrying in %.1fs (attempt %s/%s)",
                        status,
                        sleep_for,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(sleep_for)
                    continue
                LOGGER.error("Semantic Scholar request failed: %s", err)
                break
            except requests.RequestException as err:
                if attempt < self.max_retries:
                    sleep_for = self.backoff_seconds * attempt
                    LOGGER.warning(
                        "Semantic Scholar request error: %s. Retrying in %.1fs (attempt %s/%s)",
                        err,
                        sleep_for,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(sleep_for)
                    continue
                LOGGER.error("Semantic Scholar request error: %s", err)
                break
            except ValueError as err:
                LOGGER.error("Failed to parse Semantic Scholar response: %s", err)
                break

        return []

    def _parse_response(self, data: Dict) -> List[SemanticScholarPaper]:
        results = []
        for entry in data.get("data", []) or []:
            results.append(
                SemanticScholarPaper(
                    paper_id=entry.get("paperId") or "",
                    title=(entry.get("title") or "").strip(),
                    abstract=entry.get("abstract"),
                    venue=entry.get("venue"),
                    year=entry.get("year"),
                    external_ids=entry.get("externalIds") or {},
                    url=entry.get("url"),
                    authors=[(author.get("name") or "").strip() for author in entry.get("authors") or []],
                    publication_types=entry.get("publicationTypes") or [],
                )
            )
        return results

    _cache: Dict[str, List[SemanticScholarPaper]] = {}

    @classmethod
    def _cache_key(cls, title: str, year: Optional[int]) -> str:
        return f"{title.lower().strip()}::{year or 'any'}"
