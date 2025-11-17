"""Google Scholar lookup service using the unofficial PHP API.

This module integrates with https://github.com/fredrike/googlescholar-api
which scrapes a Google Scholar profile page and returns a JSON payload.
The scraper must be hosted separately (e.g., via the Docker image
documented in that repository). This service simply consumes the JSON
response and converts it into normalized publication objects.

Because the upstream project relies on HTML scraping, it may break
without notice if Google changes their markup. Use sparingly and in
accordance with the Google Scholar terms of service.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GoogleScholarPublication:
    """Normalized representation of a publication on a Scholar profile."""

    title: str
    authors: List[str]
    venue: Optional[str]
    year: Optional[int]
    citations: Optional[int]


class GoogleScholarLookupService:
    """Client for the external googlescholar-api scraper."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/googlescholar.php",
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or requests.Session()

    def lookup(self, scholar_id: str, extra_params: Optional[Dict[str, str]] = None) -> List[GoogleScholarPublication]:
        """Return publications for the given Google Scholar profile.

        Parameters
        ----------
        scholar_id:
            The Google Scholar profile identifier (value of the `user` parameter).
        extra_params:
            Optional dictionary of query parameters forwarded to the scraper
            (e.g., to sort publications).
        """
        if not scholar_id:
            raise ValueError("scholar_id must be provided")

        params: Dict[str, str] = {"user": scholar_id}
        if extra_params:
            params.update(extra_params)

        try:
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            LOGGER.error("Google Scholar lookup failed: %s", exc)
            return []

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - malformed payload
            LOGGER.error("Failed to parse Google Scholar response: %s", exc)
            return []

        return self._parse_payload(payload)

    @staticmethod
    def _parse_payload(payload: Dict) -> List[GoogleScholarPublication]:
        publications: List[GoogleScholarPublication] = []
        for entry in payload.get("publications", []) or []:
            title = (entry.get("title") or "").strip()
            raw_authors = entry.get("authors") or ""
            authors = [author.strip() for author in raw_authors.split(",") if author.strip()]

            venue = entry.get("venue")
            if isinstance(venue, str):
                venue = venue.strip() or None

            year_value = entry.get("year")
            year = GoogleScholarLookupService._coerce_int(year_value)

            citations_value = entry.get("citations")
            citations = GoogleScholarLookupService._coerce_int(citations_value)

            publications.append(
                GoogleScholarPublication(
                    title=title,
                    authors=authors,
                    venue=venue,
                    year=year,
                    citations=citations,
                )
            )
        return publications

    @staticmethod
    def _coerce_int(value: Optional[object]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None
        return None
