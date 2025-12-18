"""
Pytest configuration for the test suite.

Defines custom markers and shared fixtures.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require dataset, models, may be slow)",
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as fast unit tests (no external dependencies)"
    )
