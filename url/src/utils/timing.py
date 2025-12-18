"""Timing and logging utilities for the retrieval pipeline."""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@contextmanager
def timer(description: str, log_level: int = logging.INFO):
    """
    Context manager for timing code blocks.
    
    Usage:
        with timer("Loading dataset"):
            dataset = load_dataset(path)
    """
    start = time.time()
    logger.log(log_level, f"⏱️  Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(log_level, f"✅ Completed: {description} ({elapsed:.2f}s)")


def timed(description: Optional[str] = None, log_level: int = logging.INFO):
    """
    Decorator for timing functions.
    
    Usage:
        @timed("Query reformulation")
        def query_reformulator(state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        nonlocal description
        if description is None:
            description = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            logger.log(log_level, f"⏱️  Starting: {description}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.log(log_level, f"✅ Completed: {description} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.log(logging.ERROR, f"❌ Failed: {description} ({elapsed:.2f}s) - {e}")
                raise
        
        return wrapper
    return decorator


def setup_logging(level: int = logging.INFO, format_str: Optional[str] = None):
    """
    Configure logging for the entire application.
    
    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string (default: includes timestamp and level)
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
