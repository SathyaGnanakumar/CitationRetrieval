"""Resource caching utilities for persisting expensive computations."""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_dataset_hash(dataset_path: str) -> str:
    """Generate a hash of the dataset file for cache invalidation."""
    with open(dataset_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def get_cache_path(dataset_path: str, cache_dir: Optional[str] = None) -> Path:
    """
    Get the cache directory path for a dataset.

    Args:
        dataset_path: Path to the dataset file
        cache_dir: Optional custom cache directory (defaults to .cache in dataset parent dir)

    Returns:
        Path to cache directory
    """
    if cache_dir:
        base_cache_dir = Path(cache_dir)
    else:
        # Default: .cache folder next to the dataset
        base_cache_dir = Path(dataset_path).parent / ".cache"

    # Create subdirectory based on dataset hash for versioning
    dataset_hash = _get_dataset_hash(dataset_path)
    cache_path = base_cache_dir / dataset_hash
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path


def save_resources(
    resources: Dict[str, Any],
    dataset_path: str,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Save built resources to disk cache.

    Args:
        resources: The resources dict to cache
        dataset_path: Path to the source dataset
        cache_dir: Optional custom cache directory
    """
    cache_path = get_cache_path(dataset_path, cache_dir)

    logger.info(f"💾 Saving resources to cache: {cache_path}")

    # Save metadata
    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_hash": _get_dataset_hash(dataset_path),
        "corpus_size": len(resources.get("corpus", [])),
        "has_bm25": "bm25" in resources,
        "has_e5": "e5" in resources,
        "has_specter": "specter" in resources,
    }

    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save corpus (lightweight)
    with open(cache_path / "corpus.pkl", "wb") as f:
        pickle.dump(resources.get("corpus", []), f)

    # Save BM25 resources (lightweight)
    if "bm25" in resources:
        logger.debug("  - Saving BM25 index...")
        bm25_data = {
            "ids": resources["bm25"]["ids"],
            "titles": resources["bm25"]["titles"],
            "texts": resources["bm25"]["texts"],
        }
        with open(cache_path / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)
        # Save the BM25 index separately using its native save method
        resources["bm25"]["index"].save(str(cache_path / "bm25_index"))

    # Save E5 resources (embeddings are heavy!)
    if "e5" in resources:
        logger.debug("  - Saving E5 embeddings...")
        import torch

        e5_data = {
            "ids": resources["e5"]["ids"],
            "titles": resources["e5"]["titles"],
            "texts": resources["e5"]["texts"],
            "model_name": resources["e5"]["model_name"],
            "device": resources["e5"]["device"],
        }
        with open(cache_path / "e5.pkl", "wb") as f:
            pickle.dump(e5_data, f)
        # Save embeddings as tensor
        torch.save(resources["e5"]["corpus_embeddings"], cache_path / "e5_embeddings.pt")

    # Save SPECTER resources (embeddings are heavy!)
    if "specter" in resources:
        logger.debug("  - Saving SPECTER embeddings...")
        import torch

        specter_data = {
            "ids": resources["specter"]["ids"],
            "titles": resources["specter"]["titles"],
            "texts": resources["specter"]["texts"],
            "model_name": resources["specter"]["model_name"],
            "device": resources["specter"]["device"],
        }
        with open(cache_path / "specter.pkl", "wb") as f:
            pickle.dump(specter_data, f)
        # Save embeddings as tensor
        torch.save(resources["specter"]["corpus_embeddings"], cache_path / "specter_embeddings.pt")

    logger.info(f"✅ Resources cached successfully!")


def load_resources(
    dataset_path: str,
    cache_dir: Optional[str] = None,
    enable_bm25: bool = True,
    enable_e5: bool = True,
    enable_specter: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load resources from disk cache if available.

    Args:
        dataset_path: Path to the source dataset
        cache_dir: Optional custom cache directory
        enable_bm25: Whether to load BM25 resources
        enable_e5: Whether to load E5 resources
        enable_specter: Whether to load SPECTER resources

    Returns:
        Loaded resources dict, or None if cache doesn't exist or is invalid
    """
    cache_path = get_cache_path(dataset_path, cache_dir)

    # Check if cache exists
    metadata_file = cache_path / "metadata.json"
    if not metadata_file.exists():
        logger.info("📦 No cache found, will build from scratch")
        return None

    # Verify cache is still valid
    with open(metadata_file) as f:
        metadata = json.load(f)

    current_hash = _get_dataset_hash(dataset_path)
    if metadata["dataset_hash"] != current_hash:
        logger.warning("⚠️  Dataset has changed, cache is stale. Rebuilding...")
        return None

    logger.info(f"📦 Loading resources from cache: {cache_path}")

    resources = {}

    # Load corpus
    with open(cache_path / "corpus.pkl", "rb") as f:
        resources["corpus"] = pickle.load(f)
    logger.debug(f"  - Loaded corpus ({len(resources['corpus'])} docs)")

    # Load BM25
    if enable_bm25 and (cache_path / "bm25.pkl").exists():
        logger.debug("  - Loading BM25 index...")
        import bm25s
        import Stemmer

        with open(cache_path / "bm25.pkl", "rb") as f:
            bm25_data = pickle.load(f)

        # Load the index
        bm25_index = bm25s.BM25.load(str(cache_path / "bm25_index"))

        resources["bm25"] = {
            **bm25_data,
            "index": bm25_index,
            "stemmer": Stemmer.Stemmer("english"),
        }

    # Load E5
    if enable_e5 and (cache_path / "e5.pkl").exists():
        logger.debug("  - Loading E5 embeddings...")
        import torch
        from sentence_transformers import SentenceTransformer

        with open(cache_path / "e5.pkl", "rb") as f:
            e5_data = pickle.load(f)

        # Load model
        model = SentenceTransformer(e5_data["model_name"])
        device = e5_data["device"]
        if device != "cpu":
            model.to(device)

        # Load embeddings
        corpus_embeddings = torch.load(cache_path / "e5_embeddings.pt")

        resources["e5"] = {
            **e5_data,
            "model": model,
            "corpus_embeddings": corpus_embeddings,
        }

    # Load SPECTER
    if enable_specter and (cache_path / "specter.pkl").exists():
        logger.debug("  - Loading SPECTER embeddings...")
        import torch
        from transformers import AutoModel, AutoTokenizer

        with open(cache_path / "specter.pkl", "rb") as f:
            specter_data = pickle.load(f)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(specter_data["model_name"])
        model = AutoModel.from_pretrained(specter_data["model_name"])

        # Load embeddings
        corpus_embeddings = torch.load(cache_path / "specter_embeddings.pt")

        resources["specter"] = {
            **specter_data,
            "model": model,
            "tokenizer": tokenizer,
            "corpus_embeddings": corpus_embeddings,
        }

    logger.info(f"✅ Resources loaded from cache!")
    return resources


def clear_cache(dataset_path: str, cache_dir: Optional[str] = None) -> None:
    """
    Clear the cache for a dataset.

    Args:
        dataset_path: Path to the source dataset
        cache_dir: Optional custom cache directory
    """
    import shutil

    cache_path = get_cache_path(dataset_path, cache_dir)
    if cache_path.exists():
        logger.info(f"🗑️  Clearing cache: {cache_path}")
        shutil.rmtree(cache_path)
        logger.info("✅ Cache cleared")
    else:
        logger.info("📦 No cache to clear")
