from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _split_docs(docs: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    ids: List[str] = []
    titles: List[str] = []
    texts: List[str] = []

    for d in docs:
        ids.append(str(d.get("id", "")))
        titles.append(d.get("title", "") or "")
        texts.append(d.get("text", "") or "")

    return ids, titles, texts


def build_bm25_resources(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import bm25s
    import Stemmer

    logger.info(f"🔨 Building BM25 index for {len(docs)} documents...")
    ids, titles, texts = _split_docs(docs)

    stemmer = Stemmer.Stemmer("english")
    tokenized_corpus = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)

    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)

    logger.info(f"✅ BM25 index built successfully")

    return {
        "ids": ids,
        "titles": titles,
        "texts": texts,
        "stemmer": stemmer,
        "index": bm25,
    }


def build_e5_resources(
    docs: List[Dict[str, Any]],
    *,
    model_name: str = "intfloat/e5-large-v2",
    device: Optional[str] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"🔨 Building E5 embeddings for {len(docs)} documents...")
    ids, titles, texts = _split_docs(docs)

    logger.info(f"📥 Loading E5 model: {model_name}")
    model = SentenceTransformer(model_name)
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    model.to(device)
    logger.info(f"💻 Using device: {device}")

    logger.info(f"🔢 Encoding {len(texts)} texts (batch_size={batch_size})...")
    corpus_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    logger.info(f"✅ E5 embeddings built successfully ({corpus_embeddings.shape})")

    return {
        "ids": ids,
        "titles": titles,
        "texts": texts,
        "model_name": model_name,
        "device": device,
        "model": model,
        "corpus_embeddings": corpus_embeddings,
    }


def build_specter_resources(
    docs: List[Dict[str, Any]],
    *,
    model_name: str = "allenai/specter2_base",
    device: Optional[str] = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    logger.info(f"🔨 Building SPECTER embeddings for {len(docs)} documents...")
    ids, titles, texts = _split_docs(docs)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    logger.info(f"💻 Using device: {device}")

    logger.info(f"📥 Loading SPECTER model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device == "cuda":
        model.to("cuda")

    num_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(
        f"🔢 Encoding {len(texts)} texts in {num_batches} batches (batch_size={batch_size})..."
    )

    corpus_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", unit="batch"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        if device == "cuda":
            inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        corpus_embeddings.append(outputs.cpu())

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    model.to("cpu")

    logger.info(f"✅ SPECTER embeddings built successfully ({corpus_embeddings.shape})")

    return {
        "ids": ids,
        "titles": titles,
        "texts": texts,
        "model_name": model_name,
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "corpus_embeddings": corpus_embeddings,
    }


def build_inmemory_resources(
    docs: List[Dict[str, Any]],
    *,
    enable_bm25: bool = True,
    enable_e5: bool = True,
    enable_specter: bool = True,
    e5_model_name: str = "intfloat/e5-large-v2",
    specter_model_name: str = "allenai/specter2_base",
) -> Dict[str, Any]:
    """
    Convenience builder that creates a single `resources` dict compatible with the workflow.

    This is intended for dev-mode and evals; production can inject external backend clients.
    """

    resources: Dict[str, Any] = {"corpus": docs}

    if enable_bm25:
        logger.info("=" * 60)
        resources["bm25"] = build_bm25_resources(docs)
    if enable_e5:
        logger.info("=" * 60)
        resources["e5"] = build_e5_resources(docs, model_name=e5_model_name)
    if enable_specter:
        logger.info("=" * 60)
        resources["specter"] = build_specter_resources(docs, model_name=specter_model_name)

    logger.info("=" * 60)
    logger.info("✅ All retrieval resources built successfully!")
    return resources
