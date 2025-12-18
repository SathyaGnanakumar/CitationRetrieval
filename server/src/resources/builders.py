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


def build_llm_reranker_resources(
    inference_engine: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build LLM reranker resources (load model once, reuse across examples).

    Args:
        inference_engine: "ollama" or "huggingface" (defaults to INFERENCE_ENGINE env var)
        model_name: Model identifier (e.g., "gemma3:4b" for Ollama, "google/gemma-2-9b-it" for HF)

    Returns:
        Dict with llm_model, inference_engine, model_name
    """
    import os

    if inference_engine is None:
        inference_engine = os.getenv("INFERENCE_ENGINE", "ollama").lower()

    if model_name is None:
        model_name = os.getenv("LOCAL_LLM", "gemma3:4b")

    logger.info(f"🔨 Building LLM Reranker resources...")
    logger.info(f"   Inference Engine: {inference_engine}")
    logger.info(f"   Model: {model_name}")

    if inference_engine == "openai":
        # OpenAI - cloud-based, no local loading needed
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        logger.info(f"🔄 Initializing OpenAI with model: {model_name}")
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        logger.info(f"✅ OpenAI ready!")

    elif inference_engine == "ollama":
        # Ollama - local inference server
        from langchain_ollama import ChatOllama

        logger.info(f"🔄 Initializing Ollama with model: {model_name}")
        llm = ChatOllama(model=model_name, temperature=0)
        logger.info(f"✅ Ollama ready!")

    else:
        # Hugging Face - load once and cache
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_huggingface import HuggingFacePipeline
        import torch

        logger.info(f"🔄 Loading Hugging Face model: {model_name}")
        logger.info(f"   This will take a few minutes on first run...")

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=1024,
            do_sample=False,
        )

        llm = HuggingFacePipeline(pipeline=gen)
        logger.info(f"✅ Hugging Face model loaded and cached!")

    return {
        "llm_model": llm,
        "inference_engine": inference_engine,
        "model_name": model_name,
    }


def build_inmemory_resources(
    docs: List[Dict[str, Any]],
    *,
    enable_bm25: bool = True,
    enable_e5: bool = True,
    enable_specter: bool = True,
    enable_llm_reranker: bool = False,
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
    if enable_llm_reranker:
        logger.info("=" * 60)
        resources["llm_reranker"] = build_llm_reranker_resources()

    logger.info("=" * 60)
    logger.info("✅ All retrieval resources built successfully!")
    return resources
