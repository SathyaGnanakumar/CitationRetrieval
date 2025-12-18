"""
Unified evaluation CLI for the retrieval system.

This script consolidates multiple evaluation modes:
1. pipeline - Full workflow evaluation with reranking
2. baselines - Individual retriever recall metrics
3. retrievers - Retriever testing and benchmarking
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import bm25s
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add parent directory to path for datasets import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.scholarcopilot import build_citation_corpus, load_dataset
from src.agents.retrievers.e5_agent import E5Retriever
from src.agents.retrievers.specter_agent import SPECTERRetriever
from src.evaluation.metrics import evaluate_retrieval
from src.resources.builders import (
    build_bm25_resources,
    build_e5_resources,
    build_inmemory_resources,
    build_specter_resources,
)
from src.resources.cache import load_resources, save_resources
from src.utils.timing import setup_logging, timer
from src.workflow import RetrievalWorkflow

logger = logging.getLogger(__name__)


# ============================================================================
# MODE 1: PIPELINE EVALUATION (Full workflow with reranking)
# ============================================================================


def extract_query_from_paper(paper_text: str, max_words: int = 50) -> str:
    """Extract a query from the paper text (first few sentences)."""
    words = paper_text.split()[:max_words]
    return " ".join(words)


def get_ground_truth_ids(paper: Dict[str, Any]) -> Set[str]:
    """
    Extract ground truth citation IDs from a paper's bib_info.

    Args:
        paper: Paper dict with 'bib_info' field

    Returns:
        Set of citation IDs that appear in this paper
    """
    relevant_ids = set()

    bib_info = paper.get("bib_info", {}) or {}
    if not isinstance(bib_info, dict):
        return relevant_ids

    for entries in bib_info.values():
        if not isinstance(entries, list):
            continue
        for citation in entries:
            if isinstance(citation, dict):
                # Try multiple ID fields
                citation_id = (
                    citation.get("citation_key") or citation.get("paper_id") or citation.get("id")
                )
                if citation_id:
                    relevant_ids.add(str(citation_id))

    return relevant_ids


def evaluate_single_query(
    workflow: RetrievalWorkflow,
    resources: Dict[str, Any],
    paper: Dict[str, Any],
    k: int = 20,
) -> Dict[str, Any]:
    """
    Evaluate the pipeline on a single query paper.

    Args:
        workflow: The retrieval workflow
        resources: Built resources (indexes, embeddings)
        paper: Query paper
        k: Number of results to retrieve

    Returns:
        Dictionary with metrics and results
    """
    # Extract query from paper
    paper_text = paper.get("paper", "")
    if not paper_text:
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

    query = extract_query_from_paper(paper_text)

    # Get ground truth
    relevant_ids = get_ground_truth_ids(paper)

    if not relevant_ids:
        logger.debug(f"Skipping paper {paper.get('paper_id', 'unknown')} - no citations")
        return None

    # Run workflow
    try:
        final_state = workflow.run(
            {
                "messages": [HumanMessage(content=query)],
                "resources": resources,
                "config": {"k": k},
            }
        )
    except Exception as e:
        logger.error(f"Error running workflow: {e}")
        return None

    # Get ranked papers
    ranked_papers = final_state.get("ranked_papers", [])

    if not ranked_papers:
        logger.warning("No papers retrieved")
        return None

    # Evaluate
    metrics = evaluate_retrieval(ranked_papers, relevant_ids, k_values=[5, 10, 20])

    return {
        "query": query[:100] + "..." if len(query) > 100 else query,
        "paper_id": paper.get("paper_id", "unknown"),
        "paper_title": paper.get("title", ""),
        "relevant_count": len(relevant_ids),
        "retrieved_count": len(ranked_papers),
        "metrics": metrics,
        "ranked_papers": ranked_papers,
    }


def run_pipeline_evaluation(
    dataset_path: str,
    num_queries: int = None,
    k: int = 20,
    bm25_only: bool = False,
    no_e5: bool = False,
    no_specter: bool = False,
    use_cache: bool = True,
    use_llm_reranker: bool = False,
) -> Dict[str, Any]:
    """
    Run full pipeline evaluation on the dataset.

    Args:
        dataset_path: Path to the dataset
        num_queries: Number of queries to evaluate (None = all)
        k: Number of results to retrieve per query
        bm25_only: Only use BM25 (skip dense embeddings)
        no_e5: Skip E5 embeddings
        no_specter: Skip SPECTER embeddings
        use_cache: Whether to use resource caching
        use_llm_reranker: Use LLM-based reranker

    Returns:
        Aggregated evaluation results
    """
    logger.info("=" * 70)
    logger.info("RETRIEVAL PIPELINE EVALUATION")
    logger.info("=" * 70)

    # Load dataset
    with timer("Loading dataset"):
        dataset = load_dataset(dataset_path)

    logger.info(f"📚 Loaded {len(dataset)} papers")

    # Build resources
    enable_e5 = not (bm25_only or no_e5)
    enable_specter = not (bm25_only or no_specter)

    if use_cache:
        resources = load_resources(
            dataset_path,
            enable_bm25=True,
            enable_e5=enable_e5,
            enable_specter=enable_specter,
        )

    if not use_cache or not resources:
        with timer("Building citation corpus"):
            docs = build_citation_corpus(dataset)
            logger.info(f"📄 Corpus size: {len(docs)} documents")

        with timer("Building retrieval resources"):
            resources = build_inmemory_resources(
                docs,
                enable_bm25=True,
                enable_e5=enable_e5,
                enable_specter=enable_specter,
            )

        if use_cache:
            with timer("Saving resources to cache"):
                save_resources(resources, dataset_path)
    else:
        logger.info(f"📦 Loaded resources from cache")

    # Initialize workflow
    workflow = RetrievalWorkflow(use_llm_reranker=use_llm_reranker)

    if use_llm_reranker:
        logger.info("🤖 Using LLM-based reranker")

    # Evaluate on subset of papers
    papers_to_evaluate = dataset[:num_queries] if num_queries else dataset
    logger.info(f"\n🔍 Evaluating on {len(papers_to_evaluate)} queries...")

    all_metrics = []
    successful_queries = 0

    for i, paper in enumerate(papers_to_evaluate, 1):
        logger.info(f"\n[{i}/{len(papers_to_evaluate)}] {paper.get('title', '')[:60]}...")

        result = evaluate_single_query(workflow, resources, paper, k=k)

        if result:
            all_metrics.append(result["metrics"])
            successful_queries += 1

            # Log individual results
            metrics = result["metrics"]
            logger.info(
                f"  R@5={metrics['R@5']:.3f}, R@10={metrics['R@10']:.3f}, MRR={metrics['MRR']:.3f}"
            )
            logger.info(f"  Found {metrics['hits']}/{metrics['total_relevant']} relevant papers")

    # Aggregate metrics
    if not all_metrics:
        logger.error("No successful queries!")
        return {}

    avg_metrics = {
        "R@5": sum(m["R@5"] for m in all_metrics) / len(all_metrics),
        "R@10": sum(m["R@10"] for m in all_metrics) / len(all_metrics),
        "R@20": sum(m["R@20"] for m in all_metrics) / len(all_metrics),
        "MRR": sum(m["MRR"] for m in all_metrics) / len(all_metrics),
        "total_queries": len(papers_to_evaluate),
        "successful_queries": successful_queries,
    }

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Evaluated {successful_queries}/{len(papers_to_evaluate)} queries")
    logger.info("")
    logger.info(f"📊 Average Metrics:")
    logger.info(f"  • Recall@5:  {avg_metrics['R@5']:.4f}")
    logger.info(f"  • Recall@10: {avg_metrics['R@10']:.4f}")
    logger.info(f"  • Recall@20: {avg_metrics['R@20']:.4f}")
    logger.info(f"  • MRR:       {avg_metrics['MRR']:.4f}")
    logger.info("=" * 70)

    return avg_metrics


# ============================================================================
# MODE 2: BASELINES EVALUATION (Individual retrievers)
# ============================================================================


def recall_at_k(retrieved_ids, gold_ids, k):
    """Calculate if any gold ID is in top-k retrieved IDs."""
    return int(any(g in retrieved_ids[:k] for g in gold_ids))


def run_baselines_evaluation(
    dataset_path: str,
    retrievers: List[str],
    k_values: tuple = (1, 5, 10),
    window: int = 200,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run baseline evaluation for individual retrievers.

    Args:
        dataset_path: Path to the dataset
        retrievers: List of retrievers to evaluate (bm25, e5, specter)
        k_values: Tuple of k values for recall@k
        window: Context window around citations
        output_dir: Output directory for results

    Returns:
        Evaluation results
    """
    logger.info("=" * 70)
    logger.info("BASELINES EVALUATION")
    logger.info("=" * 70)

    # Load dataset
    with open(dataset_path) as f:
        content = f.read().strip()
        raw_data = json.loads(content) if content.startswith("[") else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]

    logger.info(f"Loaded {len(raw_data)} source papers")

    # Build mapping: paper_id → citation_corpus_id
    paper_to_corpus_id: Dict[str, str] = {}
    for entry in raw_data:
        for records in entry["bib_info"].values():
            for rec in records:
                cid = rec.get("citation_corpus_id")
                if cid:
                    paper_to_corpus_id[entry["paper_id"]] = cid

    logger.info(f"Mapped {len(paper_to_corpus_id)} papers to citation_corpus_id")

    # Build corpus docs
    docs: List[Dict[str, Any]] = []
    for entry in raw_data:
        cid = paper_to_corpus_id.get(entry["paper_id"])
        if cid is None:
            continue
        docs.append({
            "id": cid,
            "title": entry.get("title", ""),
            "text": entry["paper"],
        })

    logger.info(f"Built corpus with {len(docs)} documents")

    # Extract evaluation examples
    CITE_PATTERN = re.compile(r"<\|cite_(\d+)\|>")

    def extract_examples(entry: Dict[str, Any], window: int):
        text = entry["paper"]
        bib = entry["bib_info"]
        examples = []

        for match in CITE_PATTERN.finditer(text):
            cite_token = match.group(0)
            if cite_token not in bib:
                continue

            gold_ids = [
                rec["citation_corpus_id"]
                for rec in bib[cite_token]
                if "citation_corpus_id" in rec
            ]
            if not gold_ids:
                continue

            s, e = match.span()
            ctx = (text[max(0, s - window):s] + text[e:e + window]).strip()
            if ctx:
                examples.append({
                    "query": ctx,
                    "gold_ids": gold_ids,
                })

        return examples

    eval_examples = []
    for entry in raw_data:
        eval_examples.extend(extract_examples(entry, window))

    if not eval_examples:
        raise RuntimeError("No evaluation queries extracted")

    queries = [e["query"] for e in eval_examples]
    gold_ids_list = [e["gold_ids"] for e in eval_examples]
    N = len(eval_examples)

    logger.info(f"Evaluating on {N} citation queries")

    # Evaluate retrievers
    results = {}
    latency = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BM25
    if "bm25" in retrievers:
        logger.info("Evaluating BM25...")
        t0 = time.time()
        bm25_res = build_bm25_resources(docs)
        latency["BM25_build"] = time.time() - t0

        t0 = time.time()
        bm25_hits = {k: 0 for k in k_values}
        for ex in eval_examples:
            q_tokens = bm25s.tokenize(
                ex["query"], stopwords="en", stemmer=bm25_res["stemmer"]
            )
            doc_ids, _ = bm25_res["index"].retrieve(q_tokens, k=max(k_values))
            retrieved = [bm25_res["ids"][i] for i in doc_ids[0]]
            for k in k_values:
                bm25_hits[k] += recall_at_k(retrieved, ex["gold_ids"], k)
        latency["BM25_eval"] = time.time() - t0
        results["BM25"] = bm25_hits

    # E5
    if "e5" in retrievers:
        logger.info("Evaluating E5 (batched)...")
        t0 = time.time()
        e5_res = build_e5_resources(docs)
        latency["E5_build"] = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            q_embs = e5_res["model"].encode(
                queries,
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        scores = q_embs @ e5_res["corpus_embeddings"].T

        e5_hits = {k: 0 for k in k_values}
        for i in range(N):
            topk = torch.topk(scores[i], k=max(k_values)).indices.tolist()
            retrieved = [e5_res["ids"][j] for j in topk]
            for k in k_values:
                e5_hits[k] += recall_at_k(retrieved, gold_ids_list[i], k)
        latency["E5_eval"] = time.time() - t0
        results["E5"] = e5_hits

    # SPECTER
    if "specter" in retrievers:
        logger.info("Evaluating SPECTER (batched, GPU)...")
        t0 = time.time()
        specter_res = build_specter_resources(docs)
        specter_res["model"].to(device)
        latency["SPECTER_build"] = time.time() - t0

        t0 = time.time()
        q_embs = []
        with torch.no_grad():
            for i in range(0, N, 32):
                batch = queries[i:i + 32]
                inputs = specter_res["tokenizer"](
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                emb = specter_res["model"](**inputs).last_hidden_state.mean(dim=1)
                q_embs.append(emb.cpu())

        q_embs = torch.nn.functional.normalize(torch.cat(q_embs), dim=1)
        corpus_embs = torch.nn.functional.normalize(
            specter_res["corpus_embeddings"], dim=1
        )
        scores = q_embs @ corpus_embs.T

        specter_hits = {k: 0 for k in k_values}
        for i in range(N):
            topk = torch.topk(scores[i], k=max(k_values)).indices.tolist()
            retrieved = [specter_res["ids"][j] for j in topk]
            for k in k_values:
                specter_hits[k] += recall_at_k(retrieved, gold_ids_list[i], k)
        latency["SPECTER_eval"] = time.time() - t0
        results["SPECTER"] = specter_hits

    # Aggregate metrics
    recall = {
        name: {f"Recall@{k}": results[name][k] / N for k in k_values}
        for name in results
    }

    logger.info("\n=== FINAL RESULTS ===")
    for name, scores in recall.items():
        logger.info(f"{name}: {scores}")

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    recall_df = pd.DataFrame(recall).T
    recall_df.to_csv(out_dir / "retrieval_recall_table.csv")
    logger.info(f"Saved recall table to {out_dir / 'retrieval_recall_table.csv'}")

    lat_df = pd.DataFrame.from_dict(latency, orient="index", columns=["seconds"])
    lat_df.to_csv(out_dir / "retrieval_latency.csv")
    logger.info(f"Saved latency table to {out_dir / 'retrieval_latency.csv'}")

    # Plot if we have Recall@5
    if 5 in k_values and len(results) > 1:
        plt.figure()
        methods = list(results.keys())
        x = [latency[f"{m}_eval"] for m in methods]
        y = [recall[m]["Recall@5"] for m in methods]

        plt.scatter(x, y)
        for i, m in enumerate(methods):
            plt.annotate(m, (x[i], y[i]))

        plt.xlabel("Evaluation Latency (seconds)")
        plt.ylabel("Recall@5")
        plt.title("Latency vs Recall@5")
        plt.grid(True)
        plt.savefig(out_dir / "latency_vs_recall.png", dpi=300)
        plt.close()
        logger.info(f"Saved latency plot to {out_dir / 'latency_vs_recall.png'}")

    return {"recall": recall, "latency": latency}


# ============================================================================
# MODE 3: RETRIEVERS TESTING (Batch query testing)
# ============================================================================


def run_retrievers_test(
    dataset_path: str,
    test_type: str,
    e5_model: str,
    specter_model: str,
    num_queries: int,
    k: int,
) -> None:
    """
    Run retriever testing with configurable models.

    Args:
        dataset_path: Path to dataset
        test_type: Type of test (single, batch, cli, consistency, all)
        e5_model: E5 model name
        specter_model: SPECTER model name
        num_queries: Number of queries to test
        k: Top-k results
    """
    logger.info("=" * 70)
    logger.info("RETRIEVERS TESTING")
    logger.info("=" * 70)

    # Load dataset and build corpus
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    logger.info(f"Building citation corpus from {len(dataset)} papers")
    corpus = build_citation_corpus(dataset)
    logger.info(f"Corpus built: {len(corpus)} documents")

    # Build resources
    logger.info(f"Building E5 resources (model: {e5_model})")
    e5_resources = build_e5_resources(corpus, model_name=e5_model, batch_size=16)

    logger.info(f"Building SPECTER resources (model: {specter_model})")
    specter_resources = build_specter_resources(corpus, model_name=specter_model, batch_size=8)

    # Initialize retrievers
    e5_retriever = E5Retriever(e5_resources["model"], e5_resources.get("device"))
    specter_retriever = SPECTERRetriever(
        specter_resources["model"],
        specter_resources["tokenizer"],
        specter_resources.get("device"),
    )

    # Generate test queries
    base_queries = [
        "transformer architecture",
        "neural network models",
        "language representation learning",
        "deep learning for NLP",
        "attention mechanisms",
        "bidirectional transformers",
        "pre-training methods",
        "image recognition",
        "residual learning",
        "sequence transduction",
    ]
    queries = (base_queries * ((num_queries // len(base_queries)) + 1))[:num_queries]

    def test_single_query():
        logger.info("\n" + "=" * 70)
        logger.info("Testing Single Query")
        logger.info("=" * 70)

        query = queries[0]
        logger.info(f"Query: {query}")

        # E5
        e5_results = e5_retriever.single_query(
            query,
            e5_resources["corpus_embeddings"],
            e5_resources["ids"],
            e5_resources["titles"],
            k=k,
        )
        logger.info(f"\nE5: Retrieved {len(e5_results)} results")
        for i, r in enumerate(e5_results[:3], 1):
            logger.info(f"  {i}. {r['title'][:60]}... (score: {r['score']:.4f})")

        # SPECTER
        specter_results = specter_retriever.single_query(
            query,
            specter_resources["corpus_embeddings"],
            specter_resources["ids"],
            specter_resources["titles"],
            k=k,
        )
        logger.info(f"\nSPECTER: Retrieved {len(specter_results)} results")
        for i, r in enumerate(specter_results[:3], 1):
            logger.info(f"  {i}. {r['title'][:60]}... (score: {r['score']:.4f})")

    def test_batch_query():
        logger.info("\n" + "=" * 70)
        logger.info(f"Testing Batch Query ({num_queries} queries)")
        logger.info("=" * 70)

        # E5
        logger.info("\n[E5] Processing batch...")
        start_time = time.time()
        e5_results = e5_retriever.batch_query(
            queries,
            e5_resources["corpus_embeddings"],
            e5_resources["ids"],
            e5_resources["titles"],
            k=k,
        )
        e5_time = time.time() - start_time
        logger.info(f"✓ Processed {len(e5_results)} queries in {e5_time:.2f}s")
        logger.info(f"✓ Average time per query: {e5_time/num_queries:.3f}s")

        # SPECTER
        logger.info("\n[SPECTER] Processing batch...")
        start_time = time.time()
        specter_results = specter_retriever.batch_query(
            queries,
            specter_resources["corpus_embeddings"],
            specter_resources["ids"],
            specter_resources["titles"],
            k=k,
        )
        specter_time = time.time() - start_time
        logger.info(f"✓ Processed {len(specter_results)} queries in {specter_time:.2f}s")
        logger.info(f"✓ Average time per query: {specter_time/num_queries:.3f}s")

    def test_consistency():
        logger.info("\n" + "=" * 70)
        logger.info("Testing Consistency: Single Query vs Batch Query")
        logger.info("=" * 70)

        query = queries[0]

        # E5
        e5_single = e5_retriever.single_query(
            query,
            e5_resources["corpus_embeddings"],
            e5_resources["ids"],
            e5_resources["titles"],
            k=k,
        )
        e5_batch = e5_retriever.batch_query(
            [query],
            e5_resources["corpus_embeddings"],
            e5_resources["ids"],
            e5_resources["titles"],
            k=k,
        )[0]

        assert len(e5_single) == len(e5_batch), "E5: Results should have same length"
        for s, b in zip(e5_single, e5_batch):
            assert s["id"] == b["id"], "E5: Result IDs should match"
            assert abs(s["score"] - b["score"]) < 1e-5, "E5: Scores should match"
        logger.info("✓ E5: Single query and batch query produce identical results")

        # SPECTER
        specter_single = specter_retriever.single_query(
            query,
            specter_resources["corpus_embeddings"],
            specter_resources["ids"],
            specter_resources["titles"],
            k=k,
        )
        specter_batch = specter_retriever.batch_query(
            [query],
            specter_resources["corpus_embeddings"],
            specter_resources["ids"],
            specter_resources["titles"],
            k=k,
        )[0]

        assert len(specter_single) == len(specter_batch), "SPECTER: Results should have same length"
        for s, b in zip(specter_single, specter_batch):
            assert s["id"] == b["id"], "SPECTER: Result IDs should match"
            assert abs(s["score"] - b["score"]) < 1e-5, "SPECTER: Scores should match"
        logger.info("✓ SPECTER: Single query and batch query produce identical results")

    # Run tests based on test_type
    if test_type == "all":
        test_single_query()
        test_batch_query()
        test_consistency()
    elif test_type == "single":
        test_single_query()
    elif test_type == "batch":
        test_batch_query()
    elif test_type == "consistency":
        test_consistency()
    elif test_type == "cli":
        test_single_query()
        test_batch_query()

    logger.info("\n" + "=" * 70)
    logger.info("✓ All tests completed successfully!")
    logger.info("=" * 70)


# ============================================================================
# MAIN CLI
# ============================================================================


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI for retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline evaluation
  python evaluate.py --mode pipeline --dataset PATH --num-queries 100

  # Individual retriever baselines
  python evaluate.py --mode baselines --dataset PATH --retrievers bm25,e5,specter

  # Retriever batch testing
  python evaluate.py --mode retrievers --test-type batch --num-queries 10
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["pipeline", "baselines", "retrievers"],
        required=True,
        help="Evaluation mode: pipeline, baselines, or retrievers",
    )

    # Common arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("DATASET_DIR", ""),
        help="Path to ScholarCopilot JSON dataset",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (default: all for pipeline/baselines, 10 for retrievers)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of results to retrieve per query (default: 20 for pipeline, 5 for baselines, 3 for retrievers)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (rebuild from scratch). Default: use cache.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Pipeline-specific arguments
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="[pipeline] Disable dense embeddings (only use BM25)",
    )
    parser.add_argument(
        "--no-e5",
        action="store_true",
        help="[pipeline] Disable E5 dense embeddings",
    )
    parser.add_argument(
        "--no-specter",
        action="store_true",
        help="[pipeline] Disable SPECTER dense embeddings",
    )
    parser.add_argument(
        "--llm-reranker",
        action="store_true",
        help="[pipeline] Use LLM-based reranker instead of cross-encoder",
    )

    # Baselines-specific arguments
    parser.add_argument(
        "--retrievers",
        type=str,
        default="bm25,e5,specter",
        help="[baselines] Comma-separated list of retrievers to evaluate (default: bm25,e5,specter)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="[baselines] Output directory for results (default: results)",
    )

    # Retrievers-specific arguments
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["all", "single", "batch", "cli", "consistency"],
        default="all",
        help="[retrievers] Type of test to run (default: all)",
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default="intfloat/e5-base-v2",
        help="[retrievers] E5 model name (default: intfloat/e5-base-v2)",
    )
    parser.add_argument(
        "--specter-model",
        type=str,
        default="allenai/specter2_base",
        help="[retrievers] SPECTER model name (default: allenai/specter2_base)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    # Validate dataset path
    if not args.dataset:
        raise SystemExit("Missing dataset path. Provide --dataset or set DATASET_DIR.")

    # Set defaults for k based on mode
    if args.k is None:
        if args.mode == "pipeline":
            args.k = 20
        elif args.mode == "baselines":
            args.k = 5
        elif args.mode == "retrievers":
            args.k = 3

    # Set defaults for num_queries based on mode
    if args.num_queries is None and args.mode == "retrievers":
        args.num_queries = 10

    # Run appropriate mode
    if args.mode == "pipeline":
        run_pipeline_evaluation(
            dataset_path=args.dataset,
            num_queries=args.num_queries,
            k=args.k,
            bm25_only=args.bm25_only,
            no_e5=args.no_e5,
            no_specter=args.no_specter,
            use_cache=not args.no_cache,
            use_llm_reranker=args.llm_reranker,
        )
    elif args.mode == "baselines":
        retrievers = [r.strip() for r in args.retrievers.split(",")]
        k_values = (1, args.k, 10) if args.k <= 10 else (1, 5, args.k)
        run_baselines_evaluation(
            dataset_path=args.dataset,
            retrievers=retrievers,
            k_values=k_values,
            output_dir=args.output_dir,
        )
    elif args.mode == "retrievers":
        run_retrievers_test(
            dataset_path=args.dataset,
            test_type=args.test_type,
            e5_model=args.e5_model,
            specter_model=args.specter_model,
            num_queries=args.num_queries,
            k=args.k,
        )


if __name__ == "__main__":
    main()
