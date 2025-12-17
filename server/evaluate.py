"""
Evaluation script for the retrieval pipeline.

Evaluates the system on the ScholarCopilot dataset by:
1. Using each paper as a query
2. Treating citations in that paper's bib_info as ground truth
3. Measuring R@5, R@10, R@20, and MRR
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.corpus.scholarcopilot import build_citation_corpus, load_dataset
from src.evaluation.metrics import evaluate_retrieval, print_evaluation_results
from src.resources.builders import build_inmemory_resources
from src.resources.cache import load_resources, save_resources
from src.utils.timing import setup_logging, timer
from src.workflow import RetrievalWorkflow

logger = logging.getLogger(__name__)


def extract_query_from_paper(paper_text: str, max_words: int = 50) -> str:
    """Extract a query from the paper text (first few sentences)."""
    # Take first N words as the query
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


def run_evaluation(
    dataset_path: str,
    num_queries: int = None,
    k: int = 20,
    bm25_only: bool = False,
    use_cache: bool = True,
    use_llm_reranker: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluation on the dataset.

    Args:
        dataset_path: Path to the dataset
        num_queries: Number of queries to evaluate (None = all)
        k: Number of results to retrieve per query
        bm25_only: Only use BM25 (skip dense embeddings)
        use_cache: Whether to use resource caching

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
    enable_e5 = not bm25_only
    enable_specter = not bm25_only

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


def main():
    load_dotenv()
    setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate the retrieval pipeline")
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
        help="Number of queries to evaluate (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of results to retrieve per query (default: 20)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Disable dense embeddings (only use BM25). Default: use all retrievers.",
    )
    parser.add_argument(
        "--no-e5",
        action="store_true",
        help="Disable E5 dense embeddings. Default: enabled.",
    )
    parser.add_argument(
        "--no-specter",
        action="store_true",
        help="Disable SPECTER dense embeddings. Default: enabled.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (rebuild from scratch). Default: use cache.",
    )
    parser.add_argument(
        "--llm-reranker",
        action="store_true",
        help="Use LLM-based reranker instead of cross-encoder. Default: cross-encoder.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dataset:
        raise SystemExit("Missing dataset path. Provide --dataset or set DATASET_DIR.")

    # Check which features are enabled
    enable_e5 = not (args.bm25_only or args.no_e5)
    enable_specter = not (args.bm25_only or args.no_specter)

    if args.bm25_only:
        logger.info("⚠️  Dense embeddings disabled (BM25 only)")
    else:
        enabled_retrievers = ["BM25"]
        if enable_e5:
            enabled_retrievers.append("E5")
        if enable_specter:
            enabled_retrievers.append("SPECTER")
        logger.info(f"🔧 Enabled retrievers: {', '.join(enabled_retrievers)}")

    run_evaluation(
        dataset_path=args.dataset,
        num_queries=args.num_queries,
        k=args.k,
        bm25_only=args.bm25_only,
        use_cache=not args.no_cache,
        use_llm_reranker=args.llm_reranker,
    )


if __name__ == "__main__":
    main()
