"""
Aggregator agent for merging results from multiple retrievers.

This agent collects results from BM25, E5, and SPECTER retrievers,
deduplicates by paper ID, and prepares the candidate set for reranking.
"""

import logging
from typing import Any, Dict, List
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def _normalize_scores(results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        results: List of retrieval results with scores
        source: Name of the retriever source

    Returns:
        Results with normalized scores
    """
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero
    score_range = max_score - min_score
    if score_range == 0:
        return results

    normalized = []
    for result in results:
        normalized_result = dict(result)
        normalized_result["normalized_score"] = (result["score"] - min_score) / score_range
        normalized_result["original_score"] = result["score"]
        normalized.append(normalized_result)

    return normalized


def _reciprocal_rank_fusion(
    paper_id_to_results: Dict[str, List[Dict[str, Any]]], k: int = 60
) -> List[Dict[str, Any]]:
    """
    Apply Reciprocal Rank Fusion (RRF) to merge rankings from multiple retrievers.

    RRF score for a document = sum(1 / (k + rank_i)) across all retrievers
    where rank_i is the rank position in retriever i (1-indexed).

    Args:
        paper_id_to_results: Map from paper ID to list of results from different retrievers
        k: Constant for RRF formula (default: 60, as per original paper)

    Returns:
        Fused results sorted by RRF score
    """
    rrf_scores = {}

    for paper_id, results_list in paper_id_to_results.items():
        rrf_score = 0.0
        for result in results_list:
            # Rank is 1-indexed position in original retriever results
            rank = result.get("rank", 1)
            rrf_score += 1.0 / (k + rank)

        rrf_scores[paper_id] = rrf_score

    # Sort papers by RRF score descending
    sorted_papers = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final results
    fused_results = []
    for paper_id, rrf_score in sorted_papers:
        # Take the first occurrence (arbitrary choice for metadata)
        paper_data = paper_id_to_results[paper_id][0].copy()
        paper_data["rrf_score"] = rrf_score
        paper_data["retriever_count"] = len(paper_id_to_results[paper_id])
        paper_data["sources"] = [r["source"] for r in paper_id_to_results[paper_id]]
        fused_results.append(paper_data)

    return fused_results


def aggregator(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate results from all retrievers.

    This agent:
    1. Collects results from bm25_results, e5_results, specter_results
    2. Stores raw results in retriever_results dict
    3. Deduplicates papers by ID
    4. Applies Reciprocal Rank Fusion (RRF) for score combination
    5. Outputs candidate_papers for downstream reranking

    Expects:
        state["bm25_results"]: List[Dict] from BM25 retriever
        state["e5_results"]: List[Dict] from E5 retriever
        state["specter_results"]: List[Dict] from SPECTER retriever
        state["config"]["aggregation_method"]: Optional, "rrf" or "simple" (default: "rrf")
        state["config"]["rrf_k"]: Optional, RRF constant (default: 60)

    Returns:
        Dict with:
        - retriever_results: Dict mapping retriever name to its results
        - candidate_papers: Deduplicated and fused list of papers
        - messages: AIMessage for LangGraph visibility
    """
    logger.info("🔀 Aggregator starting...")

    # Collect results from all retrievers
    bm25_results = state.get("bm25_results") or []
    e5_results = state.get("e5_results") or []
    specter_results = state.get("specter_results") or []

    logger.debug(
        f"Retrieved results - BM25: {len(bm25_results)}, E5: {len(e5_results)}, SPECTER: {len(specter_results)}"
    )

    # Store raw results
    retriever_results = {
        "bm25": bm25_results,
        "e5": e5_results,
        "specter": specter_results,
    }

    # Check if we have any results
    total_results = len(bm25_results) + len(e5_results) + len(specter_results)
    if total_results == 0:
        logger.warning("⚠️  No results from any retriever!")
        return {
            "retriever_results": retriever_results,
            "candidate_papers": [],
            "messages": [AIMessage(name="aggregator", content="No results from any retriever")],
        }

    # Add rank information to each result (1-indexed)
    for results in [bm25_results, e5_results, specter_results]:
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

    # Get aggregation method from config
    config = state.get("config", {}) or {}
    aggregation_method = config.get("aggregation_method", "rrf")
    logger.debug(f"Using aggregation method: {aggregation_method}")

    if aggregation_method == "rrf":
        # Reciprocal Rank Fusion approach
        # Group results by paper ID
        paper_id_to_results: Dict[str, List[Dict[str, Any]]] = {}

        for results in [bm25_results, e5_results, specter_results]:
            for result in results:
                paper_id = result["id"]
                if paper_id not in paper_id_to_results:
                    paper_id_to_results[paper_id] = []
                paper_id_to_results[paper_id].append(result)

        # Apply RRF
        rrf_k = config.get("rrf_k", 60)
        candidate_papers = _reciprocal_rank_fusion(paper_id_to_results, k=rrf_k)

        message_content = (
            f"Aggregated {total_results} results into {len(candidate_papers)} unique papers "
            f"using RRF (k={rrf_k})"
        )
        logger.info(f"✅ {message_content}")

    else:
        # Simple aggregation: deduplicate and keep highest score per paper
        paper_id_to_best: Dict[str, Dict[str, Any]] = {}

        for results in [bm25_results, e5_results, specter_results]:
            normalized = _normalize_scores(results, results[0]["source"] if results else "")

            for result in normalized:
                paper_id = result["id"]
                score = result.get("normalized_score", result.get("score", 0))

                if paper_id not in paper_id_to_best or score > paper_id_to_best[paper_id].get(
                    "score", 0
                ):
                    result_copy = result.copy()
                    result_copy["score"] = score
                    paper_id_to_best[paper_id] = result_copy

        # Sort by score descending
        candidate_papers = sorted(
            paper_id_to_best.values(), key=lambda x: x.get("score", 0), reverse=True
        )

        message_content = (
            f"Aggregated {total_results} results into {len(candidate_papers)} unique papers "
            f"using simple max-score deduplication"
        )
        logger.info(f"✅ {message_content}")

    return {
        "retriever_results": retriever_results,
        "candidate_papers": candidate_papers,
        "messages": [AIMessage(name="aggregator", content=message_content)],
    }
