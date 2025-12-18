import ast
import logging
from typing import Any, Dict, List, Optional

import torch
from langchain_core.messages import AIMessage
from sentence_transformers import util

logger = logging.getLogger(__name__)


def _get_queries(state: Dict[str, Any]) -> List[str]:
    queries = state.get("queries")
    if isinstance(queries, list) and queries:
        return [str(q) for q in queries if str(q).strip()]

    try:
        msg = state["messages"][-1].content
        parsed = ast.literal_eval(msg) if isinstance(msg, str) else msg
    except Exception:
        return []

    if isinstance(parsed, dict):
        qs = parsed.get("queries", [])
        return [str(q) for q in qs if str(q).strip()] if isinstance(qs, list) else []
    if isinstance(parsed, list):
        return [str(q) for q in parsed if str(q).strip()]
    return []


class E5Retriever:
    """E5 dense retrieval retriever with single and batch query support."""

    def __init__(self, model, device: Optional[str] = None):
        """
        Initialize with E5 model.

        Args:
            model: SentenceTransformer model instance
            device: Device to use ('cuda', 'cpu', etc.). If None, auto-detects.
        """
        self.model = model
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        if self.device != "cpu":
            self.model.to(self.device)

    def single_query(
        self,
        query: str,
        corpus_embeddings: torch.Tensor,
        ids: List[str],
        titles: List[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process a single query and return top-k results.

        Args:
            query: Single query string
            corpus_embeddings: Pre-computed corpus embeddings (corpus_size, hidden_dim)
            ids: List of document IDs
            titles: List of document titles
            k: Number of top results to return

        Returns:
            List of result dicts with 'id', 'title', 'score', 'source'
        """
        # Encode single query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute similarity
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_k = torch.topk(scores, k=min(k, scores.shape[0]))

        # Format results
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            i = int(idx)
            results.append(
                {
                    "id": ids[i],
                    "title": titles[i],
                    "score": float(score),
                    "source": "e5",
                }
            )
        return results

    def batch_query(
        self,
        queries: List[str],
        corpus_embeddings: torch.Tensor,
        ids: List[str],
        titles: List[str],
        k: int = 5,
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple queries in batch and return top-k results for each.

        Args:
            queries: List of query strings
            corpus_embeddings: Pre-computed corpus embeddings (corpus_size, hidden_dim)
            ids: List of document IDs
            titles: List of document titles
            k: Number of top results to return per query
            batch_size: Optional batch size for encoding (default: all at once)

        Returns:
            List of result lists, one per query. Each inner list contains top-k results.
        """
        # Batch encode all queries
        query_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size or len(queries),
        )

        # Compute similarity for all queries: (num_queries, corpus_size)
        scores = util.cos_sim(query_embeddings, corpus_embeddings)

        # Get top-k for each query
        all_results = []
        for query_scores in scores:
            top_k = torch.topk(query_scores, k=min(k, query_scores.shape[0]))
            query_results = []
            for score, idx in zip(top_k.values, top_k.indices):
                i = int(idx)
                query_results.append(
                    {
                        "id": ids[i],
                        "title": titles[i],
                        "score": float(score),
                        "source": "e5",
                    }
                )
            all_results.append(query_results)

        return all_results


def e5_agent(state: Dict[str, Any]):
    """
    E5 dense retrieval agent (dataset-agnostic).

    Wrapper function for backward compatibility with workflow.
    Uses E5Retriever class internally.

    Expects:
    - `state["queries"]`: list[str]
    - `state["resources"]["e5"]`: built via src.resources.builders.build_e5_resources()
    """
    logger.info("🔍 E5 retriever starting...")

    queries = _get_queries(state)
    if not queries:
        logger.warning("⚠️  E5 received no queries")
        return {"messages": [AIMessage(content="E5 received no queries", name="e5")]}

    resources = state.get("resources", {}) or {}
    e5_res = resources.get("e5")
    if not e5_res:
        logger.error("❌ E5 resources not found")
        return {"messages": [AIMessage(content="E5_ERROR: missing e5 resources", name="e5")]}

    k = int((state.get("config", {}) or {}).get("k", 5))
    logger.debug(f"Query: {queries[0][:100]}...")
    logger.debug(f"Retrieving top-{k} results")

    # Initialize retriever
    retriever = E5Retriever(e5_res["model"], e5_res.get("device"))

    # Use batch_query if multiple queries, single_query otherwise
    if len(queries) > 1:
        logger.debug(f"Processing {len(queries)} queries in batch")
        results_per_query = retriever.batch_query(
            queries,
            e5_res["corpus_embeddings"],
            e5_res["ids"],
            e5_res["titles"],
            k=k,
        )
        # For backward compatibility, return results from first query
        # TODO: Consider returning per-query results in future
        results = results_per_query[0]
    else:
        results = retriever.single_query(
            queries[0],
            e5_res["corpus_embeddings"],
            e5_res["ids"],
            e5_res["titles"],
            k=k,
        )

    if results:
        scores = [r["score"] for r in results]
        logger.info(
            f"✅ E5 retrieved {len(results)} results (scores: {scores[0]:.3f} to {scores[-1]:.3f})"
        )
    else:
        logger.warning("⚠️  E5 returned no results")

    return {
        "e5_results": results,
        "messages": [AIMessage(content=str(results), name="e5")],
    }
