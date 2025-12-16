import ast
from typing import Any, Dict, List, Optional

import torch
from langchain_core.messages import AIMessage


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


class SPECTERRetriever:
    """SPECTER2 dense retrieval retriever with single and batch query support."""

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        """
        Initialize with SPECTER model and tokenizer.

        Args:
            model: AutoModel instance for SPECTER
            tokenizer: AutoTokenizer instance for SPECTER
            device: Device to use ('cuda', 'cpu', etc.). If None, auto-detects.
        """
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def single_query(
        self,
        query: str,
        corpus_embeddings: torch.Tensor,
        ids: List[str],
        titles: List[str],
        k: int = 5,
        max_length: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Process a single query and return top-k results.

        Args:
            query: Single query string
            corpus_embeddings: Pre-computed corpus embeddings (corpus_size, hidden_dim)
            ids: List of document IDs
            titles: List of document titles
            k: Number of top results to return
            max_length: Maximum token length for query

        Returns:
            List of result dicts with 'id', 'title', 'score', 'source'
        """
        # Move model to device if needed
        if self.device == "cuda":
            self.model.to("cuda")

        # Tokenize and encode
        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            q_emb = self.model(**inputs).last_hidden_state.mean(dim=1)

        corpus_embeddings = corpus_embeddings.to(self.device)

        # Compute similarity
        # q_emb: (1, hidden_dim), corpus_embeddings: (corpus_size, hidden_dim)
        scores = torch.cosine_similarity(
            q_emb.expand_as(corpus_embeddings), corpus_embeddings, dim=1
        )
        top_indices = torch.topk(scores, k=min(k, scores.shape[0])).indices.tolist()

        # Format results
        results = []
        for i in top_indices:
            results.append(
                {
                    "id": ids[i],
                    "title": titles[i],
                    "score": float(scores[i]),
                    "source": "specter",
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
        max_length: int = 512,
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple queries in batch and return top-k results for each.

        Args:
            queries: List of query strings
            corpus_embeddings: Pre-computed corpus embeddings (corpus_size, hidden_dim)
            ids: List of document IDs
            titles: List of document titles
            k: Number of top results to return per query
            max_length: Maximum token length for queries

        Returns:
            List of result lists, one per query. Each inner list contains top-k results.
        """
        # Move model to device if needed
        if self.device == "cuda":
            self.model.to("cuda")

        # Batch tokenize all queries
        inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        # Batch forward pass
        with torch.no_grad():
            q_embs = self.model(**inputs).last_hidden_state.mean(dim=1)

        corpus_embeddings = corpus_embeddings.to(self.device)

        # Compute similarity for all queries
        # q_embs: (num_queries, hidden_dim)
        # corpus_embeddings: (corpus_size, hidden_dim)
        # Expand for broadcasting: (num_queries, 1, hidden_dim) vs (1, corpus_size, hidden_dim)
        q_embs_expanded = q_embs.unsqueeze(1)  # (num_queries, 1, hidden_dim)
        corpus_expanded = corpus_embeddings.unsqueeze(0)  # (1, corpus_size, hidden_dim)
        scores = torch.cosine_similarity(q_embs_expanded, corpus_expanded, dim=2)
        # scores: (num_queries, corpus_size)

        # Get top-k for each query
        all_results = []
        for query_scores in scores:
            top_indices = torch.topk(query_scores, k=min(k, query_scores.shape[0])).indices.tolist()
            query_results = []
            for i in top_indices:
                query_results.append(
                    {
                        "id": ids[i],
                        "title": titles[i],
                        "score": float(query_scores[i]),
                        "source": "specter",
                    }
                )
            all_results.append(query_results)

        return all_results


def specter_agent(state: Dict[str, Any]):
    """
    SPECTER2 dense retrieval agent (dataset-agnostic).

    Wrapper function for backward compatibility with workflow.
    Uses SPECTERRetriever class internally.

    Expects:
    - `state["queries"]`: list[str]
    - `state["resources"]["specter"]`: built via src.resources.builders.build_specter_resources()
    """
    queries = _get_queries(state)
    if not queries:
        return {"messages": [AIMessage(content="SPECTER received no queries", name="specter")]}

    resources = state.get("resources", {}) or {}
    sp_res = resources.get("specter")
    if not sp_res:
        return {
            "messages": [
                AIMessage(content="SPECTER_ERROR: missing specter resources", name="specter")
            ]
        }

    k = int((state.get("config", {}) or {}).get("k", 5))

    # Initialize retriever
    retriever = SPECTERRetriever(sp_res["model"], sp_res["tokenizer"], sp_res.get("device", "cpu"))

    # Use batch_query if multiple queries, single_query otherwise
    if len(queries) > 1:
        results_per_query = retriever.batch_query(
            queries,
            sp_res["corpus_embeddings"],
            sp_res["ids"],
            sp_res["titles"],
            k=k,
        )
        # For backward compatibility, return results from first query
        # TODO: Consider returning per-query results in future
        results = results_per_query[0]
    else:
        results = retriever.single_query(
            queries[0],
            sp_res["corpus_embeddings"],
            sp_res["ids"],
            sp_res["titles"],
            k=k,
        )

    return {
        "specter_results": results,
        "messages": [AIMessage(content=str(results), name="specter")],
    }
