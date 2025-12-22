from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.prompts.llm_reranker import LLMRerankerPrompt

load_dotenv()

logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from httpx (used by Ollama)
logging.getLogger("httpx").setLevel(logging.WARNING)


def llm_reranker(state: Dict[str, Any], closed_source: bool = False):
    """
    LLM-based reranking agent.
    Takes candidate papers from state and reranks them using an LLM.

    Args:
        state: MessagesState containing query and candidate_papers
        closed_source: Whether to force OpenAI usage (overrides INFERENCE_ENGINE)

    Returns:
        Dictionary with ranked_papers containing (paper, relevance_score) tuples
    """
    logger.info("🤖 LLM Reranker starting...")

    # Debug: log all state keys
    logger.debug(f"State keys available: {list(state.keys())}")

    query = state.get("query", "")
    candidate_papers = state.get("candidate_papers", [])
    resources = state.get("resources", {})

    logger.debug(f"Query: {query[:100] if query else 'NONE'}...")
    logger.debug(f"Candidate papers count: {len(candidate_papers)}")

    if not candidate_papers:
        logger.warning("⚠️  No candidate papers to rerank")
        return {"ranked_papers": []}

    # Check if LLM model is cached in resources (PERFORMANCE OPTIMIZATION)
    llm_reranker_res = resources.get("llm_reranker")

    if llm_reranker_res and "llm_model" in llm_reranker_res:
        # Use cached model (FAST PATH - no loading overhead)
        llm = llm_reranker_res["llm_model"]
        model_id = llm_reranker_res.get("model_name", "cached")
        logger.info(f"🚀 Using cached LLM model: {model_id}")
    else:
        # Fallback: load model on-the-fly (SLOW PATH - only for backwards compatibility)
        logger.info(f"⚠️  No LLM model in resources - loading on-the-fly (SLOW!)")
        logger.info(f"   💡 Tip: Enable llm_reranker in build_inmemory_resources for better performance")

        # Determine inference engine
        use_openai = closed_source or os.getenv("USE_OPENAI_RERANKER", "false").lower() in ["true", "1", "yes"]
        inference_engine = os.getenv("INFERENCE_ENGINE", "ollama").lower()

        model_id = os.getenv("LOCAL_LLM", "gemma3:4b")

        if use_openai or inference_engine == "openai":
            # OpenAI - cloud-based
            openai_model = os.getenv("OPENAI_RERANKER_MODEL", "gpt-5-mini-2025-08-07")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            logger.info(f"🤖 Using OpenAI with model: {openai_model}")
            # Note: GPT-5 models only support temperature=1 (default)
            if "gpt-5" in openai_model.lower():
                # GPT-5 models: must use temperature=1 (the only supported value)
                llm = ChatOpenAI(
                    model=openai_model,
                    api_key=api_key,
                    temperature=1.0
                )
                logger.info(f"   Using temperature=1.0 (GPT-5 requirement)")
            else:
                # Other models: use temperature=0 for deterministic outputs
                llm = ChatOpenAI(model=openai_model, api_key=api_key, temperature=0)
            logger.info(f"✅ OpenAI ready!")

        elif inference_engine == "ollama":
            # Using Ollama for local inference (faster, less memory)
            logger.info(f"🔄 Using Ollama model: {model_id}")
            llm = ChatOllama(model=model_id, temperature=0)
            logger.info(f"✅ Ollama ready!")

        elif inference_engine == "huggingface":
            # Using Hugging Face model loaded locally via transformers pipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            logger.info(f"🔄 Loading Hugging Face model: {model_id}...")
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=1024,  # Increased for reasoning + JSON output
                do_sample=False,
            )

            llm = HuggingFacePipeline(pipeline=gen)
            logger.info(f"✅ Hugging Face model loaded!")
        else:
            raise ValueError(f"Invalid INFERENCE_ENGINE: {inference_engine}. Must be 'ollama', 'huggingface', or 'openai'")

    prompt = LLMRerankerPrompt(query=query, candidate_papers=candidate_papers).get_prompt()
    logger.debug(f"Prompt length: {len(prompt)} characters")

    logger.info(f"🔄 Calling LLM to rerank {len(candidate_papers)} papers...")

    # Call LLM
    response = llm.invoke(prompt)

    # Handle different response formats
    # - ChatOllama/ChatOpenAI return object with .content attribute
    # - HuggingFacePipeline may return string directly or dict with 'text' key
    if isinstance(response, str):
        response_text = response
    elif hasattr(response, 'content'):
        response_text = response.content
    elif isinstance(response, dict) and 'text' in response:
        response_text = response['text']
    elif isinstance(response, list) and len(response) > 0:
        # Some pipelines return a list of dicts
        if isinstance(response[0], dict) and 'generated_text' in response[0]:
            response_text = response[0]['generated_text']
        else:
            response_text = str(response)
    else:
        response_text = str(response)

    logger.debug(f"LLM response received, length: {len(response_text)} characters")
    logger.debug(f"First 200 chars: {response_text[:200]}...")
    logger.debug(f"Last 200 chars: ...{response_text[-200:]}")

    # Extract JSON from response
    try:
        # Try to find JSON array in the response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            json_str = json_match.group()
            logger.debug(f"✓ Found JSON array ({len(json_str)} chars)")
            rankings = json.loads(json_str)
        else:
            logger.warning(f"⚠️  No JSON array pattern found, trying to parse entire response")
            rankings = json.loads(response_text)

        logger.debug(f"✓ Successfully parsed {len(rankings)} rankings")

        # Build ranked results, tracking which papers have been added
        # Match the format of cross-encoder reranker: list of dicts with rerank_score
        ranked_papers = []
        seen_indices = set()

        # Add papers from LLM ranking (skip duplicates)
        for item in rankings:
            idx = item["index"] - 1  # Convert to 0-based index
            score = item["score"]

            if 0 <= idx < len(candidate_papers) and idx not in seen_indices:
                paper_dict = dict(candidate_papers[idx])
                paper_dict["rerank_score"] = float(score)
                ranked_papers.append(paper_dict)
                seen_indices.add(idx)

        # Add any papers that weren't ranked by the LLM (with score 0.0)
        unranked_count = 0
        for i, paper in enumerate(candidate_papers):
            if i not in seen_indices:
                paper_dict = dict(paper)
                paper_dict["rerank_score"] = 0.0
                ranked_papers.append(paper_dict)
                seen_indices.add(i)
                unranked_count += 1

        if unranked_count > 0:
            logger.warning(f"⚠️  {unranked_count}/{len(candidate_papers)} papers not ranked by LLM (assigned score 0.0)")

        if ranked_papers:
            top_score = ranked_papers[0]["rerank_score"]
            bottom_score = ranked_papers[-1]["rerank_score"]
            logger.info(
                f"✅ LLM reranking complete: {len(ranked_papers)} papers (scores: {top_score:.3f} to {bottom_score:.3f})"
            )

        return {"ranked_papers": ranked_papers}

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"⚠️  Error parsing LLM response: {e}")
        logger.debug(f"Response was: {response_text[:500]}...")
        logger.warning(f"❌ Fallback: returning papers in original order with score 0.1")
        # Fallback: return papers in original order with low scores
        fallback_papers = []
        for paper in candidate_papers:
            paper_dict = dict(paper)
            paper_dict["rerank_score"] = 0.1
            fallback_papers.append(paper_dict)
        return {"ranked_papers": fallback_papers}


def estimate_rerank_tokens(query: str, candidate_papers: List[Dict[str, Any]]) -> int:
    """
    Estimate the number of tokens required for a reranking prompt.

    Conservative estimation:
    - Base prompt: ~200 tokens
    - Query: ~length/4 (rough approximation)
    - Each paper: ~75 tokens (title + score)
    - Response buffer: ~500 tokens

    Args:
        query: Query string
        candidate_papers: List of candidate papers

    Returns:
        Estimated token count
    """
    base_tokens = 200
    query_tokens = len(query) // 4  # Rough approximation: 1 token ≈ 4 chars
    paper_tokens = len(candidate_papers) * 75  # ~75 tokens per paper entry
    response_buffer = 500  # Buffer for response

    total = base_tokens + query_tokens + paper_tokens + response_buffer
    return total


def get_model_context_limit(model_name: str, inference_engine: str) -> int:
    """
    Get the context limit for a given model.

    Returns a conservative estimate to avoid hitting limits.

    Args:
        model_name: Model identifier
        inference_engine: Engine being used (ollama, huggingface, openai)

    Returns:
        Context limit in tokens
    """
    # Conservative limits (leaving 20% buffer for safety)
    LIMITS = {
        # Gemma 3 models - 128K context
        "gemma3:4b": 102_000,  # 80% of 128K
        "gemma3:12b": 102_000,
        "google/gemma-3-4b-it": 102_000,
        "google/gemma-2-2b-it": 64_000,  # Gemma 2 has 8K base
        # Llama 3 models - 128K context
        "llama3.1:8b": 102_000,
        "meta-llama/Llama-3.2-3B-Instruct": 102_000,
        # Qwen models - 128K context
        "qwen3:8b": 102_000,
        # Mistral models - vary by version
        "mistral:7b": 64_000,  # 32K base, conservative
        "mistralai/Mistral-7B-Instruct-v0.3": 64_000,
        # OpenAI GPT-5 models - 272K input tokens
        "gpt-5-mini-2025-08-07": 217_000,  # 80% of 272K
        "gpt-5-2025-08-07": 217_000,
        # OpenAI GPT-4 models (legacy)
        "gpt-4o-mini": 102_000,  # 128K context
        "gpt-4o": 102_000,
    }

    limit = LIMITS.get(model_name)

    if limit is None:
        # Default based on inference engine
        if inference_engine == "openai":
            logger.warning(f"Unknown OpenAI model '{model_name}', assuming 102K limit")
            return 102_000
        else:
            logger.warning(f"Unknown model '{model_name}', assuming 64K limit")
            return 64_000

    return limit


def batch_llm_reranker(
    queries: List[str],
    candidate_papers_list: List[List[Dict[str, Any]]],
    resources: Dict[str, Any],
    max_workers: int = None,
    closed_source: bool = False
) -> List[List[Dict[str, Any]]]:
    """
    Context-aware batch LLM-based reranking for multiple queries.

    Processes multiple queries in parallel using ThreadPoolExecutor for improved throughput.
    Each query is reranked independently. Automatically respects model context limits.

    Context Limits (reference):
    - gemma3:4b: 128K tokens (~64 concurrent queries with 20 papers each)
    - qwen3:8b: 128K tokens (~64 concurrent queries)
    - llama3.1:8b: 128K tokens (~64 concurrent queries)
    - gpt-5-mini: 272K input tokens (~136 concurrent queries)

    Args:
        queries: List of query strings
        candidate_papers_list: List of candidate paper lists (one per query)
        resources: Resources dictionary containing llm_reranker model
        max_workers: Maximum number of parallel workers (None = auto-detect based on model)
        closed_source: Whether to force OpenAI usage

    Returns:
        List of ranked paper lists (one per query)

    Example:
        >>> queries = ["transformers", "attention mechanisms"]
        >>> candidates = [[...papers for query 1...], [...papers for query 2...]]
        >>> results = batch_llm_reranker(queries, candidates, resources)
        >>> # results[0] = ranked papers for query 1
        >>> # results[1] = ranked papers for query 2
    """
    if len(queries) != len(candidate_papers_list):
        raise ValueError(
            f"Mismatched lengths: {len(queries)} queries but {len(candidate_papers_list)} candidate lists"
        )

    # Auto-detect max_workers based on model and context limits
    if max_workers is None:
        max_workers = int(os.getenv("LLM_BATCH_MAX_WORKERS", "5"))

    # Get model info for context limit checking
    use_openai = closed_source or os.getenv("USE_OPENAI_RERANKER", "false").lower() in ["true", "1", "yes"]
    inference_engine = os.getenv("INFERENCE_ENGINE", "ollama").lower()

    if use_openai or inference_engine == "openai":
        model_name = os.getenv("OPENAI_RERANKER_MODEL", "gpt-5-mini-2025-08-07")
        inference_engine = "openai"
    else:
        model_name = os.getenv("LOCAL_LLM", "gemma3:4b")

    # Check if cached model is available
    if resources.get("llm_reranker") and "llm_model" in resources["llm_reranker"]:
        cached_model_name = resources["llm_reranker"].get("model_name", model_name)
        logger.info(f"🚀 Batch LLM Reranker using cached model: {cached_model_name}")
    else:
        logger.info(f"🚀 Batch LLM Reranker using model: {model_name}")

    # Get context limit for this model
    context_limit = get_model_context_limit(model_name, inference_engine)

    # Estimate token usage for a typical query
    if queries and candidate_papers_list:
        sample_tokens = estimate_rerank_tokens(queries[0], candidate_papers_list[0])
        max_safe_parallel = max(1, int((context_limit * 0.8) / sample_tokens))  # 80% safety margin

        # Adjust max_workers if needed
        if max_workers > max_safe_parallel:
            logger.warning(
                f"⚠️  Reducing max_workers from {max_workers} to {max_safe_parallel} "
                f"to respect {model_name} context limit ({context_limit:,} tokens)"
            )
            max_workers = max_safe_parallel

    logger.info(f"   Processing {len(queries)} queries with {max_workers} parallel workers")
    logger.info(f"   Model context limit: {context_limit:,} tokens (~{sample_tokens} tokens/query)")

    # Validate resources contain LLM model
    if not resources.get("llm_reranker") or "llm_model" not in resources["llm_reranker"]:
        logger.warning("⚠️  No cached LLM model in resources - batching will be slower")
        logger.warning("   💡 Tip: Enable llm_reranker in build_inmemory_resources for better performance")

    start_time = time.time()
    results = [None] * len(queries)
    total_tokens_estimate = sum(
        estimate_rerank_tokens(q, c) for q, c in zip(queries, candidate_papers_list)
    )

    logger.info(f"   Estimated total tokens: {total_tokens_estimate:,}")

    # Define worker function
    def process_single_query(idx: int, query: str, candidates: List[Dict[str, Any]]) -> tuple:
        """Process a single query and return (index, result)"""
        try:
            # Create state for single query
            state = {
                "query": query,
                "candidate_papers": candidates,
                "resources": resources
            }

            # Run single query reranking
            result = llm_reranker(state, closed_source=closed_source)
            ranked_papers = result.get("ranked_papers", [])

            return (idx, ranked_papers)

        except Exception as e:
            logger.error(f"❌ Error processing query {idx}: {e}")
            # Return candidates with fallback score
            fallback = [dict(p) for p in candidates]
            for p in fallback:
                p["rerank_score"] = 0.1
            return (idx, fallback)

    # Process queries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_query, i, q, c): i
            for i, (q, c) in enumerate(zip(queries, candidate_papers_list))
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            idx, ranked_papers = future.result()
            results[idx] = ranked_papers
            completed += 1

            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (len(queries) - completed) / rate if rate > 0 else 0
                logger.info(
                    f"   Progress: {completed}/{len(queries)} queries "
                    f"({rate:.1f} queries/sec, ETA: {eta:.1f}s)"
                )

    elapsed = time.time() - start_time
    avg_time = elapsed / len(queries)
    throughput = len(queries) / elapsed

    logger.info(f"✅ Batch reranking complete!")
    logger.info(f"   Total time: {elapsed:.2f}s")
    logger.info(f"   Avg per query: {avg_time:.2f}s")
    logger.info(f"   Throughput: {throughput:.1f} queries/sec")
    logger.info(f"   Speedup vs sequential: ~{max_workers * 0.8:.1f}x (theoretical)")

    return results
