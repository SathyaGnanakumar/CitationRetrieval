from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re
import logging
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
