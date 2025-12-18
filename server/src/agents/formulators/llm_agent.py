from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re
import logging

from src.prompts.llm_reranker import LLMRerankerPrompt

load_dotenv()

# Suppress verbose HTTP logs from httpx (used by Ollama)
logging.getLogger("httpx").setLevel(logging.WARNING)


def llm_reranker(state: Dict[str, Any], closed_source: bool = False):
    """
    LLM-based reranking agent.
    Takes candidate papers from state and reranks them using an LLM.

    Args:
        state: MessagesState containing query and candidate_papers
        model_name: Optional model name to use (defaults to LOCAL_LLM env var)

    Returns:
        Dictionary with ranked_papers containing (paper, relevance_score) tuples
    """
    # State is a dict at runtime (despite the type annotation)
    print(f"🤖 LLM Reranker starting")
    print(f"   State type: {type(state)}")
    print(f"   State keys: {list(state.keys())[:15]}")  # Show first 15 keys

    query = state.get("query", "")
    candidate_papers = state.get("candidate_papers", [])
    resources = state.get("resources", {})

    print(f"   Query: {query[:80]}..." if query else "   Query: (empty)")
    print(f"   Candidates in state: {len(candidate_papers)}")

    if not candidate_papers:
        print(f"⚠️  No candidate papers to rerank")
        return {"ranked_papers": []}

    # Check if LLM model is cached in resources (PERFORMANCE OPTIMIZATION)
    llm_reranker_res = resources.get("llm_reranker")

    if llm_reranker_res and "llm_model" in llm_reranker_res:
        # Use cached model (FAST PATH - no loading overhead)
        llm = llm_reranker_res["llm_model"]
        model_id = llm_reranker_res.get("model_name", "cached")
        print(f"🚀 Using cached LLM model: {model_id}")
    else:
        # Fallback: load model on-the-fly (SLOW PATH - only for backwards compatibility)
        print(f"⚠️  LLM model not found in resources - loading on-the-fly (SLOW!)")
        print(f"   💡 Tip: Enable llm_reranker in build_inmemory_resources for better performance")

        model_id = os.getenv("LOCAL_LLM", "gemma3:4b")
        inference_engine = os.getenv("INFERENCE_ENGINE", "ollama").lower()

        if closed_source:
            llm = ChatOpenAI("gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
            print(f"🤖 Using OpenAI GPT-4")
        elif inference_engine == "ollama":
            # Using Ollama for local inference (faster, less memory)
            print(f"🔄 Using Ollama with model: {model_id}")
            llm = ChatOllama(model=model_id, temperature=0)
            print(f"✅ Ollama ready!")
        else:
            # Using Hugging Face model loaded locally via transformers pipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            print(f"🔄 Loading Hugging Face model: {model_id}...")
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
            print(f"✅ Hugging Face model loaded!")

    prompt = LLMRerankerPrompt(query=query, candidate_papers=candidate_papers).get_prompt()

    print(f"🤖 LLM Reranking with model: {model_id}...")

    # Call LLM
    response = llm.invoke(prompt)
    response_text = response.content

    print(f"📝 LLM response length: {len(response_text)} chars")
    print(f"📝 First 200 chars: {response_text[:200]}...")
    print(f"📝 Last 200 chars: ...{response_text[-200:]}")

    # Extract JSON from response
    try:
        # Try to find JSON array in the response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            json_str = json_match.group()
            print(f"✓ Found JSON array ({len(json_str)} chars)")
            rankings = json.loads(json_str)
        else:
            print(f"⚠️  No JSON array pattern found, trying to parse entire response")
            rankings = json.loads(response_text)

        print(f"✓ Successfully parsed {len(rankings)} rankings")

        # Build ranked results, tracking which papers have been added
        ranked_papers = []
        seen_indices = set()

        # Add papers from LLM ranking (skip duplicates)
        for item in rankings:
            idx = item["index"] - 1  # Convert to 0-based index
            score = item["score"]

            if 0 <= idx < len(candidate_papers) and idx not in seen_indices:
                ranked_papers.append((candidate_papers[idx], score))
                seen_indices.add(idx)

        # Add any papers that weren't ranked by the LLM (with score 0.0)
        unranked_count = 0
        for i, paper in enumerate(candidate_papers):
            if i not in seen_indices:
                ranked_papers.append((paper, 0.0))
                seen_indices.add(i)
                unranked_count += 1

        if unranked_count > 0:
            print(f"⚠️  {unranked_count}/{len(candidate_papers)} papers not ranked by LLM (assigned score 0.0)")

        return {"ranked_papers": ranked_papers}

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"❌ Error parsing LLM response: {e}")
        print(f"❌ Response was: {response_text[:500]}...")
        print(f"❌ Fallback: returning papers in original order with score 0.1")
        # Fallback: return papers in original order with low scores
        return {"ranked_papers": [(paper, 0.1) for paper in candidate_papers]}
