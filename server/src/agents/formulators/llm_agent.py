from typing import List, Dict, Any

# from langchain_ollama import ChatOllama  # Commented out - using Hugging Face instead
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re

from src.prompts.llm_reranker import LLMRerankerPrompt

load_dotenv()


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

    print(f"   Query: {query[:80]}..." if query else "   Query: (empty)")
    print(f"   Candidates in state: {len(candidate_papers)}")

    if not candidate_papers:
        print(f"⚠️  No candidate papers to rerank")
        return {"ranked_papers": []}

    # Use specified model or default from env
    model_id = os.getenv("LOCAL_LLM", "google/gemma-3-4b-it")  # Using gemma-3-4b-it by default

    if closed_source:
        llm = ChatOpenAI("gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        # Using Hugging Face model loaded locally via transformers pipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        print(f"🔄 Loading {model_id}...")
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
            max_new_tokens=256,
            do_sample=False,
        )

        llm = HuggingFacePipeline(pipeline=gen)
        print(f"✅ Model loaded!")

        # # OLLAMA VERSION (commented out):
        # from langchain_ollama import ChatOllama
        # llm = ChatOllama(model=model, temperature=0)

    prompt = LLMRerankerPrompt(query=query, candidate_papers=candidate_papers).get_prompt()

    print(f"🤖 LLM Reranking with model: {model_id}...")

    # Call LLM
    response = llm.invoke(prompt)
    response_text = response.content

    # Extract JSON from response
    try:
        # Try to find JSON array in the response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            rankings = json.loads(json_match.group())
        else:
            rankings = json.loads(response_text)

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
        for i, paper in enumerate(candidate_papers):
            if i not in seen_indices:
                ranked_papers.append((paper, 0.0))
                seen_indices.add(i)

        return {"ranked_papers": ranked_papers}

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"⚠️  Error parsing LLM response: {e}")
        print(f"Response was: {response_text}")
        # Fallback: return papers in original order with low scores
        return {"ranked_papers": [(paper, 0.1) for paper in candidate_papers]}
