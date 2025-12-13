from typing import List, Dict, Any
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re

from src.prompts.llm_reranker import LLMRerankerPrompt

load_dotenv()


def llm_reranker(state: MessagesState, closed_source: bool = False):
    """
    LLM-based reranking agent.
    Takes candidate papers from state and reranks them using an LLM.

    Args:
        state: MessagesState containing query and candidate_papers
        model_name: Optional model name to use (defaults to LOCAL_LLM env var)

    Returns:
        Dictionary with ranked_papers containing (paper, relevance_score) tuples
    """
    query = getattr(state, "query", "")
    candidate_papers = getattr(state, "candidate_papers", [])

    if not candidate_papers:
        return {"ranked_papers": []}

    # Use specified model or default from env
    model = os.getenv("LOCAL_LLM", "gemma3:4b")
    if closed_source:
        llm = ChatOpenAI("gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        llm = ChatOllama(model=model, temperature=0)

    prompt = LLMRerankerPrompt(query=query, candidate_papers=candidate_papers).build_prompt()

    print(f"🤖 LLM Reranking with model: {model}...")

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
