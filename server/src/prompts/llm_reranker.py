from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


class LLMRerankerPrompt:
    """Prompt builder for the LLM reranker.

    Can use either:
    1. Default hardcoded prompt (original)
    2. DSPy-optimized prompt from file (if USE_DSPY_PROMPT=true)

    The DSPy prompt was generated from a preliminary optimization run on 10 citations
    using GPT-4o mini, demonstrating improved reasoning patterns.
    """

    def __init__(
        self,
        query: str,
        candidate_papers: Sequence[Dict[str, Any]],
        use_dspy_prompt: Optional[bool] = None
    ):
        self.query = query
        self.candidate_papers = candidate_papers

        # Check environment variable if not explicitly set
        if use_dspy_prompt is None:
            use_dspy_prompt = os.getenv("USE_DSPY_PROMPT", "false").lower() in ["true", "1", "yes", "y"]

        self.use_dspy_prompt = use_dspy_prompt

    def get_prompt(self) -> str:
        if self.use_dspy_prompt:
            return self.build_dspy_prompt(query=self.query, candidate_papers=self.candidate_papers)
        return self.build_prompt(query=self.query, candidate_papers=self.candidate_papers)

    @staticmethod
    def build_prompt(query: str, candidate_papers: Sequence[Dict[str, Any]]) -> str:
        prompt = f"""You are an expert research paper recommender. Given a query and a list of candidate papers, your task is to rerank these papers based on their relevance to the query.

Query: "{query}"

Candidate Papers:
"""

        for i, paper in enumerate(candidate_papers, 1):
            title = str(paper.get("title", "")).replace("\n", " ").strip()
            retriever_score = paper.get("score", 0.0)
            try:
                retriever_score_f = float(retriever_score)
            except (TypeError, ValueError):
                retriever_score_f = 0.0

            prompt += f"\n{i}. {title} (retriever score: {retriever_score_f:.4f})"

        prompt += """

Please analyze each paper's relevance to the query and return ONLY a JSON array with the paper indices in order of relevance (most relevant first). Also provide a relevance score from 0 to 1 for each paper.

Return your response in this exact JSON format:
[
  {"index": 1, "score": 0.95},
  {"index": 3, "score": 0.87},
  {"index": 2, "score": 0.65}
]

Return ONLY the JSON array, no additional text or explanation."""

        return prompt

    @staticmethod
    def build_dspy_prompt(query: str, candidate_papers: Sequence[Dict[str, Any]]) -> str:
        """
        DSPy-optimized prompt for reranking.

        Loads the base DSPy prompt from llm_prompt.txt and adapts it for the reranking task.
        This prompt was generated from optimization on 10 citations using GPT-4o mini.
        """
        # Load DSPy base prompt
        prompt_file = Path(__file__).parent / "llm_prompt.txt"

        if not prompt_file.exists():
            # Fallback to default if DSPy prompt file not found
            return LLMRerankerPrompt.build_prompt(query, candidate_papers)

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                dspy_base_prompt = f.read().strip()
        except Exception:
            # Fallback to default on error
            return LLMRerankerPrompt.build_prompt(query, candidate_papers)

        # Build the reranking-specific prompt using DSPy instructions
        prompt = f"""{dspy_base_prompt}

---

Reranking Task:

You are performing citation retrieval. Given a query (which may be a citation context from a paper) and a list of candidate papers, your task is to rerank these papers based on their relevance to the query.

Query: "{query}"

Candidate Papers:
"""

        for i, paper in enumerate(candidate_papers, 1):
            title = str(paper.get("title", "")).replace("\n", " ").strip()
            retriever_score = paper.get("score", 0.0)
            try:
                retriever_score_f = float(retriever_score)
            except (TypeError, ValueError):
                retriever_score_f = 0.0

            prompt += f"\n{i}. {title} (retriever score: {retriever_score_f:.4f})"

        prompt += """

Instructions:
1. Analyze the query to understand what specific claim, method, or result is being referenced
2. Evaluate each candidate paper to determine if it matches the context
3. Consider both semantic relevance and the retriever scores
4. Rank papers by their likelihood of being the correct citation

Return ONLY a JSON array with the paper indices in order of relevance (most relevant first), with relevance scores from 0 to 1.

Required JSON format:
[
  {"index": 1, "score": 0.95},
  {"index": 3, "score": 0.87},
  {"index": 2, "score": 0.65}
]

Return ONLY the JSON array, no additional text or explanation."""

        return prompt
