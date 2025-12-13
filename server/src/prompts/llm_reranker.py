from __future__ import annotations

from typing import Any, Dict, Sequence


class LLMRerankerPrompt:
    """Prompt builder for the LLM reranker."""

    def __init__(self, query: str, candidate_papers: Sequence[Dict[str, Any]]):
        self.query = query
        self.candidate_papers = candidate_papers

    def get_prompt(self) -> str:
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
