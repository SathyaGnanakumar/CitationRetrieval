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
        # NEW PROMPT: Citation retrieval with context analysis
        prompt = f"""You are an expert citation retrieval system as described in the paper "Multi-Agent System for Reliable Citation Retrieval". Your goal is to autonomously retrieve, verify, and recommend academic references given a query or document excerpt.

Task: Given a citation context from a scientific paper (where a citation is missing), identify the correct paper from a list of candidates. Analyze the context to understand the specific claim, method, or result being cited. Then, evaluate each candidate paper to see if it matches the context. Finally, select the best matching paper.

Citation Context: {query}

Candidate Papers:
"""

        for i, paper in enumerate(candidate_papers, 1):
            title = str(paper.get("title", "")).replace("\n", " ").strip()
            abstract = (
                str(paper.get("abstract", "No abstract available")).replace("\n", " ").strip()
            )

            prompt += f"\n{i}. Title: {title}\n   Abstract: {abstract}\n"

        prompt += f"""
CRITICAL: You MUST rank ALL {len(candidate_papers)} papers in your JSON output. Do not skip any papers.

Think step-by-step about relevance, then output a JSON array ranking ALL papers from most to least relevant.

Format: [{{"index": 1, "score": 0.95}}, {{"index": 2, "score": 0.80}}, ...]

Rules:
- Include EVERY paper (indices 1 to {len(candidate_papers)})
- Most relevant papers: score 0.8-1.0
- Somewhat relevant: score 0.4-0.7
- Not relevant: score 0.0-0.3
- Output ONLY the JSON array, nothing else

Your JSON array with all {len(candidate_papers)} papers:"""

        return prompt

        # OLD PROMPT (commented out for reference):
        # prompt = f"""You are an expert research paper recommender. Given a query and a list of candidate papers, your task is to rerank these papers based on their relevance to the query.
        #
        # Query: "{query}"
        #
        # Candidate Papers:
        # """
        #
        # for i, paper in enumerate(candidate_papers, 1):
        #     title = str(paper.get("title", "")).replace("\n", " ").strip()
        #     retriever_score = paper.get("score", 0.0)
        #     try:
        #         retriever_score_f = float(retriever_score)
        #     except (TypeError, ValueError):
        #         retriever_score_f = 0.0
        #
        #     prompt += f"\n{i}. {title} (retriever score: {retriever_score_f:.4f})"
        #
        # prompt += """
        #
        # Please analyze each paper's relevance to the query and return ONLY a JSON array with the paper indices in order of relevance (most relevant first). Also provide a relevance score from 0 to 1 for each paper.
        #
        # Return your response in this exact JSON format:
        # [
        #   {"index": 1, "score": 0.95},
        #   {"index": 3, "score": 0.87},
        #   {"index": 2, "score": 0.65}
        # ]
        #
        # Return ONLY the JSON array, no additional text or explanation."""
        #
        # return prompt
