"""
CiteAgent Multi-Agent Pipeline Template

This is a template for implementing the CiteAgent system within the evaluation framework.

Architecture:
    Query → Agent 1: Retrieval (top-100)
          → Agent 2: Reranking (top-20)
          → Agent 3: LLM Selection (top-k)
"""

from typing import List, Dict, Any, Optional
from .base_model import BaseRetrievalModel
from .dense_model import DenseRetrievalModel

# Import your LLM and reranking libraries here
# from langchain_openai import ChatOpenAI
# from sentence_transformers import CrossEncoder


class CiteAgentModel(BaseRetrievalModel):
    """
    Multi-agent citation retrieval system.

    Combines:
    1. Retrieval Agent: Dense retrieval for broad candidate selection
    2. Reranking Agent: Cross-encoder or dense reranking
    3. Selection Agent: LLM-based final selection with reasoning
    """

    def __init__(
        self,
        retriever_model: str = "allenai/specter2",
        reranker_model: Optional[str] = None,
        llm_model: str = "gpt-4",
        stage1_k: int = 100,
        stage2_k: int = 20,
        use_llm_selection: bool = True
    ):
        """
        Args:
            retriever_model: Dense retrieval model (SPECTER2, E5, etc.)
            reranker_model: Reranking model (cross-encoder or None)
            llm_model: LLM for final selection (gpt-4, claude, etc.)
            stage1_k: Number of candidates from retrieval
            stage2_k: Number of candidates after reranking
            use_llm_selection: Whether to use LLM for final selection
        """
        super().__init__("CiteAgent")

        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.use_llm_selection = use_llm_selection

        # Agent 1: Retrieval
        print(f"🔧 Initializing Agent 1: Retrieval ({retriever_model})")
        self.retriever = DenseRetrievalModel(retriever_model)

        # Agent 2: Reranking (optional)
        if reranker_model:
            print(f"🔧 Initializing Agent 2: Reranking ({reranker_model})")
            self.reranker = self._init_reranker(reranker_model)
        else:
            self.reranker = None

        # Agent 3: LLM Selection (optional)
        if use_llm_selection:
            print(f"🔧 Initializing Agent 3: LLM Selection ({llm_model})")
            self.llm = self._init_llm(llm_model)
        else:
            self.llm = None

        print("✅ CiteAgent initialized")

    def _init_reranker(self, model_name: str):
        """
        Initialize reranking model.

        Options:
        1. Cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-12-v2)
        2. Dense reranker (different embedding model)
        3. Custom reranking logic
        """
        # Example with cross-encoder
        # from sentence_transformers import CrossEncoder
        # return CrossEncoder(model_name)

        # For now, return None (implement when ready)
        return None

    def _init_llm(self, model_name: str):
        """
        Initialize LLM for final selection.

        Options:
        1. OpenAI (gpt-4, gpt-3.5-turbo)
        2. Anthropic (claude-3-opus, claude-3-sonnet)
        3. Open-source (Llama, Mixtral via vLLM)
        """
        # Example with OpenAI
        # from langchain_openai import ChatOpenAI
        # return ChatOpenAI(model=model_name, temperature=0)

        # For now, return None (implement when ready)
        return None

    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Multi-stage retrieval pipeline.

        Args:
            query: Citation context
            corpus: Candidate papers
            k: Final number of results

        Returns:
            Top-k ranked papers
        """
        # Stage 1: Dense Retrieval (broad candidate selection)
        candidates = self._stage1_retrieval(query, corpus)

        # Stage 2: Reranking (optional)
        if self.reranker:
            candidates = self._stage2_reranking(query, candidates)

        # Stage 3: LLM Selection (optional)
        if self.llm:
            candidates = self._stage3_llm_selection(query, candidates, k)
        else:
            candidates = candidates[:k]

        return candidates

    def _stage1_retrieval(
        self,
        query: str,
        corpus: List[Dict]
    ) -> List[Dict]:
        """
        Agent 1: Retrieve top candidates using dense retrieval.

        Goal: Cast a wide net to ensure correct citation is in candidates
        """
        candidates = self.retriever.retrieve(
            query=query,
            corpus=corpus,
            k=min(self.stage1_k, len(corpus))
        )

        # Add metadata
        for cand in candidates:
            cand['stage1_score'] = cand['score']
            cand['passed_stage1'] = True

        return candidates

    def _stage2_reranking(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Agent 2: Rerank candidates with more sophisticated model.

        Options:
        1. Cross-encoder reranking
        2. LLM-based scoring
        3. Ensemble of multiple retrieval methods
        """
        if not self.reranker:
            return candidates[:self.stage2_k]

        # TODO: Implement reranking
        # Example with cross-encoder:
        # pairs = [[query, c['text']] for c in candidates]
        # scores = self.reranker.predict(pairs)
        # for cand, score in zip(candidates, scores):
        #     cand['stage2_score'] = score
        # candidates = sorted(candidates, key=lambda x: x['stage2_score'], reverse=True)

        # For now, just return top stage2_k
        return candidates[:self.stage2_k]

    def _stage3_llm_selection(
        self,
        query: str,
        candidates: List[Dict],
        k: int
    ) -> List[Dict]:
        """
        Agent 3: LLM-based final selection with reasoning.

        The LLM analyzes the query context and candidates to make
        an informed decision about which citations are most relevant.
        """
        if not self.llm:
            return candidates[:k]

        # TODO: Implement LLM selection
        # Example prompt:
        prompt = self._build_selection_prompt(query, candidates, k)

        # # Get LLM response
        # response = self.llm.invoke(prompt)
        # selected_indices = self._parse_llm_response(response)

        # # Reorder candidates based on LLM selection
        # selected = [candidates[i] for i in selected_indices[:k]]

        # For now, just return top k
        return candidates[:k]

    def _build_selection_prompt(
        self,
        query: str,
        candidates: List[Dict],
        k: int
    ) -> str:
        """
        Build prompt for LLM to select best citations.

        The prompt should:
        1. Provide the citation context
        2. List candidate papers with titles and abstracts
        3. Ask LLM to select top-k most relevant
        4. Request reasoning for each selection
        """
        prompt = f"""You are a citation expert helping select the most relevant papers for a citation context.

CITATION CONTEXT:
{query}

CANDIDATE PAPERS:
"""
        for i, cand in enumerate(candidates, 1):
            prompt += f"\n{i}. {cand['title']}\n"
            if cand.get('abstract'):
                prompt += f"   Abstract: {cand['abstract'][:200]}...\n"

        prompt += f"""

TASK:
Select the top {k} most relevant papers for this citation context.
For each selection, provide:
1. Paper number (1-{len(candidates)})
2. Relevance score (0-10)
3. Brief reasoning

Respond in JSON format:
{{
  "selections": [
    {{"paper_num": 1, "score": 9.5, "reasoning": "..."}},
    ...
  ]
}}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> List[int]:
        """Parse LLM response to extract selected paper indices"""
        # TODO: Implement JSON parsing of LLM response
        # import json
        # data = json.loads(response)
        # return [sel['paper_num'] - 1 for sel in data['selections']]
        return list(range(len(response)))

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'model_name': 'CiteAgent',
            'retriever': self.retriever.model_name,
            'reranker': self.reranker if self.reranker else None,
            'llm': 'gpt-4' if self.llm else None,
            'stage1_k': self.stage1_k,
            'stage2_k': self.stage2_k,
            'use_llm_selection': self.use_llm_selection
        }


# Example usage
if __name__ == "__main__":
    """
    Example of how to use CiteAgent in evaluation.
    """
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from evaluation import CitationEvaluator, CitationDataLoader

    # Load data
    loader = CitationDataLoader("datasets/scholar_copilot_eval_data_1k.json")
    loader.load_data()
    examples = loader.extract_examples()[:10]  # Test on 10 examples

    # Initialize CiteAgent
    citeagent = CiteAgentModel(
        retriever_model="allenai/specter2",
        reranker_model=None,  # Implement when ready
        use_llm_selection=False  # Implement when ready
    )

    # Evaluate
    evaluator = CitationEvaluator()
    results = evaluator.evaluate_model(
        model=citeagent,
        examples=examples,
        top_k=10
    )

    print(results['metrics'])

    # Compare with baseline
    from .bm25_model import BM25Model

    models = {
        'BM25': BM25Model(),
        'CiteAgent': citeagent
    }

    comparison = evaluator.compare_models(models, examples)
    print(comparison)
