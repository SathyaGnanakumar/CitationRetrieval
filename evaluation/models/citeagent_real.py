"""
Real CiteAgent Implementation - LLM + Semantic Scholar API

This is the REAL CiteAgent that uses:
- LLM-powered reasoning (GPT-4o, Claude, or Together AI models)
- Semantic Scholar API for paper search
- Multi-step agent actions (search, read, select)

Requirements:
- API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, or TOGETHER_API_KEY)
- S2_API_KEY for Semantic Scholar
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
from .base_model import BaseRetrievalModel

# Load environment variables from .env file
load_dotenv()

# Add cite_agent to path
CITE_AGENT_PATH = Path(__file__).parent.parent.parent / "baselines" / "cite_agent" / "src"
sys.path.insert(0, str(CITE_AGENT_PATH))

try:
    from retriever.agent import LLMSelfAskAgentPydantic, PaperNotFoundError
    from utils.data_model import Output
    from utils.str_matcher import is_similar
except ImportError as e:
    raise ImportError(
        f"Could not import CiteAgent modules. Make sure baselines/cite_agent/ is available. Error: {e}"
    )


class CiteAgentAPI(BaseRetrievalModel):
    """
    Real CiteAgent using LLM + Semantic Scholar API.

    This implementation:
    - Searches Semantic Scholar API (not local corpus)
    - Uses LLM for multi-step reasoning
    - Performs actions: search_relevance, search_citation_count, read, select
    - Much slower but more accurate than simple baselines

    Cost: ~$0.01-0.05 per query depending on LLM backend
    Speed: 15-30 seconds per query
    """

    def __init__(
        self,
        llm_backend: str = "gpt-4o",
        search_limit: int = 10,
        max_actions: int = 15,
        temperature: float = 0.0,
        prompt_name: str = "few_shot_search"
    ):
        """
        Initialize Real CiteAgent.

        Args:
            llm_backend: LLM to use ("gpt-4o", "claude-3-5-sonnet-20241022", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
            search_limit: Number of papers to return per search
            max_actions: Maximum agent actions per query
            temperature: LLM temperature
            prompt_name: Prompt template to use
        """
        super().__init__(f"CiteAgent-{llm_backend}")

        self.llm_backend = llm_backend
        self.search_limit = search_limit
        self.max_actions = max_actions
        self.temperature = temperature
        self.prompt_name = prompt_name

        # Check API keys
        self._check_api_keys()

        # Initialize agent
        print(f"🤖 Initializing Real CiteAgent with {llm_backend}...")
        try:
            self.agent = LLMSelfAskAgentPydantic(
                model_name=llm_backend,
                temperature=temperature,
                search_limit=search_limit,
                only_open_access=False,  # Allow all papers
                use_web_search=False,
                prompt_name=prompt_name,
                pydantic_object=Output
            )
            print(f"✅ CiteAgent initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CiteAgent: {e}")

    def _check_api_keys(self):
        """Check if required API keys are present"""
        # Check LLM API key
        if self.llm_backend.startswith("gpt"):
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Please set it in .env file or environment variables."
                )
        elif "claude" in self.llm_backend:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Please set it in .env file or environment variables."
                )
        elif "llama" in self.llm_backend.lower() or "mixtral" in self.llm_backend.lower():
            if not os.getenv("TOGETHER_API_KEY"):
                raise ValueError(
                    "TOGETHER_API_KEY not found in environment. "
                    "Please set it in .env file or environment variables."
                )

        # Check Semantic Scholar API key
        if not os.getenv("S2_API_KEY"):
            print("⚠️  Warning: S2_API_KEY not found. Semantic Scholar API may rate limit.")

    def _extract_year(self, query: str) -> str:
        """
        Extract publication year from citation context.

        Args:
            query: Citation context text

        Returns:
            Year as string, or empty string if not found
        """
        # Look for 4-digit years between 1900 and 2030
        year_pattern = r'\b(19\d{2}|20[0-2]\d|2030)\b'
        matches = re.findall(year_pattern, query)

        if matches:
            # Return the most recent year found
            return max(matches)
        return ""

    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve papers using Real CiteAgent.

        Args:
            query: Citation context
            corpus: List of candidate papers (used for matching back)
            k: Number to retrieve

        Returns:
            Top-k ranked papers from corpus
        """
        if not corpus:
            return []

        # Extract year from context
        year = self._extract_year(query)

        # Reset agent for new query
        corpus_titles = [paper.get('title', '') for paper in corpus]
        self.agent.reset(source_papers_title=corpus_titles)

        try:
            # Run agent to find the cited paper
            selected_paper = self.agent(excerpt=query, year=year, max_actions=self.max_actions)

            # selected_paper is a PaperSearchResult from Semantic Scholar
            # Now we need to match it back to the corpus

            # Try to find exact or similar match in corpus
            matched_papers = []
            for idx, corpus_paper in enumerate(corpus):
                corpus_title = corpus_paper.get('title', '')

                # Check if titles are similar
                if is_similar(selected_paper.title, corpus_title, threshold=0.8):
                    result_paper = corpus_paper.copy()
                    result_paper['score'] = 1.0 - (idx * 0.01)  # Slight preference for earlier matches
                    result_paper['rank'] = 1
                    result_paper['agent_selection'] = selected_paper.title
                    matched_papers.append(result_paper)

            if matched_papers:
                # Return the matched paper(s)
                return matched_papers[:k]
            else:
                # Agent found a paper, but it's not in corpus
                # Return corpus ranked by title similarity to agent's selection
                scored_papers = []
                for corpus_paper in corpus:
                    corpus_title = corpus_paper.get('title', '')
                    # Simple similarity score
                    score = self._calculate_similarity(selected_paper.title, corpus_title)
                    result_paper = corpus_paper.copy()
                    result_paper['score'] = score
                    result_paper['agent_selection'] = selected_paper.title
                    result_paper['note'] = 'Agent selected paper not in corpus, showing closest matches'
                    scored_papers.append(result_paper)

                # Sort by score
                scored_papers.sort(key=lambda x: x['score'], reverse=True)

                # Add ranks
                for i, paper in enumerate(scored_papers[:k]):
                    paper['rank'] = i + 1

                return scored_papers[:k]

        except PaperNotFoundError as e:
            print(f"⚠️  CiteAgent error: {e}")
            # Return empty or fallback to corpus order
            return corpus[:k]

        except ValueError as e:
            if "Max actions reached" in str(e):
                print(f"⚠️  CiteAgent reached max actions ({self.max_actions}), no selection made")
            else:
                print(f"⚠️  CiteAgent error: {e}")
            return corpus[:k]

        except Exception as e:
            print(f"⚠️  Unexpected CiteAgent error: {e}")
            return corpus[:k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Word overlap
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'model_name': f'CiteAgent-{self.llm_backend}',
            'llm_backend': self.llm_backend,
            'search_limit': self.search_limit,
            'max_actions': self.max_actions,
            'temperature': self.temperature,
            'prompt_name': self.prompt_name,
            'method': 'llm_agent_with_semantic_scholar_api',
            'cost_per_query': '$0.01-0.05 (estimated)',
            'note': 'Real CiteAgent - uses LLM + Semantic Scholar API'
        }
