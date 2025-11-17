"""
LLM baseline that selects citations from the local corpus.

This replaces the CiteAgent baseline with a simpler approach:
- Prefilter candidates using a lightweight lexical similarity
- Ask an LLM (ChatGPT/Claude/Together) to pick the top-k most relevant
"""

from typing import List, Dict, Any, Tuple
import os
import json
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

from .base_model import BaseRetrievalModel


class LLMCitationChooser(BaseRetrievalModel):
	"""
	Simple LLM-based chooser that ranks/chooses citations from the provided corpus.

	Flow per query:
	1) Prefilter local corpus to top-N using lightweight lexical scoring
	2) Prompt an LLM with the context and the N candidates (titles + short abstracts)
	3) Parse the LLM JSON response indicating selected candidate indices
	4) Return mapped top-k items from the original corpus with ranks
	"""

	def __init__(
		self,
		llm_backend: str = "gpt-4o",
		temperature: float = 0.0,
		max_candidates: int = 100,
		# Accept extra kwargs for compatibility with original CiteAgent constructor
		search_limit: int | None = None,
		max_actions: int | None = None,
		prompt_name: str | None = None,
	):
		super().__init__(f"LLM-Baseline-{llm_backend}")
		self.llm_backend = llm_backend
		self.temperature = temperature
		self.max_candidates = max_candidates
		# Ignored but kept for signature compatibility
		self._compat_search_limit = search_limit
		self._compat_max_actions = max_actions
		self._compat_prompt_name = prompt_name

		# Load .env if present
		load_dotenv()

		self._client = self._init_llm_client(llm_backend, temperature)

	def retrieve(
		self,
		query: str,
		corpus: List[Dict],
		k: int = 10
	) -> List[Dict]:
		if not corpus:
			return []

		# Step 1: prefilter corpus to reduce token usage
		subset, index_map = self._prefilter_candidates(query, corpus, self.max_candidates)

		# Step 2: ask the LLM to select the best citations
		selection_indices = self._ask_llm_for_selection(query, subset, k)

		# Step 3: build results mapped to original corpus indices
		results: List[Dict] = []
		used = set()
		rank_counter = 1
		for cand_idx in selection_indices:
			if 0 <= cand_idx < len(subset):
				orig_idx = index_map[cand_idx]
				if orig_idx in used:
					continue
				item = corpus[orig_idx].copy()
				item['score'] = float(1.0 - (rank_counter - 1) * 0.01)
				item['rank'] = rank_counter
				results.append(item)
				used.add(orig_idx)
				rank_counter += 1
				if len(results) >= k:
					break

		# Fallback: if LLM didn't return enough items, pad with remaining prefiltered
		if len(results) < k:
			for local_idx, cand in enumerate(subset):
				orig_idx = index_map[local_idx]
				if orig_idx in used:
					continue
				item = corpus[orig_idx].copy()
				item['score'] = float(1.0 - (rank_counter - 1) * 0.01)
				item['rank'] = rank_counter
				results.append(item)
				used.add(orig_idx)
				rank_counter += 1
				if len(results) >= k:
					break

		return results

	def _prefilter_candidates(
		self,
		query: str,
		corpus: List[Dict],
		max_candidates: int
	) -> Tuple[List[Dict], List[int]]:
		"""
		Lightweight lexical/keyword scoring to select a manageable subset.
		Returns the subset and a mapping to original indices.
		"""
		query_lower = query.lower()
		query_words = set(query_lower.split())
		stopwords = {
			'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
			'by', 'from', 'as', 'is', 'was', 'are', 'were', 'and', 'or'
		}
		query_words = query_words - stopwords

		def make_bigrams(text: str) -> set[str]:
			tokens = text.split()
			return set([" ".join(pair) for pair in zip(tokens, tokens[1:])])

		def score_entry(entry: Dict) -> float:
			title = (entry.get('title') or '').lower()
			abstract = (entry.get('abstract') or '')[:800].lower()
			text = f"{title} {abstract}"
			word_overlap = 0.0
			if query_words:
				word_overlap = len(query_words & set(text.split())) / max(1, len(query_words))
			sequence_score = SequenceMatcher(None, query_lower, text).ratio()
			# Bigram overlap to emphasize phrasal matches beyond BM25-like signals
			query_bigrams = make_bigrams(query_lower)
			text_bigrams = make_bigrams(text)
			bigram_overlap = 0.0
			if query_bigrams:
				bigram_overlap = len(query_bigrams & text_bigrams) / max(1, len(query_bigrams))
			return 0.45 * word_overlap + 0.35 * sequence_score + 0.20 * bigram_overlap

		scored = []
		for idx, entry in enumerate(corpus):
			s = score_entry(entry)
			scored.append((s, idx, entry))
		scored.sort(key=lambda x: x[0], reverse=True)

		top = scored[:min(max_candidates, len(scored))]
		subset = [e for _, _, e in top]
		index_map = [i for _, i, _ in top]
		return subset, index_map

	def _build_prompt(self, query: str, candidates: List[Dict], k: int) -> str:
		lines = []
		lines.append("You are an expert at selecting the most relevant academic citation for a given context.")
		lines.append("Given a citation context and a set of candidate references, select the top items that best match the context.")
		lines.append("")
		lines.append("Return ONLY a compact JSON object with an array of 0-based indices of the chosen candidates, ordered from most to least relevant.")
		lines.append("The schema is: {\"indices\": [i0, i1, ...]}. Do not include explanations.")
		lines.append("If you cannot decide, still return a best-effort list of indices. Never return an empty object.")
		lines.append("")
		lines.append("Citation context:")
		lines.append(query.strip())
		lines.append("")
		lines.append("Candidates:")
		for i, cand in enumerate(candidates):
			title = (cand.get('title') or '').strip()
			abstract = (cand.get('abstract') or '').strip().replace("\n", " ")
			if len(abstract) > 220:
				abstract = abstract[:220] + "..."
			lines.append(f"{i}. {title}")
			if abstract:
				lines.append(f"   Abstract: {abstract}")
		lines.append("")
		lines.append(f"Select the top {k} most relevant candidates.")
		lines.append("Respond with JSON only:")
		lines.append('{"indices": [0, 3, 5]}')
		return "\n".join(lines)

	def _ask_llm_for_selection(self, query: str, candidates: List[Dict], k: int) -> List[int]:
		# Guard
		if not candidates:
			return []

		prompt = self._build_prompt(query, candidates, k)
		try:
			content = self._invoke_llm(prompt)
		except Exception:
			return list(range(min(k, len(candidates))))

		# Try strict JSON first
		parsed = self._parse_indices_json(content)
		if parsed:
			return [i for i in parsed if 0 <= i < len(candidates)][:k]

		# Fallback: extract integers from text
		nums = [int(n) for n in re.findall(r"\b\d+\b", content)]
		return [i for i in nums if 0 <= i < len(candidates)][:k]

	def _parse_indices_json(self, text: str) -> List[int] | None:
		text = (text or "").strip()
		# Normalize common code-fence wrappers
		if text.startswith("```") and text.endswith("```"):
			text = text.strip("`").strip()
			# Remove possible language identifiers like ```json
			text = re.sub(r'^\s*json\s*', '', text, flags=re.IGNORECASE)
		# Try direct JSON parse
		try:
			obj = json.loads(text)
			if isinstance(obj, dict) and isinstance(obj.get("indices"), list):
				ints = []
				for v in obj["indices"]:
					if isinstance(v, int):
						ints.append(v)
				return ints
		except Exception:
			pass
		# Try to locate a JSON object in the text
		m = re.search(r"\{.*\}", text, flags=re.DOTALL)
		if m:
			try:
				obj = json.loads(m.group(0))
				if isinstance(obj, dict) and isinstance(obj.get("indices"), list):
					ints = []
					for v in obj["indices"]:
						if isinstance(v, int):
							ints.append(v)
					return ints
			except Exception:
				return None
		return None

	def _init_llm_client(self, backend: str, temperature: float):
		"""
		Initialize an LLM client via LangChain interfaces.
		Supports:
		  - OpenAI (gpt-*)
		  - Anthropic (claude-*)
		  - Together (meta-llama*, mixtral*, etc.)
		"""
		if backend.startswith("gpt"):
			api_key = os.getenv("OPENAI_API_KEY")
			if not api_key:
				raise ValueError("OPENAI_API_KEY not found in environment.")
			try:
				from langchain_openai import ChatOpenAI
			except ImportError as e:
				raise ImportError(f"langchain-openai not installed: {e}")
			return ChatOpenAI(model=backend, temperature=temperature)

		if "claude" in backend:
			api_key = os.getenv("ANTHROPIC_API_KEY")
			if not api_key:
				raise ValueError("ANTHROPIC_API_KEY not found in environment.")
			try:
				from langchain_anthropic import ChatAnthropic
			except ImportError as e:
				raise ImportError(f"langchain-anthropic not installed: {e}")
			return ChatAnthropic(model=backend, temperature=temperature)

		# Default to Together for other backend strings
		api_key = os.getenv("TOGETHER_API_KEY")
		if not api_key:
			raise ValueError("TOGETHER_API_KEY not found in environment for non-OpenAI/Anthropic model.")
		try:
			from langchain_together import ChatTogether
		except ImportError as e:
			raise ImportError(f"langchain-together not installed: {e}")
		return ChatTogether(model=backend, temperature=temperature)

	def _invoke_llm(self, prompt: str) -> str:
		"""
		Invoke the underlying LLM client with a single-turn prompt.
		"""
		# LangChain chat models accept a list of messages; use system+user minimal structure
		messages = [
			("system", "You return strictly formatted JSON and nothing else."),
			("user", prompt)
		]
		# Most LC chat models expose invoke() with list of tuples or Messages; coerce via dicts
		try:
			# Newer LC chat models accept plain strings via invoke as well; prefer messages
			response = self._client.invoke(messages)  # type: ignore
		except Exception:
			# Fallback to sending plain string
			response = self._client.invoke(prompt)  # type: ignore

		# Extract text content from LC message-like object
		content = ""
		if hasattr(response, "content"):
			content = response.content  # type: ignore
		elif isinstance(response, dict) and "content" in response:
			content = str(response["content"])
		else:
			content = str(response)
		return content or ""

	def get_config(self) -> Dict[str, Any]:
		return {
			'model_name': self.model_name,
			'llm_backend': self.llm_backend,
			'temperature': self.temperature,
			'max_candidates': self.max_candidates,
			'method': 'llm_selection_over_local_corpus'
		}


