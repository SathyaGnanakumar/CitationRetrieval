"""Model wrappers for citation retrieval evaluation"""

from .base_model import BaseRetrievalModel
from .bm25_model import BM25Model
from .dense_model import DenseRetrievalModel
from .citeagent_model import CiteAgentLocal

# Replace CiteAgent with a simple LLM baseline over local corpus
from .llm_baseline import LLMCitationChooser as CiteAgentAPI

__all__ = ["BaseRetrievalModel", "BM25Model", "DenseRetrievalModel", "CiteAgentLocal", "CiteAgentAPI"]
