"""Model wrappers for citation retrieval evaluation"""

from .base_model import BaseRetrievalModel
from .bm25_model import BM25Model
from .dense_model import DenseRetrievalModel
from .citeagent_model import CiteAgentLocal

# Try to import real CiteAgent (requires API keys)
try:
    from .citeagent_real import CiteAgentAPI
    __all__ = ["BaseRetrievalModel", "BM25Model", "DenseRetrievalModel", "CiteAgentLocal", "CiteAgentAPI"]
except ImportError as e:
    print(f"Warning: Could not import CiteAgentAPI: {e}")
    __all__ = ["BaseRetrievalModel", "BM25Model", "DenseRetrievalModel", "CiteAgentLocal"]
