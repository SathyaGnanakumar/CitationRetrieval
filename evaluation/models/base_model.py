"""
Base class for all retrieval models.

Defines the interface that all models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetrievalModel(ABC):
    """
    Abstract base class for citation retrieval models.

    All models must implement the retrieve() method.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name identifier for this model
        """
        self.model_name = model_name

    @abstractmethod
    def retrieve(
        self,
        query: str,
        corpus: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant citations for a query.

        Args:
            query: Citation context (text around citation marker)
            corpus: List of candidate citations, each dict should have:
                - 'title': str
                - 'abstract': str (optional)
                - 'text': str (combined title + abstract)
            k: Number of candidates to retrieve

        Returns:
            List of top-k citations (dicts from corpus), ranked by relevance
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration.

        Returns:
            Dict with model parameters and settings
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
