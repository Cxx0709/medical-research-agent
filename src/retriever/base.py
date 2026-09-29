"""
Abstract Base Retriever for Medical Literature
"""

from abc import ABC, abstractmethod
from typing import List
from src.models import Paper


class BaseRetriever(ABC):
    """Abstract interface for medical paper retrieval."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search medical literature given a query or search terms."""
        pass

    @abstractmethod
    async def asearch(self, query: str, max_results: int = 5) -> List[Paper]:
        """Asynchronously search medical literature."""
        pass
