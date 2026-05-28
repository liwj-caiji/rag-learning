"""Generator abstract interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.documents import Document


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Document],
        intent: str,
        target_dish: Optional[str] = None,
    ) -> str:
        """Generate a response from retrieved context."""
