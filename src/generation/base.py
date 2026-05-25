"""Generator abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Generator(ABC):
    """Abstract generator for RAG responses."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Dict],
        intent: str,
        target_dish: Optional[str] = None,
    ) -> str:
        """Generate a response from retrieved context."""
