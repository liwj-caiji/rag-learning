"""Generator abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Generator(ABC):
    """Abstract generator for RAG responses."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Dict],
        intent: str,
    ) -> str:
        """Generate a response from retrieved context."""
