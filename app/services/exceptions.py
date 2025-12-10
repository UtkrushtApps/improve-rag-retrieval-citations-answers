"""Custom exceptions for the RAG service."""

from __future__ import annotations


class ChromaUnavailableError(RuntimeError):
    """Raised when ChromaDB is not reachable or returns an unexpected error."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
