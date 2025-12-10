"""Request and response models for the RAG API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming payload for a RAG query.

    Attributes
    ----------
    question:
        User's natural language question.
    max_sources:
        Optional cap on the number of context chunks to return. If not
        provided, a sensible default from configuration is used.
    """

    question: str = Field(..., min_length=1)
    max_sources: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of context chunks to return. "
            "If omitted, the service default is used."
        ),
    )


class SourceChunk(BaseModel):
    """Metadata about a retrieved chunk used as context for an answer."""

    id: str = Field(..., description="Chunk or document id in ChromaDB.")
    citation_id: int = Field(
        ..., description="Stable 1-based index used in the answer text."
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score.")
    source: Optional[str] = Field(
        None,
        description="Logical source identifier (e.g. file path or URL).",
    )
    rank: int = Field(..., ge=1, description="Rank position after retrieval.")
    text: str = Field(..., description="The actual text content of the chunk.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata stored in Chroma for this chunk.",
    )


class QueryResponse(BaseModel):
    """Full RAG response including the answer and its citations."""

    answer: str = Field(
        ..., description="Natural language answer, with a Sources section."
    )
    sources: List[SourceChunk] = Field(
        ..., description="Chunks retrieved from the vector store."
    )


class InternalChunk(BaseModel):
    """Internal representation of a retrieved chunk.

    This model is not exposed on the API but is used inside the RAG
    service to organize and score chunks before formatting them for the
    client.
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
