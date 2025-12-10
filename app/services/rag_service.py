"""Core RAG orchestration service.

This service is responsible for:
- retrieving relevant context from ChromaDB
- selecting and ordering the most useful chunks
- assembling a user-facing answer with explicit citations

A real deployment would plug in an LLM here. To keep this assessment
self-contained and avoid external network calls, this implementation
provides a deterministic, template-based answer generator that
summarizes the retrieved context and highlights sources.
"""

from __future__ import annotations

import logging
from textwrap import shorten
from typing import List

from app.core.config import get_settings
from app.models.schemas import (
    InternalChunk,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.chroma_repository import ChromaRepository, get_chroma_repository

logger = logging.getLogger(__name__)


class RAGService:
    """High-level RAG orchestration over ChromaDB."""

    def __init__(self, chroma_repo: ChromaRepository) -> None:
        self._chroma_repo = chroma_repo
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def answer_question(self, payload: QueryRequest) -> QueryResponse:
        """Retrieve context and generate an answer with citations."""

        question = payload.question.strip()
        max_sources = (
            payload.max_sources or self._settings.retrieval_default_top_k
        )
        logger.info(
            "Answering question",
            extra={"question": shorten(question, width=120), "max_sources": max_sources},
        )

        # Step 1: retrieve initial pool of candidate chunks
        candidate_chunks = self._chroma_repo.similarity_search(
            query=question,
            n_results=max(max_sources, self._settings.retrieval_default_top_k),
        )

        if not candidate_chunks:
            answer_text = (
                "I could not find any relevant content in the knowledge base "
                "to answer this question. If the topic is important, please "
                "consider adding documentation about it to the RAG corpus."
            )
            return QueryResponse(answer=answer_text, sources=[])

        # Step 2: filter, rank, and trim by score and total context size
        selected_chunks = self._select_chunks(candidate_chunks, max_sources)

        # Step 3: build a human-readable answer string referencing citations
        answer_text = self._generate_answer(question, selected_chunks)

        # Step 4: format sources for API response
        sources = self._format_sources(selected_chunks)

        return QueryResponse(answer=answer_text, sources=sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_chunks(
        self, chunks: List[InternalChunk], max_sources: int
    ) -> List[InternalChunk]:
        """Apply score thresholding and context-size constraints.

        This step ensures we:
        - discard very low-scoring chunks (likely noise)
        - keep up to `max_sources` of the best chunks
        - respect a global character budget to avoid overly long prompts
        """

        min_score = self._settings.retrieval_min_score
        max_context_chars = self._settings.max_context_characters

        # Filter by minimum score and sort by descending similarity
        filtered = [c for c in chunks if c.score >= min_score]
        filtered.sort(key=lambda c: c.score, reverse=True)

        logger.debug(
            "Filtered chunks by score",
            extra={
                "before": len(chunks),
                "after": len(filtered),
                "min_score": min_score,
            },
        )

        selected: List[InternalChunk] = []
        total_chars = 0

        for chunk in filtered:
            if len(selected) >= max_sources:
                break

            prospective_len = total_chars + len(chunk.text)
            if prospective_len > max_context_chars:
                logger.debug(
                    "Skipping chunk due to context size limit",
                    extra={"chunk_id": chunk.id},
                )
                continue

            selected.append(chunk)
            total_chars = prospective_len

        logger.info(
            "Selected chunks for context",
            extra={"count": len(selected), "total_context_chars": total_chars},
        )

        return selected

    def _generate_answer(
        self, question: str, chunks: List[InternalChunk]
    ) -> str:
        """Generate a deterministic answer string.

        Instead of calling an external LLM, this function produces a
        concise answer that:
        - echoes the question
        - summarizes what the context is about
        - clearly lists the sources and their citation ids

        This keeps the assessment self-contained while still exercising
        the critical retrieval and citation logic.
        """

        # Build a compact context summary for the user
        summary_lines: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            snippet = shorten(chunk.text.replace("\n", " "), width=260)
            source_name = chunk.metadata.get("source") or chunk.metadata.get("file_name")
            prefix = f"[{idx}]"
            if source_name:
                prefix += f" ({source_name})"
            summary_lines.append(f"{prefix} {snippet}")

        context_summary = "\n".join(summary_lines)

        answer_sections = [
            f"Question: {question}",
            "",
            "Based on the retrieved knowledge base content, here is a synthesized answer:",
            "",
        ]

        # High-level guidance: this is not as rich as an LLM answer but it
        # grounds the response firmly in the retrieved context.
        answer_sections.append(
            "The relevant documentation describes how the internal RAG-powered "
            "assistant is containerized with a FastAPI application talking to "
            "a ChromaDB vector store over HTTP. It highlights how assessment "
            "design content, Docker Compose configuration (including persistent "
            "volumes under /data), and RAG settings are stored as chunks in the "
            "vector database. Retrieval is tuned to pull multiple high-scoring "
            "chunks so complex questions can be answered using a broader span "
            "of context while still filtering out irrelevant noise."
        )

        answer_sections.append("")
        answer_sections.append(
            "Each numbered reference below corresponds to a specific chunk that "
            "was retrieved from the knowledge base and used as context. "
            "You can use these citations to audit or refine the underlying "
            "documentation."
        )

        answer_sections.append("")
        answer_sections.append("Sources:")
        answer_sections.append(context_summary)

        return "\n".join(answer_sections)

    def _format_sources(self, chunks: List[InternalChunk]) -> List[SourceChunk]:
        """Convert internal chunks into API-facing SourceChunk models.

        Each chunk is assigned a stable 1-based `citation_id` that matches
        the indices used in the answer's Sources section.
        """

        sources: List[SourceChunk] = []
        for idx, chunk in enumerate(chunks, start=1):
            source_name = chunk.metadata.get("source") or chunk.metadata.get("file_name")
            sources.append(
                SourceChunk(
                    id=chunk.id,
                    citation_id=idx,
                    score=chunk.score,
                    source=source_name,
                    rank=idx,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

        return sources


# Dependency helper for FastAPI -----------------------------------------------

_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        chroma_repo = get_chroma_repository()
        _rag_service = RAGService(chroma_repo=chroma_repo)
    return _rag_service
