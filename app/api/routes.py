"""API routes for the RAG assistant.

Defines the query endpoint that engineers and content authors interact with.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    payload: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a natural-language question using RAG over ChromaDB.

    The response includes:
    - `answer`: natural language answer text
    - `sources`: the retrieved chunks with scores and metadata
    The answer string also contains a human-readable "Sources" section
    that references the same citation ids for traceability.
    """

    return await rag_service.answer_question(payload)
