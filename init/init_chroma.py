"""One-time initialization script to populate Chroma with sample data.

In the real deployment this script would run in a dedicated Docker
service that depends_on the Chroma container. It connects over HTTP and
idempotently inserts a small corpus of domain documents covering:
- assessment design at Utkrusht
- Dockerized deployments with Docker Compose
- RAG configurations and best practices

This file is included to demonstrate how the application expects the
vector store to be structured. It is safe to run multiple times; inserts
are skipped when ids already exist.
"""

from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_sample_corpus() -> List[dict]:
    """Return a small sample corpus of RAG-related documents.

    In production, this would be replaced by a real loader that ingests
    markdown, notebooks, or other internal docs into chunks.
    """

    docs = []

    docs.append(
        {
            "id": "assessment-design-1",
            "text": (
                "Utkrusht assessments are designed to be modular and reusable. "
                "Each assessment specifies a clear task description, expected "
                "outcomes, and competencies such as Retrieval_Augmented_Generation."
            ),
            "metadata": {"source": "docs/assessments/overview.md", "topic": "assessments"},
        }
    )

    docs.append(
        {
            "id": "docker-compose-1",
            "text": (
                "The RAG assistant is deployed with Docker Compose. "
                "The FastAPI app, ChromaDB, and an initialization service "
                "are defined as separate services. ChromaDB mounts a named "
                "Docker volume at /data to persist vector store state."
            ),
            "metadata": {"source": "docs/deploy/docker-compose.md", "topic": "docker"},
        }
    )

    docs.append(
        {
            "id": "rag-config-1",
            "text": (
                "RAG configuration includes the number of results (top_k) "
                "retrieved from ChromaDB, similarity thresholds, and context "
                "size limits. Increasing top_k and enforcing a minimum score "
                "helps answer complex questions with enough relevant context "
                "while avoiding unrelated noise."
            ),
            "metadata": {"source": "docs/rag/configuration.md", "topic": "rag"},
        }
    )

    docs.append(
        {
            "id": "observability-1",
            "text": (
                "Production deployments of the RAG assistant should expose "
                "health endpoints, structured logging with request ids, and "
                "metrics about retrieval latency and result counts. This helps "
                "engineering teams debug connectivity issues with ChromaDB."
            ),
            "metadata": {"source": "docs/ops/observability.md", "topic": "ops"},
        }
    )

    return docs


def main() -> None:
    settings = get_settings()

    logger.info(
        "Connecting to Chroma for initialization",
        extra={"host": settings.chroma_host, "port": settings.chroma_port},
    )

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name=settings.chroma_collection)

    existing_ids = set(collection.get()["ids"])

    docs = _build_sample_corpus()

    new_docs = [d for d in docs if d["id"] not in existing_ids]

    if not new_docs:
        logger.info("Chroma collection already contains all sample documents; nothing to do")
        return

    logger.info("Adding %d new documents to Chroma", len(new_docs))

    collection.add(
        ids=[d["id"] for d in new_docs],
        documents=[d["text"] for d in new_docs],
        metadatas=[d["metadata"] for d in new_docs],
    )

    logger.info("Initialization completed successfully")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
