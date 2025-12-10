"""ChromaDB repository abstraction.

This layer hides the details of talking to ChromaDB over HTTP and
exposes a simple retrieval API tailored for the RAG service.

Key responsibilities:
- Maintain a single `HttpClient` instance per process
- Guarantee the collection exists
- Provide a scored similarity search that returns normalized scores
"""

from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.models.schemas import InternalChunk
from app.services.exceptions import ChromaUnavailableError

logger = logging.getLogger(__name__)


class ChromaRepository:
    """Thin wrapper around the Chroma HTTP client.

    Parameters
    ----------
    host:
        Hostname or IP where Chroma is exposed.
    port:
        TCP port where Chroma listens (typically 8000 inside Docker).
    collection_name:
        Name of the logical collection used for RAG documents.
    """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name

        logger.info(
            "Initializing Chroma HttpClient",
            extra={"host": host, "port": port, "collection": collection_name},
        )

        try:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Create or retrieve the collection. This is idempotent.
            self._collection = self._client.get_or_create_collection(
                name=collection_name
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to initialize Chroma HttpClient")
            raise ChromaUnavailableError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def healthcheck(self) -> int:
        """Return Chroma's heartbeat value or raise if unavailable."""

        try:
            heartbeat = self._client.heartbeat()
            logger.debug("Chroma heartbeat", extra={"heartbeat": heartbeat})
            return int(heartbeat)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Chroma heartbeat failed")
            raise ChromaUnavailableError("ChromaDB is not reachable") from exc

    def similarity_search(
        self,
        query: str,
        n_results: int,
    ) -> List[InternalChunk]:
        """Run a similarity search against Chroma.

        The raw distances returned by Chroma are converted into
        similarity scores in the range [0, 1], where higher is better.
        """

        settings = get_settings()
        n_results = max(1, min(n_results, settings.retrieval_max_k))

        logger.debug(
            "Running Chroma similarity search",
            extra={"query": query, "n_results": n_results},
        )

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Chroma query failed")
            raise ChromaUnavailableError("ChromaDB query failed") from exc

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0] if raw.get("ids") else [str(i) for i in range(len(documents))]

        chunks: List[InternalChunk] = []

        if not documents:
            logger.info("Chroma returned no documents for query", extra={"query": query})
            return chunks

        # Normalize distances to similarity scores in [0, 1]. For cosine
        # distance (the default), distance is in [0, 2] where 0 is
        # identical. We map this to 1 - (d / 2).
        for idx, doc in enumerate(documents):
            text = doc or ""
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else 0.0

            similarity = max(0.0, min(1.0, 1.0 - (float(distance) / 2.0)))
            chunk_id = ids[idx] if idx < len(ids) else str(idx)

            chunks.append(
                InternalChunk(
                    id=chunk_id,
                    text=text,
                    score=similarity,
                    metadata=metadata or {},
                )
            )

        logger.debug(
            "Chroma search returned chunks",
            extra={"count": len(chunks)},
        )

        return chunks


# Dependency injection helpers -------------------------------------------------

_chroma_repo: ChromaRepository | None = None


def get_chroma_repository() -> ChromaRepository:
    """FastAPI dependency for a singleton ChromaRepository.

    This avoids reconnecting to Chroma for every request while keeping
    testability (the dependency can be overridden in tests).
    """

    global _chroma_repo
    if _chroma_repo is None:
        cfg = get_settings()
        _chroma_repo = ChromaRepository(
            host=cfg.chroma_host,
            port=cfg.chroma_port,
            collection_name=cfg.chroma_collection,
        )
    return _chroma_repo
