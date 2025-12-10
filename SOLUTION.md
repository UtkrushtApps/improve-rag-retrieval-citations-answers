# Solution Steps

1. Create a configuration module to centralize environment-driven settings:
- Add `app/core/config.py` with a `Settings` class (subclassing `BaseSettings`).
- Include Chroma connection params (`CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`).
- Add retrieval tuning options (`retrieval_default_top_k`, `retrieval_max_k`, `retrieval_min_score`, `max_context_characters`).
- Expose a cached `get_settings()` function using `functools.lru_cache` so the same config instance is reused across the app.

2. Set up structured logging and observability:
- Implement `app/core/logging_config.py` with a `configure_logging()` function.
- Use `logging.config.dictConfig` to configure a root handler with a structured formatter (single-line logs including timestamp, level, logger, message, and `request_id`).
- Add a global logging filter that injects a default `request_id` when not present, so log parsing is consistent.

3. Define Pydantic models for the API and internal use:
- Create `app/models/schemas.py`.
- Implement `QueryRequest` with `question: str` and optional `max_sources: int` (with validation constraints).
- Implement `SourceChunk` to describe retrieved chunks (id, citation_id, score, source, rank, text, metadata).
- Implement `QueryResponse` with `answer: str` and `sources: List[SourceChunk]`.
- Implement `InternalChunk` as an internal model (id, text, score, metadata) used by the service layer when handling retrieval results.

4. Implement a robust ChromaDB repository abstraction:
- Create `app/services/chroma_repository.py`.
- In the constructor, read settings via `get_settings()` and initialize a `chromadb.HttpClient` with `ChromaSettings(anonymized_telemetry=False)`.
- Use `client.get_or_create_collection(name=collection_name)` to ensure the RAG collection exists.
- Add a `healthcheck()` method that calls `client.heartbeat()` and returns the integer heartbeat, raising a custom `ChromaUnavailableError` if anything fails.
- Add a `similarity_search(query: str, n_results: int)` method that:
  - Clamps `n_results` between 1 and `settings.retrieval_max_k`.
  - Calls `collection.query(...)` with `include=["documents", "metadatas", "distances"]`.
  - Iterates over the returned documents/meta/distances and converts distances into normalized similarity scores in [0,1] (e.g., for cosine distance, use `1 - d/2`).
  - Returns a list of `InternalChunk` instances.
- Provide a singleton-style `get_chroma_repository()` function that FastAPI can use as a dependency, caching the repository in a module-level variable.

5. Define a small custom exception hierarchy:
- Add `app/services/exceptions.py` with a `ChromaUnavailableError(RuntimeError)` that stores a human-readable `message`.
- Use this exception within the Chroma repository to signal connectivity or query errors.

6. Implement the RAG orchestration service with improved retrieval and citations:
- Create `app/services/rag_service.py` with a `RAGService` class accepting a `ChromaRepository`.
- Implement `answer_question(self, payload: QueryRequest) -> QueryResponse` that:
  - Strips the input question and determines `max_sources` (either from request or `settings.retrieval_default_top_k`).
  - Calls `chroma_repo.similarity_search()` to get an initial pool of candidate `InternalChunk`s.
  - If no chunks are returned, respond with a fallback answer explaining that nothing relevant was found and an empty `sources` list.
  - Otherwise, call a private `_select_chunks()` method to filter by `settings.retrieval_min_score`, sort by score descending, cap at `max_sources`, and enforce a global `settings.max_context_characters` budget.
  - Call `_generate_answer(question, selected_chunks)` to build a human-readable answer string that includes a "Sources:" section with numbered entries matching the retrieved chunks.
  - Call `_format_sources(selected_chunks)` to convert internal chunks into `SourceChunk` models with stable `citation_id`s.
  - Return a `QueryResponse` with the generated answer and formatted sources.
- Implement `_select_chunks()` to apply score and length constraints as described.
- Implement `_generate_answer()` as a deterministic, template-based summarizer that:
  - Echoes the question.
  - Provides high-level guidance about the RAG and deployment context.
  - Builds a per-chunk summary list where each line is prefixed with `[n]` and optionally the `source` metadata.
  - Appends a "Sources:" section listing these lines, so the answer clearly references which chunks were used.
- Implement `_format_sources()` to map each selected chunk to a `SourceChunk` with `citation_id` equal to its 1-based index.
- Add a module-level `get_rag_service()` that uses `get_chroma_repository()` to provide a singleton `RAGService` instance for FastAPI dependency injection.

7. Wire up the FastAPI application entrypoint and routes:
- Create `app/api/routes.py` with an `APIRouter` at prefix `/rag`.
- Add a `POST /rag/query` endpoint that accepts `QueryRequest`, injects `RAGService` via `Depends(get_rag_service)`, and returns the result of `rag_service.answer_question()` as `QueryResponse`.
- Create `app/main.py` with a `create_app()` function that:
  - Calls `configure_logging()` early.
  - Instantiates `FastAPI` with metadata from settings.
  - Adds CORS middleware allowing all origins/headers/methods for internal use.
  - Adds an `http` middleware that generates or propagates `X-Request-ID`, stores it in `request.state`, and logs start/end of each request including duration.
  - Defines a `GET /health` endpoint that injects `Settings` and `ChromaRepository`, calls `chroma_repo.healthcheck()`, and returns app/chroma status.
  - Registers an exception handler for `ChromaUnavailableError` that returns HTTP 503 with a clear JSON body and logs the failure.
  - Includes the RAG router under `/api`.
- Expose the app instance as `app = create_app()` for use by Uvicorn or Gunicorn.

8. Add a simple Chroma initialization script to seed the vector store:
- Create `init/init_chroma.py` which reads config via `get_settings()`.
- Connect to Chroma with `chromadb.HttpClient` using `settings.chroma_host` and `settings.chroma_port`.
- Call `get_or_create_collection(settings.chroma_collection)`.
- Build a small in-memory corpus covering assessment design, Docker Compose deployment, RAG configuration, and observability using `_build_sample_corpus()`.
- Fetch existing ids from the collection and only add new documents that are missing, making the script idempotent.
- Use `collection.add(ids=..., documents=..., metadatas=...)` to insert documents.
- Log progress so operators can see when initialization completes.
- Configure this script to be run by a one-time init container/service in Docker Compose (outside this codebase).

9. Verify behavior and tune retrieval parameters for complex questions:
- Start the Chroma container and run the `init_chroma.py` script to populate the collection.
- Run the FastAPI app container and hit `GET /health` to confirm connectivity to Chroma.
- Issue `POST /api/rag/query` requests with realistic questions about assessment design, Dockerized deployments, and RAG configuration.
- If answers feel under- or over-specified, adjust `RETRIEVAL_DEFAULT_TOP_K`, `RETRIEVAL_MAX_K`, and `RETRIEVAL_MIN_SCORE` via environment variables to balance breadth of context and precision.
- Confirm that each response includes an `answer` with a clear "Sources:" section and a `sources` array with matching `citation_id`s, enabling stakeholders to trace exactly which chunks grounded the answer.

