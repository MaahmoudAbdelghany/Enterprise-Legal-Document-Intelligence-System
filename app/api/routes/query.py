"""Legal RAG query API route handlers for LexisGraph.

Provides REST endpoints for querying indexed legal documents:
- Synchronous grounded Q&A with multi-vector late-interaction ColPali retrieval
- Server-Sent Events (SSE) streaming token output
- Metadata filtering (document IDs, filenames, page ranges, tables, categories)
- Citations and source provenance attribution
"""

from __future__ import annotations

import asyncio
import datetime
import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from qdrant_client.http import models as qmodels

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_legal_rag_chain,
    get_qdrant_store,
)
from app.config import Settings
from app.core.generation.chain import (
    ContextFormattingError,
    LegalGenerationError,
    LegalRAGChain,
    LegalRAGInput,
    LegalRAGOutput,
    NoLLMProviderConfiguredError,
    SourceDocument,
)
from app.core.ingestion.colpali_embedder import (
    ColPaliEmbedder,
    ColPaliEmbeddingError,
    EmptyInputError,
    QueryEmbedding,
)
from app.core.retrieval.qdrant_store import (
    QdrantSearchResult,
    QdrantStore,
    QdrantStoreError,
    SearchError,
)
from app.models.schemas import (
    HTTPErrorResponse,
    LegalLanguage,
    QueryFilter,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
    SourceReference,
    StreamTokenChunk,
    TaskType,
)

router = APIRouter(prefix="/query", tags=["Query"])


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def _build_qdrant_filter(
    filters: QueryFilter | None,
) -> tuple[list[str] | None, tuple[int, int] | None, qmodels.Filter | None]:
    """Convert a QueryFilter schema into parameters suitable for QdrantStore.search.

    Args:
        filters: Optional QueryFilter submitted by the client.

    Returns:
        Tuple of:
            - document_ids (list of str or None)
            - page_range (tuple of [min_page, max_page] or None)
            - extra_filter (qmodels.Filter or None)
    """
    if filters is None:
        return None, None, None

    doc_ids = filters.document_ids or None
    page_range: tuple[int, int] | None = None
    if filters.page_range and len(filters.page_range) == 2:
        page_range = (filters.page_range[0], filters.page_range[1])

    extra_conditions: list[Any] = []

    # Filename filter
    if filters.filenames:
        clean_filenames = [f.strip() for f in filters.filenames if f and f.strip()]
        if len(clean_filenames) == 1:
            extra_conditions.append(
                qmodels.FieldCondition(
                    key="filename",
                    match=qmodels.MatchValue(value=clean_filenames[0]),
                )
            )
        elif len(clean_filenames) > 1:
            extra_conditions.append(
                qmodels.FieldCondition(
                    key="filename",
                    match=qmodels.MatchAny(any=clean_filenames),
                )
            )

    # Has tables filter
    if filters.has_tables is not None:
        extra_conditions.append(
            qmodels.FieldCondition(
                key="has_tables",
                match=qmodels.MatchValue(value=filters.has_tables),
            )
        )

    # Legal category filter
    if filters.legal_category and filters.legal_category.strip():
        extra_conditions.append(
            qmodels.FieldCondition(
                key="metadata.legal_category",
                match=qmodels.MatchValue(value=filters.legal_category.strip()),
            )
        )

    # Tags filter
    if filters.tags:
        clean_tags = [t.strip() for t in filters.tags if t and t.strip()]
        if clean_tags:
            extra_conditions.append(
                qmodels.FieldCondition(
                    key="metadata.tags",
                    match=qmodels.MatchAny(any=clean_tags),
                )
            )

    extra_filter: qmodels.Filter | None = None
    if extra_conditions:
        extra_filter = qmodels.Filter(must=extra_conditions)

    return doc_ids, page_range, extra_filter


def _calculate_confidence(search_results: list[QdrantSearchResult]) -> float:
    """Derive an aggregate answer confidence score based on retrieved context scores.

    Args:
        search_results: List of retrieved QdrantSearchResult items.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not search_results:
        return 0.5  # Neutral confidence when answering without document context

    scores = [r.score for r in search_results if r.score is not None]
    if not scores:
        return 0.5

    # Use highest relevance score bounded between 0.0 and 1.0
    top_score = max(scores)
    return round(float(min(1.0, max(0.0, top_score))), 3)


def _convert_sources(sources: list[SourceDocument]) -> list[SourceReference]:
    """Convert internal SourceDocument dataclasses to public SourceReference schemas."""
    return [
        SourceReference(
            document_id=s.document_id,
            page_number=s.page_number,
            score=round(s.score, 4),
            filename=s.filename,
            text_snippet=s.text_snippet,
            has_tables=s.has_tables,
            metadata=s.metadata,
        )
        for s in sources
    ]


async def _execute_retrieval(
    question: str,
    top_k: int,
    filters: QueryFilter | None,
    embedder: ColPaliEmbedder,
    qdrant: QdrantStore,
) -> tuple[list[QdrantSearchResult], float, float]:
    """Execute asynchronous ColPali multi-vector embedding and Qdrant search.

    Args:
        question: User query string.
        top_k: Max results to retrieve.
        filters: Retrieval filters.
        embedder: ColPaliEmbedder dependency.
        qdrant: QdrantStore dependency.

    Returns:
        Tuple of (search_results, embedding_latency_seconds, search_latency_seconds).
    """
    # 1. Embed query
    t0 = time.perf_counter()
    try:
        query_embedding = await asyncio.to_thread(embedder.embed_query, question)
    except EmptyInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid question: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ColPali query embedding failed: {e}",
        ) from e
    t_embed = time.perf_counter() - t0

    # 2. Search Qdrant
    doc_ids, page_range, extra_filter = _build_qdrant_filter(filters)
    t1 = time.perf_counter()
    try:
        search_results = await asyncio.to_thread(
            qdrant.search,
            query=query_embedding,
            top_k=top_k,
            document_ids=doc_ids,
            page_range=page_range,
            extra_filter=extra_filter,
        )
    except SearchError as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed: {e}",
        ) from e
    except QdrantStoreError as e:
        logger.error(f"Qdrant store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector store unavailable: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected retrieval failure: {e}",
        ) from e
    t_search = time.perf_counter() - t1

    return search_results, t_embed, t_search


# ------------------------------------------------------------------------------
# Route Handlers
# ------------------------------------------------------------------------------

@router.post(
    "",
    response_model=QueryResponse,
    responses={
        200: {"description": "Legal grounded answer with sources and confidence score"},
        400: {"model": HTTPErrorResponse, "description": "Invalid query parameters"},
        500: {"model": HTTPErrorResponse, "description": "Generation or retrieval failure"},
        503: {"model": HTTPErrorResponse, "description": "LLM provider not configured"},
    },
    summary="Execute legal RAG query",
    description=(
        "Performs multi-vector ColPali retrieval and LCEL generation for Arabic and "
        "multilingual legal questions. Returns grounded answer with citations. "
        "If `stream=True` in request body, streams tokens via Server-Sent Events (SSE)."
    ),
)
async def query_documents(
    request: QueryRequest,
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: ColPaliEmbedder = Depends(get_colpali_embedder),
    chain: LegalRAGChain = Depends(get_legal_rag_chain),
) -> QueryResponse | StreamingResponse:
    """Execute legal question-answering query against indexed documents."""
    start_total = time.perf_counter()

    logger.info(
        f"Processing query: question='{request.question[:60]}...', "
        f"strategy='{request.strategy}', stream={request.stream}, top_k={request.top_k}"
    )

    # Execute ColPali embedding + Qdrant search
    search_results, t_embed, t_search = await _execute_retrieval(
        question=request.question,
        top_k=request.top_k,
        filters=request.filters,
        embedder=embedder,
        qdrant=qdrant,
    )

    # Resolve language preference
    lang_pref = "auto"
    if request.language == LegalLanguage.AR:
        lang_pref = "ar"
    elif request.language == LegalLanguage.EN:
        lang_pref = "en"

    # Resolve legal task domain
    task_domain = (
        request.task_type.value
        if hasattr(request.task_type, "value")
        else str(request.task_type)
    )

    rag_input = LegalRAGInput(
        question=request.question,
        context=search_results,
        language=lang_pref,
        legal_domain=task_domain,
    )

    # --------------------------------------------------------------------------
    # Streaming Response (SSE)
    # --------------------------------------------------------------------------
    if request.stream:
        async def event_generator() -> AsyncIterator[str]:
            source_refs: list[SourceReference] = []
            try:
                # Format sources in advance for final emission
                if request.include_sources:
                    from app.core.generation.chain import format_legal_context

                    _, norm_sources = format_legal_context(search_results)
                    source_refs = _convert_sources(norm_sources)

                async for token in chain.astream(rag_input):
                    chunk = StreamTokenChunk(token=token, is_final=False)
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Final completion chunk with sources
                final_chunk = StreamTokenChunk(
                    token="",
                    is_final=True,
                    sources=source_refs if request.include_sources else None,
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            except NoLLMProviderConfiguredError as e:
                logger.error(f"Streaming failed - no LLM configured: {e}")
                err_data = json.dumps({"error": "LLM provider not configured", "detail": str(e)})
                yield f"data: {err_data}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming generation: {e}")
                err_data = json.dumps({"error": "Streaming generation failed", "detail": str(e)})
                yield f"data: {err_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------------------------------------------------------
    # Synchronous Response
    # --------------------------------------------------------------------------
    try:
        rag_output: LegalRAGOutput = await chain.ainvoke(rag_input)
    except NoLLMProviderConfiguredError as e:
        logger.error(f"Query generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured. Please set GROQ_API_KEY or OPENAI_API_KEY in .env.",
        ) from e
    except ContextFormattingError as e:
        logger.error(f"Context formatting failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Context formatting error: {e}",
        ) from e
    except LegalGenerationError as e:
        logger.error(f"Legal generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected query error: {e}",
        ) from e

    elapsed_total = time.perf_counter() - start_total
    confidence = _calculate_confidence(search_results)

    # Strategy used mapping
    strategy_used = (
        request.strategy
        if request.strategy != RetrievalStrategy.AUTO
        else RetrievalStrategy.VISUAL
    )

    sources = _convert_sources(rag_output.sources) if request.include_sources else []

    return QueryResponse(
        question=request.question,
        answer=rag_output.answer,
        sources=sources,
        confidence=confidence,
        strategy_used=strategy_used,
        task_type=request.task_type,
        language=rag_output.language,
        model_used=rag_output.model_used,
        latency_seconds=round(elapsed_total, 4),
        created_at=datetime.datetime.now(datetime.timezone.utc),
        metadata={
            "retrieved_page_count": len(search_results),
            "embedding_latency_seconds": round(t_embed, 4),
            "search_latency_seconds": round(t_search, 4),
            "generation_latency_seconds": round(rag_output.latency_seconds, 4),
            "total_pipeline_latency_seconds": round(elapsed_total, 4),
        },
    )


@router.post(
    "/stream",
    summary="Stream legal RAG query via SSE",
    description="Dedicated convenience endpoint for Server-Sent Events (SSE) streaming answers.",
)
async def query_stream(
    request: QueryRequest,
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: ColPaliEmbedder = Depends(get_colpali_embedder),
    chain: LegalRAGChain = Depends(get_legal_rag_chain),
) -> StreamingResponse:
    """Stream answers as Server-Sent Events (SSE)."""
    request.stream = True
    return await query_documents(
        request=request,
        qdrant=qdrant,
        embedder=embedder,
        chain=chain,
    )
