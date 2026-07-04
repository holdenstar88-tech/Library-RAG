from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.config import Settings, get_settings
from app.generation.rag import clarify_answer, generate_answer, make_llm
from app.ingestion.chunking import split_documents
from app.ingestion.loaders import load_documents_from_dir
from app.models import CatalogSearchRequest, CatalogSearchResponse, ChatResponse, HealthResponse, SourceItem
from app.retrieval.hybrid import (
    apply_filters,
    build_bm25_index,
    bm25_search,
    compute_confidence,
    infer_metadata_filters,
    metadata_matches,
    rrf_fuse,
)
from app.services.metadata_store import MetadataStore
from app.services.vector_index_service import IndexSyncResult, VectorIndexService
from app.services.vector_search_service import get_vector_search_service
from app.services.vector_store_manager import get_vector_store_manager


logger = logging.getLogger(__name__)


@dataclass
class SearchArtifacts:
    documents: list[Any]
    confidence: float
    filters: dict[str, str]
    fallback: bool


class LibraryRAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._history: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.settings.chat_history_size)
        )
        self.metadata_store = MetadataStore(self.settings.metadata_db_path)
        self.index_service = VectorIndexService(self.settings)
        self.vector_search_service = get_vector_search_service()
        self.vector_store_manager = get_vector_store_manager()
        self.llm = make_llm(
            self._shared_api_key(),
            self.settings.deepseek_base_url,
            self.settings.deepseek_model,
        )
        self.last_sync_result: IndexSyncResult | None = None
        self.vector_store_ready = False
        try:
            self.last_sync_result = self.index_service.sync_directory(self.settings.data_dir)
        except Exception as exc:
            logger.warning("Incremental sync failed, falling back to loaded documents: %s", exc)
        self._refresh_corpus()

    def _shared_api_key(self) -> str:
        return self.settings.deepseek_api_key or self.settings.api_key

    def _refresh_corpus(self) -> None:
        documents = self.metadata_store.load_active_documents()
        if not documents:
            raw_documents = load_documents_from_dir(self.settings.data_dir)
            documents = split_documents(
                raw_documents,
                chunk_size=self.settings.chunk_max_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
        self.documents = documents
        self.bm25_index, self.bm25_documents, _ = build_bm25_index(self.documents)
        self.vector_store_ready = self.vector_store_manager.is_ready()

    def refresh(self) -> None:
        self._refresh_corpus()

    def sync(self) -> IndexSyncResult:
        result = self.index_service.sync_directory(self.settings.data_dir)
        self.last_sync_result = result
        self._refresh_corpus()
        return result

    def health(self) -> HealthResponse:
        vector_store_ready = self.vector_store_manager.is_ready()
        self.vector_store_ready = vector_store_ready
        return HealthResponse(
            status="ok",
            vector_store_ready=vector_store_ready,
            documents_loaded=len(self.documents),
            collection_name=self.vector_store_manager.resolved_collection_name,
        )

    def history(self, session_id: str) -> list[dict[str, str]]:
        return list(self._history[session_id])

    def append_history(self, session_id: str, role: str, content: str) -> None:
        self._history[session_id].append({"role": role, "content": content})

    def _categories(self) -> list[str]:
        categories = {
            str((document.metadata or {}).get("category") or "").strip()
            for document in self.documents
        }
        return sorted(category for category in categories if category)

    def _source_item(self, document: Any, rank: int, score: float | None = None) -> SourceItem:
        metadata = document.metadata or {}
        return SourceItem(
            rank=rank,
            score=float(score if score is not None else max(0.0, 1.0 - rank * 0.1)),
            book_id=metadata.get("book_id"),
            title=metadata.get("title") or metadata.get("h2") or metadata.get("h1"),
            author=metadata.get("author"),
            isbn=metadata.get("isbn"),
            call_number=metadata.get("call_number"),
            category=metadata.get("category"),
            subjects=metadata.get("subjects"),
            main_characters=metadata.get("main_characters"),
            plot_summary=metadata.get("plot_summary"),
            shelf=metadata.get("shelf"),
            shelf_code=metadata.get("shelf_code"),
            shelf_row=metadata.get("shelf_row"),
            shelf_col=metadata.get("shelf_col"),
            floor=metadata.get("floor"),
            area=metadata.get("area"),
            copy_count=metadata.get("copy_count"),
            available_count=metadata.get("available_count"),
            availability=metadata.get("availability"),
            borrow_rule=metadata.get("borrow_rule"),
            open_time=metadata.get("open_time"),
            source=metadata.get("source_name") or metadata.get("source"),
            excerpt=document.page_content[:240],
            metadata=metadata,
        )

    def retrieve(self, question: str) -> SearchArtifacts:
        filters = infer_metadata_filters(question)
        prefiltered_documents = [document for document in self.documents if metadata_matches(document, filters)]
        if filters and prefiltered_documents:
            bm25_index, bm25_documents, _ = build_bm25_index(prefiltered_documents)
        else:
            bm25_index, bm25_documents = self.bm25_index, self.bm25_documents
        bm25_results = apply_filters(
            bm25_search(bm25_index, bm25_documents, question, self.settings.search_top_k),
            filters,
        )
        vector_results = apply_filters(
            self.vector_search_service.search_similar_documents(question, self.settings.search_top_k * 4),
            filters,
        )
        fused = rrf_fuse(bm25_results, vector_results)
        fused = fused[: self.settings.rerank_top_k]
        confidence = compute_confidence(fused, filters)
        return SearchArtifacts(
            documents=[item.document for item in fused],
            confidence=confidence,
            filters=filters,
            fallback=not fused,
        )

    def _request_filters(self, payload: CatalogSearchRequest) -> dict[str, str]:
        filters = infer_metadata_filters(payload.query)
        direct_fields = {
            "category": payload.category,
            "author": payload.author,
            "title": payload.title,
            "isbn": payload.isbn,
            "book_id": payload.book_id,
            "call_number": payload.call_number,
            "main_characters": payload.main_character,
            "subjects": payload.subject,
        }
        for key, value in direct_fields.items():
            if value:
                filters[key] = value
        if payload.available_only:
            filters["available_only"] = "true"
        return filters

    def search_catalog(self, payload: CatalogSearchRequest) -> CatalogSearchResponse:
        query = payload.query.strip()
        filters = self._request_filters(payload)
        candidates = [document for document in self.documents if metadata_matches(document, filters)]
        exact_fields = {"book_id", "isbn", "call_number"}
        exact = []
        if exact_fields.intersection(filters):
            exact = candidates
        if query:
            bm25_index, bm25_documents, _ = build_bm25_index(candidates or self.documents)
            bm25_results = apply_filters(bm25_search(bm25_index, bm25_documents, query, payload.limit), filters)
            vector_results = apply_filters(
                self.vector_search_service.search_similar_documents(query, min(payload.limit * 3, 50)),
                filters,
            )
            fused = rrf_fuse(
                [item for item in bm25_results if item.document not in exact],
                [item for item in vector_results if item.document not in exact],
            )
            documents = exact + [item.document for item in fused]
        else:
            documents = candidates
        seen: set[str] = set()
        unique = []
        for document in documents:
            metadata = document.metadata or {}
            key = str(metadata.get("book_id") or metadata.get("isbn") or metadata.get("chunk_id") or id(document))
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        results = [self._source_item(document, rank) for rank, document in enumerate(unique[: payload.limit], start=1)]
        return CatalogSearchResponse(
            query=query,
            results=results,
            total=len(unique),
            categories=self._categories(),
            fallback=not results,
        )

    def answer(self, question: str, session_id: str | None = None) -> ChatResponse:
        sid = session_id or str(uuid4())
        self.append_history(sid, "user", question)
        artifacts = self.retrieve(question)
        if artifacts.fallback or artifacts.confidence < self.settings.similarity_threshold:
            answer_text = clarify_answer(question, artifacts.documents)
            fallback = True
        else:
            answer_text, fallback = generate_answer(self.llm, question, artifacts.documents, self.history(sid))
        self.append_history(sid, "assistant", answer_text)

        sources = [self._source_item(document, rank) for rank, document in enumerate(artifacts.documents, start=1)]
        return ChatResponse(
            session_id=sid,
            answer=answer_text,
            sources=sources,
            confidence=artifacts.confidence,
            fallback=fallback,
        )


_service: LibraryRAGService | None = None


def get_service() -> LibraryRAGService:
    global _service
    if _service is None:
        _service = LibraryRAGService()
    return _service
