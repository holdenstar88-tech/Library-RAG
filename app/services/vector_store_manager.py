from __future__ import annotations

import time
from functools import lru_cache
from typing import Iterable

from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.services.vector_embedding_service import get_vector_embedding_service


class VectorStoreManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._vectorstore: Milvus | None = None

    def _resolved_collection_name(self) -> str:
        base_name = self.settings.milvus_collection.strip() or "library_books"
        namespace = get_vector_embedding_service().collection_namespace
        suffix = f"__{namespace}"
        if base_name.endswith(suffix):
            return base_name
        return f"{base_name}{suffix}"

    @property
    def resolved_collection_name(self) -> str:
        return self._resolved_collection_name()

    def _build_vectorstore(self) -> Milvus:
        return Milvus(
            embedding_function=get_vector_embedding_service().embeddings,
            collection_name=self._resolved_collection_name(),
            connection_args={
                "uri": self.settings.resolved_milvus_uri,
                "timeout": self.settings.resolved_milvus_timeout,
            },
            auto_id=False,
            text_field="content",
            vector_field="vector",
            primary_field="id",
            metadata_field="metadata",
            drop_old=self.settings.milvus_drop_old,
        )

    @property
    def vectorstore(self) -> Milvus:
        if self._vectorstore is None:
            self._vectorstore = self._build_vectorstore()
        return self._vectorstore

    def probe_ready(self) -> bool:
        try:
            store = self._build_vectorstore()
            collection = getattr(store, "col", None)
            if collection is not None:
                try:
                    collection.load()
                except Exception:
                    pass
            self._vectorstore = store
            return True
        except Exception:
            return False

    def wait_until_ready(self, timeout_seconds: int = 300, interval_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if self.probe_ready():
                    return True
                time.sleep(interval_seconds)
            except Exception as exc:
                last_error = exc
                time.sleep(interval_seconds)
        if last_error is not None:
            raise last_error
        return False

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        if not documents:
            return []
        normalized_documents = [
            Document(page_content=str(document.page_content), metadata=dict(document.metadata or {}))
            for document in documents
        ]
        self.vectorstore.add_documents(normalized_documents, ids=ids)
        return ids

    def delete_ids(self, ids: Iterable[str]) -> None:
        ids = [str(item) for item in ids if item]
        if not ids:
            return
        try:
            self.vectorstore.delete(ids=ids)
            return
        except Exception:
            pass
        collection = getattr(self.vectorstore, "col", None)
        if collection is None:
            return
        quoted = ", ".join(f'"{item}"' for item in ids)
        collection.delete(expr=f"id in [{quoted}]")

    def similarity_search_with_score(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=top_k)

    def flush(self) -> None:
        collection = getattr(self.vectorstore, "col", None)
        if collection is not None:
            try:
                collection.flush()
            except Exception:
                pass

    def is_ready(self) -> bool:
        return self.probe_ready()


@lru_cache(maxsize=1)
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager()
