from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.ingestion.chunking import split_documents
from app.ingestion.loaders import load_documents_from_file
from app.services.metadata_store import MetadataStore, infer_source_type, normalize_source_path
from app.services.vector_store_manager import get_vector_store_manager


@dataclass(frozen=True)
class IndexSyncResult:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    indexed_chunks: int = 0


class VectorIndexService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.metadata_store = MetadataStore(self.settings.metadata_db_path)
        self.vector_store_manager = get_vector_store_manager()

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _file_mtime(self, path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    def _document_field(self, metadata: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = metadata.get(key)
            if value:
                text = str(value).strip()
                if text:
                    return text
        return None

    def _prepare_chunks(
        self,
        path: Path,
        file_hash: str,
        file_mtime: str,
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        documents = load_documents_from_file(path)
        chunks = split_documents(
            documents,
            chunk_size=self.settings.chunk_max_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        normalized_path = normalize_source_path(path)
        source_type = infer_source_type(path)
        payload_documents: list[Document] = []
        chunk_payloads: list[dict[str, Any]] = []

        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            chunk_id = str(uuid4())
            metadata.update(
                {
                    "_source": normalized_path,
                    "source_path": normalized_path,
                    "source_name": path.name,
                    "source_type": source_type,
                    "file_hash": file_hash,
                    "file_mtime": file_mtime,
                }
            )
            title = self._document_field(metadata, "title", "h2", "h1", "source_name")
            author = self._document_field(metadata, "author", "writer")
            isbn = self._document_field(metadata, "isbn")
            category = self._document_field(metadata, "category", "type")
            shelf = self._document_field(metadata, "shelf", "location")
            availability = self._document_field(metadata, "availability", "status")
            borrow_rule = self._document_field(metadata, "borrow_rule", "borrow_rules")
            open_time = self._document_field(metadata, "open_time", "opening_hours")
            metadata.update(
                {
                    "title": title,
                    "author": author,
                    "isbn": isbn,
                    "category": category,
                    "shelf": shelf,
                    "availability": availability,
                    "borrow_rule": borrow_rule,
                    "open_time": open_time,
                }
            )
            payload_documents.append(Document(page_content=str(chunk.page_content).strip(), metadata=metadata))
            chunk_payloads.append(
                {
                    "chunk_id": chunk_id,
                    "vector_id": chunk_id,
                    "content": str(chunk.page_content).strip(),
                    "metadata": metadata,
                }
            )
        return payload_documents, chunk_payloads

    def sync_directory(self, data_dir: Path | None = None) -> IndexSyncResult:
        root = Path(data_dir or self.settings.data_dir)
        if not root.exists():
            return IndexSyncResult()

        if not self.vector_store_manager.wait_until_ready():
            raise RuntimeError("Milvus vector store is not ready")

        seen_sources: set[str] = set()
        created = 0
        updated = 0
        deleted = 0
        skipped = 0
        indexed_chunks = 0

        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            normalized_path = normalize_source_path(path)
            if path.suffix.lower() not in {".json", ".md", ".markdown", ".txt", ".csv", ".pdf"}:
                continue
            seen_sources.add(normalized_path)
            file_hash = self._hash_file(path)
            file_mtime = self._file_mtime(path)
            current = self.metadata_store.get_active_document(path)
            if current and str(current["file_hash"]) == file_hash:
                skipped += 1
                continue

            payload_documents, chunk_payloads = self._prepare_chunks(path, file_hash, file_mtime)
            if not chunk_payloads:
                if current:
                    deleted_chunk_ids = self.metadata_store.delete_source(path)
                    self.vector_store_manager.delete_ids(deleted_chunk_ids)
                    deleted += 1
                continue

            ids = [str(payload["chunk_id"]) for payload in chunk_payloads]
            self.vector_store_manager.add_documents(payload_documents, ids=ids)

            try:
                _doc_id, new_chunk_ids, previous_chunk_ids, action = self.metadata_store.upsert_source_version(
                    source_path=path,
                    source_name=path.name,
                    source_type=infer_source_type(path),
                    file_hash=file_hash,
                    file_mtime=file_mtime,
                    chunks=chunk_payloads,
                )
            except Exception:
                self.vector_store_manager.delete_ids(ids)
                raise

            if previous_chunk_ids:
                self.vector_store_manager.delete_ids(previous_chunk_ids)
            self.vector_store_manager.flush()

            if action == "created":
                created += 1
            elif action == "updated":
                updated += 1
            indexed_chunks += len(new_chunk_ids or ids)

        for source_path in self.metadata_store.list_active_source_paths():
            if source_path in seen_sources:
                continue
            deleted_chunk_ids = self.metadata_store.delete_source(source_path)
            self.vector_store_manager.delete_ids(deleted_chunk_ids)
            deleted += 1

        self.vector_store_manager.flush()
        return IndexSyncResult(
            created=created,
            updated=updated,
            deleted=deleted,
            skipped=skipped,
            indexed_chunks=indexed_chunks,
        )


def get_vector_index_service() -> VectorIndexService:
    return VectorIndexService()
