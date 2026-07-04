from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.documents import Document


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source_path(path: str | Path) -> str:
    return Path(path).resolve().as_posix().lower()


def infer_source_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "text"


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    vector_id: str
    source_path: str
    source_name: str
    source_type: str
    file_hash: str
    file_mtime: str
    version: int
    chunk_index: int
    chunk_hash: str
    content: str
    book_id: str | None
    title: str | None
    author: str | None
    isbn: str | None
    call_number: str | None
    category: str | None
    subjects: str | None
    main_characters: str | None
    plot_summary: str | None
    shelf: str | None
    shelf_code: str | None
    shelf_row: int | None
    shelf_col: int | None
    floor: str | None
    area: str | None
    copy_count: int | None
    available_count: int | None
    availability: str | None
    borrow_rule: str | None
    open_time: str | None
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None
    metadata: dict[str, Any]


class MetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_mtime TEXT,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    vector_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_mtime TEXT,
                    version INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    book_id TEXT,
                    title TEXT,
                    author TEXT,
                    isbn TEXT,
                    call_number TEXT,
                    category TEXT,
                    subjects TEXT,
                    main_characters TEXT,
                    plot_summary TEXT,
                    shelf TEXT,
                    shelf_code TEXT,
                    shelf_row INTEGER,
                    shelf_col INTEGER,
                    floor TEXT,
                    area TEXT,
                    copy_count INTEGER,
                    available_count INTEGER,
                    availability TEXT,
                    borrow_rule TEXT,
                    open_time TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
                """
            )
            self._ensure_chunk_columns(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_isbn ON chunks(isbn)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_call_number ON chunks(call_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category)")
            conn.commit()

    def _ensure_chunk_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        columns = {
            "book_id": "TEXT",
            "call_number": "TEXT",
            "subjects": "TEXT",
            "main_characters": "TEXT",
            "plot_summary": "TEXT",
            "shelf_code": "TEXT",
            "shelf_row": "INTEGER",
            "shelf_col": "INTEGER",
            "floor": "TEXT",
            "area": "TEXT",
            "copy_count": "INTEGER",
            "available_count": "INTEGER",
        }
        for name, column_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {column_type}")

    def get_active_document(self, source_path: str | Path) -> sqlite3.Row | None:
        normalized = normalize_source_path(source_path)
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM documents
                WHERE source_path = ? AND status = 'active'
                ORDER BY version DESC, created_at DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()

    def list_active_source_paths(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT source_path
                FROM documents
                WHERE status = 'active'
                ORDER BY source_path
                """
            ).fetchall()
        return [str(row["source_path"]) for row in rows]

    def list_active_chunks(self) -> list[StoredChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE status = 'active'
                ORDER BY source_path, version, chunk_index, created_at
                """
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def list_active_documents(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM documents
                WHERE status = 'active'
                ORDER BY source_path, version DESC, created_at DESC
                """
            ).fetchall()

    def upsert_source_version(
        self,
        *,
        source_path: str | Path,
        source_name: str,
        source_type: str,
        file_hash: str,
        file_mtime: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[str, list[str], list[str], str]:
        normalized = normalize_source_path(source_path)
        now = utc_now()
        with self._connect() as conn:
            current = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE source_path = ? AND status = 'active'
                ORDER BY version DESC, created_at DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
            if current and str(current["file_hash"]) == file_hash:
                return str(current["doc_id"]), [], [], "unchanged"

            previous_chunk_ids: list[str] = []
            version = 1
            if current:
                previous_chunk_ids = [
                    str(row["chunk_id"])
                    for row in conn.execute(
                        """
                        SELECT chunk_id
                        FROM chunks
                        WHERE doc_id = ? AND status = 'active'
                        """,
                        (current["doc_id"],),
                    ).fetchall()
                ]
                conn.execute(
                    """
                    UPDATE documents
                    SET status = 'superseded', updated_at = ?, deleted_at = ?
                    WHERE doc_id = ?
                    """,
                    (now, now, current["doc_id"]),
                )
                conn.execute(
                    """
                    UPDATE chunks
                    SET status = 'superseded', updated_at = ?, deleted_at = ?
                    WHERE doc_id = ? AND status = 'active'
                    """,
                    (now, now, current["doc_id"]),
                )
                version = int(current["version"]) + 1
            else:
                row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) AS max_version FROM documents WHERE source_path = ?",
                    (normalized,),
                ).fetchone()
                version = int(row["max_version"]) + 1 if row else 1

            doc_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, source_name, source_type, file_hash, file_mtime,
                    version, status, created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                """,
                (doc_id, normalized, source_name, source_type, file_hash, file_mtime, version, now, now),
            )

            chunk_ids: list[str] = []
            for index, chunk in enumerate(chunks):
                chunk_id = str(chunk.get("chunk_id") or uuid4())
                vector_id = str(chunk.get("vector_id") or chunk_id)
                metadata = dict(chunk.get("metadata") or {})
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, vector_id, source_path, source_name, source_type,
                        file_hash, file_mtime, version, chunk_index, chunk_hash, content,
                        book_id, title, author, isbn, call_number, category, subjects,
                        main_characters, plot_summary, shelf, shelf_code, shelf_row,
                        shelf_col, floor, area, copy_count, available_count, availability,
                        borrow_rule, open_time,
                        status, created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                    """,
                    (
                        chunk_id,
                        doc_id,
                        vector_id,
                        normalized,
                        source_name,
                        source_type,
                        file_hash,
                        file_mtime,
                        version,
                        int(metadata.get("chunk_index", index)),
                        str(metadata.get("chunk_hash") or ""),
                        str(chunk.get("content") or ""),
                        metadata.get("book_id"),
                        metadata.get("title"),
                        metadata.get("author"),
                        metadata.get("isbn"),
                        metadata.get("call_number"),
                        metadata.get("category"),
                        metadata.get("subjects"),
                        metadata.get("main_characters"),
                        metadata.get("plot_summary"),
                        metadata.get("shelf"),
                        metadata.get("shelf_code"),
                        metadata.get("shelf_row"),
                        metadata.get("shelf_col"),
                        metadata.get("floor"),
                        metadata.get("area"),
                        metadata.get("copy_count"),
                        metadata.get("available_count"),
                        metadata.get("availability"),
                        metadata.get("borrow_rule"),
                        metadata.get("open_time"),
                        now,
                        now,
                    ),
                )
                chunk_ids.append(chunk_id)
            conn.commit()
            return doc_id, chunk_ids, previous_chunk_ids, "created" if not current else "updated"

    def delete_source(self, source_path: str | Path) -> list[str]:
        normalized = normalize_source_path(source_path)
        now = utc_now()
        with self._connect() as conn:
            current = conn.execute(
                """
                SELECT *
                FROM documents
                WHERE source_path = ? AND status = 'active'
                ORDER BY version DESC, created_at DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
            if not current:
                return []
            chunk_ids = [
                str(row["chunk_id"])
                for row in conn.execute(
                    """
                    SELECT chunk_id
                    FROM chunks
                    WHERE doc_id = ? AND status = 'active'
                    """,
                    (current["doc_id"],),
                ).fetchall()
            ]
            conn.execute(
                """
                UPDATE documents
                SET status = 'deleted', updated_at = ?, deleted_at = ?
                WHERE doc_id = ?
                """,
                (now, now, current["doc_id"]),
            )
            conn.execute(
                """
                UPDATE chunks
                SET status = 'deleted', updated_at = ?, deleted_at = ?
                WHERE doc_id = ? AND status = 'active'
                """,
                (now, now, current["doc_id"]),
            )
            conn.commit()
            return chunk_ids

    def load_active_documents(self) -> list[Document]:
        documents: list[Document] = []
        for chunk in self.list_active_chunks():
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "_source": chunk.source_path,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "vector_id": chunk.vector_id,
                    "source_path": chunk.source_path,
                    "source_name": chunk.source_name,
                    "source_type": chunk.source_type,
                    "file_hash": chunk.file_hash,
                    "file_mtime": chunk.file_mtime,
                    "version": chunk.version,
                    "chunk_index": chunk.chunk_index,
                    "chunk_hash": chunk.chunk_hash,
                    "status": chunk.status,
                    "created_at": chunk.created_at,
                    "updated_at": chunk.updated_at,
                }
            )
            documents.append(Document(page_content=chunk.content, metadata=metadata))
        return documents

    def _row_to_chunk(self, row: sqlite3.Row) -> StoredChunk:
        metadata = {
            "doc_id": row["doc_id"],
            "chunk_id": row["chunk_id"],
            "vector_id": row["vector_id"],
            "source_path": row["source_path"],
            "source_name": row["source_name"],
            "source_type": row["source_type"],
            "file_hash": row["file_hash"],
            "file_mtime": row["file_mtime"],
            "version": row["version"],
            "chunk_index": row["chunk_index"],
            "chunk_hash": row["chunk_hash"],
            "book_id": row["book_id"],
            "title": row["title"],
            "author": row["author"],
            "isbn": row["isbn"],
            "call_number": row["call_number"],
            "category": row["category"],
            "subjects": row["subjects"],
            "main_characters": row["main_characters"],
            "plot_summary": row["plot_summary"],
            "shelf": row["shelf"],
            "shelf_code": row["shelf_code"],
            "shelf_row": row["shelf_row"],
            "shelf_col": row["shelf_col"],
            "floor": row["floor"],
            "area": row["area"],
            "copy_count": row["copy_count"],
            "available_count": row["available_count"],
            "availability": row["availability"],
            "borrow_rule": row["borrow_rule"],
            "open_time": row["open_time"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted_at": row["deleted_at"],
        }
        return StoredChunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            vector_id=str(row["vector_id"]),
            source_path=str(row["source_path"]),
            source_name=str(row["source_name"]),
            source_type=str(row["source_type"]),
            file_hash=str(row["file_hash"]),
            file_mtime=str(row["file_mtime"] or ""),
            version=int(row["version"]),
            chunk_index=int(row["chunk_index"]),
            chunk_hash=str(row["chunk_hash"]),
            content=str(row["content"]),
            book_id=row["book_id"],
            title=row["title"],
            author=row["author"],
            isbn=row["isbn"],
            call_number=row["call_number"],
            category=row["category"],
            subjects=row["subjects"],
            main_characters=row["main_characters"],
            plot_summary=row["plot_summary"],
            shelf=row["shelf"],
            shelf_code=row["shelf_code"],
            shelf_row=row["shelf_row"],
            shelf_col=row["shelf_col"],
            floor=row["floor"],
            area=row["area"],
            copy_count=row["copy_count"],
            available_count=row["available_count"],
            availability=row["availability"],
            borrow_rule=row["borrow_rule"],
            open_time=row["open_time"],
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            deleted_at=row["deleted_at"],
            metadata=metadata,
        )
