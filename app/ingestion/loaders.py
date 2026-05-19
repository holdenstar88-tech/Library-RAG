from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_metadata(value: object) -> str:
    text = _normalize_text(value)
    return text if text else ""


def _source_metadata(path: Path, source_type: str) -> dict[str, str]:
    return {
        "source": path.name,
        "source_name": path.name,
        "source_path": str(path.resolve()),
        "source_type": source_type,
    }


def _record_to_document(record: dict[str, object], source_path: Path, index: int) -> Document:
    title = _safe_metadata(record.get("title") or record.get("book_name"))
    author = _safe_metadata(record.get("author") or record.get("writer"))
    isbn = _safe_metadata(record.get("isbn"))
    category = _safe_metadata(record.get("category") or record.get("type"))
    shelf = _safe_metadata(record.get("shelf") or record.get("location"))
    availability = _safe_metadata(record.get("availability") or record.get("status"))
    borrow_rule = _safe_metadata(record.get("borrow_rule") or record.get("borrow_rules"))
    open_time = _safe_metadata(record.get("open_time") or record.get("opening_hours"))
    faq = _safe_metadata(record.get("faq"))
    summary = _safe_metadata(record.get("summary") or record.get("description"))

    content = "\n".join(
        part
        for part in [
            f"书名: {title}" if title else "",
            f"作者: {author}" if author else "",
            f"ISBN: {isbn}" if isbn else "",
            f"分类: {category}" if category else "",
            f"馆藏位置: {shelf}" if shelf else "",
            f"馆藏状态: {availability}" if availability else "",
            f"借阅规则: {borrow_rule}" if borrow_rule else "",
            f"开放时间: {open_time}" if open_time else "",
            f"简介: {summary}" if summary else "",
            f"FAQ: {faq}" if faq else "",
        ]
        if part
    ).strip()

    metadata = {
        **_source_metadata(source_path, "json"),
        "doc_type": "book_record",
        "record_index": index,
        "title": title,
        "author": author,
        "isbn": isbn,
        "category": category,
        "shelf": shelf,
        "availability": availability,
        "borrow_rule": borrow_rule,
        "open_time": open_time,
    }
    return Document(page_content=content or json.dumps(record, ensure_ascii=False), metadata=metadata)


def _load_json(path: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, object]]
    if isinstance(payload, dict):
        if "books" in payload and isinstance(payload["books"], list):
            records = [item for item in payload["books"] if isinstance(item, dict)]
        else:
            records = [payload]
    elif isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    else:
        records = []
    return [_record_to_document(record, path, index) for index, record in enumerate(records)]


def _load_markdown(path: Path) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    documents = splitter.split_text(path.read_text(encoding="utf-8"))
    for document in documents:
        document.metadata.update(_source_metadata(path, "markdown"))
    return documents


def _load_txt(path: Path) -> list[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    for document in documents:
        document.metadata.update(_source_metadata(path, "txt"))
    return documents


def _load_csv(path: Path) -> list[Document]:
    loader = CSVLoader(file_path=str(path), encoding="utf-8")
    documents = loader.load()
    for document in documents:
        document.metadata.update(_source_metadata(path, "csv"))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    for document in documents:
        document.metadata.update(_source_metadata(path, "pdf"))
    return documents


def load_documents_from_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".md", ".markdown"}:
        return _load_markdown(path)
    if suffix == ".txt":
        return _load_txt(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={**_source_metadata(path, "text"), "doc_type": "text"},
        )
    ]


def load_documents_from_dir(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    if not data_dir.exists():
        return documents
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        documents.extend(load_documents_from_file(path))
    return documents
