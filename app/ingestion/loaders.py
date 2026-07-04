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


def _safe_int(value: object, default: int = 0) -> int:
    text = _normalize_text(value)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"Expected integer value, got {text!r}") from None


def _list_metadata(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_text(item) for item in value if _normalize_text(item)]
    text = _normalize_text(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,，;；、|]", text) if part.strip()]


def _join_list(values: list[str]) -> str:
    return "、".join(values)


def _source_metadata(path: Path, source_type: str) -> dict[str, str]:
    return {
        "source": path.name,
        "source_name": path.name,
        "source_path": str(path.resolve()),
        "source_type": source_type,
    }


REQUIRED_BOOK_FIELDS = ("book_id", "title", "category", "shelf_code", "shelf_row", "shelf_col")


def validate_book_record(record: dict[str, object], source_path: Path, index: int) -> None:
    aliases = {
        "title": ("title", "book_name"),
        "category": ("category", "type"),
        "shelf_code": ("shelf_code", "shelf"),
        "shelf_row": ("shelf_row", "row"),
        "shelf_col": ("shelf_col", "col", "column"),
    }
    missing = []
    for field in REQUIRED_BOOK_FIELDS:
        keys = aliases.get(field, (field,))
        if not any(_normalize_text(record.get(key)) for key in keys):
            missing.append(field)
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"{source_path.name} record #{index + 1} missing required field(s): {fields}")


def _record_to_document(record: dict[str, object], source_path: Path, index: int) -> Document:
    validate_book_record(record, source_path, index)
    book_id = _safe_metadata(record.get("book_id") or record.get("barcode") or record.get("accession_no"))
    title = _safe_metadata(record.get("title") or record.get("book_name"))
    author = _safe_metadata(record.get("author") or record.get("writer"))
    isbn = _safe_metadata(record.get("isbn"))
    category = _safe_metadata(record.get("category") or record.get("type"))
    call_number = _safe_metadata(record.get("call_number") or record.get("classification_no"))
    subjects = _list_metadata(record.get("subjects") or record.get("keywords"))
    main_characters = _list_metadata(record.get("main_characters") or record.get("characters") or record.get("protagonists"))
    plot_summary = _safe_metadata(record.get("plot_summary") or record.get("book_summary") or record.get("summary") or record.get("description"))
    shelf_code = _safe_metadata(record.get("shelf_code") or record.get("shelf"))
    shelf_row = _safe_int(record.get("shelf_row") or record.get("row"), default=1)
    shelf_col = _safe_int(record.get("shelf_col") or record.get("col") or record.get("column"), default=1)
    floor = _safe_metadata(record.get("floor"))
    area = _safe_metadata(record.get("area"))
    copy_count = _safe_int(record.get("copy_count"), default=1)
    available_count = _safe_int(record.get("available_count"), default=copy_count)
    shelf = f"{shelf_code}书架 第{shelf_row}行 第{shelf_col}列"
    availability = _safe_metadata(record.get("availability") or record.get("status"))
    borrow_rule = _safe_metadata(record.get("borrow_rule") or record.get("borrow_rules"))
    open_time = _safe_metadata(record.get("open_time") or record.get("opening_hours"))
    faq = _safe_metadata(record.get("faq"))

    content = "\n".join(
        part
        for part in [
            f"馆藏编号: {book_id}" if book_id else "",
            f"书名: {title}" if title else "",
            f"作者: {author}" if author else "",
            f"ISBN: {isbn}" if isbn else "",
            f"索书号: {call_number}" if call_number else "",
            f"分类: {category}" if category else "",
            f"主题词: {_join_list(subjects)}" if subjects else "",
            f"主角: {_join_list(main_characters)}" if main_characters else "",
            f"馆藏位置: {shelf}" if shelf else "",
            f"楼层: {floor}" if floor else "",
            f"区域: {area}" if area else "",
            f"馆藏状态: {availability}" if availability else "",
            f"馆藏册数: {copy_count}",
            f"可借册数: {available_count}",
            f"借阅规则: {borrow_rule}" if borrow_rule else "",
            f"开放时间: {open_time}" if open_time else "",
            f"书籍大意: {plot_summary}" if plot_summary else "",
            f"FAQ: {faq}" if faq else "",
        ]
        if part
    ).strip()

    metadata = {
        **_source_metadata(source_path, "json"),
        "doc_type": "book_record",
        "record_index": index,
        "book_id": book_id,
        "title": title,
        "author": author,
        "isbn": isbn,
        "call_number": call_number,
        "category": category,
        "subjects": _join_list(subjects),
        "main_characters": _join_list(main_characters),
        "plot_summary": plot_summary,
        "shelf": shelf,
        "shelf_code": shelf_code,
        "shelf_row": shelf_row,
        "shelf_col": shelf_col,
        "floor": floor,
        "area": area,
        "copy_count": copy_count,
        "available_count": available_count,
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
    documents: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, record in enumerate(reader):
            documents.append(_record_to_document(dict(record), path, index))
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
