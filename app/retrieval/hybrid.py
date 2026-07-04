from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


ISBN_RE = re.compile(r"(?i)\bisbn[:：]?\s*([0-9xX-]{10,20})\b")
BARE_ISBN_RE = re.compile(r"(?<![A-Za-z0-9-])([0-9][0-9xX-]{9,19})(?![A-Za-z0-9-])")
BOOK_ID_RE = re.compile(r"(?i)\b(?:book[_\s-]?id|馆藏编号|条码号|编号)[:：]?\s*([A-Z0-9-]{3,32})\b")
CALL_NUMBER_RE = re.compile(r"(?:索书号|分类号|call\s*number)[:：]?\s*([A-Za-z0-9./-]{2,40})", re.I)
TITLE_RE = re.compile(r"书名[:：]\s*([^\s，。,;；]{2,60})")
AUTHOR_RE = re.compile(r"作者[:：]\s*([^\s，。,;；]{2,40})")
CATEGORY_RE = re.compile(r"分类[:：]\s*([^\s，。,;；]{2,40})")
CHARACTER_RE = re.compile(r"(?:主角|人物)[:：]?\s*([^\s，。,;；]{2,40})")
SUBJECT_RE = re.compile(r"(?:主题|主题词|内容)[:：]?\s*([^\s，。,;；]{2,60})")

POPULAR_CATEGORIES = [
    "文学",
    "历史",
    "科幻",
    "计算机",
    "艺术",
    "哲学",
    "社科",
    "教育",
    "医学",
    "自然科学",
    "经济管理",
]


@dataclass(frozen=True)
class RetrievedChunk:
    document: Document
    score: float
    source: str
    rank: int


def normalize_text(text: str) -> list[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return tokens or list(text.lower())


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    source_path = metadata.get("source_path") or metadata.get("source") or ""
    chunk_index = metadata.get("chunk_index", 0)
    chunk_hash = metadata.get("chunk_hash") or ""
    return f"{source_path}::{chunk_index}::{chunk_hash}"


def infer_metadata_filters(question: str) -> dict[str, str]:
    filters: dict[str, str] = {}
    book_id_match = BOOK_ID_RE.search(question)
    if book_id_match:
        filters["book_id"] = book_id_match.group(1).upper()
    isbn_match = ISBN_RE.search(question) or BARE_ISBN_RE.search(question)
    if isbn_match:
        normalized_isbn = isbn_match.group(1).replace("-", "")
        if len(normalized_isbn) in {10, 13}:
            filters["isbn"] = normalized_isbn
    call_number_match = CALL_NUMBER_RE.search(question)
    if call_number_match:
        filters["call_number"] = call_number_match.group(1)
    title_match = TITLE_RE.search(question)
    if title_match:
        filters["title"] = title_match.group(1)
    author_match = AUTHOR_RE.search(question)
    if author_match:
        filters["author"] = author_match.group(1)
    category_match = CATEGORY_RE.search(question)
    if category_match:
        filters["category"] = category_match.group(1)
    else:
        for category in POPULAR_CATEGORIES:
            if category in question:
                filters["category"] = category
                break
    character_match = CHARACTER_RE.search(question)
    if character_match:
        filters["main_characters"] = character_match.group(1)
    subject_match = SUBJECT_RE.search(question)
    if subject_match:
        filters["subjects"] = subject_match.group(1)
    if "可借" in question or "能借" in question or "在馆" in question:
        filters["available_only"] = "true"
    return filters


def metadata_matches(document: Document, filters: dict[str, str]) -> bool:
    if not filters:
        return True
    metadata = document.metadata or {}
    for key, expected in filters.items():
        if key == "available_only":
            if str(expected).lower() != "true":
                continue
            available_count = metadata.get("available_count")
            if available_count is not None:
                try:
                    if int(available_count) > 0:
                        continue
                except (TypeError, ValueError):
                    pass
            availability = str(metadata.get("availability", "") or "")
            if "可借" not in availability and "在馆" not in availability:
                return False
            continue
        value = str(metadata.get(key, "") or "")
        if key in {"isbn", "book_id"}:
            if expected.replace("-", "") not in value.replace("-", ""):
                return False
        elif expected.lower() not in value.lower():
            return False
    return True


def build_bm25_index(documents: Iterable[Document]) -> tuple[BM25Okapi, list[Document], list[list[str]]]:
    docs = list(documents)
    tokenized = [normalize_text(doc.page_content) for doc in docs]
    index = BM25Okapi(tokenized) if tokenized else BM25Okapi([["empty"]])
    return index, docs, tokenized


def bm25_search(index: BM25Okapi, documents: list[Document], query: str, top_k: int) -> list[RetrievedChunk]:
    if not documents:
        return []
    query_tokens = normalize_text(query)
    scores = index.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    results: list[RetrievedChunk] = []
    for rank, idx in enumerate(ranked, start=1):
        score = float(scores[idx])
        results.append(RetrievedChunk(document=documents[int(idx)], score=score, source="bm25", rank=rank))
    return results


def vector_search(vectorstore: Any, query: str, top_k: int) -> list[RetrievedChunk]:
    if vectorstore is None:
        return []
    try:
        scored = vectorstore.similarity_search_with_score(query, k=top_k)
    except Exception:
        return []
    results: list[RetrievedChunk] = []
    for rank, (document, score) in enumerate(scored, start=1):
        results.append(RetrievedChunk(document=document, score=float(score), source="vector", rank=rank))
    return results


def rrf_fuse(*result_sets: Iterable[RetrievedChunk], k: int = 60) -> list[RetrievedChunk]:
    fused: dict[str, dict[str, Any]] = {}
    for source_results in result_sets:
        for item in source_results:
            key = document_key(item.document)
            bucket = fused.setdefault(
                key,
                {
                    "document": item.document,
                    "score": 0.0,
                    "sources": set(),
                    "best_rank": math.inf,
                },
            )
            bucket["score"] += 1.0 / (k + item.rank)
            bucket["sources"].add(item.source)
            bucket["best_rank"] = min(bucket["best_rank"], item.rank)
    ranked = sorted(
        fused.values(),
        key=lambda item: (item["score"], len(item["sources"])),
        reverse=True,
    )
    return [
        RetrievedChunk(
            document=item["document"],
            score=float(item["score"]),
            source="+".join(sorted(item["sources"])),
            rank=index,
        )
        for index, item in enumerate(ranked, start=1)
    ]


def apply_filters(documents: Iterable[RetrievedChunk], filters: dict[str, str]) -> list[RetrievedChunk]:
    filtered = [item for item in documents if metadata_matches(item.document, filters)]
    return filtered


def compute_confidence(results: list[RetrievedChunk], filters: dict[str, str]) -> float:
    if not results:
        return 0.0
    confidence = 0.45
    if len(results) >= 1:
        confidence += 0.12
    if len(results) >= 2:
        confidence += 0.08
    if len(results) >= 3:
        confidence += 0.05
    source_set: set[str] = set()
    for item in results[:3]:
        source_set.update(part for part in item.source.split("+") if part)
    if len(source_set) >= 2:
        confidence += 0.18
    if filters:
        confidence += 0.12
    if len(results) == 1:
        confidence -= 0.05
    return round(confidence, 3)
