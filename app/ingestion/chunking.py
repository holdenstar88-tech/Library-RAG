from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_documents(documents: Iterable[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        base_metadata = dict(document.metadata)
        split_docs = splitter.split_documents([document])
        for index, chunk in enumerate(split_docs):
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_hash": _content_hash(chunk.page_content),
                    "chunk_length": len(chunk.page_content),
                }
            )
            chunks.append(Document(page_content=chunk.page_content.strip(), metadata=metadata))
    return chunks

