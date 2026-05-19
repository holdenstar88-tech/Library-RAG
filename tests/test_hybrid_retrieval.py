from app.retrieval.hybrid import infer_metadata_filters, normalize_text, rrf_fuse
from langchain_core.documents import Document
from app.retrieval.hybrid import RetrievedChunk


def test_infer_filters_by_isbn():
    filters = infer_metadata_filters("请问 ISBN 9787302523190 的书在哪里？")
    assert filters["isbn"] == "9787302523190"


def test_rrf_prefers_documents_present_in_both_sources():
    doc_a = Document(page_content="A", metadata={"source_path": "a", "chunk_index": 0, "chunk_hash": "1"})
    doc_b = Document(page_content="B", metadata={"source_path": "b", "chunk_index": 0, "chunk_hash": "2"})
    fused = rrf_fuse(
        [RetrievedChunk(document=doc_a, score=1.0, source="bm25", rank=1), RetrievedChunk(document=doc_b, score=0.9, source="bm25", rank=2)],
        [RetrievedChunk(document=doc_a, score=0.95, source="vector", rank=1)],
    )
    assert fused[0].document.page_content == "A"

