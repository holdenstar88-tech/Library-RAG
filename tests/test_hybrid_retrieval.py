from app.retrieval.hybrid import infer_metadata_filters, metadata_matches, normalize_text, rrf_fuse
from langchain_core.documents import Document
from app.retrieval.hybrid import RetrievedChunk


def test_infer_filters_by_isbn():
    filters = infer_metadata_filters("请问 ISBN 9787302523190 的书在哪里？")
    assert filters["isbn"] == "9787302523190"


def test_infer_filters_by_book_id_call_number_and_character():
    filters = infer_metadata_filters("馆藏编号 LIB-2026-0001 索书号 I242.4/W80 主角 孙悟空")
    assert filters["book_id"] == "LIB-2026-0001"
    assert "isbn" not in filters
    assert filters["call_number"] == "I242.4/W80"
    assert filters["main_characters"] == "孙悟空"


def test_infer_popular_category_and_available_only():
    filters = infer_metadata_filters("找一本可借的科幻书")
    assert filters["category"] == "科幻"
    assert filters["available_only"] == "true"


def test_metadata_matches_available_count_and_subjects():
    document = Document(
        page_content="三体文明",
        metadata={"category": "科幻", "subjects": "外星文明、宇宙社会学", "available_count": 2},
    )
    assert metadata_matches(document, {"category": "科幻", "subjects": "外星文明", "available_only": "true"})


def test_rrf_prefers_documents_present_in_both_sources():
    doc_a = Document(page_content="A", metadata={"source_path": "a", "chunk_index": 0, "chunk_hash": "1"})
    doc_b = Document(page_content="B", metadata={"source_path": "b", "chunk_index": 0, "chunk_hash": "2"})
    fused = rrf_fuse(
        [RetrievedChunk(document=doc_a, score=1.0, source="bm25", rank=1), RetrievedChunk(document=doc_b, score=0.9, source="bm25", rank=2)],
        [RetrievedChunk(document=doc_a, score=0.95, source="vector", rank=1)],
    )
    assert fused[0].document.page_content == "A"
