from pathlib import Path

import pytest

from app.ingestion.loaders import load_documents_from_file


def test_json_book_record_includes_opac_fields(tmp_path: Path):
    path = tmp_path / "books.json"
    path.write_text(
        """
        [
          {
            "book_id": "LIB-1",
            "title": "西游记",
            "author": "吴承恩",
            "category": "文学",
            "subjects": ["神魔", "取经"],
            "main_characters": ["孙悟空"],
            "plot_summary": "师徒四人西行取经。",
            "shelf_code": "F",
            "shelf_row": 1,
            "shelf_col": 2
          }
        ]
        """,
        encoding="utf-8",
    )

    document = load_documents_from_file(path)[0]

    assert document.metadata["book_id"] == "LIB-1"
    assert document.metadata["main_characters"] == "孙悟空"
    assert document.metadata["shelf"] == "F书架 第1行 第2列"
    assert "书籍大意: 师徒四人西行取经。" in document.page_content


def test_csv_book_record_is_validated_and_loaded(tmp_path: Path):
    path = tmp_path / "books.csv"
    path.write_text(
        "book_id,title,category,shelf_code,shelf_row,shelf_col,subjects,main_characters,plot_summary\n"
        "LIB-2,三体,科幻,C,2,5,外星文明;宇宙社会学,叶文洁;汪淼,人类与三体文明建立联系。\n",
        encoding="utf-8",
    )

    document = load_documents_from_file(path)[0]

    assert document.metadata["category"] == "科幻"
    assert document.metadata["subjects"] == "外星文明、宇宙社会学"
    assert document.metadata["main_characters"] == "叶文洁、汪淼"


def test_book_record_requires_core_location_fields(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('[{"title": "缺字段"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field"):
        load_documents_from_file(path)
