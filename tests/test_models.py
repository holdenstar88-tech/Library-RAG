from app.models import CatalogSearchRequest, CatalogSearchResponse


def test_catalog_search_pagination_defaults_match_opac_style() -> None:
    payload = CatalogSearchRequest()

    assert payload.page == 1
    assert payload.limit == 20


def test_catalog_search_response_exposes_pagination_metadata() -> None:
    response = CatalogSearchResponse(
        query="科幻",
        total=45,
        page=2,
        limit=20,
        total_pages=3,
        has_prev=True,
        has_next=True,
    )

    assert response.total_pages == 3
    assert response.has_prev is True
    assert response.has_next is True
