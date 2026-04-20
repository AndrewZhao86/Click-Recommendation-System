from click_rec.models import Base


def test_metadata_contains_phase2_tables() -> None:
    expected = {"item", "user_account", "click_event", "search_query", "co_click"}
    assert expected.issubset(Base.metadata.tables.keys())


def test_item_has_vector_and_tsvector_columns() -> None:
    item = Base.metadata.tables["item"]
    assert "embedding" in item.columns
    assert "tsv" in item.columns
