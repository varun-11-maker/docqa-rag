from app.citations import mark_cited, parse_citation_indices


def test_parse_single_and_adjacent_citations():
    assert parse_citation_indices("Hello [1] world.") == {1}
    assert parse_citation_indices("A [1][2][3] B") == {1, 2, 3}
    assert parse_citation_indices("No citations here") == set()


def test_mark_cited_from_answer():
    sources = [
        {"doc": "a.pdf", "page": 1, "text": "one", "score": 0.9},
        {"doc": "a.pdf", "page": 2, "text": "two", "score": 0.8},
        {"doc": "a.pdf", "page": 3, "text": "three", "score": 0.7},
    ]
    out = mark_cited("Fact [1] and more [3].", sources, mode="llm")
    assert [s["cited"] for s in out] == [True, False, True]
    assert out[0]["page"] == 1
    assert out[1]["doc"] == "a.pdf"


def test_extractive_marks_first_source():
    sources = [
        {"doc": "a.md", "page": None, "text": "top", "score": 0.5},
        {"doc": "a.md", "page": None, "text": "other", "score": 0.4},
    ]
    out = mark_cited("top [1]", sources, mode="extractive")
    assert out[0]["cited"] is True


def test_no_citations_marks_top_source():
    sources = [
        {"doc": "a.pdf", "page": 4, "text": "alpha", "score": 0.6},
        {"doc": "a.pdf", "page": 5, "text": "beta", "score": 0.5},
    ]
    out = mark_cited("I could not find a citation marker.", sources, mode="llm")
    assert out[0]["cited"] is True
    assert out[1]["cited"] is False
