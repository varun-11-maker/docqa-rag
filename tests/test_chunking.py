from app.chunking import chunk_page, chunk_pages, chunk_text, split_sentences

TEXT = " ".join(f"Sentence number {i} talks about refunds and shipping." for i in range(60))

CODE = """def foo():
    return 1

def bar():
    return 2
"""


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []
    assert chunk_pages([{"page": 1, "text": "   \n\n "}]) == []


def test_chunks_respect_size_limit():
    for c in chunk_text(TEXT, max_words=50, overlap=10):
        assert len(c.split()) <= 50 + 10


def test_no_sentence_is_lost():
    joined = "\n".join(chunk_text(TEXT, max_words=50, overlap=10))
    for s in split_sentences(TEXT):
        assert s in joined


def test_consecutive_chunks_overlap_within_page():
    chunks = chunk_page(TEXT, page=1, max_words=50, overlap=15)
    assert len(chunks) > 1
    assert all(c["page"] == 1 for c in chunks)
    last_sentence = split_sentences(chunks[0]["text"])[-1]
    assert last_sentence in chunks[1]["text"]


def test_preserves_line_breaks_and_indentation():
    chunks = chunk_pages([{"page": None, "text": CODE}])
    assert chunks
    joined = "\n".join(c["text"] for c in chunks)
    assert "def foo():\n    return 1" in joined
    assert "\n    return" in joined


def test_one_page_per_chunk():
    pages = [
        {"page": 1, "text": TEXT},
        {"page": 2, "text": " ".join(f"Other topic sentence {i}." for i in range(40))},
    ]
    chunks = chunk_pages(pages, max_words=50, overlap=10)
    assert chunks
    for c in chunks:
        assert c["page"] in {1, 2}
    # no chunk mixes pages — page field is singular
    assert {c["page"] for c in chunks} == {1, 2}
