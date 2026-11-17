from app.extract import clean_page_text, extract_text


def test_clean_page_text_removes_digit_only_lines():
    raw = "Hello world\n42\nMore text\n  7  \nEnd"
    cleaned = clean_page_text(raw)
    assert "42" not in cleaned.splitlines()
    assert "7" not in [ln.strip() for ln in cleaned.splitlines()]
    assert "Hello world" in cleaned
    assert "More text" in cleaned
    assert "End" in cleaned


def test_clean_keeps_lines_with_digits_and_words():
    raw = "See page 12 for details\n12\nDone"
    cleaned = clean_page_text(raw)
    assert "See page 12 for details" in cleaned
    assert cleaned.splitlines()[-1] == "Done"
    assert "\n12\n" not in f"\n{cleaned}\n"


def test_extract_text_file_has_no_page():
    pages = extract_text("notes.md", b"# Title\n\nHello")
    assert len(pages) == 1
    assert pages[0]["page"] is None
    assert "Hello" in pages[0]["text"]


def test_extract_rejects_unsupported_type():
    try:
        extract_text("x.docx", b"abc")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Unsupported" in str(e)
