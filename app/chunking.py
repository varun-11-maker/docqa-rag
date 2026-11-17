"""Sentence-aware chunking that preserves original whitespace and page boundaries."""
import re

_SEP = re.compile(r"(?<=[.!?])[ \t]+|\n{2,}")


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of segments over the original string."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SEP.finditer(text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentence_spans(text)]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_page(
    text: str,
    page: int | None,
    max_words: int = 180,
    overlap: int = 30,
) -> list[dict]:
    """Chunk one page's text into {"text", "page"} items, preserving line breaks."""
    spans = split_sentence_spans(text)
    if not spans:
        return []

    chunks: list[dict] = []
    cur: list[tuple[int, int]] = []
    count = 0

    for span in spans:
        words = _word_count(text[span[0] : span[1]])
        if cur and count + words > max_words:
            chunks.append({"text": text[cur[0][0] : cur[-1][1]], "page": page})
            carry: list[tuple[int, int]] = []
            carried = 0
            for prev in reversed(cur):
                n = _word_count(text[prev[0] : prev[1]])
                if carried + n > overlap:
                    break
                carry.insert(0, prev)
                carried += n
            cur, count = carry, carried
        cur.append(span)
        count += words

    if cur:
        chunks.append({"text": text[cur[0][0] : cur[-1][1]], "page": page})
    return chunks


def chunk_pages(
    pages: list[dict],
    max_words: int = 180,
    overlap: int = 30,
) -> list[dict]:
    """Chunk each page separately so every chunk has exactly one page number."""
    out: list[dict] = []
    for p in pages:
        out.extend(chunk_page(p.get("text", ""), p.get("page"), max_words, overlap))
    return out


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    """Convenience wrapper: chunk a single page-less string into text list."""
    return [c["text"] for c in chunk_page(text, None, max_words, overlap)]
