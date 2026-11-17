"""Turn uploaded files into page-aware text segments."""
import io
import re

from pypdf import PdfReader

TEXT_TYPES = (".txt", ".md", ".csv")
_DIGIT_ONLY = re.compile(r"^\d+$")


def clean_page_text(text: str) -> str:
    """Drop lines that are only digits (stray page numbers / footers)."""
    lines = text.splitlines()
    kept = [ln for ln in lines if not _DIGIT_ONLY.match(ln.strip())]
    return "\n".join(kept)


def extract_text(filename: str, data: bytes) -> list[dict]:
    """Return [{"page": int | None, "text": str}, ...].

    PDFs yield one entry per page (1-based). TXT/MD/CSV yield a single
    entry with page=None.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages: list[dict] = []
        for i, page in enumerate(reader.pages, 1):
            raw = page.extract_text() or ""
            text = clean_page_text(raw)
            if text.strip():
                pages.append({"page": i, "text": text})
        return pages
    if name.endswith(TEXT_TYPES):
        return [{"page": None, "text": data.decode("utf-8", errors="ignore")}]
    raise ValueError(f"Unsupported file type: {filename}. Use PDF, TXT, MD or CSV.")
