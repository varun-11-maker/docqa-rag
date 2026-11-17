"""Citation parsing helpers (pure; no model downloads)."""
import re

_CITE = re.compile(r"\[(\d+)\]")


def parse_citation_indices(answer: str) -> set[int]:
    """Return 1-based citation numbers found in the answer, e.g. [1], [2][3]."""
    return {int(n) for n in _CITE.findall(answer)}


def mark_cited(answer: str, sources: list[dict], mode: str = "llm") -> list[dict]:
    """Attach cited=true/false to each source.

    Extractive mode always marks source 1. If the LLM cites nothing, mark the
    top source as cited so the UI still has something to show.
    """
    indices = parse_citation_indices(answer)
    out: list[dict] = []
    for i, src in enumerate(sources, 1):
        item = {
            "doc": src["doc"],
            "page": src.get("page"),
            "text": src["text"],
            "score": src.get("score"),
            "cited": i in indices,
        }
        if "chunk" in src:
            item["chunk"] = src["chunk"]
        out.append(item)

    if not out:
        return out

    if mode == "extractive":
        out[0]["cited"] = True
    elif not any(s["cited"] for s in out):
        out[0]["cited"] = True
    return out
