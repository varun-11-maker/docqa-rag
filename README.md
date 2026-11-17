# DocQ&A: Retrieval-Augmented Document Chatbot

Ask questions about your own PDFs and text files and get **grounded answers with citations**. Built with FastAPI, a vector database and Gemini.

**Problem it solves:** teams lose hours searching policies, contracts and manuals, and plain chatbots invent answers. DocQ&A only answers from the uploaded documents, shows the exact passages it used, and says so when the answer isn't there.

## How it works

```
Upload (PDF/TXT/MD) -> extract text -> sentence-aware chunks (with overlap)
        -> embeddings (sentence-transformers) -> ChromaDB (persistent)

Question -> embed -> top-K similar chunks -> relevance threshold
        -> Gemini answers using ONLY those chunks, citing [1] [2]
        -> answer + source passages + latency returned to the UI
```

## Key design decisions

| Decision | Why |
|---|---|
| Semantic search with embeddings | Finds "money back" when the document says "refund" |
| Local embedding model | Free, private, no extra API key; swap via `EMBEDDING_MODEL` |
| Relevance threshold (`MIN_SCORE`) | Off-topic questions skip the LLM call: cheaper and no hallucination |
| Strict grounding prompt + citations | Every claim is checkable against its source |
| Passages treated as untrusted data | Mitigates prompt injection from uploaded documents |
| Extractive fallback without API key | The app still runs for demos and local development |
| Persistent store + Docker | Documents survive restarts; one-command deploy |

## Quick start

```bash
git clone <your-repo-url> && cd docqa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY
uvicorn app.main:app --reload   # open http://localhost:8000
```

Or with Docker: `docker compose up --build`

Try it with `sample_docs/refund_policy.md`, then ask "How long do refunds take?".

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model, whether the LLM is enabled |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents` | Upload one or more files (multipart) |
| DELETE | `/api/documents/{id}` | Remove a document |
| POST | `/api/ask` | `{"question": "..."}` returns `answer`, `sources`, `mode`, `latency_ms` |

Interactive docs are at `/docs`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Enables written answers |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model used |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `TOP_K` | 4 | Passages sent to the LLM |
| `MIN_SCORE` | 0.25 | Minimum similarity to attempt an answer |

## Tests

```bash
pytest
```

## Limitations and roadmap

- Scanned PDFs need OCR (planned: Tesseract step).
- Single shared knowledge base; no per-user access control yet (planned: auth + per-tenant collections).
- Planned: streaming answers, hybrid BM25 + vector search, reranking, DOCX/Notion/Google Drive ingestion, evaluation harness.
