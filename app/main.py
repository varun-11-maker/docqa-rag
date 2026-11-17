import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from app.chunking import chunk_pages  # noqa: E402
from app.citations import mark_cited  # noqa: E402
from app.extract import extract_text  # noqa: E402
from app.llm import MODEL, generate, llm_enabled  # noqa: E402
from app.store import Store  # noqa: E402

TOP_K = int(os.getenv("TOP_K", 4))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.25))
MAX_UPLOAD = 10 * 1024 * 1024
DATA_DIR = os.getenv("DATA_DIR", "./data")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    state["store"] = Store(DATA_DIR, os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    yield


app = FastAPI(title="DocQ&A", version="1.0.0", lifespan=lifespan)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class FeedbackBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=20000)
    rating: Literal["up", "down"]


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL, "llm_enabled": llm_enabled()}


@app.get("/api/documents")
def list_documents():
    return state["store"].list_docs()


@app.post("/api/documents")
def upload(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        name = f.filename
        if not name:
            raise HTTPException(400, "Uploaded file is missing a filename.")
        data = f.file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{name} is larger than 10 MB.")
        try:
            pages = extract_text(name, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        chunks = chunk_pages(pages)
        if not chunks:
            raise HTTPException(400, f"No readable text found in {name}. Scanned PDFs need OCR.")
        doc_id = hashlib.sha1(name.encode()).hexdigest()[:12]
        state["store"].add(doc_id, name, chunks)
        added.append({"id": doc_id, "name": name, "chunks": len(chunks)})
    return added


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    state["store"].delete(doc_id)
    return {"deleted": doc_id}


@app.post("/api/ask")
def ask(body: AskBody):
    t0 = time.perf_counter()
    hits = [h for h in state["store"].search(body.question, TOP_K) if h["score"] >= MIN_SCORE]
    if not hits:  # skip the LLM call entirely: cheaper and can't hallucinate
        return {"answer": "I couldn’t find that in the uploaded documents.", "sources": [],
                "mode": "no_match", "latency_ms": round((time.perf_counter() - t0) * 1000)}
    try:
        answer, mode = generate(body.question, hits)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    sources = mark_cited(answer, hits, mode)
    return {"answer": answer, "sources": sources, "mode": mode,
            "latency_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/api/feedback")
def feedback(body: FeedbackBody):
    path = Path(DATA_DIR) / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": body.question,
        "answer": body.answer,
        "rating": body.rating,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
