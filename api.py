import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.parser import parse_epub
from src.chunker import chunk_chapters
from src.embedder import embed_chunks
from src.enricher import enrich_chunks
from src.store_firestore import store_chunks_firestore, delete_book_firestore
from src.retriever_firestore import retrieve_hybrid_firestore, rerank
from src.answer import answer
from src.book_catalog import (
    book_exists_firestore, register_book_firestore,
    delete_book_catalog_firestore, list_books_firestore,
)

load_dotenv()

app = FastAPI(title="Pergunte ao Livro API")


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), enrich: bool = False, provider: str = "anthropic"):
    if not file.filename.endswith(".epub"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um .epub")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        book_meta, chapters = parse_epub(tmp_path)
        book_id = book_meta["id"]

        delete_book_firestore(book_id)
        delete_book_catalog_firestore(book_id)

        chunks = chunk_chapters(chapters, book_id=book_id)
        if enrich:
            chunks = enrich_chunks(chunks, chapters, provider=provider)
        embedded_chunks = embed_chunks(chunks)
        store_chunks_firestore(embedded_chunks)
        register_book_firestore(book_meta, chunk_count=len(embedded_chunks), enriched=enrich)
    finally:
        os.unlink(tmp_path)

    return {
        "status": "ok",
        "book_id": book_id,
        "title": book_meta.get("title"),
        "chapters": len(chapters),
        "chunks": len(embedded_chunks),
    }


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    model: str = "claude-sonnet-4-20250514"
    book_id: str | None = None


@app.post("/ask")
async def ask(req: AskRequest):
    chunks = retrieve_hybrid_firestore(req.query, n_results=req.top_k, book_id=req.book_id)

    if not chunks:
        raise HTTPException(status_code=404, detail="Nenhum trecho relevante encontrado")

    reranked = rerank(req.query, chunks)
    response = answer(req.query, reranked, model=req.model)

    sources = [
        {"chapter_id": c["chapter_id"], "book_id": c.get("book_id"), "relevance_score": c.get("relevance_score", 0.0)}
        for c in reranked
    ]

    return {"answer": response, "sources": sources}


@app.get("/books")
async def books():
    return list_books_firestore()
