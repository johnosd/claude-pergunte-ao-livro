import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.parser import parse_epub
from src.chunker import chunk_chapters
from src.embedder import embed_chunks
from src.enricher import enrich_chunks
from src.store_firestore import store_chunks_firestore, delete_collection_firestore
from src.retriever_firestore import retrieve_hybrid_firestore, rerank
from src.answer import answer

load_dotenv()

app = FastAPI(title="Pergunte ao Livro API")

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), enrich: bool = False):
    # Valida que é um EPUB
    if not file.filename.endswith(".epub"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um .epub")

    # Salva em arquivo temporário para o parser conseguir ler
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        delete_collection_firestore()

        chapters = parse_epub(tmp_path)
        chunks = chunk_chapters(chapters)
        if enrich:
            chunks = enrich_chunks(chunks, chapters)
        embedded_chunks = embed_chunks(chunks)
        store_chunks_firestore(embedded_chunks)
    finally:
        os.unlink(tmp_path)  # remove o arquivo temporário sempre

    return {"status": "ok", "chapters": len(chapters), "chunks": len(embedded_chunks)}


class AskRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ask")
async def ask(req: AskRequest):
    chunks = retrieve_hybrid_firestore(req.query, n_results=req.top_k)

    if not chunks:
        raise HTTPException(status_code=404, detail="Nenhum trecho relevante encontrado")

    reranked = rerank(req.query, chunks)
    response = answer(req.query, reranked)

    sources = [
        {"chapter_id": c["chapter_id"], "relevance_score": c.get("relevance_score", 0.0)}
        for c in reranked
    ]

    return {"answer": response, "sources": sources}
