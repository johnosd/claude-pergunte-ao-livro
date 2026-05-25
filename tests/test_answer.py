from src.retriever import retrieve, rerank
from src.chunker import chunk_text, chunk_chapters
from src.parser import parse_epub
from src.embedder import embed_chunks
from src.store import store_chunks, get_collection
from src.answer import answer
import time


def test_answer():
    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")
    chunks = chunk_chapters(chapters) # pega só os 3 primeiros para teste
    embedded_chunks = embed_chunks(chunks[:10])

    # armazena no Chroma
    store_chunks(embedded_chunks)
    time.sleep(20)  # aguarda rate limit da Voyage AI (plano gratuito: 3 RPM)
    results = retrieve("quem é Kate Blackwell?", n_results=3)
    reranked = rerank("quem é Kate Blackwell?", results)
    response = answer("quem é Kate Blackwell?", reranked)
    
    assert len(reranked) > 0

    assert "text" in reranked[0]
    assert "distance" in reranked[0]
    assert "chapter_id" in reranked[0]
    assert "relevance_score" in reranked[0]

    print(f"Relevance score: {reranked[0]['relevance_score']:.4f}")
    print(f"Trecho: {reranked[0]['text'][:200]}")
    print(f"Resposta: {response}")


