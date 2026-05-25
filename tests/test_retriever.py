from src.retriever import retrieve
from src.chunker import chunk_text, chunk_chapters
from src.parser import parse_epub
from src.embedder import embed_chunks
from src.store import store_chunks, get_collection

def test_retriever():

    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")
    chunks = chunk_chapters(chapters) # pega só os 3 primeiros para teste
    embedded_chunks = embed_chunks(chunks[:10])

    # armazena no Chroma
    store_chunks(embedded_chunks)
    results = retrieve("quem é Kate Blackwell?", n_results=3)

    assert len(results) > 0
    assert "text" in results[0]
    assert "distance" in results[0]
    assert "chapter_id" in results[0]
    
    print(f"Chunks encontrados: {len(results)}")
    print(f"Distância do mais relevante: {results[0]['distance']:.4f}")
    print(f"Trecho: {results[0]['text'][:200]}")


