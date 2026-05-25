from src.embedder import embed_chunks
from src.chunker import chunk_chapters
from src.parser import parse_epub

def test_embed_chunks():
    """ testa apenas 3 chunks para não gastar muito da cota da API"""
    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")
    chunks = chunk_chapters(chapters, chunk_size=500, overlap=50)
    sample_chunks = chunks[:3]  # pega só os 3 primeiros para teste
    embedded_chunks = embed_chunks(sample_chunks)
    assert len(embedded_chunks) == 3
    assert "embedding" in embedded_chunks[0]
    assert isinstance(embedded_chunks[0]["embedding"], list)
    print(f"Embedding do primeiro chunk: {embedded_chunks[0]['embedding'][:5]}...")
    print(f"Total de chunks: {len(embedded_chunks)}")
    print(f"ID do primeiro chunk: {embedded_chunks[0]['id']}")
