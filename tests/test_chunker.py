from src.chunker import chunk_text, chunk_chapters
from src.parser import parse_epub

def test_chunk_text():
    text = "palavra " * 200  # 1600 caracteres
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    assert len(chunks) > 1
    assert len(chunks[0]) == 500
    assert chunks[1][:50] == chunks[0][-50:]  # verifica overlap
    print(f"Total de chunks: {len(chunks)}")

def test_chunk_chapters():
    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")

    chunks = chunk_chapters(chapters, chunk_size=500, overlap=50)

    assert len(chunks) > 0
    assert "id" in chunks[0]
    assert "text" in chunks[0]
    assert "chapter_id" in chunks[0]
    print(f"Total de chunks: {len(chunks)}")
    print(f"Primeiro ID: {chunks[0]['id']}")
    print(f"Início do texto: {chunks[0]['text'][:200]}")
    print(f"ID do capítulo: {chunks[0]['chapter_id']}")