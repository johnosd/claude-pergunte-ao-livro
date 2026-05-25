from src.chunker import chunk_text, chunk_chapters
from src.parser import parse_epub
from src.embedder import embed_chunks
from src.store import store_chunks, get_collection




def test_store_chunks():
    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")
    chunks = chunk_chapters(chapters)
    sample_chunks = chunks[:3]  # pega só os 3 primeiros para teste
    embedded_chunks = embed_chunks(sample_chunks)

    # armazena no Chroma
    store_chunks(embedded_chunks)

    # Verifica se foram salvos
    collection = get_collection()
    assert collection.count() == 3
    print(f"Total de chunks armazenados: {collection.count()}")
