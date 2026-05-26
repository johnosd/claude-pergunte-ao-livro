import sys
from dotenv import load_dotenv
from src.parser import parse_epub
from src.metadata_fetcher import fetch_all_metadata
from src.chunker import chunk_chapters_hierarchical
from src.enricher import enrich_chunks
from src.embedder import embed_chunks
from src.store import store_chunks, store_parent_chunks, delete_book, delete_parents_for_book
from src.book_catalog import book_exists, register_book, delete_book_catalog

load_dotenv()


def remove_book(book_id: str):
    delete_book(book_id)
    delete_parents_for_book(book_id)
    delete_book_catalog(book_id)
    print(f"Livro '{book_id}' removido do Chroma e do catálogo.")


def ingest_book(file_path: str, reset: bool = False, enrich: bool = True, provider: str = "anthropic", threads: int = 10):
    print(f"Lendo metadados do EPUB: {file_path}")
    book_meta, chapters = parse_epub(file_path)
    book_id = book_meta["id"]
    print(f"  book_id : {book_id}")
    print(f"  título  : {book_meta.get('title')}")
    print(f"  autor   : {book_meta.get('author')}")

    print("Buscando metadados externos...")
    all_metadata = fetch_all_metadata(book_meta)
    from src.metadata_fetcher import format_metadata_summary
    print(format_metadata_summary(all_metadata))

    if reset:
        print("Removendo versão anterior do livro...")
        remove_book(book_id)
    elif book_exists(book_id):
        print(f"  Livro já ingerido. Use --reset para reprocessar. (book_id: {book_id})")
        return

    print(f"  {len(chapters)} capítulos encontrados")

    print("Dividindo em chunks (parent-child)...")
    parents, children = chunk_chapters_hierarchical(chapters, book_id=book_id)
    print(f"  {len(parents)} blocos pai, {len(children)} chunks filho gerados")

    print("Armazenando blocos pai no SQLite...")
    store_parent_chunks(parents)

    if enrich:
        print(f"Enriquecendo chunks filho (provider: {provider}, threads: {threads})...")
        children = enrich_chunks(children, chapters, provider=provider, threads=threads)
        print(f"  {len(children)} chunks enriquecidos")

    print("Gerando embeddings (pode demorar)...")
    embedded_chunks = embed_chunks(children)
    print(f"  {len(embedded_chunks)} embeddings gerados")

    print("Armazenando no Chroma...")
    store_chunks(embedded_chunks)

    register_book(book_meta, chunk_count=len(children), enriched=enrich, all_metadata=all_metadata)
    print(f"Concluído! Livro '{book_meta.get('title')}' disponível com book_id: {book_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.ingest <epub> [--reset] [--no-enrich] [--provider=anthropic|deepseek|...] [--threads=5]")
        sys.exit(1)
    reset = "--reset" in sys.argv
    enrich = "--no-enrich" not in sys.argv
    provider = next((a.split("=")[1] for a in sys.argv if a.startswith("--provider=")), "anthropic")
    threads = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--threads=")), 5))
    ingest_book(sys.argv[1], reset=reset, enrich=enrich, provider=provider, threads=threads)
