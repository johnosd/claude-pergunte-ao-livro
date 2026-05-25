import sys
import chromadb
from dotenv import load_dotenv
from src.parser import parse_epub
from src.chunker import chunk_chapters
from src.embedder import embed_chunks
from src.store import store_chunks

load_dotenv()

def reset_chroma(persist_dir: str = "data/chroma"):
    # Apaga a coleção se existir, para evitar IDs duplicados
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("book_chunks")
    except Exception:
        pass
    print("  Chroma resetado.")

def ingest_book(file_path: str, reset: bool = False):
    if reset:
        print("Resetando Chroma...")
        reset_chroma()

    print(f"Parseando EPUB: {file_path}")
    chapters = parse_epub(file_path)
    print(f"  {len(chapters)} capítulos encontrados")

    print("Dividindo em chunks...")
    chunks = chunk_chapters(chapters)
    print(f"  {len(chunks)} chunks gerados")

    print("Gerando embeddings (pode demorar)...")
    embedded_chunks = embed_chunks(chunks)
    print(f"  {len(embedded_chunks)} embeddings gerados")

    print("Armazenando no Chroma...")
    store_chunks(embedded_chunks)
    print("Concluído!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/ingest.py <caminho-do-epub> [--reset]")
        sys.exit(1)
    reset = "--reset" in sys.argv
    ingest_book(sys.argv[1], reset=reset)


