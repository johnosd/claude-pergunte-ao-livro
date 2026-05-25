import chromadb

def get_collection(persist_dir: str = "data/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="book_chunks")

def store_chunks(chunks: list[dict], persist_dir: str = "data/chroma"):
    collection = get_collection(persist_dir)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"chapter_id": chunk["chapter_id"]} for chunk in chunks]
    )
