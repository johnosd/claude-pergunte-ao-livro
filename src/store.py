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
        metadatas=[{"chapter_id": chunk["chapter_id"], "book_id": chunk["book_id"]} for chunk in chunks],
    )


def delete_book(book_id: str, persist_dir: str = "data/chroma"):
    collection = get_collection(persist_dir)
    results = collection.get(where={"book_id": book_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
