import sqlite3
import chromadb

_DB_PATH = "data/books.db"


def _parent_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id  TEXT PRIMARY KEY,
            book_id    TEXT NOT NULL,
            chapter_id TEXT NOT NULL,
            text       TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_parent_chunks(parents: list[dict]):
    with _parent_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO parent_chunks (parent_id, book_id, chapter_id, text) VALUES (?, ?, ?, ?)",
            [(p["id"], p["book_id"], p["chapter_id"], p["text"]) for p in parents],
        )


def get_parents_by_ids(parent_ids: list[str]) -> dict[str, str]:
    if not parent_ids:
        return {}
    with _parent_conn() as conn:
        placeholders = ",".join("?" * len(parent_ids))
        rows = conn.execute(
            f"SELECT parent_id, text FROM parent_chunks WHERE parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    return {row["parent_id"]: row["text"] for row in rows}


def delete_parents_for_book(book_id: str):
    with _parent_conn() as conn:
        conn.execute("DELETE FROM parent_chunks WHERE book_id = ?", (book_id,))


def get_collection(persist_dir: str = "data/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="book_chunks")


def store_chunks(chunks: list[dict], persist_dir: str = "data/chroma"):
    collection = get_collection(persist_dir)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{
            "chapter_id": chunk["chapter_id"],
            "chapter_title": chunk.get("chapter_title", chunk["chapter_id"]),
            "book_id": chunk["book_id"],
            **( {"parent_id": chunk["parent_id"]} if chunk.get("parent_id") else {} ),
        } for chunk in chunks],
    )


def delete_book(book_id: str, persist_dir: str = "data/chroma"):
    collection = get_collection(persist_dir)
    results = collection.get(where={"book_id": book_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
