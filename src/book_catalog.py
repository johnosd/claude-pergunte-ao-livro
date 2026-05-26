import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/books.db"
BOOKS_COLLECTION = "books"


# ── SQLite (caminho local / Chroma) ──────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id              TEXT PRIMARY KEY,
            title                TEXT,
            author               TEXT,
            isbn                 TEXT,
            ingested_at          TEXT,
            chunk_count          INTEGER,
            enriched             INTEGER,
            metadata_epub        TEXT,
            metadata_google      TEXT,
            metadata_openlibrary TEXT
        )
    """)
    for col in ("metadata_epub", "metadata_google", "metadata_openlibrary"):
        try:
            conn.execute(f"ALTER TABLE books ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    return conn


def book_exists(book_id: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM books WHERE book_id = ?", (book_id,)).fetchone()
        return row is not None


def register_book(book_meta: dict, chunk_count: int, enriched: bool, all_metadata: dict | None = None):
    all_metadata = all_metadata or {}
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO books
               (book_id, title, author, isbn, ingested_at, chunk_count, enriched,
                metadata_epub, metadata_google, metadata_openlibrary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                book_meta["id"],
                book_meta.get("title"),
                book_meta.get("author"),
                book_meta.get("isbn"),
                datetime.now(timezone.utc).isoformat(),
                chunk_count,
                int(enriched),
                json.dumps(all_metadata.get("epub", {}),        ensure_ascii=False),
                json.dumps(all_metadata.get("google", {}),      ensure_ascii=False),
                json.dumps(all_metadata.get("openlibrary", {}), ensure_ascii=False),
            ),
        )


def update_book_metadata(book_id: str, all_metadata: dict):
    with _get_conn() as conn:
        conn.execute(
            """UPDATE books
               SET metadata_epub = ?, metadata_google = ?, metadata_openlibrary = ?
               WHERE book_id = ?""",
            (
                json.dumps(all_metadata.get("epub", {}),        ensure_ascii=False),
                json.dumps(all_metadata.get("google", {}),      ensure_ascii=False),
                json.dumps(all_metadata.get("openlibrary", {}), ensure_ascii=False),
                book_id,
            ),
        )


def delete_book_catalog(book_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM books WHERE book_id = ?", (book_id,))


def list_books() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM books ORDER BY ingested_at DESC").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for col in ("metadata_epub", "metadata_google", "metadata_openlibrary"):
                if d.get(col):
                    try:
                        d[col] = json.loads(d[col])
                    except (json.JSONDecodeError, TypeError):
                        d[col] = {}
            result.append(d)
        return result


# ── Firestore (caminho cloud / API) ──────────────────────────────────────────

def book_exists_firestore(book_id: str) -> bool:
    from src.store_firestore import get_db
    doc = get_db().collection(BOOKS_COLLECTION).document(book_id).get()
    return doc.exists


def register_book_firestore(book_meta: dict, chunk_count: int, enriched: bool, all_metadata: dict | None = None):
    from src.store_firestore import get_db
    all_metadata = all_metadata or {}
    get_db().collection(BOOKS_COLLECTION).document(book_meta["id"]).set({
        "title": book_meta.get("title"),
        "author": book_meta.get("author"),
        "isbn": book_meta.get("isbn"),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "enriched": enriched,
        "metadata_epub": all_metadata.get("epub", {}),
        "metadata_google": all_metadata.get("google", {}),
        "metadata_openlibrary": all_metadata.get("openlibrary", {}),
    })


def delete_book_catalog_firestore(book_id: str):
    from src.store_firestore import get_db
    get_db().collection(BOOKS_COLLECTION).document(book_id).delete()


def list_books_firestore() -> list[dict]:
    from src.store_firestore import get_db
    return [
        {"book_id": doc.id, **doc.to_dict()}
        for doc in get_db().collection(BOOKS_COLLECTION).stream()
    ]
