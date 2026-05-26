from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

COLLECTION = "book_chunks"


def get_db():
    return firestore.Client()


def store_chunks_firestore(chunks: list[dict]):
    db = get_db()
    col = db.collection(COLLECTION)
    batch = db.batch()

    for i, chunk in enumerate(chunks):
        ref = col.document(chunk["id"])
        batch.set(ref, {
            "text": chunk["text"],
            "chapter_id": chunk["chapter_id"],
            "book_id": chunk["book_id"],
            "embedding": Vector(chunk["embedding"]),
        })
        if (i + 1) % 500 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()


def get_all_chunks_firestore(book_id: str | None = None) -> list[dict]:
    db = get_db()
    query = db.collection(COLLECTION)
    if book_id:
        query = query.where("book_id", "==", book_id)
    return [
        {"text": d.get("text"), "chapter_id": d.get("chapter_id"), "book_id": d.get("book_id")}
        for d in query.stream()
    ]


def delete_book_firestore(book_id: str):
    db = get_db()
    batch = db.batch()
    count = 0
    for doc in db.collection(COLLECTION).where("book_id", "==", book_id).stream():
        batch.delete(doc.reference)
        count += 1
        if count % 500 == 0:
            batch.commit()
            batch = db.batch()
    if count % 500 != 0:
        batch.commit()


def delete_collection_firestore():
    db = get_db()
    for doc in db.collection(COLLECTION).stream():
        doc.reference.delete()
