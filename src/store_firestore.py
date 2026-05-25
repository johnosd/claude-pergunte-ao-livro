from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

COLLECTION = "book_chunks"

def get_db():
    # Usa GOOGLE_APPLICATION_CREDENTIALS do ambiente
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
            "embedding": Vector(chunk["embedding"])
        })
        # Firestore tem limite de 500 operações por batch
        if (i + 1) % 500 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()

def get_all_chunks_firestore() -> list[dict]:
    # Carrega todos os docs para busca lexical (BM25)
    db = get_db()
    return [
        {"text": d.get("text"), "chapter_id": d.get("chapter_id")}
        for d in db.collection(COLLECTION).stream()
    ]

def delete_collection_firestore():
    # Apaga todos os documentos da coleção (reset antes de reingerir)
    db = get_db()
    for doc in db.collection(COLLECTION).stream():
        doc.reference.delete()
