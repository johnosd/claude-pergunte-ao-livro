from rank_bm25 import BM25Okapi
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from src.store_firestore import get_db, get_all_chunks_firestore, COLLECTION
from src.clients import voyage_client
from src.retriever import rerank


def retrieve_semantic_firestore(query: str, n_results: int = 10, book_id: str | None = None) -> list[dict]:
    result = voyage_client.embed([query], model="voyage-3.5", input_type="query")
    query_embedding = result.embeddings[0]

    db = get_db()
    col = db.collection(COLLECTION)
    # find_nearest não suporta where — buscamos mais resultados e filtramos
    fetch = n_results if not book_id else n_results * 4
    docs = col.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=fetch,
    ).get()

    chunks = []
    for doc in docs:
        if book_id and doc.get("book_id") != book_id:
            continue
        chunks.append({
            "text": doc.get("text"),
            "chapter_id": doc.get("chapter_id"),
            "book_id": doc.get("book_id"),
            "distance": doc.get("distance", 0.0),
        })
        if len(chunks) == n_results:
            break

    return chunks


def retrieve_lexical_firestore(query: str, n_results: int = 10, book_id: str | None = None) -> list[dict]:
    all_chunks = get_all_chunks_firestore(book_id=book_id)
    tokenized_docs = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [
        {
            "text": all_chunks[idx]["text"],
            "chapter_id": all_chunks[idx]["chapter_id"],
            "book_id": all_chunks[idx].get("book_id"),
            "bm25_score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def retrieve_hybrid_firestore(query: str, n_results: int = 10, book_id: str | None = None) -> list[dict]:
    semantic = retrieve_semantic_firestore(query, n_results=n_results, book_id=book_id)
    lexical = retrieve_lexical_firestore(query, n_results=n_results, book_id=book_id)

    seen = set()
    combined = []
    for chunk in semantic + lexical:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            combined.append(chunk)
    return combined
