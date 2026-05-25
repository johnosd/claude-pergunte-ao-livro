import voyageai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from src.store_firestore import get_db, get_all_chunks_firestore, COLLECTION
from src.retriever import rerank  # reutiliza o reranker existente

load_dotenv()

vo = voyageai.Client()

def retrieve_semantic_firestore(query: str, n_results: int = 10) -> list[dict]:
    # Converte a pergunta em embedding
    result = vo.embed([query], model="voyage-3.5", input_type="query")
    query_embedding = result.embeddings[0]

    # Busca os vetores mais próximos no Firestore (requer índice vetorial criado)
    db = get_db()
    col = db.collection(COLLECTION)
    docs = col.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=n_results
    ).get()

    return [
        {
            "text": doc.get("text"),
            "chapter_id": doc.get("chapter_id"),
            "distance": doc.get("distance", 0.0)
        }
        for doc in docs
    ]

def retrieve_lexical_firestore(query: str, n_results: int = 10) -> list[dict]:
    # Carrega todos os chunks e aplica BM25 em memória
    all_chunks = get_all_chunks_firestore()
    tokenized_docs = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [
        {
            "text": all_chunks[idx]["text"],
            "chapter_id": all_chunks[idx]["chapter_id"],
            "bm25_score": float(scores[idx])
        }
        for idx in top_indices
    ]

def retrieve_hybrid_firestore(query: str, n_results: int = 10) -> list[dict]:
    # Combina busca semântica e lexical, deduplica pelo texto
    semantic = retrieve_semantic_firestore(query, n_results=n_results)
    lexical = retrieve_lexical_firestore(query, n_results=n_results)

    seen = set()
    combined = []
    for chunk in semantic + lexical:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            combined.append(chunk)

    return combined
