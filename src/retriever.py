from src.store import get_collection
from src.clients import voyage_client
from rank_bm25 import BM25Okapi


def retrieve(query: str, n_results: int = 5, book_id: str | None = None) -> list[dict]:
    result = voyage_client.embed([query], model="voyage-3.5", input_type="query")
    query_embedding = result.embeddings[0]

    collection = get_collection()
    kwargs = dict(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if book_id:
        kwargs["where"] = {"book_id": book_id}

    results = collection.query(**kwargs)

    return [
        {
            "text": doc,
            "chapter_id": results["metadatas"][0][i]["chapter_id"],
            "book_id": results["metadatas"][0][i].get("book_id"),
            "distance": results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]


def retrieve_lexical(query: str, n_results: int = 10, book_id: str | None = None) -> list[dict]:
    collection = get_collection()
    kwargs = dict(include=["documents", "metadatas"])
    if book_id:
        kwargs["where"] = {"book_id": book_id}

    all_docs = collection.get(**kwargs)
    tokenized_docs = [doc.lower().split() for doc in all_docs["documents"]]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [
        {
            "text": all_docs["documents"][idx],
            "chapter_id": all_docs["metadatas"][idx]["chapter_id"],
            "book_id": all_docs["metadatas"][idx].get("book_id"),
            "bm25_score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def retrieve_hybrid(query: str, n_results: int = 10, book_id: str | None = None) -> list[dict]:
    semantic = retrieve(query, n_results=n_results, book_id=book_id)
    lexical = retrieve_lexical(query, n_results=n_results, book_id=book_id)

    seen = set()
    combined = []
    for chunk in semantic + lexical:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            combined.append(chunk)
    return combined


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    documents = [chunk["text"] for chunk in chunks]
    result = voyage_client.rerank(query, documents, model="rerank-2", top_k=top_k)
    reranked = []
    for item in result.results:
        chunk = chunks[item.index]
        chunk["relevance_score"] = item.relevance_score
        reranked.append(chunk)
    return reranked
