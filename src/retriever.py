import voyageai
from dotenv import load_dotenv
from src.store import store_chunks, get_collection
from rank_bm25 import BM25Okapi

# Inicializa o cliente Voyage AI (lê VOYAGE_API_KEY do .env)    

load_dotenv()

# Inicializa o cliente Voyage AI
vo = voyageai.Client()

def retrieve(query:str, n_results: int = 5) -> list[dict]:
    # Converte a pergunta em vertor (mesmo modelo usado nos chunks)
    result = vo.embed([query], model="voyage-3.5", input_type="query")
    query_embedding = result.embeddings[0]

    # Busca os chunks mais proximos no Chroma
    collection = get_collection()
    results = collection.query(
        query_embeddings=query_embedding, 
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
        )
    
    # Monta lista de resultados com texto e distancia
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "chapter_id": results["metadatas"][0][i]["chapter_id"],
            "distance": results["distances"][0][i]
        })
        
    return chunks
    
def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    # Extrai so os textos para o reranker
    documents = [chunk["text"] for chunk in chunks]

    # Reranker le a pergunta + cada chunk e calcula relevancia real
    result = vo.rerank(query, documents, model="rerank-2", top_k=top_k)

    # monta resultado reordenado com score de relevancia
    reranked = []
    for item in result.results:
        chunk = chunks[item.index]
        chunk["relevance_score"] = item.relevance_score
        reranked.append(chunk)
        
    return reranked

def retrieve_lexical(query: str, n_results: int = 10) -> list[dict]:
    # Carrega todos os chunks do Chroma para construir o índice BM25
    collection = get_collection()
    all_docs = collection.get(include=["documents", "metadatas"])

    # Tokeniza os documentos (BM25 trabalha com palavras, não vetores)
    tokenized_docs = [doc.lower().split() for doc in all_docs["documents"]]
    bm25 = BM25Okapi(tokenized_docs)

    # Calcula score de cada chunk para a pergunta
    scores = bm25.get_scores(query.lower().split())

    # Pega os índices dos top-n mais relevantes
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [{
        "text": all_docs["documents"][idx],
        "chapter_id": all_docs["metadatas"][idx]["chapter_id"],
        "bm25_score": float(scores[idx])
    } for idx in top_indices]

def retrieve_hybrid(query: str, n_results: int = 10) -> list[dict]:
    # Busca semântica + lexical
    semantic = retrieve(query, n_results=n_results)
    lexical = retrieve_lexical(query, n_results=n_results)

    # Combina e deduplica pelo texto
    seen = set()
    combined = []
    for chunk in semantic + lexical:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            combined.append(chunk)

    return combined
