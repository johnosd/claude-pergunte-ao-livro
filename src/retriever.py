import voyageai
from dotenv import load_dotenv
from src.store import store_chunks, get_collection

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
    
    
    