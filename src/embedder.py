import time
from src.clients import voyage_client

def embed_chunks(chunks: list[dict], batch_size: int = 128) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = voyage_client.embed(batch, model="voyage-3.5", input_type="document")

        for j, embedding in enumerate(result.embeddings):
            chunks[i + j]["embedding"] = embedding

        if i + batch_size < len(texts):
            print(f"  Batch {i // batch_size + 1} concluído. Aguardando 2s...")
            time.sleep(2)

    return chunks
