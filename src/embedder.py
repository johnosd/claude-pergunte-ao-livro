import voyageai
from dotenv import load_dotenv

load_dotenv()

# Inicializa o cliente Voyage AI (lê VOYAGE_API_KEY do .env)
vo = voyageai.Client()

def embed_chunks(chunks: list[dict], batch_size: int = 128) -> list[dict]:
    # Extrai só os textos — a API não recebe o dict inteiro
    texts = [chunk["text"] for chunk in chunks]

    # Processa em grupos de 128 para respeitar o limite da API
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # fatia do índice i até i+128

        # Envia o batch para a Voyage AI e recebe os vetores
        result = vo.embed(batch, model="voyage-3.5", input_type="document")

        # Injeta o embedding de volta no chunk original (pelo índice)
        for j, embedding in enumerate(result.embeddings):
            chunks[i + j]["embedding"] = embedding

    return chunks
