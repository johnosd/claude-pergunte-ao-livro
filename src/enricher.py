from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from src.clients import PROVIDERS, get_client

PROMPT = (
    "Trecho:\n<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Escreva 1-2 frases situando este trecho no documento "
    "para melhorar busca semântica. Apenas o contexto."
)

def _get_context_anthropic(client, model: str, chapter_text: str, chunk_text: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=150,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{chapter_text}\n</document>",
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": PROMPT.format(chunk_text=chunk_text)
                }
            ]
        }],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )
    return response.content[0].text.strip()

def _get_context_openai_compat(client, model: str, chapter_text: str, chunk_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                f"<document>\n{chapter_text}\n</document>\n\n"
                + PROMPT.format(chunk_text=chunk_text)
            )
        }]
    )
    return response.choices[0].message.content.strip()

def enrich_chunks(chunks: list[dict], chapters: list[dict], provider: str = "anthropic", threads: int = 5) -> list[dict]:
    if provider not in PROVIDERS:
        raise ValueError(f"Provider inválido: {provider}. Opções: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    model = config.get("enrich_model")
    if not model:
        raise ValueError(f"Provider '{provider}' não suporta enriquecimento. Use: anthropic, openai, gemini, qwen")

    client = get_client(provider)

    if config["type"] == "anthropic":
        get_context = lambda ch, ck: _get_context_anthropic(client, model, ch, ck)
    else:
        get_context = lambda ch, ck: _get_context_openai_compat(client, model, ch, ck)

    chapter_map = {c["id"]: c["text"] for c in chapters}
    counter = {"done": 0}
    lock = Lock()

    def process_chunk(chunk):
        chapter_text = chapter_map.get(chunk["chapter_id"], "")
        context = get_context(chapter_text, chunk["text"])
        chunk["text"] = f"{context}\n\n{chunk['text']}"
        with lock:
            counter["done"] += 1
            if counter["done"] % 50 == 0:
                print(f"  {counter['done']}/{len(chunks)} chunks enriquecidos")
        return chunk

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        results = [None] * len(chunks)
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return results
