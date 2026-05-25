import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

load_dotenv()

PROVIDERS = {
    "anthropic": {
        "type": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "type": "openai_compat",
        "model": "gpt-4o-mini",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "type": "openai_compat",
        "model": "gemini-2.0-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "deepseek": {
        "type": "openai_compat",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "type": "openai_compat",
        "model": "qwen2.5-7b-instruct",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "QWEN_API_KEY",
    },
}

PROMPT = (
    "Trecho:\n<chunk>\n{chunk_text}\n</chunk>\n\n"
    "Escreva 1-2 frases situando este trecho no documento "
    "para melhorar busca semântica. Apenas o contexto."
)

def _get_context_anthropic(client, chapter_text: str, chunk_text: str) -> str:
    response = client.messages.create(
        model=PROVIDERS["anthropic"]["model"],
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

    if config["type"] == "anthropic":
        client = anthropic.Anthropic()
        get_context = lambda chapter_text, chunk_text: _get_context_anthropic(client, chapter_text, chunk_text)
    else:
        api_key = os.getenv(config["key_env"])
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
        get_context = lambda chapter_text, chunk_text: _get_context_openai_compat(client, config["model"], chapter_text, chunk_text)

    chapter_map = {c["id"]: c["text"] for c in chapters}

    # Contador thread-safe para o progresso
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

    # Processa em paralelo — threads=5 é o recomendado pela Anthropic
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

        # Reconstrói a lista na ordem original
        results = [None] * len(chunks)
        for future in as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()

    return results
