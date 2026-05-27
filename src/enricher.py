import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from threading import Lock

from src.clients import PROVIDERS, get_client

BATCH_SIZE = 10

_PROMPT = (
    "Analise os trechos abaixo e para cada um escreva 1-2 frases situando o trecho "
    "no documento para melhorar busca semântica.\n\n"
    "{items}\n\n"
    "Responda EXATAMENTE neste formato, sem texto adicional:\n"
    "{fmt}"
)


def _batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _parse_contexts(text: str, n: int) -> list[str]:
    results = []
    for i in range(1, n + 1):
        m = re.search(rf"CONTEXTO_{i}:\s*(.+?)(?=CONTEXTO_\d+:|$)", text, re.DOTALL)
        results.append(m.group(1).strip() if m else "")
    return results


def _build_prompt(chunk_texts: list[str]) -> str:
    items = "\n\n".join(f"TRECHO_{i+1}:\n{t}" for i, t in enumerate(chunk_texts))
    fmt = "\n".join(f"CONTEXTO_{i+1}: <contexto aqui>" for i in range(len(chunk_texts)))
    return _PROMPT.format(items=items, fmt=fmt)


def _enrich_batch_anthropic(client, model: str, chapter_text: str, chunk_texts: list[str]) -> list[str]:
    response = client.messages.create(
        model=model,
        max_tokens=200 * len(chunk_texts),
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{chapter_text}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": _build_prompt(chunk_texts)},
            ],
        }],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    return _parse_contexts(response.content[0].text, len(chunk_texts))


def _enrich_batch_openai_compat(client, model: str, chapter_text: str, chunk_texts: list[str]) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        max_tokens=200 * len(chunk_texts),
        messages=[{
            "role": "user",
            "content": f"<document>\n{chapter_text}\n</document>\n\n{_build_prompt(chunk_texts)}",
        }],
    )
    return _parse_contexts(response.choices[0].message.content, len(chunk_texts))


def enrich_chunks(
    chunks: list[dict],
    chapters: list[dict],
    provider: str = "anthropic",
    threads: int = 15,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    if provider not in PROVIDERS:
        raise ValueError(f"Provider inválido: {provider}. Opções: {list(PROVIDERS.keys())}")

    config = PROVIDERS[provider]
    model = config.get("enrich_model")
    if not model:
        raise ValueError(f"Provider '{provider}' não suporta enriquecimento.")

    client = get_client(provider)
    if config["type"] == "anthropic":
        enrich_batch = lambda ch, ck: _enrich_batch_anthropic(client, model, ch, ck)
    else:
        enrich_batch = lambda ch, ck: _enrich_batch_openai_compat(client, model, ch, ck)

    chapter_map = {c["id"]: c["text"] for c in chapters}

    chapter_order: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        chapter_order[chunk["chapter_id"]].append(i)

    batches: list[tuple[str, list[int]]] = []
    for chapter_id, indices in chapter_order.items():
        chapter_text = chapter_map.get(chapter_id, "")
        for batch_indices in _batched(indices, batch_size):
            batches.append((chapter_text, batch_indices))

    results = list(chunks)
    counter = {"done": 0}
    lock = Lock()

    def process_batch(chapter_text: str, indices: list[int]):
        chunk_texts = [chunks[i]["text"] for i in indices]
        contexts = enrich_batch(chapter_text, chunk_texts)
        for idx, context in zip(indices, contexts):
            if context:
                results[idx] = {**chunks[idx], "text": f"{context}\n\n{chunks[idx]['text']}"}
        with lock:
            counter["done"] += len(indices)
            if counter["done"] % 50 < len(indices):
                print(f"  {counter['done']}/{len(chunks)} chunks enriquecidos")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_batch, ch_text, idxs) for ch_text, idxs in batches]
        for future in as_completed(futures):
            future.result()

    return results
