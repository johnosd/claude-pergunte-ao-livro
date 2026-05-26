from src.clients import PROVIDERS, get_client, get_provider_for_model

def answer(query: str, chunks: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    context = "\n\n".join([
        f"[Trecho {i+1} - Capítulo {c['chapter_id']}]:\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""Responda a pergunta abaixo usando APENAS os trechos do livro fornecidos.
Cite de qual trecho veio cada informação.
Se a resposta não estiver nos trechos, diga que não encontrou no livro.

TRECHOS:
{context}

PERGUNTA: {query}"""

    provider = get_provider_for_model(model)
    client = get_client(provider)

    if PROVIDERS[provider]["type"] == "anthropic":
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
