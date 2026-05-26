from src.clients import PROVIDERS, get_client, get_provider_for_model

def answer(query: str, chunks: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    context = "\n\n".join([
        f"[Trecho {i+1} - Capítulo {c['chapter_id']}]:\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""Você é um assistente que responde perguntas sobre livros de forma natural e fluida, como se estivesse contando a história para alguém.

Use APENAS as informações dos trechos abaixo para responder. Não invente detalhes que não estejam nos trechos.
Responda em prosa corrida, sem citar números de trechos ou capítulos. Se a informação não estiver nos trechos, diga de forma natural que não encontrou esse detalhe no livro.

TRECHOS DO LIVRO:
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
