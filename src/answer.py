from src.clients import PROVIDERS, get_client, get_provider_for_model

def answer(query: str, chunks: list[dict], model: str = "claude-sonnet-4-20250514") -> str:
    context = "\n\n".join([
        f"[{c.get('book_id', 'livro')} > {c.get('chapter_title', c['chapter_id'])}]:\n{c['text']}"
        for c in chunks
    ])

    prompt = f"""
    Você é um assistente que responde perguntas sobre livros de forma conversasional, natural e fluida, como se estivesse contando a história para alguém.

    Use APENAS as informações dos trechos abaixo para responder. Não invente detalhes que não estejam nos trechos.
    Ao citar uma informação relevante, indique a fonte no formato [livro > capítulo].
    Se a evidência nos trechos for insuficiente para responder, diga exatamente o que falta em vez de inventar.
    Não misture informações de livros diferentes sem indicar a fonte de cada ponto.

    TRECHOS DO LIVRO:
    {context}

    PERGUNTA: {query}

    """

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
