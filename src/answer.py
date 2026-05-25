import anthropic
from dotenv import load_dotenv

load_dotenv()

# Inicializa o cliente Anthropic
client = anthropic.Anthropic()

def answer(query: str, chunks: list[dict]) -> str:
    # Monta o contexto com os chunks rerankeados
    context = "\n\n".join([
        f"[Trecho {i+1} - Capítulo {c['chapter_id']}]:\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    # Prompt instrui o Claude a usar apenas o contexto fornecido
    prompt = f"""Responda a pergunta abaixo usando APENAS os trechos do livro fornecidos.
Cite de qual trecho veio cada informação.
Se a resposta não estiver nos trechos, diga que não encontrou no livro.

TRECHOS:
{context}

PERGUNTA: {query}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text
