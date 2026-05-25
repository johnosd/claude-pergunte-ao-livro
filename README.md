# Pergunte ao Livro

Faça perguntas para qualquer livro EPUB sem carregar o livro inteiro no contexto.

Sistema RAG (Retrieval-Augmented Generation) com enriquecimento contextual multi-provider, busca híbrida (semântica + lexical) e reranking. Responde com citações dos trechos do livro via Claude.

---

## Arquitetura

O sistema tem dois fluxos independentes: **ingestão** (roda uma vez) e **pergunta** (roda sempre).

### Fluxo de Ingestão

```
livro.epub
    │
    ▼
[ parser.py ]
Extrai texto limpo de cada capítulo
(filtra CSS, imagens, páginas em branco)
    │
    ▼
[ chunker.py ]
Divide cada capítulo em pedaços de 500 chars
com overlap de 50 chars entre eles
Ex: 8 capítulos → 1746 chunks
    │
    ▼
[ enricher.py ]  ← opcional (flag --enrich)
Para cada chunk, envia o capítulo inteiro + chunk
para um LLM e gera 1-2 frases de contexto.
O contexto é prefixado ao chunk antes do embedding.

Suporta múltiplos providers (--provider=...):
  anthropic → Claude Haiku + prompt caching (-90% custo)
  openai    → GPT-4o mini
  gemini    → Gemini 2.0 Flash Lite
  deepseek  → DeepSeek V3
  qwen      → Qwen 2.5 7B

Processamento paralelo (--threads=5) reduz
o tempo de ~30min para ~6min.

  Antes: "ele não pôde se perdoar"
  Depois: "Este trecho descreve Jamie McGregor no
  momento em que descobre a traição de seu sócio
  em Klipdrift, África do Sul.

  ele não pôde se perdoar"
    │
    ▼
[ embedder.py ]
Envia os chunks em batches para Voyage AI
Cada chunk vira um vetor de números (embedding)
que representa seu significado semântico
    │
    ▼
[ store.py / Chroma ]  ou  [ store_firestore.py ]
Salva em disco: id + embedding + texto + capítulo
Persiste entre sessões — não precisa reprocessar
```

### Fluxo de Pergunta (Hybrid Search)

```
"Quem é Kate Blackwell?"
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
[ Busca Semântica ]           [ Busca Lexical ]
Converte a pergunta           Calcula score BM25
em embedding e busca          por palavras-chave
os vetores mais próximos      nos chunks do Chroma
no Chroma / Firestore
    │                                 │
    └──────────────┬──────────────────┘
                   │
                   ▼
         [ Combina + Deduplica ]
         Une os resultados das duas buscas
         Remove duplicatas (~15-20 chunks únicos)
                   │
                   ▼
            [ reranker ]
            Voyage AI rerank-2 lê a pergunta
            + cada chunk juntos e calcula
            relevância real → top-3
                   │
                   ▼
             [ answer.py ]
             Monta prompt com os 3 trechos
             e envia para Claude Sonnet
                   │
                   ▼
        Resposta com citações dos trechos
```

### Por que Hybrid Search?

| Situação | Semântica | Lexical | Hybrid |
|---|---|---|---|
| "quem fundou a empresa" (sem essa palavra no livro) | Encontra | Não encontra | Encontra |
| "Kruger-Brent" (nome próprio exato) | Pode perder | Encontra | Encontra |
| Melhor dos dois | - | - | Sim |

### Por que Contextual Enrichment?

Baseado na [pesquisa da Anthropic sobre Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval):

| Técnica | Pass@10 | Redução de falhas |
|---|---|---|
| RAG baseline | 87% | — |
| + Contextual Enrichment | 92% | -40% |
| + Hybrid Search | 93% | -49% |
| + Reranking | 95% | -63% |

---

## Stack

| Componente | Tecnologia |
|---|---|
| EPUB parsing | `ebooklib` + `beautifulsoup4` |
| Enriquecimento contextual | `anthropic` / `openai` (multi-provider) |
| Embeddings | Voyage AI `voyage-3.5` |
| Busca lexical | `rank_bm25` |
| Reranker | Voyage AI `rerank-2` |
| Vector Database | `chromadb` (local) ou Google Firestore |
| LLM | Anthropic `claude-sonnet-4-20250514` |
| API REST | `fastapi` + `uvicorn` |
| CLI | `click` |

---

## Estrutura do Projeto

```
pergunte-ao-livro/
├── .env.example
├── requirements.txt
├── cli.py                        # CLI: ingest e ask
├── api.py                        # API REST: POST /ingest e POST /ask
├── data/
│   ├── books/                    # coloque seus EPUBs aqui
│   └── chroma/                   # banco vetorial local (gerado automaticamente)
├── src/
│   ├── parser.py                 # EPUB → texto limpo por capítulo
│   ├── chunker.py                # texto → chunks com overlap e metadata
│   ├── enricher.py               # chunks → chunks enriquecidos (multi-provider, paralelo)
│   ├── embedder.py               # chunks → embeddings via Voyage AI (batch)
│   ├── store.py                  # Chroma: salva e consulta chunks
│   ├── store_firestore.py        # Firestore: salva e consulta chunks
│   ├── retriever.py              # hybrid search + rerank (Chroma)
│   ├── retriever_firestore.py    # hybrid search + rerank (Firestore)
│   └── answer.py                 # prompt + Claude → resposta com citações
├── scripts/
│   └── ingest.py                 # pipeline completo de ingestão
└── tests/
    ├── test_parser.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_store.py
    └── test_retriever.py
```

---

## Instalação

```bash
git clone https://github.com/johnosd/pergunte-ao-livro.git
cd pergunte-ao-livro
pip install -r requirements.txt
cp .env.example .env
```

Edite o `.env`:

```env
ANTHROPIC_API_KEY=sua_chave_aqui
VOYAGE_API_KEY=sua_chave_aqui

# Providers opcionais para enriquecimento contextual
OPENAI_API_KEY=sua_chave_aqui
GEMINI_API_KEY=sua_chave_aqui
DEEPSEEK_API_KEY=sua_chave_aqui
QWEN_API_KEY=sua_chave_aqui

# Só para Firestore
GOOGLE_APPLICATION_CREDENTIALS=caminho/para/service-account.json
```

---

## Como usar

### CLI

**Ingerir o livro:**

```bash
# Sem enriquecimento (mais rápido)
python -m scripts.ingest "data/books/seu-livro.epub" --reset

# Com enriquecimento — Anthropic Haiku (padrão, prompt caching)
python -m scripts.ingest "data/books/seu-livro.epub" --reset --enrich

# Com enriquecimento — DeepSeek (mais barato)
python -m scripts.ingest "data/books/seu-livro.epub" --reset --enrich --provider=deepseek

# Com enriquecimento — Gemini (mais barato ainda)
python -m scripts.ingest "data/books/seu-livro.epub" --reset --enrich --provider=gemini

# Ajustar threads (padrão: 5)
python -m scripts.ingest "data/books/seu-livro.epub" --reset --enrich --threads=3
```

**Comparativo de providers para enriquecimento (1746 chunks):**

| Provider | Custo estimado | Tempo (~5 threads) |
|---|---|---|
| `anthropic` | ~$0.05 (com cache) | ~6 min |
| `gemini` | ~$0.01 | ~4 min |
| `deepseek` | ~$0.02 | ~5 min |
| `qwen` | ~$0.005 | ~4 min |
| `openai` | ~$0.04 | ~5 min |

**Fazer perguntas:**

```bash
python cli.py ask "Quem é Kate Blackwell?"
python cli.py ask "Qual é o tema central do livro?"
python cli.py ask "O que acontece no início da história?"
```

### API REST

**Subir o servidor:**

```bash
uvicorn api:app --reload
```

**Ingerir via API:**

```bash
# Sem enriquecimento
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/books/seu-livro.epub"

# Com enriquecimento (provider padrão: anthropic)
curl -X POST "http://localhost:8000/ingest?enrich=true" \
  -F "file=@data/books/seu-livro.epub"
```

**Fazer perguntas via API:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Quem é Kate Blackwell?", "top_k": 5}'
```

Resposta:
```json
{
  "answer": "Kate Blackwell é uma filantropa...",
  "sources": [
    {"chapter_id": "id18", "relevance_score": 0.91},
    {"chapter_id": "id22", "relevance_score": 0.78}
  ]
}
```

---

## Roadmap

- [x] Parser EPUB
- [x] Chunking com overlap e metadata
- [x] Enriquecimento contextual multi-provider (Anthropic, OpenAI, Gemini, DeepSeek, Qwen)
- [x] Processamento paralelo do enriquecimento (ThreadPoolExecutor)
- [x] Prompt caching no enriquecimento Anthropic (-90% custo)
- [x] Embeddings em batch com Voyage AI
- [x] Busca semântica (vetorial)
- [x] Busca lexical (BM25)
- [x] Hybrid Search (semântica + lexical)
- [x] Reranking com Voyage AI
- [x] Resposta via Claude com citações
- [x] CLI com Click
- [x] API REST com FastAPI
- [x] Pipeline de ingestão completo
- [ ] Migração do vector DB para Google Firestore

---

## Licença

MIT
