# Pergunte ao Livro

Faça perguntas para qualquer livro EPUB sem carregar o livro inteiro no contexto.

Sistema RAG (Retrieval-Augmented Generation) com suporte a múltiplos livros, enriquecimento contextual multi-provider, busca híbrida (semântica + lexical) e reranking. Responde com citações dos trechos do livro via LLM de sua escolha.

---

## Arquitetura

O sistema tem dois fluxos independentes: **ingestão** (roda uma vez por livro) e **pergunta** (roda sempre).

### Fluxo de Ingestão

```
livro.epub
    │
    ▼
[ parser.py ]
Extrai metadados do EPUB (título, autor, ISBN)
Gera book_id estável: ISBN → slug(title-author)
Extrai texto limpo de cada capítulo
Detecta título do capítulo (primeiro h1/h2/h3/h4)
(filtra CSS, imagens, páginas em branco)
    │
    ▼
[ metadata_fetcher.py ]
Enriquece os metadados do livro consultando
duas APIs externas em paralelo:
  • Google Books API (requer BOOKS_API_KEY)
  • Open Library API  (gratuita, sem chave)
Busca por ISBN quando disponível, ou por
título + autor como fallback.
As três fontes (epub, google, openlibrary)
são armazenadas separadamente — nada é perdido.
Erros de rede são silenciados: o pipeline
não falha se as APIs estiverem indisponíveis.
    │
    ▼
[ book_catalog.py ]
Verifica se book_id já existe no catálogo
→ Se sim: avisa e interrompe (sem reprocessar)
→ Se não: continua o pipeline
Catálogo local: SQLite (data/books.db)
Catálogo cloud: Firestore collection "books"
    │
    ▼
[ chunker.py ]
Divide cada capítulo em pedaços de 500 chars
com overlap de 50 chars entre eles
ID do chunk: {book_id}__{chapter_id}__chunk_{n}
Propaga chapter_title para cada chunk
Ex: 8 capítulos → 1746 chunks
    │
    ▼
[ enricher.py ]  ← ativo por padrão (desativar com --no-enrich)
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
Salva: id + embedding + texto + chapter_id + chapter_title + book_id
Persiste entre sessões — múltiplos livros coexistem
    │
    ▼
[ book_catalog.py ]
Registra o livro no catálogo com metadados:
título, autor, ISBN, chunk_count, enriched, ingested_at
metadata_epub / metadata_google / metadata_openlibrary
(armazenados como JSON, deserializados na leitura)
```

### Fluxo de Pergunta (Hybrid Search)

```
"Quem é Kate Blackwell?"  [--book master-of-the-game]
    │
    │  (filtra pelo book_id se --book fornecido)
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
[ Busca Semântica ]           [ Busca Lexical ]
Converte a pergunta           Calcula score BM25
em embedding e busca          por palavras-chave
os 50 vetores mais próximos   nos chunks do Chroma
no Chroma / Firestore         (top-50)
    │                                 │
    └──────────────┬──────────────────┘
                   │
                   ▼
         [ RRF — Reciprocal Rank Fusion ]
         Combina os dois rankings por posição
         (não por score bruto) → ~100 candidatos
         ordenados por relevância combinada
                   │
                   ▼
            [ reranker ]
            Voyage AI rerank-2 lê a pergunta
            + cada chunk juntos e calcula
            relevância real → top-6
                   │
                   ▼
             [ answer.py ]
             Monta prompt com os 6 trechos no formato
             [livro > capítulo] e envia para o LLM.
             LLM cita fonte e declara insuficiência
             quando evidência não sustenta a resposta.
                   │
                   ▼
        Resposta com citações [livro > capítulo]
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
| Metadados externos | Google Books API + Open Library API (stdlib `urllib`) |
| Catálogo de livros | SQLite (local) / Firestore collection `books` (cloud) |
| Enriquecimento contextual | Multi-provider: Anthropic, DeepSeek, OpenAI, Gemini, Qwen |
| Embeddings | Voyage AI `voyage-3.5` |
| Busca lexical | `rank_bm25` |
| Reranker | Voyage AI `rerank-2` |
| Vector Database | `chromadb` (local) ou Google Firestore |
| LLM | Multi-provider: Anthropic, DeepSeek, OpenAI, Gemini, Qwen |
| Interface web | `streamlit` (local e Streamlit Community Cloud) |
| API REST | `fastapi` + `uvicorn` |
| CLI | `click` |

---

## Estrutura do Projeto

```
pergunte-ao-livro/
├── .env.example
├── requirements.txt
├── app.py                        # Interface web Streamlit
├── cli.py                        # CLI: ingest, ask, books, remove-book
├── api.py                        # API REST: /ingest, /ask, /books
├── data/
│   ├── books/                    # coloque seus EPUBs aqui
│   ├── books.db                  # catálogo SQLite (gerado automaticamente)
│   └── chroma/                   # banco vetorial local (gerado automaticamente)
├── src/
│   ├── clients.py                # fonte única de todos os clientes de API
│   ├── book_catalog.py           # catálogo de livros: SQLite (local) e Firestore (cloud)
│   ├── parser.py                 # EPUB → metadados + texto limpo por capítulo
│   ├── metadata_fetcher.py       # Google Books + Open Library → metadados complementares
│   ├── chunker.py                # texto → chunks com book_id, overlap e metadata
│   ├── enricher.py               # chunks → chunks enriquecidos (multi-provider, paralelo)
│   ├── embedder.py               # chunks → embeddings via Voyage AI (batch)
│   ├── store.py                  # Chroma: salva, consulta e deleta chunks por livro
│   ├── store_firestore.py        # Firestore: salva, consulta e deleta chunks por livro
│   ├── retriever.py              # hybrid search + rerank (Chroma)
│   ├── retriever_firestore.py    # hybrid search + rerank (Firestore)
│   └── answer.py                 # prompt + LLM → resposta com citações
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

# Providers opcionais para enriquecimento contextual e respostas
OPENAI_API_KEY=sua_chave_aqui
GEMINI_API_KEY=sua_chave_aqui
DEEPSEEK_API_KEY=sua_chave_aqui
QWEN_API_KEY=sua_chave_aqui

# Metadados externos (opcional — Open Library não precisa de chave)
BOOKS_API_KEY=sua_chave_aqui   # Google Books API

# Só para Firestore
GOOGLE_APPLICATION_CREDENTIALS=caminho/para/service-account.json
```

---

## Como usar

### Interface Web (Streamlit)

```bash
streamlit run app.py
```

Abre em `http://localhost:8501` com três abas:

| Aba | O que faz |
|---|---|
| 💬 **Perguntar** | Campo de pergunta, filtro por livro, seleção de modelo, exibe resposta e trechos usados |
| 📥 **Ingerir** | Upload de EPUB, opção de enriquecimento com seleção de provider |
| 📖 **Livros** | Lista livros ingeridos com metadados e botão de remoção |

**Deploy no Streamlit Community Cloud** (gratuito):
1. Suba o projeto no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte o repositório
3. Configure as variáveis de ambiente (API keys) em **Settings → Secrets**
4. Aponte o arquivo principal para `app.py`

> Para o deploy cloud, configure o Firestore — o SQLite e o Chroma local não persistem no Streamlit Cloud.

---

### CLI

**Ingerir livros:**

```bash
# Com enriquecimento contextual (padrão — recomendado)
python cli.py ingest "data/books/livro-a.epub"
python cli.py ingest "data/books/livro-a.epub" --provider=deepseek
python cli.py ingest "data/books/livro-a.epub" --provider=gemini

# Sem enriquecimento (mais rápido, menor qualidade de retrieval)
python cli.py ingest "data/books/livro-a.epub" --no-enrich
```

> Se tentar ingerir um livro já ingerido, o sistema avisa e para:
> `Livro já ingerido. Use 'python cli.py remove-book <book_id>' para removê-lo antes.`

**Comparativo de providers para enriquecimento (1746 chunks):**

| Provider | Custo estimado | Tempo (~5 threads) |
|---|---|---|
| `anthropic` | ~$0.05 (com cache) | ~6 min |
| `gemini` | ~$0.01 | ~4 min |
| `deepseek` | ~$0.02 | ~5 min |
| `qwen` | ~$0.005 | ~4 min |
| `openai` | ~$0.04 | ~5 min |

**Buscar metadados externos para um livro já ingerido:**

```bash
python cli.py fetch-metadata master-of-the-game
```

Consulta Google Books e Open Library e atualiza as colunas `metadata_google` e `metadata_openlibrary` no catálogo sem reprocessar chunks ou embeddings. Útil para livros ingeridos antes dessa funcionalidade existir.

**Remover um livro:**

```bash
python cli.py remove-book master-of-the-game
```

Remove o livro do Chroma e do catálogo SQLite. Outros livros não são afetados.

**Listar livros ingeridos:**

```bash
python cli.py books
```

```
book_id                                  título                               chunks  enriquecido
master-of-the-game                       Master of the Game                     1746  sim
senhor-dos-aneis-jrr-tolkien             O Senhor dos Anéis                     2100  não
```

**Fazer perguntas:**

```bash
# Busca em todos os livros (modelo padrão: Claude Sonnet)
python cli.py ask "Quem é Kate Blackwell?"

# Filtrar por livro específico
python cli.py ask "Quem é Kate Blackwell?" --book master-of-the-game

# Escolher outro modelo
python cli.py ask "Quem é Kate Blackwell?" --model deepseek-chat

# Combinando filtros
python cli.py ask "Quem é Kate?" --book master-of-the-game --model deepseek-chat --top-k 10
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
  -F "file=@data/books/livro.epub"

# Com enriquecimento
curl -X POST "http://localhost:8000/ingest?enrich=true&provider=deepseek" \
  -F "file=@data/books/livro.epub"
```

Resposta:
```json
{
  "status": "ok",
  "book_id": "master-of-the-game",
  "title": "Master of the Game",
  "chapters": 8,
  "chunks": 1746
}
```

**Listar livros via API:**

```bash
curl http://localhost:8000/books
```

**Fazer perguntas via API:**

```bash
# Busca em todos os livros
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Quem é Kate Blackwell?", "top_k": 5}'

# Filtrar por livro + escolher modelo
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Quem é Kate Blackwell?", "book_id": "master-of-the-game", "model": "deepseek-chat"}'
```

Resposta:
```json
{
  "answer": "Kate Blackwell é uma filantropa...",
  "sources": [
    {"chapter_id": "id18", "book_id": "master-of-the-game", "relevance_score": 0.91},
    {"chapter_id": "id22", "book_id": "master-of-the-game", "relevance_score": 0.78}
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
- [x] Resposta via LLM multi-provider com citações (Claude, DeepSeek, OpenAI, Gemini, Qwen)
- [x] CLI com Click
- [x] API REST com FastAPI
- [x] Pipeline de ingestão completo
- [x] Migração do vector DB para Google Firestore
- [x] Centralização de todos os clientes de API em `clients.py`
- [x] Suporte a múltiplos livros com book_id estável (ISBN → slug)
- [x] Catálogo de livros: SQLite (local) e Firestore collection (cloud)
- [x] Detecção de duplicatas — não reprocessa livro já ingerido
- [x] `--reset` apaga só o livro sendo reingerido, não os outros
- [x] Filtro por livro em perguntas (`--book` no CLI, `book_id` na API)
- [x] Interface web com Streamlit (ingerir, perguntar, listar e remover livros)
- [x] Deploy no Streamlit Community Cloud
- [x] Enriquecimento de metadados com Google Books API e Open Library API
- [x] Armazenamento separado das três fontes de metadados (epub, google, openlibrary)
- [x] Comando `fetch-metadata` para atualizar metadados sem reingerir o livro
- [x] Extração do título do capítulo (h1/h2/h3) para metadados do chunk
- [x] RRF (Reciprocal Rank Fusion) substituindo deduplicação simples no hybrid search
- [x] Candidate pool de 50 candidatos por retriever (era 5–10) para aumentar recall antes do reranker
- [x] Reranker top-k de 3 → 6 chunks para melhor cobertura de contexto
- [x] Citação estruturada no prompt: `[livro > capítulo]` com cláusula de abstenção explícita
- [x] Enriquecimento contextual ativado por padrão (`--no-enrich` para desativar)

---

## Licença

MIT
