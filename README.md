# Pergunte ao Livro

Faça perguntas para qualquer livro EPUB sem carregar o livro inteiro no contexto.

Sistema RAG (Retrieval-Augmented Generation) com busca híbrida (semântica + lexical) e reranking. Responde com citações dos trechos do livro via Claude.

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
[ embedder.py ]
Envia os chunks em batches para Voyage AI
Cada chunk vira um vetor de números (embedding)
que representa seu significado semântico
    │
    ▼
[ store.py / Chroma ]
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
no Chroma
    │                                 │
    └──────────────┬──────────────────┘
                   │
                   ▼
         [ Combina + Deduplica ]
         Une os resultados das duas buscas
         Remove duplicatas
         (~15-20 chunks únicos)
                   │
                   ▼
            [ reranker.py ]
            Voyage AI rerank-2 lê a pergunta
            + cada chunk juntos e calcula
            relevância real → top-3
                   │
                   ▼
             [ answer.py ]
             Monta prompt com os 3 trechos
             e envia para Claude
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

---

## Stack

| Componente | Tecnologia |
|---|---|
| EPUB parsing | `ebooklib` + `beautifulsoup4` |
| Embeddings | Voyage AI `voyage-3.5` |
| Busca lexical | `rank_bm25` |
| Reranker | Voyage AI `rerank-2` |
| Vector Database | `chromadb` (local) |
| LLM | `claude-sonnet-4-20250514` |
| CLI | `click` |

---

## Estrutura do Projeto

```
pergunte-ao-livro/
├── .env.example
├── requirements.txt
├── cli.py                    # entry point: ingest e ask
├── data/
│   ├── books/                # coloque seus EPUBs aqui
│   └── chroma/               # banco vetorial (gerado automaticamente)
├── src/
│   ├── parser.py             # EPUB → texto limpo por capítulo
│   ├── chunker.py            # texto → chunks com overlap e metadata
│   ├── embedder.py           # chunks → embeddings via Voyage AI (batch)
│   ├── store.py              # Chroma: salva e consulta chunks
│   ├── retriever.py          # busca semântica + lexical + rerank
│   └── answer.py             # prompt + Claude → resposta com citações
├── scripts/
│   └── ingest.py             # pipeline completo de ingestão
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
```

---

## Como usar

### 1. Ingerir o livro (roda uma vez)

```bash
python -m scripts.ingest "data/books/seu-livro.epub" --reset
```

### 2. Fazer perguntas

```bash
python cli.py ask "Quem é Kate Blackwell?"
python cli.py ask "Qual é o tema central do livro?"
python cli.py ask "O que acontece no início da história?"
```

### 3. Trocar de livro

```bash
python -m scripts.ingest "data/books/outro-livro.epub" --reset
```

---

## Roadmap

- [x] Parser EPUB
- [x] Chunking com overlap e metadata
- [x] Embeddings em batch com Voyage AI
- [x] Armazenamento no Chroma
- [x] Busca semântica (vetorial)
- [x] Busca lexical (BM25)
- [x] Hybrid Search (semântica + lexical)
- [x] Reranking com Voyage AI
- [x] Resposta via Claude com citações
- [x] CLI com Click
- [x] Pipeline de ingestão completo
- [ ] Bonus: Portar vector DB para Google Firestore

---

## Licença

MIT
