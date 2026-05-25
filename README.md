# 📚 Pergunte ao Livro

> Faça perguntas para qualquer livro EPUB sem carregar o livro inteiro no contexto.

**Pergunte ao Livro** é um sistema RAG (Retrieval-Augmented Generation) que permite qualquer pessoa fazer upload de um livro `.epub` e conversar com ele via linha de comando. O sistema encontra os trechos mais relevantes do livro e usa Claude para gerar respostas precisas com citações.

---

## 🧠 Como funciona

Em vez de enviar o livro inteiro para o modelo (caro, lento, limitado pelo context window), o sistema:

1. **Parseia** o EPUB e extrai o texto limpo por capítulo
2. **Divide** o texto em chunks com overlap para preservar contexto
3. **Vetoriza** cada chunk com Voyage AI (`voyage-3.5`)
4. **Armazena** os vetores no Chroma (banco de dados local)
5. **Recebe** uma pergunta do usuário
6. **Busca** os chunks mais similares à pergunta
7. **Reranka** os resultados com Voyage AI (`rerank-2.5`) para maior precisão
8. **Responde** via Claude com base apenas nos trechos relevantes

```
EPUB → parser → chunks → embeddings → Chroma
                                          ↓
Pergunta → embedding → busca top-5 → rerank top-3 → Claude → Resposta
```

---

## 🛠️ Stack

| Componente | Tecnologia |
|------------|------------|
| EPUB parsing | `ebooklib` + `beautifulsoup4` |
| Tokenização | `tiktoken` |
| Embeddings | Voyage AI `voyage-3.5` |
| Reranker | Voyage AI `rerank-2.5` |
| Vector Database | `chromadb` (local) |
| LLM | Anthropic Claude `claude-sonnet-4-20250514` |
| CLI | `click` |

---

## 📁 Estrutura do Projeto

```
pergunte-ao-livro/
│
├── .env.example                  # template das API keys
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── books/                    # coloque seus EPUBs aqui (ignorado pelo git)
│
├── src/
│   ├── __init__.py
│   ├── parser.py                 # lê EPUB, extrai texto limpo por capítulo
│   ├── chunker.py                # divide texto em chunks com overlap
│   ├── embedder.py               # gera embeddings via Voyage AI
│   ├── store.py                  # Chroma: armazena, consulta, persiste
│   ├── retriever.py              # busca top-k + rerank via Voyage AI
│   └── answer.py                 # monta prompt + chama Claude
│
├── cli.py                        # entry point principal
│
├── scripts/
│   └── ingest.py                 # pipeline: epub → chroma (roda uma vez)
│
└── tests/
    ├── test_parser.py
    ├── test_chunker.py
    └── test_retriever.py
```

---

## ⚙️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/pergunte-ao-livro.git
cd pergunte-ao-livro
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:

```env
ANTHROPIC_API_KEY=sua_chave_aqui
VOYAGE_API_KEY=sua_chave_aqui
```

---

## 🚀 Como usar

### Passo 1 — Ingerir o livro (roda uma vez)

```bash
python scripts/ingest.py --epub data/books/seu-livro.epub
```

Isso vai:
- Parsear o EPUB
- Gerar os chunks
- Criar os embeddings via Voyage AI
- Salvar no Chroma em disco

### Passo 2 — Fazer perguntas

```bash
python cli.py ask "Quem é Kate Blackwell?"
python cli.py ask "Qual é o tema central do livro?"
python cli.py ask "O que acontece no capítulo 3?"
```

### Passo 3 — Trocar de livro

```bash
python scripts/ingest.py --epub data/books/outro-livro.epub --reset
```

---

## 📦 requirements.txt

```
anthropic
voyageai
chromadb
ebooklib
beautifulsoup4
tiktoken
click
python-dotenv
```

---

## 🗺️ Roadmap

- [x] Estrutura do projeto
- [x] Parser EPUB com extração por capítulo (`src/parser.py`)
- [x] Chunking com overlap e metadata (`src/chunker.py`)
- [x] Geração de embeddings em batch (Voyage AI) (`src/embedder.py`)
- [x] Armazenamento no Chroma (`src/store.py`)
- [x] Retrieval top-k por similaridade vetorial (`src/retriever.py`)
- [ ] Reranking com Voyage AI (`rerank-2.5`)
- [ ] Resposta via Claude com citações (`src/answer.py`)
- [ ] CLI completo com Click (`cli.py`)
- [ ] Pipeline de ingestão completo (`scripts/ingest.py`)
- [ ] **Bonus:** Portar vector DB para Google Firestore

---

## 💡 Decisões de Arquitetura

**Por que RAG e não enviar o livro inteiro?**
Um livro médio tem ~150k tokens. O context window do Claude suporta até 200k, mas enviar tudo a cada pergunta é caro e lento. RAG busca apenas os 3 trechos relevantes — reduz custo em ~98%.

**Por que Voyage AI para embeddings?**
É o provedor recomendado oficialmente pela Anthropic para uso com Claude. O modelo `voyage-3.5` é estado da arte em recuperação semântica.

**Por que reranker?**
Busca vetorial por similaridade coseno é boa mas não perfeita. O reranker `rerank-2.5` faz uma segunda passagem semântica e melhora significativamente a precisão dos trechos enviados ao Claude.

**Por que Chroma?**
Simples, local, sem setup externo. Dados persistem em disco. Fácil de substituir por Firestore ou Pinecone sem mudar o restante do código (ver `src/store.py`).

---

## 📄 Licença

MIT