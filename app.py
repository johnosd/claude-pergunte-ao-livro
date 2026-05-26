import tempfile
import os
import streamlit as st

st.set_page_config(page_title="Pergunte ao Livro", page_icon="📚", layout="wide")
st.title("📚 Pergunte ao Livro")

tab_ask, tab_ingest, tab_books = st.tabs(["💬 Perguntar", "📥 Ingerir", "📖 Livros"])


# ── Perguntar ────────────────────────────────────────────────────────────────

with tab_ask:
    from src.book_catalog import list_books
    from src.retriever import retrieve_hybrid, rerank
    from src.answer import answer
    from src.clients import PROVIDERS

    books = list_books()
    book_options = {"Todos os livros": None} | {
        f"{b['title'] or b['book_id']}": b["book_id"] for b in books
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Pergunta", placeholder="Quem é Kate Blackwell?")
    with col2:
        selected_book_label = st.selectbox("Livro", list(book_options.keys()))
        book_id = book_options[selected_book_label]

    model_options = [
        "claude-sonnet-4-20250514",
        "deepseek-chat",
        "deepseek-reasoner",
        "gpt-4o-mini",
        "gemini-2.0-flash-lite",
    ]
    model = st.selectbox("Modelo", model_options)
    top_k = st.slider("Chunks recuperados", min_value=3, max_value=20, value=5)

    if st.button("Perguntar", type="primary", disabled=not query):
        with st.spinner("Buscando e gerando resposta..."):
            chunks = retrieve_hybrid(query, n_results=top_k, book_id=book_id)
            if not chunks:
                st.warning("Nenhum trecho relevante encontrado.")
            else:
                reranked = rerank(query, chunks)
                response = answer(query, reranked, model=model)
                st.markdown("### Resposta")
                st.write(response)
                with st.expander("Trechos utilizados"):
                    for i, c in enumerate(reranked):
                        st.markdown(f"**Trecho {i+1}** — capítulo `{c['chapter_id']}` | score `{c.get('relevance_score', 0):.2f}`")
                        parts = c["text"].split("\n\n", 1)
                        if len(parts) == 2:
                            st.info(f"🔍 **Contexto enriquecido:** {parts[0]}")
                            st.caption(parts[1][:400] + ("..." if len(parts[1]) > 400 else ""))
                        else:
                            st.caption(parts[0][:400] + ("..." if len(parts[0]) > 400 else ""))
                        st.divider()


# ── Ingerir ──────────────────────────────────────────────────────────────────

with tab_ingest:
    from src.clients import PROVIDERS

    st.subheader("Ingerir livro EPUB")
    uploaded = st.file_uploader("Selecione um arquivo EPUB", type=["epub"])

    enrich = st.checkbox("Enriquecimento contextual", value=True)
    enrich_providers = [p for p, cfg in PROVIDERS.items() if cfg.get("enrich_model")]
    provider = st.selectbox("Provider de enriquecimento", enrich_providers, disabled=not enrich)

    if st.button("Ingerir", type="primary", disabled=not uploaded):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            from scripts.ingest import ingest_book
            with st.spinner("Processando... (pode demorar vários minutos com enriquecimento)"):
                ingest_book(tmp_path, enrich=enrich, provider=provider if enrich else None)
            st.success("Livro ingerido com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(str(e))
        finally:
            os.unlink(tmp_path)


# ── Livros ───────────────────────────────────────────────────────────────────

with tab_books:
    from src.book_catalog import list_books, update_book_metadata
    from src.metadata_fetcher import fetch_all_metadata, format_metadata_summary
    from scripts.ingest import remove_book

    books = list_books()

    if not books:
        st.info("Nenhum livro ingerido ainda. Use a aba **Ingerir** para começar.")
    else:
        st.subheader(f"{len(books)} livro(s) ingerido(s)")
        for b in books:
            with st.expander(f"📖 {b['title'] or b['book_id']}"):
                col_info, col_actions = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**book_id:** `{b['book_id']}`")
                    st.markdown(f"**Autor:** {b['author'] or '—'}")
                    st.markdown(f"**ISBN:** {b['isbn'] or '—'}")
                    st.markdown(f"**Chunks:** {b['chunk_count']} | **Enriquecido:** {'sim' if b['enriched'] else 'não'}")
                    st.caption(f"Ingerido em: {b['ingested_at']}")

                with col_actions:
                    if st.button("🔍 Buscar Metadados", key=f"fetch_{b['book_id']}"):
                        with st.spinner("Consultando APIs..."):
                            epub_meta = {"id": b["book_id"], "title": b["title"],
                                         "author": b["author"], "isbn": b["isbn"]}
                            all_metadata = fetch_all_metadata(epub_meta)
                            update_book_metadata(b["book_id"], all_metadata)
                        st.success(format_metadata_summary(all_metadata))
                        st.rerun()

                    if st.button("🗑 Remover", key=f"remove_{b['book_id']}"):
                        remove_book(b["book_id"])
                        st.success(f"'{b['title'] or b['book_id']}' removido.")
                        st.rerun()

                # Metadados das três fontes
                g  = b.get("metadata_google") or {}
                ol = b.get("metadata_openlibrary") or {}
                ep = b.get("metadata_epub") or {}

                has_external = g or ol
                st.divider()
                meta_tab_epub, meta_tab_google, meta_tab_ol = st.tabs(["📄 EPUB", "🔵 Google Books", "🟠 Open Library"])

                with meta_tab_epub:
                    if ep:
                        for k, v in ep.items():
                            st.markdown(f"**{k}:** {v}")
                    else:
                        st.caption("Sem dados do EPUB.")

                with meta_tab_google:
                    if g:
                        for k, v in g.items():
                            if k == "description":
                                st.markdown(f"**Descrição:** {v}")
                            elif isinstance(v, list):
                                st.markdown(f"**{k}:** {', '.join(str(i) for i in v)}")
                            else:
                                st.markdown(f"**{k}:** {v}")
                    else:
                        st.caption("Sem dados do Google Books. Clique em 'Buscar Metadados'.")

                with meta_tab_ol:
                    if ol:
                        for k, v in ol.items():
                            if isinstance(v, list):
                                st.markdown(f"**{k}:** {', '.join(str(i) for i in v)}")
                            else:
                                st.markdown(f"**{k}:** {v}")
                    else:
                        st.caption("Sem dados do Open Library. Clique em 'Buscar Metadados'.")
