import click
from src.retriever import retrieve_hybrid, rerank, expand_to_parents
from src.answer import answer



@click.group()
def cli():
    pass

@cli.command()
@click.argument("epub_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--no-enrich", "no_enrich", is_flag=True, help="Desativa enriquecimento contextual (ativado por padrão)")
@click.option("--provider", default="anthropic", show_default=True, help="Provider para enriquecimento: anthropic, deepseek, openai, gemini, qwen")
def ingest(epub_path, no_enrich, provider):
    if not epub_path.endswith(".epub"):
        raise click.BadParameter("O arquivo deve ter extensão .epub", param_hint="'EPUB_PATH'")
    from scripts.ingest import ingest_book
    ingest_book(epub_path, enrich=not no_enrich, provider=provider)

@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Número de chunks recuperados")
@click.option("--model", default="claude-sonnet-4-20250514", help="Modelo LLM (ex: deepseek-chat)")
@click.option("--book", default=None, help="Filtrar por book_id (ex: master-of-the-game)")
def ask(query, top_k, model, book):
    chunks = retrieve_hybrid(query, n_results=top_k, book_id=book)
    expanded = expand_to_parents(chunks)
    reranked = rerank(query, expanded, top_k=top_k)
    response = answer(query, reranked, model=model)
    click.echo(response)

@cli.command("fetch-metadata")
@click.argument("book_id")
def fetch_metadata(book_id):
    from src.book_catalog import book_exists, list_books as _list_books, update_book_metadata
    from src.metadata_fetcher import fetch_all_metadata

    if not book_exists(book_id):
        raise click.ClickException(f"Livro '{book_id}' não encontrado. Use 'python cli.py books' para ver os disponíveis.")

    book = next(b for b in _list_books() if b["book_id"] == book_id)
    epub_meta = {
        "id":     book["book_id"],
        "title":  book["title"],
        "author": book["author"],
        "isbn":   book["isbn"],
    }

    click.echo(f"Buscando metadados para: {book['title']!r}")
    all_metadata = fetch_all_metadata(epub_meta)

    from src.metadata_fetcher import format_metadata_summary
    click.echo(format_metadata_summary(all_metadata))

    update_book_metadata(book_id, all_metadata)
    click.echo(f"Metadados atualizados para '{book_id}'.")


@cli.command("remove-book")
@click.argument("book_id")
def remove_book(book_id):
    from src.book_catalog import book_exists, list_books as _list_books
    from scripts.ingest import remove_book as _remove_book
    if not book_exists(book_id):
        raise click.ClickException(f"Livro '{book_id}' não encontrado. Use 'python cli.py books' para ver os disponíveis.")
    _remove_book(book_id)

@cli.command("books")
def list_books():
    from src.book_catalog import list_books as _list_books
    books = _list_books()
    if not books:
        click.echo("Nenhum livro ingerido.")
        return
    click.echo(f"{'book_id':<40} {'título':<35} {'chunks':>6}  {'enriquecido'}")
    click.echo("-" * 90)
    for b in books:
        enriched = "sim" if b["enriched"] else "não"
        click.echo(f"{b['book_id']:<40} {(b['title'] or '-'):<35} {b['chunk_count']:>6}  {enriched}")

if __name__ == "__main__":
    cli()
