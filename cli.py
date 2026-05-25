import click
from src.retriever import retrieve_hybrid, rerank
from src.answer import answer

@click.group()
def cli():
    pass

@cli.command()
@click.argument("epub_path")
@click.option("--reset", is_flag=True, help="Apaga o Chroma antes de ingerir")
def ingest(epub_path, reset):
    from scripts.ingest import ingest_book
    ingest_book(epub_path, reset=reset)

@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Número de chunks recuperados")
def ask(query, top_k):
    chunks = retrieve_hybrid(query, n_results=top_k)
    reranked = rerank(query, chunks)
    response = answer(query, reranked)
    click.echo(response)

if __name__ == "__main__":
    cli()
