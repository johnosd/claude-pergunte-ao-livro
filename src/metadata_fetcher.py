import json
import os
import urllib.error
import urllib.parse
import urllib.request


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "claude-pergunte-ao-livro/1.0 (https://github.com/johnosd/pergunte-ao-livro; ccplaytv@gmail.com)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def fetch_google_books(isbn: str | None, title: str | None, author: str | None) -> dict:
    api_key = os.getenv("BOOKS_API_KEY", "")
    base = "https://www.googleapis.com/books/v1/volumes"

    if isbn:
        q = f"isbn:{isbn.replace('-', '')}"
    elif title:
        q = f"intitle:{title}"
        if author:
            q += f" inauthor:{author}"
    else:
        return {}

    params = {"q": q, "maxResults": "1"}
    if api_key:
        params["key"] = api_key
    url = f"{base}?{urllib.parse.urlencode(params)}"

    data = _get(url)
    items = data.get("items") or []
    if not items:
        return {}

    info = items[0].get("volumeInfo", {})
    result = {}
    for field in ("title", "authors", "description", "categories", "publishedDate", "publisher", "pageCount"):
        if field in info:
            result[field] = info[field]

    ids = info.get("industryIdentifiers", [])
    isbn13 = next((i["identifier"] for i in ids if i.get("type") == "ISBN_13"), None)
    if isbn13:
        result["isbn"] = isbn13

    return result


def fetch_open_library(isbn: str | None, title: str | None, author: str | None) -> dict:
    if isbn:
        clean = isbn.replace("-", "").replace(" ", "")
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{clean}&format=json&jscmd=data"
        data = _get(url)
        book = data.get(f"ISBN:{clean}")
        if book:
            return _parse_openlibrary_book(book)

    if title:
        params = {"title": title, "limit": "1"}
        if author:
            params["author"] = author
        url = f"https://openlibrary.org/search.json?{urllib.parse.urlencode(params)}"
        data = _get(url)
        docs = data.get("docs") or []
        if docs:
            return _parse_openlibrary_search(docs[0])

    return {}


def _parse_openlibrary_book(book: dict) -> dict:
    result = {}
    if "title" in book:
        result["title"] = book["title"]
    if "by_statement" in book:
        result["by_statement"] = book["by_statement"]
    if "publish_date" in book:
        result["publish_date"] = book["publish_date"]
    if "number_of_pages" in book:
        result["number_of_pages"] = book["number_of_pages"]

    publishers = book.get("publishers") or []
    if publishers:
        result["publishers"] = [p.get("name", p) if isinstance(p, dict) else p for p in publishers]

    subjects = book.get("subjects") or []
    if subjects:
        result["subjects"] = [s.get("name", s) if isinstance(s, dict) else s for s in subjects]

    excerpt = (book.get("excerpts") or [{}])[0]
    if excerpt.get("text"):
        result["description"] = excerpt["text"]

    return result


def _parse_openlibrary_search(doc: dict) -> dict:
    result = {}
    for field in ("title", "first_publish_year", "number_of_pages_median"):
        if field in doc:
            result[field] = doc[field]

    if "publisher" in doc:
        result["publishers"] = doc["publisher"][:3]
    if "subject" in doc:
        result["subjects"] = doc["subject"][:10]
    if "author_name" in doc:
        result["by_statement"] = ", ".join(doc["author_name"])

    return result


def fetch_all_metadata(epub_meta: dict) -> dict:
    isbn = epub_meta.get("isbn")
    title = epub_meta.get("title")
    author = epub_meta.get("author")

    google = {}
    openlibrary = {}

    try:
        google = fetch_google_books(isbn, title, author)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass

    # Se o EPUB não tinha ISBN mas o Google Books encontrou, usa esse ISBN
    # para o Open Library — aumenta o hit rate para livros em português
    isbn_for_ol = isbn or google.get("isbn")

    try:
        openlibrary = fetch_open_library(isbn_for_ol, title, author)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass

    return {
        "epub": epub_meta,
        "google": google,
        "openlibrary": openlibrary,
    }


def format_metadata_summary(all_metadata: dict) -> str:
    lines = []

    g = all_metadata.get("google", {})
    if g:
        found = []
        for key, label in [("title", "título"), ("authors", "autores"), ("publishedDate", "ano"),
                           ("publisher", "editora"), ("pageCount", "páginas"),
                           ("categories", "categorias"), ("isbn", "isbn")]:
            if g.get(key):
                val = g[key]
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val[:3])
                found.append(f"{label}: {val}")
        desc = g.get("description", "")
        if desc:
            found.append(f"descrição: {desc[:80]}{'...' if len(desc) > 80 else ''}")
        lines.append("  [Google Books] encontrado — " + " | ".join(found))
    else:
        lines.append("  [Google Books] não encontrado")

    ol = all_metadata.get("openlibrary", {})
    if ol:
        found = []
        for key, label in [("title", "título"), ("by_statement", "autor"), ("publish_date", "ano"),
                           ("publishers", "editora"), ("number_of_pages", "páginas"),
                           ("subjects", "assuntos")]:
            if ol.get(key):
                val = ol[key]
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val[:3])
                found.append(f"{label}: {val}")
        lines.append("  [Open Library] encontrado — " + " | ".join(found))
    else:
        lines.append("  [Open Library] não encontrado")

    return "\n".join(lines)
