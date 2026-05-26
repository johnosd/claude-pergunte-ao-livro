import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def _generate_book_id(book) -> str:
    identifiers = book.get_metadata("DC", "identifier") or []
    for value, _ in identifiers:
        val = str(value).replace("-", "").replace(" ", "")
        if re.match(r"97[89]\d{10}", val):
            return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:64]

    titles = book.get_metadata("DC", "title") or []
    creators = book.get_metadata("DC", "creator") or []
    title = titles[0][0] if titles else "unknown"
    author = creators[0][0] if creators else "unknown"
    raw = f"{title}-{author}".lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:64]


def parse_epub(file_path: str) -> tuple[dict, list[dict]]:
    book = epub.read_epub(file_path)

    titles = book.get_metadata("DC", "title") or []
    creators = book.get_metadata("DC", "creator") or []
    identifiers = book.get_metadata("DC", "identifier") or []

    isbn = next(
        (str(v) for v, _ in identifiers
         if re.match(r"97[89]\d{10}", str(v).replace("-", "").replace(" ", ""))),
        None,
    )

    book_meta = {
        "id": _generate_book_id(book),
        "title": titles[0][0] if titles else None,
        "author": creators[0][0] if creators else None,
        "isbn": isbn,
    }

    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text().strip()
            if len(text) < 100:
                continue
            chapters.append({"id": item.get_id(), "text": text})

    return book_meta, chapters
