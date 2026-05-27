def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_chapters_hierarchical(
    chapters: list[dict],
    book_id: str,
    child_size: int = 450,
    parent_size: int = 1000,
    child_overlap: int = 50,
    parent_overlap: int = 100,
) -> tuple[list[dict], list[dict]]:
    parents, children = [], []
    for chapter in chapters:
        chapter_id = chapter["id"]
        chapter_title = chapter.get("title", chapter_id)
        for pi, parent_text in enumerate(chunk_text(chapter["text"], parent_size, overlap=parent_overlap)):
            parent_id = f"{book_id}__{chapter_id}__parent_{pi}"
            parents.append({
                "id": parent_id,
                "text": parent_text,
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "book_id": book_id,
            })
            for ci, child_text in enumerate(chunk_text(parent_text, child_size, child_overlap)):
                children.append({
                    "id": f"{parent_id}__child_{ci}",
                    "text": child_text,
                    "parent_id": parent_id,
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title,
                    "book_id": book_id,
                })
    return parents, children


def chunk_chapters(chapters: list[dict], book_id: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for chapter in chapters:
        for i, text in enumerate(chunk_text(chapter["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{book_id}__{chapter['id']}__chunk_{i}",
                "text": text,
                "chapter_id": chapter["id"],
                "chapter_title": chapter.get("title", chapter["id"]),
                "book_id": book_id,
            })
    return chunks
