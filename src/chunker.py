def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_chapters(chapters: list[dict], book_id: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for chapter in chapters:
        for i, text in enumerate(chunk_text(chapter["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{book_id}__{chapter['id']}__chunk_{i}",
                "text": text,
                "chapter_id": chapter["id"],
                "book_id": book_id,
            })
    return chunks
