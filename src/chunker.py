def chunk_text(text, chunk_size=500, overlap=50):
    """
    Divide o texto em pedaços (chunks) de tamanho especificado, com sobreposição.
    
    Args:
        text (str): O texto a ser dividido.
        chunk_size (int): O tamanho máximo de cada chunk.
        overlap (int): O número de caracteres que se sobrepõem entre os chunks.
        
    Returns:
        list: Uma lista de chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap  # Move o início para criar sobreposição
    return chunks

def chunk_chapters(chapters: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for chapter in chapters:
        texts = chunk_text(chapter["text"], chunk_size, overlap)
        for i, text in enumerate(texts):
            chunks.append({
                "id": f"{chapter['id']}_chunk_{i}",
                "text": text,
                "chapter_id": chapter["id"]
            })
    return chunks
