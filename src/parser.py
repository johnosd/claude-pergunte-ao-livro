import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def parse_epub(file_path):
    book = epub.read_epub(file_path)
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text().strip()
            if len(text) < 100:
                continue  # Considera apenas capítulos com mais de 100 caracteres

            chapters.append(
                {
                    "id": item.get_id(),
                    "text": text
                }
            )
    
    return chapters