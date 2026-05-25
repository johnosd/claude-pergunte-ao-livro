from src.parser import parse_epub

def test_parse_epub():
    chapters = parse_epub("data/books/O Reverso Da Medalha.epub")
    
    assert len(chapters) > 0
    assert "id" in chapters[0]
    assert "text" in chapters[0]
    assert len(chapters[0]["text"]) > 100
    
    print(f"Capítulos encontrados: {len(chapters)}")
    print(f"Primeiro ID: {chapters[0]['id']}")
    print(f"Início do texto: {chapters[0]['text'][:200]}")
