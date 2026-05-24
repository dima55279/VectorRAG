import re
from typing import List, Dict


HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def semantic_markdown_chunking(
    text: str,
    max_tokens: int = 600,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80
) -> List[Dict]:
    """
    Улучшенное чанкирование для юридических markdown-файлов.
    Возвращает формат, совместимый с текущим indexer.py
    """
    if not text or not text.strip():
        return [{"title": "document", "content": text.strip()}]

    matches = list(HEADER_PATTERN.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        header_level = len(match.group(1))
        header_text = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        # Если секция слишком большая — разбиваем
        if len(section_text.split()) > max_tokens * 1.2:   # небольшой запас
            sub_chunks = _split_large_section(
                section_text, 
                max_tokens=max_tokens,
                overlap=chunk_overlap,
                min_size=min_chunk_size
            )
            for sub_content in sub_chunks:
                chunks.append({
                    "title": f"{header_text} (часть)",
                    "content": sub_content
                })
        else:
            chunks.append({
                "title": header_text,
                "content": section_text
            })

    # Если нет заголовков — разбиваем весь текст
    if not matches and text.strip():
        sub_chunks = _split_large_section(
            text.strip(), 
            max_tokens=max_tokens,
            overlap=chunk_overlap,
            min_size=min_chunk_size
        )
        for sub in sub_chunks:
            chunks.append({
                "title": "document",
                "content": sub
            })

    # Фильтрация мелких чанков
    chunks = [ch for ch in chunks if len(ch["content"].split()) >= min_chunk_size]

    return chunks or [{"title": "document", "content": text.strip()}]


def _split_large_section(
    text: str, 
    max_tokens: int = 600, 
    overlap: int = 120,
    min_size: int = 80
) -> List[str]:
    """Разбивает большой текст на пересекающиеся чанки."""
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        end = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:end])
        
        if len(chunk.split()) >= min_size:
            chunks.append(chunk)
        
        i += max_tokens - overlap
        if i >= len(words) - min_size:
            break

    return chunks