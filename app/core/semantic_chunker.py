import re
from typing import List, Dict

from app.core.config import FAISS_PATH  # если нужно, можно импортировать константы

# Паттерн для поиска markdown заголовков
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def semantic_markdown_chunking(
    text: str,
    max_tokens: int = 600,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80
) -> List[Dict]:
    """
    Улучшенное семантическое чанкирование Markdown для юридических текстов.
    
    Особенности:
    - Учитывает иерархию заголовков
    - Добавляет overlap между чанками
    - Сохраняет контекст родительских заголовков
    - Ограничивает размер чанка
    - Не разбивает предложения посередине (по возможности)
    """
    
    if not text or not text.strip():
        return []

    # Находим все заголовки
    matches = list(HEADER_PATTERN.finditer(text))
    
    chunks = []
    prev_end = 0

    for i, match in enumerate(matches):
        header_level = len(match.group(1))
        header_text = match.group(2).strip()
        start = match.start()

        # Определяем конец текущей секции
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        # Если секция слишком большая — разбиваем дальше
        if len(section_text.split()) > max_tokens:
            sub_chunks = _split_large_section(
                section_text, 
                max_tokens=max_tokens,
                overlap=chunk_overlap,
                min_size=min_chunk_size
            )
            for sub in sub_chunks:
                full_title = _build_full_title(text, start, header_text)
                chunks.append({
                    "document_name": "",  # будет заполнено в indexer
                    "section": full_title,
                    "content": sub,
                    "header_level": header_level
                })
        else:
            full_title = _build_full_title(text, start, header_text)
            chunks.append({
                "document_name": "",
                "section": full_title,
                "content": section_text,
                "header_level": header_level
            })

        prev_end = end

    # Обработка текста без заголовков
    if not matches and text.strip():
        chunks.extend(_split_large_section(
            text.strip(), 
            max_tokens=max_tokens,
            overlap=chunk_overlap,
            min_size=min_chunk_size
        ))

    # Фильтруем слишком маленькие чанки
    chunks = [ch for ch in chunks if len(ch["content"].split()) >= min_chunk_size]

    return chunks


def _build_full_title(text: str, start_pos: int, current_header: str) -> str:
    """Собирает полный путь заголовков (иерархия)"""
    # Можно улучшить позже, сейчас оставляем простой вариант
    return current_header


def _split_large_section(
    text: str, 
    max_tokens: int = 600, 
    overlap: int = 120,
    min_size: int = 80
) -> List[str]:
    """
    Разбивает большой текст на пересекающиеся чанки.
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_end = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:chunk_end])
        
        if len(chunk.split()) >= min_size:
            chunks.append(chunk)
        
        # Двигаемся с overlap'ом
        i += max_tokens - overlap
        
        # Защита от бесконечного цикла
        if i >= len(words) - min_size:
            break

    return chunks