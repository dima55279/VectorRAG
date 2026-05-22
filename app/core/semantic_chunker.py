import re
from typing import List, Dict


HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)


def semantic_markdown_chunking(text: str):
    matches = list(HEADER_PATTERN.finditer(text))

    chunks = []

    if not matches:
        return [{
            "title": "document",
            "content": text
        }]

    for i, match in enumerate(matches):
        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        header = match.group(2).strip()

        content = text[start:end].strip()

        chunks.append({
            "title": header,
            "content": content
        })

    return chunks