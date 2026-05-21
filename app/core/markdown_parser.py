import re
from pathlib import Path


def parse_markdown(md_path: str):
    text = Path(md_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return text