from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.utils.logger import get_logger
from app.utils.progress import ProgressManager

from app.core.markdown_parser import parse_markdown
from app.core.semantic_chunker import (
    semantic_markdown_chunking
)

from app.core.embeddings import embed_texts
from app.core.vector_store import FAISSStore
from app.core.bm25_index import BM25Index


logger = get_logger(__name__)


def process_file(md_path):

    logger.info(f"Processing: {md_path}")

    text = parse_markdown(md_path)

    chunks = semantic_markdown_chunking(text)

    docs = []

    for chunk in chunks:

        docs.append({
            "document_name": Path(md_path).stem,
            "section": chunk["title"],
            "content": chunk["content"]
        })

    return docs


def build_index(md_dir):

    md_files = list(Path(md_dir).glob("*.md"))

    all_docs = []

    with ThreadPoolExecutor(max_workers=8) as executor:

        results = executor.map(
            process_file,
            md_files
        )

        for docs in ProgressManager.track(
            results,
            desc="Indexing markdown files",
            total=len(md_files)
        ):
            all_docs.extend(docs)

    logger.info("Creating embeddings")

    texts = [
        doc["content"]
        for doc in all_docs
    ]

    embeddings = embed_texts(texts)

    dimension = len(embeddings[0])

    store = FAISSStore(dimension)

    store.add(embeddings, all_docs)

    logger.info("Saving FAISS")

    store.save()

    bm25 = BM25Index()

    bm25.build(all_docs)

    bm25.save()

    logger.info("BM25 saved")

    logger.info("Indexing completed")