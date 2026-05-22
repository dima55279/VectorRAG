from sentence_transformers import SentenceTransformer
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None


def normalize(vectors):
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def get_embedding_model():
    global _model
    if _model is None:
        logger.info("Loading embedding model")
        _model = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device="cpu"
        )
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts, batch_size=16):
    model = get_embedding_model()

    texts = [f"passage: {t}" for t in texts]

    embeddings = []
    logger.info(f"Generating embeddings for {len(texts)} chunks")

    # === Исправленный прогресс-бар с fallback ===
    try:
        from app.utils.progress import ProgressManager
        progress_iter = ProgressManager.track(
            range(0, len(texts), batch_size),
            desc="Embedding batches"
        )
    except Exception:
        # Fallback если ProgressManager не импортируется
        logger.warning("ProgressManager not available, using simple range")
        progress_iter = range(0, len(texts), batch_size)

    for i in progress_iter:
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    embeddings = normalize(embeddings)
    logger.info("Embeddings completed")
    return embeddings


def embed_query(query):
    model = get_embedding_model()
    query = f"query: {query}"
    embedding = model.encode(query)
    embedding = np.array(embedding)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding