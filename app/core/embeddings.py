from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from app.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_lock = None  # для thread-safety


def get_embedding_model():
    global _model, _lock
    if _lock is None:
        import threading
        _lock = threading.Lock()

    if _model is None:
        with _lock:  # защита от race condition
            if _model is None:
                logger.info("Loading embedding model...")
                torch.set_default_dtype(torch.float32)
                
                _model = SentenceTransformer(
                    "intfloat/multilingual-e5-base",
                    device="cpu",
                    trust_remote_code=True
                )
                logger.info("Embedding model loaded")
    return _model


def normalize(vectors):

    vectors = np.array(vectors)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / norms


def embed_texts(
    texts,
    batch_size=16
):

    model = get_embedding_model()

    texts = [
        f"passage: {t}"
        for t in texts
    ]

    embeddings = []

    logger.info(
        f"Generating embeddings "
        f"for {len(texts)} chunks"
    )

    for i in ProgressManager.track(
        range(0, len(texts), batch_size),
        desc="Embedding batches"
    ):

        batch = texts[i:i + batch_size]

        batch_embeddings = model.encode(
            batch
        )

        embeddings.extend(
            batch_embeddings
        )

    embeddings = normalize(embeddings)

    logger.info("Embeddings completed")

    return embeddings


def embed_query(query):

    model = get_embedding_model()

    query = f"query: {query}"

    embedding = model.encode(query)

    embedding = np.array(embedding)

    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding
