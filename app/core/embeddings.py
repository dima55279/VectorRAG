from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import EMBEDDING_MODEL
from app.utils.logger import get_logger


logger = get_logger(__name__)

_model = None

def normalize(vectors):

    vectors = np.array(vectors)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    return vectors / norms


def get_embedding_model():

    global _model

    if _model is None:

        logger.info(
            f"Loading embeddings model: "
            f"{EMBEDDING_MODEL}"
        )

        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        logger.info("Embedding model loaded")

    return _model


def embed_texts(texts):

    model = get_embedding_model()

    embeddings = model.encode(
        texts
    )

    return normalize(embeddings)


def embed_query(query):

    model = get_embedding_model()

    query = f"query: {query}"

    embedding = model.encode(query)

    embedding = np.array(embedding)

    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding
