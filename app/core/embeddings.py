from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# В начале файла после импортов
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPU(s)")
    device = "cuda" if torch.cuda.device_count() > 1 else "cuda:0"
else:
    device = "cpu"

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
        logger.info("Loading embedding model...")

        torch.set_default_dtype(torch.float32)

        base_model = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device = "cuda:1",                    # сначала грузим на CPU
            trust_remote_code=True
        )
        """
        # Multi-GPU поддержка
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"✅ Найдено {torch.cuda.device_count()} GPU. Используем DataParallel.")
            # Правильный способ для SentenceTransformer
            base_model = base_model.to('cuda')
            base_model = torch.nn.DataParallel(base_model)
            # Важно: сохраняем оригинальный метод encode
            base_model.encode = base_model.module.encode
        elif torch.cuda.is_available():
            print("✅ Используется одна GPU")
            base_model = base_model.to('cuda')
        else:
            print("⚠️ CUDA недоступен, работаем на CPU")
        """
        _model = base_model
        logger.info("Embedding model loaded successfully")
    
    return _model


def embed_texts(texts, batch_size=16):
    model = get_embedding_model()

    texts = [f"passage: {t}" for t in texts]

    embeddings = []
    logger.info(f"Generating embeddings for {len(texts)} chunks")

    try:
        from app.utils.progress import ProgressManager
        progress = ProgressManager.track(
            range(0, len(texts), batch_size),
            desc="Embedding batches"
        )
    except:
        progress = range(0, len(texts), batch_size)

    for i in progress:
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