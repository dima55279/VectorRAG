from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "jina-embeddings-v5-text-small"
)

TOP_K = int(os.getenv("TOP_K", 10))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 1))

FAISS_PATH = "data/faiss_index"
