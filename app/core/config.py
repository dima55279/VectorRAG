from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-base"
)

TOP_K = int(os.getenv("TOP_K", 5))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))

FAISS_PATH = "data/faiss_index"
