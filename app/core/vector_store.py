import faiss
import pickle
import numpy as np
from pathlib import Path

from app.core.config import FAISS_PATH


class FAISSStore:

    def __init__(self, dimension):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(self, embeddings, docs):
        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.documents.extend(docs)

    def search(self, embedding, k=5):
        scores, ids = self.index.search(
            np.array([embedding]).astype("float32"),
            k
        )

        results = []

        for idx, score in zip(ids[0], scores[0]):

            if idx == -1:
                continue

            doc = self.documents[idx]

            results.append({
                "score": float(score),
                "document": doc
            })

        return results

    def save(self):
        Path(FAISS_PATH).mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            f"{FAISS_PATH}/index.faiss"
        )

        with open(f"{FAISS_PATH}/docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls):

        index = faiss.read_index(
            f"{FAISS_PATH}/index.faiss"
        )

        with open(f"{FAISS_PATH}/docs.pkl", "rb") as f:
            docs = pickle.load(f)

        store = cls(index.d)

        store.index = index
        store.documents = docs

        return store