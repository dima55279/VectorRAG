from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path


BM25_PATH = "data/faiss_index/bm25.pkl"


class BM25Index:

    def __init__(self):

        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

    def build(self, docs):

        self.documents = docs

        self.tokenized_docs = [
            d["content"].lower().split()
            for d in docs
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def search(self, query, k=5):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                "score": score,
                "document": doc
            }
            for doc, score in ranked[:k]
        ]

    def save(self):

        with open(BM25_PATH, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls):

        with open(BM25_PATH, "rb") as f:
            return pickle.load(f)