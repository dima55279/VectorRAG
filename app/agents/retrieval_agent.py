from collections import defaultdict

from app.core.embeddings import embed_query
from app.core.vector_store import FAISSStore
from app.core.bm25_index import BM25Index
from app.core.config import TOP_K

from app.utils.logger import get_logger


logger = get_logger(__name__)


class RetrievalAgent:

    def __init__(self):

        self.faiss_store = FAISSStore.load()

        self.bm25 = BM25Index.load()

    def reciprocal_rank_fusion(
        self,
        dense_results,
        sparse_results
    ):

        scores = defaultdict(float)

        docs = {}

        for rank, item in enumerate(dense_results):

            key = item["document"]["content"]

            scores[key] += 1 / (rank + 60)

            docs[key] = item

        for rank, item in enumerate(sparse_results):

            key = item["document"]["content"]

            scores[key] += 1 / (rank + 60)

            docs[key] = item

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            docs[key]
            for key, _ in ranked[:TOP_K]
        ]

    def run(self, state):

        question = state["question"]

        logger.info(f"Retrieving: {question}")

        query_embedding = embed_query(question)

        dense_results = self.faiss_store.search(
            query_embedding,
            k=TOP_K
        )

        sparse_results = self.bm25.search(
            question,
            k=TOP_K
        )

        fused = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )

        state["retrieved_docs"] = fused

        return state