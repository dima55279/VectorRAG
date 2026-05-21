from app.core.reranker import Reranker

from app.utils.logger import get_logger


logger = get_logger(__name__)


class RerankerAgent:

    def __init__(self):

        self.reranker = Reranker()

    def run(self, state):

        logger.info("Running reranker")

        query = state["question"]

        docs = state["retrieved_docs"]

        reranked = self.reranker.rerank(
            query,
            docs
        )

        state["retrieved_docs"] = reranked

        return state