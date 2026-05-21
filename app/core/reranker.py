from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "BAAI/bge-reranker-large"
        )

    def rerank(self, query, docs, top_k=5):

        pairs = [
            (query, d["document"]["content"])
            for d in docs
        ]

        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                **doc,
                "rerank_score": float(score)
            }
            for doc, score in reranked[:top_k]
        ]