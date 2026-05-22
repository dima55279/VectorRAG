import torch
from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        # === Фиксы для torch meta tensor ошибки ===
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, 'backends'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        print("🔄 Загрузка Reranker...")

        self.model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32}
        )
        
        print("✅ Reranker успешно загружен")

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, d["document"]["content"]) for d in docs]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"⚠️ Ошибка в reranker.predict: {e}")
            # Fallback — возвращаем как есть
            return docs[:top_k]

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            {**doc, "rerank_score": float(score)}
            for doc, score in reranked[:top_k]
        ]