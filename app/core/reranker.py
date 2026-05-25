import torch
from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        print("🔄 Загрузка Reranker (Qwen/Qwen3-Reranker-4B)...")

        # Фикс совместимости
        torch.set_default_dtype(torch.float32)

        # Используем только одну GPU (cuda:0)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"✅ Используется устройство: {device}")

        self.model = CrossEncoder(
            "Qwen/Qwen3-Reranker-4B",
            device=device,           # ← явно указываем cuda:0
            trust_remote_code=True
        )

        print("✅ Reranker успешно загружен")

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, d["document"]["content"]) for d in docs]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"⚠️ Ошибка reranker.predict: {e}")
            return docs[:top_k]  # fallback

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            {**doc, "rerank_score": float(score)}
            for doc, score in reranked[:top_k]
        ]