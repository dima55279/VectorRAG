import torch

# В начале файла после импортов
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPU(s)")
    device = "cuda" if torch.cuda.device_count() > 1 else "cuda:0"
else:
    device = "cpu"

from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        print("🔄 Загрузка Reranker (BAAI/bge-reranker-v2-m3)...")
        
        # Фиксы для проблем с torch
        torch.set_default_dtype(torch.float32)
        
        self.model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cpu",
            trust_remote_code=True
        )

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("Using DataParallel for reranker")
            self.model.model = torch.nn.DataParallel(self.model.model)
            # Сохраняем predict
            original_predict = self.model.predict
            self.model.predict = lambda *args, **kwargs: original_predict(*args, **kwargs)
        
        print("✅ Reranker успешно загружен")

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, d["document"]["content"]) for d in docs]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"⚠️ Ошибка при rerank.predict: {e}")
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