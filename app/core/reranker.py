import torch
from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        print("🔄 Загрузка reranker...")

        torch.set_default_dtype(torch.float32)

        # Загружаем модель
        self.model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cpu",           # сначала на CPU
            trust_remote_code=True
        )

        # Переносим на GPU
        if torch.cuda.is_available():
            print(f"✅ Перенос reranker на GPU (найдено {torch.cuda.device_count()} GPU)")
            self.model = self.model.to('cuda')
            
            # DataParallel только если больше одной GPU
            if torch.cuda.device_count() > 1:
                print("🔀 Используем DataParallel")
                self.model.model = torch.nn.DataParallel(self.model.model)
        
        print("✅ Reranker успешно загружен")

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        pairs = [(query, d["document"]["content"]) for d in docs]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"⚠️ Ошибка reranker.predict: {e}")
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