# app/agents/csv_agent.py
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from app.agents.orchestrator import OrchestratorAgent
from app.utils.logger import get_logger


logger = get_logger(__name__)


class CSVAgent:

    def __init__(self, max_workers=8):
        self.orchestrator = OrchestratorAgent()
        self.max_workers = max_workers

    def process_row(self, row):
        try:
            result = self.orchestrator.run_question(row["question"])
            return {
                "question": row["question"],
                "answer": result["answer"],
                "document": str(result.get("documents", []))
            }
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\n❌ Ошибка при обработке вопроса:\n{row['question']}")
            print(f"Тип ошибки: {type(e).__name__}")
            print(error_msg[:1500])   # первые 1500 символов трассировки
            
            return {
                "question": row["question"],
                "answer": f"Ошибка: {type(e).__name__}",
                "document": "[]"
            }

    def process_csv(self, input_csv: str, output_csv: str):
        logger.info(f"Загрузка CSV: {input_csv}")
        
        df = pd.read_csv(input_csv)
        rows = df.to_dict("records")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            outputs = executor.map(self.process_row, rows)
            
            for result in tqdm(
                outputs, 
                total=len(rows), 
                desc="Processing questions"
            ):
                results.append(result)

        result_df = pd.DataFrame(results)
        
        # Сохраняем
        result_df.to_csv(output_csv, index=False)
        
        logger.info(f"Результат сохранён: {output_csv}")
        return result_df