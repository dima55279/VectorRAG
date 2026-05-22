# app/pipelines/csv_pipeline.py
from app.agents.csv_agent import CSVAgent
from app.utils.io import load_csv, save_csv


def run_csv(input_csv: str, output_csv: str):
    agent = CSVAgent() 
    
    df = load_csv(input_csv)
    
    result_df = agent.process_csv(
        input_csv=input_csv, 
        output_csv=output_csv
    )
    
    print(f"✅ Обработка завершена. Результат сохранён в: {output_csv}")
    print(f"Обработано вопросов: {len(result_df)}")
    
    return result_df