from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

from app.agents.orchestrator import OrchestratorAgent


class CSVAgent:

    def __init__(self):

        self.orchestrator = OrchestratorAgent()

    def process_row(self, row):

        result = self.orchestrator.run_question(
            row["question"]
        )

        return {
            "question": row["question"],
            "answer": result["answer"],
            "document": result["documents"]
        }

    def process_csv(self, input_csv, output_csv):

        df = pd.read_csv(input_csv)

        rows = df.to_dict("records")

        results = []

        with ThreadPoolExecutor(max_workers=8) as executor:

            outputs = executor.map(
                self.process_row,
                rows
            )

            for result in tqdm(
                outputs,
                total=len(rows),
                desc="Processing questions"
            ):
                results.append(result)

        out_df = pd.DataFrame(results)

        out_df.to_csv(
            output_csv,
            index=False
        )

        print(f"Saved to {output_csv}")