import pandas as pd

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from app.utils.logger import get_logger
from app.utils.progress import ProgressManager
from app.utils.io import save_csv

logger = get_logger(__name__)


class CSVAgent:

    def __init__(
        self,
        rag_pipeline,
        max_workers=10
    ):

        self.rag = rag_pipeline
        self.max_workers = max_workers

    def process_question(
        self,
        question
    ):

        try:

            result = self.rag.ask(
                question
            )

            return {
                "answer": result["answer"],
                "documents": result["documents"]
            }

        except Exception as e:

            logger.exception(
                f"CSV processing error: {e}"
            )

            return {
                "answer": (
                    "Ошибка обработки вопроса"
                ),
                "documents": []
            }

    def process_dataframe(
        self,
        df
    ):

        results = [
            None
        ] * len(df)

        futures = {}

        logger.info(
            f"Processing {len(df)} questions"
        )

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            for idx, row in df.iterrows():

                future = executor.submit(
                    self.process_question,
                    row["question"]
                )

                futures[future] = idx

            for future in ProgressManager.track(
                as_completed(futures),
                desc="Processing CSV",
                total=len(futures)
            ):

                idx = futures[future]

                result = future.result()

                results[idx] = {
                    "question":
                        df.iloc[idx]["question"],

                    "answer":
                        result["answer"],

                    "document":
                        str(result["documents"])
                }

        return pd.DataFrame(results)

    def process_csv(
        self,
        input_path,
        output_path
    ):

        logger.info(
            f"Loading CSV: {input_path}"
        )

        df = pd.read_csv(input_path)

        result_df = self.process_dataframe(df)

        save_csv(
            result_df,
            output_path
        )

        logger.info(
            f"Saved results: {output_path}"
        )

        return result_df