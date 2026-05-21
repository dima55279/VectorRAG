from app.agents.csv_agent import CSVAgent

from app.utils.io import (
    load_csv,
    save_csv
)


def run_csv(input_csv, output_csv):

    agent = CSVAgent()

    df = load_csv(input_csv)

    result_df = agent.process_dataframe(df)

    save_csv(result_df, output_csv)