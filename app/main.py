import argparse

from app.indexing.indexer import build_index
from app.pipelines.chat_pipeline import run_chat
from app.pipelines.csv_pipeline import run_csv


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "index",
            "chat",
            "csv"
        ]
    )

    parser.add_argument(
        "--md_dir",
        default="data/md"
    )

    parser.add_argument(
        "--input_csv"
    )

    parser.add_argument(
        "--output_csv"
    )

    args = parser.parse_args()

    if args.mode == "index":

        build_index(args.md_dir)

    elif args.mode == "chat":

        run_chat()

    elif args.mode == "csv":

        run_csv(
            args.input_csv,
            args.output_csv
        )


if __name__ == "__main__":
    main()