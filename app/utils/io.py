from pathlib import Path
import pandas as pd


def load_markdown(path):

    return Path(path).read_text(
        encoding="utf-8",
        errors="ignore"
    )


def save_csv(df, output_path):

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )


def load_csv(path):

    return pd.read_csv(path)