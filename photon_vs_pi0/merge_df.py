import argparse
import glob
import pandas as pd
from typing import List


def main():
    ops = options()
    print(f"Opening {ops.input}")
    df = get_df(ops.input)
    print(f"Saving to {ops.output}")
    df.to_parquet(ops.output)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file pattern")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    return parser.parse_args()


def expand(input: str) -> List[str]:
    return [f for path in input.split(",") for f in glob.glob(path)]


def get_df(filename: str) -> pd.DataFrame:
    return pd.concat(
        [
            annotate_df(pd.read_parquet(fi), it)
            for it, fi in enumerate(expand(filename))
        ],
        ignore_index=True,
    )


def annotate_df(df: pd.DataFrame, file_number: int) -> pd.DataFrame:
    df["file"] = file_number
    return df


if __name__ == "__main__":
    main()
