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
    return pd.concat([pd.read_parquet(f) for f in expand(filename)], ignore_index=True)


if __name__ == "__main__":
    main()

