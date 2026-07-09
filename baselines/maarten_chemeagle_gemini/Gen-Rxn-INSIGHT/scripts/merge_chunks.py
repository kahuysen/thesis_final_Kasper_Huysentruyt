"""Merge all chunk parquet files into a single parquet file."""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge chunk parquet files.")
    parser.add_argument("--input-dir", required=True, help="Directory with chunk_XXXX.parquet files.")
    parser.add_argument("--output", required=True, help="Output parquet file path.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*.parquet")))
    print(f"Found {len(files)} chunk files.")

    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Total reactions: {len(merged)}")

    # Fix mixed-type columns that cause Arrow serialization errors
    for col in merged.columns:
        if merged[col].dtype == object:
            merged[col] = merged[col].astype(str)

    merged.to_parquet(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
