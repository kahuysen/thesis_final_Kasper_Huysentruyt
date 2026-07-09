"""Process a large reaction CSV in chunks using Rxn-INSIGHT.

Usage:
    python process_database.py --input <csv> --output-dir <dir> \
        --chunk-start <i> --chunk-end <j> [--chunk-size 10000] [--n-jobs 8]

Each invocation processes chunks [chunk-start, chunk-end) and saves one
parquet file per chunk.  A PBS array job can assign different chunk ranges
to different nodes.
"""

import argparse
import os
import sys
import time

import pandas as pd
import gen_rxn_insight as ri


def main():
    parser = argparse.ArgumentParser(description="Process reaction CSV in chunks.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for output parquet files.")
    parser.add_argument("--reaction-col", default="reaction_smiles", help="Name of the reaction SMILES column.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of reactions per chunk.")
    parser.add_argument("--chunk-start", type=int, required=True, help="First chunk index to process (0-based).")
    parser.add_argument("--chunk-end", type=int, required=True, help="Last chunk index (exclusive).")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read only the rows needed for this job's chunk range
    skip_rows = args.chunk_start * args.chunk_size
    n_rows = (args.chunk_end - args.chunk_start) * args.chunk_size

    # Read header first, then the relevant slice
    header = pd.read_csv(args.input, nrows=0)
    df_all = pd.read_csv(
        args.input,
        skiprows=range(1, skip_rows + 1),  # +1 to keep header
        nrows=n_rows,
        names=header.columns,
        header=0 if skip_rows == 0 else None,
        index_col=None,
    )

    total_read = len(df_all)
    print(f"Read {total_read} reactions (chunks {args.chunk_start}-{args.chunk_end - 1})")

    for chunk_idx in range(args.chunk_start, args.chunk_end):
        outfile = os.path.join(args.output_dir, f"chunk_{chunk_idx:04d}.parquet")
        if os.path.exists(outfile):
            print(f"Chunk {chunk_idx} already exists, skipping.")
            continue

        local_start = (chunk_idx - args.chunk_start) * args.chunk_size
        local_end = local_start + args.chunk_size
        chunk_df = df_all.iloc[local_start:local_end].reset_index(drop=True)

        if len(chunk_df) == 0:
            print(f"Chunk {chunk_idx} is empty, done.")
            break

        print(f"Processing chunk {chunk_idx} ({len(chunk_df)} reactions)...")
        t0 = time.time()

        db = ri.Database()
        try:
            result_df = db.create_database_from_df(
                chunk_df,
                args.reaction_col,
                n_jobs=args.n_jobs,
            )
            result_df.to_parquet(outfile)
            elapsed = time.time() - t0
            skipped = len(db.skipped_reactions)
            print(f"  Chunk {chunk_idx} done in {elapsed:.1f}s "
                  f"({len(result_df)} ok, {skipped} skipped)")
        except Exception as e:
            print(f"  Chunk {chunk_idx} FAILED: {e}", file=sys.stderr)

    print("All chunks complete.")


if __name__ == "__main__":
    main()
