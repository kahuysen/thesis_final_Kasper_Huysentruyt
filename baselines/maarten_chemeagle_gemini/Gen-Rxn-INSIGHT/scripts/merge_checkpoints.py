"""Merge extra classes from an old checkpoint into a new checkpoint.

Copies train_results, finetune_results, and fp_results for classes
that exist only in the old checkpoint into the new one.

Usage
-----
    python merge_checkpoints.py \
        --old old_checkpoint.json \
        --new new_checkpoint.json \
        --output merged_checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(
        description="Merge extra classes from old checkpoint into new checkpoint.",
    )
    p.add_argument("--old", required=True, help="Old checkpoint with extra classes.")
    p.add_argument("--new", required=True, help="New (Phase-5) checkpoint.")
    p.add_argument("--output", required=True, help="Output merged checkpoint.")
    p.add_argument(
        "--skip-conflict",
        action="store_true",
        default=True,
        help="Skip CONFLICT: classes (default: True).",
    )
    args = p.parse_args()

    print(f"Loading old checkpoint: {args.old}")
    with open(args.old, encoding="utf-8") as f:
        old_ckpt = json.load(f)

    print(f"Loading new checkpoint: {args.new}")
    with open(args.new, encoding="utf-8") as f:
        new_ckpt = json.load(f)

    old_train = set(old_ckpt.get("train_results", {}))
    new_train = set(new_ckpt.get("train_results", {}))
    only_old = old_train - new_train

    if args.skip_conflict:
        only_old = {c for c in only_old if not c.startswith("CONFLICT:")}

    print(f"\nOld checkpoint classes: {len(old_train)}")
    print(f"New checkpoint classes: {len(new_train)}")
    print(f"Classes to merge: {len(only_old)}")

    # Merge train_results
    n_train = 0
    for cls in sorted(only_old):
        if cls in old_ckpt.get("train_results", {}):
            new_ckpt.setdefault("train_results", {})[cls] = old_ckpt["train_results"][cls]
            n_train += 1
    print(f"  Merged {n_train} train_results entries")

    # Merge finetune_results
    n_finetune = 0
    for cls in sorted(only_old):
        if cls in old_ckpt.get("finetune_results", {}):
            new_ckpt.setdefault("finetune_results", {})[cls] = old_ckpt["finetune_results"][cls]
            n_finetune += 1
    print(f"  Merged {n_finetune} finetune_results entries")

    # Merge fp_results
    n_fp = 0
    for cls in sorted(only_old):
        if cls in old_ckpt.get("fp_results", {}):
            new_ckpt.setdefault("fp_results", {})[cls] = old_ckpt["fp_results"][cls]
            n_fp += 1
    print(f"  Merged {n_fp} fp_results entries")

    # Merge fp_results_v2 if present
    n_fp2 = 0
    for cls in sorted(only_old):
        if cls in old_ckpt.get("fp_results_v2", {}):
            new_ckpt.setdefault("fp_results_v2", {})[cls] = old_ckpt["fp_results_v2"][cls]
            n_fp2 += 1
    if n_fp2:
        print(f"  Merged {n_fp2} fp_results_v2 entries")

    # Merge eval_results if present
    n_eval = 0
    for cls in sorted(only_old):
        if cls in old_ckpt.get("eval_results", {}):
            new_ckpt.setdefault("eval_results", {})[cls] = old_ckpt["eval_results"][cls]
            n_eval += 1
    if n_eval:
        print(f"  Merged {n_eval} eval_results entries")

    final_train = set(new_ckpt.get("train_results", {}))
    print(f"\nFinal checkpoint classes: {len(final_train)}")

    # Atomic write
    print(f"\nWriting merged checkpoint: {args.output}")
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(args.output)), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(new_ckpt, f, ensure_ascii=False)
        os.replace(tmp, args.output)
    except BaseException:
        os.unlink(tmp)
        raise

    print("Done.")


if __name__ == "__main__":
    main()
