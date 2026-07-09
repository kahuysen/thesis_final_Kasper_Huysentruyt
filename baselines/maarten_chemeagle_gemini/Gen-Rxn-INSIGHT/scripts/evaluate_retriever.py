"""
DRFP + FAISS template retriever - evaluation across 4 splitting strategies.

Splits
------
random      Simple random 90/10 train/test split.
stratified  Stratified by TEMPLATE - proportional class coverage in test.
template    Cold-template: 20% of unique templates are unseen at train time.
            Tests whether DRFP similarity generalises to new reaction types.
patent      Cold-patent: 20% of unique REF values unseen at train time.
            Tests generalisation across literature sources.

Metrics (per split, per k in --k)
----------------------------------
top_k_acc        Fraction of test reactions where the ground-truth TEMPLATE
                 string appears in the top-k retrieved neighbours.
                 (Always 0 for the template split - GT is never in training.)
mrr              Mean Reciprocal Rank of GT template (0 if not found in k_max).
coverage         Fraction of test reactions where predict() finds a
                 product-correct template (via measure_template_accuracy),
                 evaluated on --coverage-n sampled reactions.
mean_rank_found  Mean neighbour rank when a correct template is found.

HPC usage
---------
python evaluate_retriever.py \\
    --data /path/to/classification_database.parquet \\
    --output results/ \\
    --n-jobs 32 \\
    --k 1 5 10 20 50 \\
    --coverage-n 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import gen_rxn_insight as ri
from gen_rxn_insight.retrieval import _encode

log = logging.getLogger(__name__)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate DRFP+FAISS retriever across 4 splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=r"/kyukon/data/gent/vo/000/gvo00008/vsc43212/models/classification_database.parquet",
    )
    p.add_argument("--output", default="results/", help="Directory for output files.")
    p.add_argument("--n-jobs", type=int, default=8, help="CPU cores for DRFP encoding.")
    p.add_argument("--k", nargs="+", type=int, default=[1, 5, 10, 20, 50],
                   help="k values for top-k accuracy.")
    p.add_argument("--test-size", type=float, default=0.1,
                   help="Test fraction for random/stratified splits.")
    p.add_argument("--cold-size", type=float, default=0.2,
                   help="Held-out fraction for template/patent splits.")
    p.add_argument("--coverage-n", type=int, default=2000,
                   help="Reactions sampled for coverage evaluation (slow).")
    p.add_argument("--seed", type=int, default=210995)
    p.add_argument("--splits", nargs="+",
                   default=["random", "stratified", "template", "patent"],
                   choices=["random", "stratified", "template", "patent"],
                   help="Which splits to evaluate.")
    p.add_argument("--n-bits", type=int, default=2048, help="DRFP fingerprint bits.")
    p.add_argument("--min-template-count", type=int, default=10,
                   help="Drop templates with fewer than this many reactions. "
                        "Templates with <10 occurrences are too rare to train "
                        "or evaluate a classifier meaningfully.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ── split functions ───────────────────────────────────────────────────────────

def random_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simple random split. Returns (train_indices, test_indices)."""
    idx = np.arange(len(df))
    return train_test_split(idx, test_size=test_size, random_state=seed)


def stratified_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified by TEMPLATE.

    sklearn requires each class to have at least ceil(1/test_size) samples
    so that the test set can contain >= 1 sample per class.  Templates below
    this threshold are always assigned to train.
    """
    counts = df["TEMPLATE"].value_counts()
    # Minimum occurrences needed: class_count * test_size >= 1
    min_count = int(np.ceil(1.0 / test_size))

    rare_mask = df["TEMPLATE"].isin(counts[counts < min_count].index)
    rare_idx  = np.where(rare_mask.values)[0]
    freq_idx  = np.where(~rare_mask.values)[0]

    log.info(
        f"  Stratified split (min_count={min_count}): "
        f"{len(freq_idx):,} reactions eligible for stratification, "
        f"{len(rare_idx):,} rare reactions forced to train"
    )

    labels = df.iloc[freq_idx]["TEMPLATE"].values
    train_freq, test_freq = train_test_split(
        freq_idx, test_size=test_size, stratify=labels, random_state=seed
    )
    train_idx = np.concatenate([train_freq, rare_idx])
    return train_idx, test_freq


def template_split(
    df: pd.DataFrame, cold_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cold-template split: a fraction of unique templates is entirely held out.

    Ground-truth templates in the test set are NEVER in the training set.
    top_k_acc will therefore be 0 by design; only coverage is meaningful.
    """
    unique_templates = df["TEMPLATE"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_templates)

    n_test = max(1, int(len(unique_templates) * cold_size))
    test_templates = set(unique_templates[:n_test])

    test_mask = df["TEMPLATE"].isin(test_templates).values
    train_idx = np.where(~test_mask)[0]
    test_idx  = np.where(test_mask)[0]

    log.info(
        f"  template split: {len(unique_templates) - n_test:,} train templates, "
        f"{n_test:,} test templates"
    )
    return train_idx, test_idx


def patent_split(
    df: pd.DataFrame, cold_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cold-patent split: a fraction of unique REF values is entirely held out."""
    unique_refs = df["REF"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_refs)

    n_test = max(1, int(len(unique_refs) * cold_size))
    test_refs = set(unique_refs[:n_test])

    test_mask = df["REF"].isin(test_refs).values
    train_idx = np.where(~test_mask)[0]
    test_idx  = np.where(test_mask)[0]

    log.info(
        f"  patent split: {len(unique_refs) - n_test:,} train patents, "
        f"{n_test:,} test patents"
    )
    return train_idx, test_idx


# ── core evaluation ───────────────────────────────────────────────────────────

def _compute_ranks(
    test_templates: list[str],
    all_neighbor_ids: np.ndarray,
    train_templates: list[str],
    k_max: int,
) -> np.ndarray:
    """Return array of shape (n_test,) with 1-based rank of GT template.

    Rank = inf if GT template not found within k_max neighbours.
    """
    ranks = np.full(len(test_templates), np.inf)
    for i, gt in enumerate(tqdm(test_templates, desc="  Computing ranks", leave=False)):
        for rank, j in enumerate(all_neighbor_ids[i]):
            if train_templates[j] == gt:
                ranks[i] = rank + 1
                break
    return ranks


def evaluate_split(
    split_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fps_all: np.ndarray,
    df: pd.DataFrame,
    k_values: list[int],
    coverage_n: int,
    n_jobs: int,
    output_dir: Path,
) -> dict:
    """Build retriever on train, evaluate on test, return metrics dict."""
    import faiss  # local import so the rest of the script works without faiss

    k_max = max(k_values)

    train_reactions = df.iloc[train_idx]["SANITIZED_REACTION"].tolist()
    test_reactions  = df.iloc[test_idx]["SANITIZED_REACTION"].tolist()
    train_templates = df.iloc[train_idx]["TEMPLATE"].tolist()
    test_templates  = df.iloc[test_idx]["TEMPLATE"].tolist()

    n_train, n_test = len(train_idx), len(test_idx)
    log.info(f"  train={n_train:,}  test={n_test:,}")

    # -- build FAISS index ----------------------------------------------------
    n_bits = fps_all.shape[1] * 8
    t0 = time.perf_counter()
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(fps_all[train_idx])
    build_time = time.perf_counter() - t0
    log.info(f"  FAISS index built in {build_time:.1f}s")

    # -- search ---------------------------------------------------------------
    t0 = time.perf_counter()
    _, all_neighbor_ids = index.search(fps_all[test_idx], k_max)
    search_time = time.perf_counter() - t0
    log.info(f"  FAISS search ({n_test:,} queries, k={k_max}) in {search_time:.2f}s")

    # -- top-k accuracy + MRR -------------------------------------------------
    ranks = _compute_ranks(test_templates, all_neighbor_ids, train_templates, k_max)

    metrics: dict = {
        "split": split_name,
        "n_train": n_train,
        "n_test": n_test,
        "build_time_s": round(build_time, 2),
        "search_time_s": round(search_time, 2),
    }
    for k in k_values:
        metrics[f"top_{k}_acc"] = round(float((ranks <= k).mean()), 4)

    finite_mask = np.isfinite(ranks)
    metrics["mrr"] = round(
        float(np.mean(1.0 / ranks[finite_mask])) if finite_mask.any() else 0.0, 4
    )
    metrics["found_in_kmax"] = round(float(finite_mask.mean()), 4)

    # -- coverage (product-correct via measure_template_accuracy) -------------
    rng = np.random.default_rng(42)
    sample_size = min(coverage_n, n_test)
    sample_local_idx = rng.choice(n_test, size=sample_size, replace=False)
    sample_reactions = [test_reactions[i] for i in sample_local_idx]

    # Wrap the already-built index in a TemplateRetriever for predict_batch
    retriever = ri.TemplateRetriever(n_bits=n_bits)
    retriever._index     = index
    retriever._templates = train_templates

    log.info(f"  Coverage eval on {sample_size:,} sampled reactions...")
    coverage_df = retriever.predict_batch(
        sample_reactions, k=k_max, n_jobs=n_jobs, progress=True
    )
    found = coverage_df["FOUND"]
    metrics["coverage"] = round(float(found.mean()), 4)
    metrics["mean_rank_when_found"] = round(
        float(coverage_df.loc[found, "NEIGHBOR_RANK"].mean()) if found.any() else float("nan"), 2
    )
    metrics["coverage_sample_n"] = sample_size

    # -- save per-split detail ------------------------------------------------
    detail_path = output_dir / f"detail_{split_name}.parquet"
    detail = pd.DataFrame({
        "SANITIZED_REACTION": sample_reactions,
        "GT_TEMPLATE":        [test_templates[i] for i in sample_local_idx],
        "PRED_TEMPLATE":      coverage_df["TEMPLATE"].values,
        "FOUND":              coverage_df["FOUND"].values,
        "NEIGHBOR_RANK":      coverage_df["NEIGHBOR_RANK"].values,
    })
    detail.to_parquet(detail_path, index=False)
    log.info(f"  Detail saved -> {detail_path}")

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "evaluate_retriever.log"),
        ],
    )

    log.info(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data)
    log.info(
        f"  {len(df):,} reactions, {df['TEMPLATE'].nunique():,} unique templates, "
        f"{df['REF'].nunique():,} unique REFs"
    )

    if args.min_template_count > 1:
        counts = df["TEMPLATE"].value_counts()
        keep = counts[counts >= args.min_template_count].index
        df = df[df["TEMPLATE"].isin(keep)].reset_index(drop=True)
        log.info(
            f"  After min_count>={args.min_template_count} filter: "
            f"{len(df):,} reactions, {df['TEMPLATE'].nunique():,} unique templates"
        )

    log.info(
        f"Encoding {len(df):,} reactions with DRFP "
        f"(n_bits={args.n_bits}, n_jobs={args.n_jobs})..."
    )
    t0 = time.perf_counter()
    fps_all = _encode(
        df["SANITIZED_REACTION"].tolist(),
        n_bits=args.n_bits,
        n_jobs=args.n_jobs,
        progress=True,
    )
    encode_time = time.perf_counter() - t0
    log.info(
        f"  Done in {encode_time/60:.1f} min - "
        f"fps shape {fps_all.shape}, {fps_all.nbytes / 1e6:.0f} MB"
    )

    fps_path = output_dir / "fps_all.npy"
    np.save(fps_path, fps_all)
    log.info(f"  Fingerprints saved -> {fps_path}")

    split_builders = {
        "random":     lambda: random_split(df, args.test_size, args.seed),
        "stratified": lambda: stratified_split(df, args.test_size, args.seed),
        "template":   lambda: template_split(df, args.cold_size, args.seed),
        "patent":     lambda: patent_split(df, args.cold_size, args.seed),
    }

    all_metrics = []

    for split_name in args.splits:
        log.info(f"\n{'='*60}")
        log.info(f"Split: {split_name.upper()}")
        log.info(f"{'='*60}")

        train_idx, test_idx = split_builders[split_name]()
        log.info(
            f"  Unique test templates: "
            f"{df.iloc[test_idx]['TEMPLATE'].nunique():,}"
        )

        metrics = evaluate_split(
            split_name=split_name,
            train_idx=train_idx,
            test_idx=test_idx,
            fps_all=fps_all,
            df=df,
            k_values=args.k,
            coverage_n=args.coverage_n,
            n_jobs=args.n_jobs,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

        log.info(f"\n  -- {split_name} results --")
        log.info(f"  [template-match, lower bound: a different but equally valid template counts as wrong]")
        for k in args.k:
            log.info(f"    top-{k:2d} template-match : {metrics[f'top_{k}_acc']:.1%}")
        log.info(f"    MRR (template-match)     : {metrics['mrr']:.4f}")
        log.info(f"    found@{max(args.k):2d} (tmpl-match)  : {metrics['found_in_kmax']:.1%}")
        log.info(f"  [coverage: TRUE metric - correct if template gives correct product]")
        log.info(
            f"    coverage@{max(args.k):2d}            : {metrics['coverage']:.1%}  "
            f"(mean rank when found: {metrics['mean_rank_when_found']})"
        )

    results_df = pd.DataFrame(all_metrics)
    results_path = output_dir / "metrics.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nMetrics saved -> {results_path}")

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, default=str)
    )

    log.info("\n" + "="*70)
    log.info("SUMMARY")
    log.info("="*70)
    display_cols = (
        ["split"] +
        [f"top_{k}_acc" for k in args.k] +
        ["mrr", "coverage", "mean_rank_when_found"]
    )
    log.info("\n" + results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
