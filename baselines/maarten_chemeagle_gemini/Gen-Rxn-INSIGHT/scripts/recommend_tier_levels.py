#!/usr/bin/env python
"""Recommend optimal tier levels for SMIRKS generalization.

Analyzes a classification database to determine whether each tier_3 class
should be kept at tier_3 or split into finer subclasses (tier_4, tier_5, ...)
based on template coverage homogeneity.

Outputs:
  - CSV report with metrics per class at every evaluated tier
  - JSON class list directly consumable by generalize_smirks_validated.py

Standalone script — no gen-rxn-insight dependency.
Requires: pandas, argparse, json, logging, collections.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (copied from generalize_smirks_validated.py)
# ---------------------------------------------------------------------------

def _parent_class(cls: str) -> str | None:
    """Derive the parent class code by dropping the last dotted segment.

    Examples: ``"1.4.2"`` -> ``"1.4"``, ``"1.4"`` -> ``"1"``, ``"1"`` -> None.
    """
    clean = cls.replace("CONFLICT:", "")
    parts = clean.split(".")
    if len(parts) <= 1:
        return None
    return ".".join(parts[:-1])


def _class_ancestors(cls: str) -> list[str]:
    """Return all ancestor prefixes for a class code, shortest first.

    Example: ``"1.4.2.3"`` -> ``["1", "1.4", "1.4.2", "1.4.2.3"]``.
    """
    clean = cls.replace("CONFLICT:", "")
    parts = clean.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def tier_depth(cls: str) -> int:
    """Return the tier depth of a class code (number of dot-separated segments).

    Examples: ``"1.3.1"`` -> 3, ``"5.1.1.3"`` -> 4.
    """
    return cls.replace("CONFLICT:", "").count(".") + 1


def tier_col_for_class(cls: str) -> str:
    """Return the DataFrame column name for a class code's tier depth.

    Examples: ``"1.3.1"`` -> ``"tier_3"``, ``"5.1.1.3"`` -> ``"tier_4"``.
    """
    return f"tier_{tier_depth(cls)}"


def detect_finest_tier(df: pd.DataFrame) -> str:
    """Find the deepest ``tier_N`` column in the DataFrame.

    Raises ``ValueError`` if no ``tier_N`` columns exist.
    """
    tier_cols = [
        c for c in df.columns
        if c.startswith("tier_") and c[5:].isdigit()
    ]
    if not tier_cols:
        raise ValueError("No tier_N columns found in DataFrame")
    return max(tier_cols, key=lambda c: int(c.split("_")[1]))


def _resolve_class_name(cls: str, named_dict: dict[str, str]) -> str:
    """Build a human-readable hierarchical name for a class code."""
    ancestors = _class_ancestors(cls)
    hier_names = []
    for anc in ancestors:
        if anc.count(".") >= 1:  # skip tier 1
            anc_name = named_dict.get(anc, "")
            if anc_name:
                hier_names.append(anc_name)
    if hier_names:
        return f"{cls} - {': '.join(hier_names)}"
    return f"{cls} - {named_dict.get(cls, cls)}"


# ---------------------------------------------------------------------------
# Pre-computed index for fast lookups
# ---------------------------------------------------------------------------

class ClassIndex:
    """Pre-grouped template counts and parent-child relationships.

    Avoids repeated full-DataFrame scans during recursive analysis.
    """

    def __init__(self, df: pd.DataFrame, template_col: str) -> None:
        tier_cols = sorted(
            (c for c in df.columns if c.startswith("tier_") and c[5:].isdigit()),
            key=lambda c: int(c.split("_")[1]),
        )
        # template_counts[(tier_col, cls)] -> Counter of template strings
        self.template_counts: dict[tuple[str, str], Counter] = {}
        # child_sizes[(tier_col, cls)] -> {child_code: count}
        self.child_sizes: dict[tuple[str, str], dict[str, int]] = {}

        for i, tcol in enumerate(tier_cols):
            next_tcol = tier_cols[i + 1] if i + 1 < len(tier_cols) else None
            grouped = df.groupby(tcol, observed=True)

            for cls_code, grp in grouped:
                if not isinstance(cls_code, str) or not cls_code.strip():
                    continue
                self.template_counts[(tcol, cls_code)] = Counter(
                    grp[template_col]
                )
                if next_tcol is not None:
                    children = grp[next_tcol].dropna()
                    if len(children) > 0:
                        self.child_sizes[(tcol, cls_code)] = (
                            children.value_counts().to_dict()
                        )

    def get_template_counts(
        self, tier_col: str, cls: str,
    ) -> Counter:
        """Return template Counter for a class, or empty Counter."""
        return self.template_counts.get((tier_col, cls), Counter())

    def get_child_sizes(
        self, tier_col: str, cls: str,
    ) -> dict[str, int]:
        """Return {child_code: count} at the next tier."""
        return self.child_sizes.get((tier_col, cls), {})


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_class_metrics(
    idx: ClassIndex,
    cls: str,
    tier_col: str,
    top_n: int,
) -> dict:
    """Compute template-coverage metrics for a single class.

    Returns dict with: n_reactions, n_templates, top_n_coverage, n_subclasses,
    child_sizes.
    """
    counts = idx.get_template_counts(tier_col, cls)
    n_reactions = sum(counts.values())

    if n_reactions == 0:
        return {
            "n_reactions": 0,
            "n_templates": 0,
            "top_n_coverage": 0.0,
            "n_subclasses": 0,
            "child_sizes": {},
        }

    n_templates = len(counts)
    top_n_count = sum(c for _, c in counts.most_common(top_n))
    top_n_coverage = top_n_count / n_reactions

    child_sizes = idx.get_child_sizes(tier_col, cls)

    return {
        "n_reactions": n_reactions,
        "n_templates": n_templates,
        "top_n_coverage": top_n_coverage,
        "n_subclasses": len(child_sizes),
        "child_sizes": child_sizes,
    }


def compute_weighted_subclass_coverage(
    idx: ClassIndex,
    cls: str,
    tier_col: str,
    top_n: int,
    min_class_size: int,
) -> dict:
    """Compute weighted average subclass coverage and coverage gain.

    Returns dict with: parent_coverage, weighted_avg_subclass_coverage,
    coverage_gain, viable_subclasses, merge_up_subclasses.
    """
    parent_metrics = compute_class_metrics(idx, cls, tier_col, top_n)
    parent_coverage = parent_metrics["top_n_coverage"]
    child_sizes = parent_metrics["child_sizes"]

    if not child_sizes:
        return {
            "parent_coverage": parent_coverage,
            "weighted_avg_subclass_coverage": parent_coverage,
            "coverage_gain": 0.0,
            "viable_subclasses": [],
            "merge_up_subclasses": [],
        }

    current_depth = int(tier_col.split("_")[1])
    next_tier_col = f"tier_{current_depth + 1}"

    viable = []
    merge_up = []
    weighted_sum = 0.0
    weighted_total = 0

    for child_cls, child_n in child_sizes.items():
        # Skip self-referencing children (same code repeated in a deeper
        # tier column, e.g. tier_5 == "1.2.1.5" under tier_4 == "1.2.1.5")
        if child_cls == cls:
            continue
        if child_n < min_class_size:
            merge_up.append(child_cls)
            continue
        viable.append(child_cls)
        child_metrics = compute_class_metrics(idx, child_cls, next_tier_col, top_n)
        weighted_sum += child_metrics["top_n_coverage"] * child_n
        weighted_total += child_n

    # Include merge-up reactions in the weighting at parent coverage
    merge_up_n = sum(child_sizes[c] for c in merge_up)
    weighted_sum += parent_coverage * merge_up_n
    weighted_total += merge_up_n

    if weighted_total > 0:
        weighted_avg = weighted_sum / weighted_total
    else:
        weighted_avg = parent_coverage

    return {
        "parent_coverage": parent_coverage,
        "weighted_avg_subclass_coverage": weighted_avg,
        "coverage_gain": weighted_avg - parent_coverage,
        "viable_subclasses": viable,
        "merge_up_subclasses": merge_up,
    }


# ---------------------------------------------------------------------------
# Recursive decision tree
# ---------------------------------------------------------------------------

def recommend_tier_for_class(
    idx: ClassIndex,
    cls: str,
    coverage_threshold: float,
    min_class_size: int,
    top_n: int,
    max_tier: int,
    *,
    _source_depth: int | None = None,
) -> dict:
    """Recommend the optimal tier level for a class (recursive).

    Returns dict with: class_code, recommended_classes, merge_up, reason.

    ``_source_depth`` overrides the tier depth derived from the class code
    (needed when the tier column contains codes with mismatched depth).
    """
    depth = _source_depth if _source_depth is not None else tier_depth(cls)
    tier_col = f"tier_{depth}"
    metrics = compute_class_metrics(idx, cls, tier_col, top_n)
    coverage = metrics["top_n_coverage"]

    result: dict = {
        "class_code": cls,
        "recommended_classes": [],
        "merge_up": {},
        "reason": "",
    }

    # Rule 1: already high coverage — keep at this tier
    if coverage >= coverage_threshold:
        result["recommended_classes"] = [cls]
        result["reason"] = (
            f"coverage {coverage:.1%} >= threshold {coverage_threshold:.0%}"
        )
        return result

    # Rule 2: at maximum allowed depth — keep (can't go deeper)
    if depth >= max_tier:
        result["recommended_classes"] = [cls]
        result["reason"] = f"at max tier depth {depth}"
        return result

    # Rule 3: only 1 subclass — splitting adds nothing
    sub_info = compute_weighted_subclass_coverage(
        idx, cls, tier_col, top_n, min_class_size,
    )
    n_viable = len(sub_info["viable_subclasses"])

    if metrics["n_subclasses"] <= 1:
        # Check if there's a deeper tier we can skip to
        # (e.g., tier_3 has 1 tier_4 child, but that child has multiple tier_5)
        if metrics["n_subclasses"] == 1:
            only_child = list(metrics["child_sizes"].keys())[0]
            child_result = recommend_tier_for_class(
                idx, only_child, coverage_threshold, min_class_size,
                top_n, max_tier,
                _source_depth=depth + 1,
            )
            # If the child recommends itself (i.e., stays), keep parent instead
            if child_result["recommended_classes"] == [only_child]:
                result["recommended_classes"] = [cls]
                result["reason"] = (
                    "only 1 subclass — no split benefit"
                )
                return result
            # Otherwise, propagate child's recommendation
            result["recommended_classes"] = child_result["recommended_classes"]
            result["merge_up"] = child_result["merge_up"]
            result["reason"] = (
                f"only 1 subclass {only_child}, propagating deeper split"
            )
            return result

        result["recommended_classes"] = [cls]
        result["reason"] = "no subclasses"
        return result

    # Rule 4: coverage gain too small — not worth splitting
    coverage_gain = sub_info["coverage_gain"]
    if coverage_gain < 0.05:
        result["recommended_classes"] = [cls]
        result["reason"] = (
            f"coverage gain {coverage_gain:.3f} < 0.05 — split not worthwhile"
        )
        return result

    # Rule 5: split into viable subclasses, recurse each
    recommended = []
    merge_up_all: dict[str, str] = {}

    for child_cls in sub_info["viable_subclasses"]:
        child_result = recommend_tier_for_class(
            idx, child_cls, coverage_threshold, min_class_size,
            top_n, max_tier,
            _source_depth=depth + 1,
        )
        recommended.extend(child_result["recommended_classes"])
        merge_up_all.update(child_result["merge_up"])

    # Small subclasses are simply dropped — too few reactions to produce
    # useful SMIRKS, and adding the parent would overlap with viable children.
    for mu_cls in sub_info["merge_up_subclasses"]:
        merge_up_all[mu_cls] = cls
    n_dropped = len(sub_info["merge_up_subclasses"])
    dropped_n_rxns = sum(
        idx.get_template_counts(f"tier_{depth + 1}", mc).total()
        for mc in sub_info["merge_up_subclasses"]
    )

    result["recommended_classes"] = recommended
    result["merge_up"] = merge_up_all
    result["reason"] = (
        f"split: coverage gain {coverage_gain:.3f}, "
        f"{n_viable} viable"
        + (f" + {n_dropped} dropped ({dropped_n_rxns} rxns)" if n_dropped else "")
    )
    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def _recommended_tier_label(
    recommended_classes: list[str],
    fallback: str,
) -> str:
    """Derive a tier label from the list of recommended classes."""
    if not recommended_classes:
        return fallback
    depths = {tier_depth(c) for c in recommended_classes}
    if len(depths) == 1:
        return f"tier_{depths.pop()}"
    return f"tier_{min(depths)}-{max(depths)}"

def build_full_report(
    df: pd.DataFrame,
    named_dict: dict[str, str],
    coverage_threshold: float,
    min_class_size: int,
    top_n: int,
    max_tier: int,
    template_col: str,
) -> tuple[pd.DataFrame, dict]:
    """Build the complete tier-level recommendation report.

    Returns (report_df, classes_json).
    """
    if "tier_3" not in df.columns:
        raise ValueError("DataFrame must have a 'tier_3' column")

    # Build pre-computed index for fast lookups
    logger.info("Building class index...")
    idx = ClassIndex(df, template_col)

    tier3_classes = sorted(
        c for c in df["tier_3"].dropna().unique() if c.strip()
    )
    logger.info("Analyzing %d tier_3 classes...", len(tier3_classes))

    all_recommended: list[str] = []
    all_merge_up: dict[str, str] = {}
    results_by_t3: dict[str, dict] = {}

    for cls in tier3_classes:
        result = recommend_tier_for_class(
            idx, cls, coverage_threshold, min_class_size,
            top_n, max_tier,
        )
        all_recommended.extend(result["recommended_classes"])
        all_merge_up.update(result["merge_up"])
        results_by_t3[cls] = result

    # Deduplicate preserving order
    all_recommended_unique = list(dict.fromkeys(all_recommended))
    all_recommended_set = set(all_recommended_unique)

    # Build report rows
    report_rows: list[dict] = []

    for cls in tier3_classes:
        result = results_by_t3[cls]
        tier_col = "tier_3"
        metrics = compute_class_metrics(idx, cls, tier_col, top_n)
        sub_info = compute_weighted_subclass_coverage(
            idx, cls, tier_col, top_n, min_class_size,
        )

        report_rows.append({
            "class_code": cls,
            "class_name": _resolve_class_name(cls, named_dict),
            "tier": tier_col,
            "n_reactions": metrics["n_reactions"],
            "n_templates": metrics["n_templates"],
            "top_n_coverage": round(metrics["top_n_coverage"], 4),
            "n_subclasses": metrics["n_subclasses"],
            "n_viable_subclasses": len(sub_info["viable_subclasses"]),
            "coverage_gain": round(sub_info["coverage_gain"], 4),
            "recommended_tier": _recommended_tier_label(
                result["recommended_classes"], tier_col,
            ),
            "merge_up": False,
            "recommended_for_processing": cls in all_recommended_set,
            "reason": result["reason"],
        })

        # Emit rows for deeper recommended classes
        for rec_cls in result["recommended_classes"]:
            if rec_cls == cls:
                continue
            rec_tier = tier_col_for_class(rec_cls)
            rec_metrics = compute_class_metrics(idx, rec_cls, rec_tier, top_n)
            rec_sub = compute_weighted_subclass_coverage(
                idx, rec_cls, rec_tier, top_n, min_class_size,
            )
            report_rows.append({
                "class_code": rec_cls,
                "class_name": _resolve_class_name(rec_cls, named_dict),
                "tier": rec_tier,
                "n_reactions": rec_metrics["n_reactions"],
                "n_templates": rec_metrics["n_templates"],
                "top_n_coverage": round(rec_metrics["top_n_coverage"], 4),
                "n_subclasses": rec_metrics["n_subclasses"],
                "n_viable_subclasses": len(rec_sub["viable_subclasses"]),
                "coverage_gain": round(rec_sub["coverage_gain"], 4),
                "recommended_tier": rec_tier,
                "merge_up": rec_cls in all_merge_up,
                "recommended_for_processing": rec_cls in all_recommended_set,
                "reason": "",
            })

        # Emit rows for dropped small subclasses
        for mu_cls in result["merge_up"]:
            mu_tier = tier_col_for_class(mu_cls)
            mu_metrics = compute_class_metrics(idx, mu_cls, mu_tier, top_n)
            report_rows.append({
                "class_code": mu_cls,
                "class_name": _resolve_class_name(mu_cls, named_dict),
                "tier": mu_tier,
                "n_reactions": mu_metrics["n_reactions"],
                "n_templates": mu_metrics["n_templates"],
                "top_n_coverage": round(mu_metrics["top_n_coverage"], 4),
                "n_subclasses": mu_metrics["n_subclasses"],
                "n_viable_subclasses": 0,
                "coverage_gain": 0.0,
                "recommended_tier": "",
                "merge_up": True,
                "recommended_for_processing": False,
                "reason": f"dropped (< {min_class_size} reactions)",
            })

    # Organise by tier
    classes_by_tier: dict[str, list[str]] = {}
    for cls in all_recommended_unique:
        tcol = tier_col_for_class(cls)
        classes_by_tier.setdefault(tcol, []).append(cls)

    classes_json = {
        "metadata": {
            "coverage_threshold": coverage_threshold,
            "min_class_size": min_class_size,
            "top_n": top_n,
            "max_tier": max_tier,
            "template_col": template_col,
        },
        "classes_by_tier": classes_by_tier,
        "dropped": list(all_merge_up.keys()),
        "all_classes": all_recommended_unique,
    }

    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        report_df = report_df.sort_values(["class_code"]).reset_index(drop=True)

    # Coverage check
    _verify_coverage(
        df, all_recommended_unique, list(all_merge_up.keys()), template_col,
    )

    return report_df, classes_json



def _verify_coverage(
    df: pd.DataFrame,
    all_classes: list[str],
    dropped: list[str],
    template_col: str,
) -> None:
    """Log coverage statistics for recommended classes."""
    covered = pd.Series(False, index=df.index)

    for cls in all_classes:
        tier_col = tier_col_for_class(cls)
        if tier_col in df.columns:
            covered |= df[tier_col] == cls

    n_total = len(df)
    n_covered = int(covered.sum())
    n_gap = n_total - n_covered

    logger.info(
        "Coverage: %d / %d reactions (%.1f%%). "
        "%d dropped small subclasses, %d uncovered reactions.",
        n_covered, n_total, 100 * n_covered / n_total,
        len(dropped), n_gap,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Recommend optimal tier levels for SMIRKS generalization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--database", required=True,
        help="Path to classification database (parquet/csv).",
    )
    p.add_argument(
        "--mapping", required=True,
        help="Path to structured_mapping.json (class code -> name).",
    )
    p.add_argument(
        "--output", default="tier_recommendations",
        help="Output prefix. Produces {output}_report.csv and {output}_classes.json.",
    )
    p.add_argument(
        "--coverage-threshold", type=float, default=0.9,
        help="Top-N coverage threshold to keep a class at its current tier.",
    )
    p.add_argument(
        "--min-class-size", type=int, default=5,
        help="Minimum reactions for a subclass to be viable (smaller ones merge up).",
    )
    p.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top templates to consider for coverage.",
    )
    p.add_argument(
        "--max-tier", type=int, default=0,
        help="Maximum tier depth to consider (0 = auto-detect from data).",
    )
    p.add_argument(
        "--template-col", default="TEMPLATE_rr0rp1_ring0",
        help="Column name for reaction templates.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load database
    db_path = Path(args.database)
    logger.info("Loading database from %s ...", db_path)
    if db_path.suffix == ".parquet":
        df = pd.read_parquet(db_path)
    elif db_path.suffix == ".csv":
        df = pd.read_csv(db_path)
    else:
        raise ValueError(f"Unsupported file format: {db_path.suffix}")
    logger.info("Loaded %d reactions.", len(df))

    # Validate template column
    if args.template_col not in df.columns:
        raise ValueError(
            f"Template column '{args.template_col}' not found. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Load mapping
    logger.info("Loading mapping from %s ...", args.mapping)
    with open(args.mapping, encoding="utf-8") as f:
        named_dict = json.load(f)

    # Determine max tier
    finest = detect_finest_tier(df)
    finest_depth = int(finest.split("_")[1])
    max_tier = args.max_tier if args.max_tier > 0 else finest_depth
    logger.info(
        "Finest tier in data: %s (depth %d). Max tier for analysis: %d.",
        finest, finest_depth, max_tier,
    )

    # Build report
    report_df, classes_json = build_full_report(
        df,
        named_dict,
        coverage_threshold=args.coverage_threshold,
        min_class_size=args.min_class_size,
        top_n=args.top_n,
        max_tier=max_tier,
        template_col=args.template_col,
    )

    # Save outputs
    report_path = f"{args.output}_report.csv"
    json_path = f"{args.output}_classes.json"

    report_df.to_csv(report_path, index=False)
    logger.info("Report saved to %s (%d rows).", report_path, len(report_df))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(classes_json, f, indent=2, ensure_ascii=False)
    logger.info("Class list saved to %s.", json_path)

    # Summary
    cbt = classes_json["classes_by_tier"]
    total_classes = len(classes_json["all_classes"])
    logger.info("--- Summary ---")
    for tier_name, cls_list in sorted(cbt.items()):
        logger.info("  %s: %d classes", tier_name, len(cls_list))
    logger.info("  Total: %d classes", total_classes)
    logger.info("  Dropped small subclasses: %d", len(classes_json["dropped"]))


if __name__ == "__main__":
    main()
