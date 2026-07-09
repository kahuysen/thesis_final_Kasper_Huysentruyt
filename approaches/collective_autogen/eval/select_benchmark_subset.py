"""Pick a balanced N-image subset from `eval/benchmark/manifest.json`.

Stratifies on (source_zip, source_gt) and reaction-count tier (low / mid /
high) so the subset spans figure types and complexity. Deterministic via
`--seed` (default 42).

Output: `eval/benchmark/manifest_subset_{N}.json` — same shape as the parent
manifest, filtered to the selected stems.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval" / "benchmark" / "manifest.json"


def _tier(n: int) -> str:
    if n <= 2:
        return "low"
    if n <= 8:
        return "mid"
    return "high"


# Balanced quota per source — sums to 30 by default.
DEFAULT_QUOTAS = {
    ("r_group_resolution_diagrams.zip", "GT1.json"): 9,
    ("article.zip", "GT2.json"): 8,
    ("review.zip", "GT1.json"): 6,
    ("article.zip", "GT3.json"): 4,
    ("review.zip", "GT2.json"): 3,
}


def select(manifest: list[dict], total: int, seed: int) -> list[dict]:
    quotas = dict(DEFAULT_QUOTAS)
    if total != sum(quotas.values()):
        # Rescale proportionally if the user asks for a different total.
        scale = total / sum(quotas.values())
        quotas = {k: max(1, round(v * scale)) for k, v in quotas.items()}

    valid = [m for m in manifest if m.get("valid")]
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in valid:
        by_source[(m["source_zip"], m["source_gt"])].append(m)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for src, n in quotas.items():
        pool = by_source.get(src, [])
        if not pool:
            continue
        # Stratify by reaction-count tier so we don't over-sample one complexity.
        by_tier: dict[str, list[dict]] = defaultdict(list)
        for m in pool:
            by_tier[_tier(m.get("reactions", 0))].append(m)
        # Round-robin pick from tiers (shuffled within each).
        for buckets in by_tier.values():
            rng.shuffle(buckets)
        tiers = [t for t in ("low", "mid", "high") if by_tier[t]]
        i = 0
        picked: list[dict] = []
        while len(picked) < n and tiers:
            t = tiers[i % len(tiers)]
            if by_tier[t]:
                picked.append(by_tier[t].pop())
            else:
                tiers.remove(t)
                continue
            i += 1
        chosen.extend(picked)

    chosen.sort(key=lambda m: m["stem"])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Subset size (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    chosen = select(manifest, args.n, args.seed)

    out = args.manifest.parent / f"manifest_subset_{args.n}.json"
    out.write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stats.
    from collections import Counter
    by_src = Counter((m["source_zip"], m["source_gt"]) for m in chosen)
    by_tier = Counter(_tier(m.get("reactions", 0)) for m in chosen)
    print(f"Selected {len(chosen)} images (seed={args.seed}):\n")
    for src, n in by_src.most_common():
        print(f"  {src[0]}/{src[1]:<14s}  {n}")
    print(f"\nReaction-count tiers: low(<=2)={by_tier['low']}  mid(3-8)={by_tier['mid']}  high(>8)={by_tier['high']}")
    print(f"\nWrote: {out.relative_to(REPO)}")
    print("\nFirst few:")
    for m in chosen[:6]:
        print(f"  {m['stem']:60s}  rxns={m['reactions']:>3}  ({m['source_zip']}/{m['source_gt']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
