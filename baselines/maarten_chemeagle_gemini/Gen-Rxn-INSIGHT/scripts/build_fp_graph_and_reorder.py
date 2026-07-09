"""Build a COMPLETE false-positive graph from the reaction database, then reorder.

Phase 3 only tested training-set reactions, producing an incomplete FP graph.
This script tests EVERY reaction against ALL SMIRKS to discover the full set
of cross-class FP relationships, then computes the optimal ordering.

For each reaction of true class B, every class A (≠B) whose SMIRKS fires
correctly produces an edge A → B (weighted by count).

Usage
-----
    python build_fp_graph_and_reorder.py \
        --smirks-db ordered_smirks_db.json \
        --reactions reaction_db.parquet \
        --checkpoint gemini_smirks_checkpoint.json \
        --output reordered_smirks_db.json \
        [--n-jobs 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

# Suppress RDKit warnings for cleaner output
RDLogger.logger().setLevel(RDLogger.ERROR)


# ── SMIRKS testing ───────────────────────────────────────────────────────────

_RXN_CACHE: dict[str, AllChem.ChemicalReaction | None] = {}


def _get_compiled_rxn(smirks: str):
    if smirks not in _RXN_CACHE:
        try:
            _RXN_CACHE[smirks] = AllChem.ReactionFromSmarts(smirks)
        except Exception:
            _RXN_CACHE[smirks] = None
    return _RXN_CACHE[smirks]


def _smirks_fires(
    react_mols: list,
    expected: set[str],
    smirks: str,
    nreact: int,
) -> bool:
    """Test if a SMIRKS fires correctly on pre-parsed reactants/products."""
    rxn_obj = _get_compiled_rxn(smirks)
    if rxn_obj is None:
        return False

    n = len(react_mols)
    if nreact > n:
        return False

    if nreact == n:
        tuples = (
            [tuple(react_mols)]
            if n == 1
            else list(itertools.permutations(react_mols))
        )
    else:
        tuples = []
        for subset in itertools.combinations(react_mols, nreact):
            if nreact == 1:
                tuples.append(subset)
            else:
                tuples.extend(itertools.permutations(subset))

    for tup in tuples:
        try:
            outcomes = rxn_obj.RunReactants(tup)
        except Exception:
            continue
        for prods in outcomes:
            try:
                smi = Chem.MolToSmiles(prods[0], isomericSmiles=False)
            except Exception:
                continue
            if smi in expected:
                return True
    return False


# ── Chunk worker: find ALL matching classes for each reaction ────────────────


def _find_all_matches_chunk(
    chunk: list[tuple[str, str]],  # (true_class, rxn_smi)
    smirks_records: list[tuple[str, str, int]],  # (class, smirks, nreact)
) -> list[tuple[str, str]]:
    """For each reaction, find all classes whose SMIRKS fire correctly.

    Returns list of (smirks_class, true_class) FP edges.
    """
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    edges: list[tuple[str, str]] = []

    for true_class, rxn_smi in chunk:
        parts = rxn_smi.split(">>")
        if len(parts) != 2:
            continue

        react_mols = []
        for s in parts[0].split("."):
            m = Chem.MolFromSmiles(s)
            if m is None:
                break
            react_mols.append(m)
        else:
            pass
        if len(react_mols) != len(parts[0].split(".")):
            continue
        if len(react_mols) > 4:
            continue

        expected: set[str] = set()
        for s in parts[1].split("."):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                expected.add(Chem.MolToSmiles(m, isomericSmiles=False))

        n_react = len(react_mols)

        for cls, smirks, nreact in smirks_records:
            # Skip own class (and parent/child)
            if (
                cls == true_class
                or true_class.startswith(cls + ".")
                or cls.startswith(true_class + ".")
            ):
                continue

            if nreact > n_react:
                continue

            if _smirks_fires(react_mols, expected, smirks, nreact):
                edges.append((cls, true_class))

    return edges


# ── Graph algorithms (reused from order_smirks_db.py) ────────────────────────


def tarjan_sccs(adj: dict[str, set[str]]) -> list[list[str]]:
    """Iterative Tarjan's SCC algorithm."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    nodes: set[str] = set(adj.keys())
    for v in adj:
        nodes.update(adj[v])

    for root in nodes:
        if root in index:
            continue
        call_stack: list[tuple[str, list[str], int]] = []
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        call_stack.append((root, list(adj.get(root, set())), 0))

        while call_stack:
            v, neighbours, ni = call_stack[-1]
            if ni < len(neighbours):
                call_stack[-1] = (v, neighbours, ni + 1)
                w = neighbours[ni]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, list(adj.get(w, set())), 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                if lowlink[v] == index[v]:
                    scc: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    sccs.append(scc)
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
    return sccs


def topo_sort_sccs(sccs, adj):
    node_to_scc = {}
    for i, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = i

    n_sccs = len(sccs)
    in_degree = defaultdict(int)
    out_adj = defaultdict(set)
    for a in adj:
        a_scc = node_to_scc.get(a)
        if a_scc is None:
            continue
        for b in adj[a]:
            b_scc = node_to_scc.get(b)
            if b_scc is None or b_scc == a_scc:
                continue
            out_adj[b_scc].add(a_scc)
            in_degree[a_scc] += 1

    queue = [i for i in range(n_sccs) if in_degree[i] == 0]
    order = []
    while queue:
        queue.sort(key=lambda x: -len(out_adj.get(x, set())))
        node = queue.pop(0)
        order.append(node)
        for succ in out_adj.get(node, set()):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) < n_sccs:
        order.extend(i for i in range(n_sccs) if i not in set(order))

    return [sccs[i] for i in order]


def greedy_order_within_scc(scc, edge_weight):
    if len(scc) <= 1:
        return list(scc)

    scc_set = set(scc)
    in_w = defaultdict(int)
    out_w = defaultdict(int)
    for a in scc:
        for b, w in edge_weight.get(a, {}).items():
            if b in scc_set:
                out_w[a] += w
                in_w[b] += w

    remaining = set(scc)
    order = []
    while remaining:
        best = max(remaining, key=lambda n: in_w[n] - out_w[n])
        order.append(best)
        remaining.remove(best)
        for b, w in edge_weight.get(best, {}).items():
            if b in remaining:
                in_w[b] -= w
        for a in list(remaining):
            w = edge_weight.get(a, {}).get(best, 0)
            if w:
                out_w[a] -= w

    return order


def compute_optimal_order(all_classes, adj, edge_weight):
    sccs = tarjan_sccs(adj)
    sorted_sccs = topo_sort_sccs(sccs, adj)
    ordered = []
    seen = set()
    for scc in sorted_sccs:
        if len(scc) == 1:
            ordered.append(scc[0])
        else:
            ordered.extend(greedy_order_within_scc(scc, edge_weight))
        seen.update(scc)
    ordered.extend(sorted(all_classes - seen))
    return ordered


def compute_stats(order, adj, edge_weight):
    pos = {cls: i for i, cls in enumerate(order)}
    total_fps = eliminated = remaining = 0
    for a in edge_weight:
        for b, w in edge_weight[a].items():
            total_fps += w
            if a not in pos or b not in pos:
                remaining += w
            elif pos[b] < pos[a]:
                eliminated += w
            else:
                remaining += w
    return {
        "total_fps": total_fps,
        "eliminated_fps": eliminated,
        "remaining_fps": remaining,
        "reduction_pct": eliminated / total_fps * 100 if total_fps else 100.0,
    }


# ── Determine true class for each reaction ──────────────────────────────────


def get_true_class(
    row: pd.Series,
    db_classes: set[str],
    tier_cols: list[str],
) -> str:
    """Find the finest tier that matches a class in the SMIRKS DB."""
    for col in reversed(tier_cols):
        val = row.get(col)
        if pd.notna(val) and str(val) in db_classes:
            return str(val)
    # Also check if a parent class matches
    for col in reversed(tier_cols):
        val = row.get(col)
        if pd.notna(val):
            val_s = str(val)
            if any(dc.startswith(val_s + ".") for dc in db_classes):
                return val_s
    return ""


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build complete FP graph and reorder SMIRKS DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smirks-db", required=True, help="Input SMIRKS DB (JSONL).")
    p.add_argument("--reactions", required=True, help="Reaction DB (parquet).")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint JSON (for class info from train_results).",
    )
    p.add_argument("--output", required=True, help="Output reordered SMIRKS DB.")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for quick testing.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    n_jobs = args.n_jobs or max(1, mp.cpu_count() // 2)

    # ── Load SMIRKS DB ───────────────────────────────────────────────────
    print(f"Loading SMIRKS DB: {args.smirks_db}")
    db_entries: list[dict] = []
    with open(args.smirks_db, encoding="utf-8") as f:
        for line in f:
            db_entries.append(json.loads(line))

    # Build records for workers: (class, smirks, nreact)
    smirks_records: list[tuple[str, str, int]] = []
    for e in db_entries:
        smirks = e["smirks"]
        nreact = len(smirks.split(">>")[0].split("."))
        smirks_records.append((e.get("class", ""), e["name"], nreact))
    # Fix: include the actual smirks string
    smirks_records = [
        (e.get("class", ""), e["smirks"], len(e["smirks"].split(">>")[0].split(".")))
        for e in db_entries
    ]

    db_classes = {r[0] for r in smirks_records if r[0]}
    print(f"  {len(db_entries)} SMIRKS, {len(db_classes)} classes")

    # ── Load reactions ───────────────────────────────────────────────────
    print(f"Loading reactions: {args.reactions}")
    df = pd.read_parquet(args.reactions)
    print(f"  {len(df):,} reactions")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
        df = df.reset_index(drop=True)
        print(f"  Sampled {len(df):,}")

    tier_cols = sorted(
        [c for c in df.columns if c.startswith("tier_") and c[5:].isdigit()],
        key=lambda c: int(c.split("_")[1]),
    )

    # ── Assign true class to each reaction ───────────────────────────────
    print("Assigning true classes...")
    reaction_data: list[tuple[str, str]] = []  # (true_class, rxn_smi)
    n_skip = 0
    for idx in tqdm(range(len(df)), desc="Assigning classes"):
        row = df.iloc[idx]
        rxn = row.get("SANITIZED_REACTION", "")
        if pd.isna(rxn) or not rxn:
            n_skip += 1
            continue
        true_cls = get_true_class(row, db_classes, tier_cols)
        if not true_cls:
            n_skip += 1
            continue
        reaction_data.append((true_cls, str(rxn)))

    print(f"  {len(reaction_data):,} reactions with known class, {n_skip:,} skipped")

    # ── Build complete FP graph ──────────────────────────────────────────
    print(f"Building complete FP graph with {n_jobs} workers...")
    print(f"  Testing {len(reaction_data):,} reactions × {len(smirks_records)} SMIRKS")
    t0 = time.time()

    chunk_size = max(20, len(reaction_data) // (n_jobs * 16))
    chunks = [
        reaction_data[i : i + chunk_size]
        for i in range(0, len(reaction_data), chunk_size)
    ]

    if n_jobs == 1:
        all_edges: list[tuple[str, str]] = []
        for chunk in tqdm(chunks, desc="FP discovery"):
            all_edges.extend(
                _find_all_matches_chunk(chunk, smirks_records)
            )
    else:
        raw = Parallel(n_jobs=n_jobs)(
            delayed(_find_all_matches_chunk)(chunk, smirks_records)
            for chunk in tqdm(chunks, desc="FP discovery")
        )
        all_edges = [e for batch in raw for e in batch]

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({len(reaction_data) / elapsed:.0f} rxn/s)")
    print(f"  Raw FP edges: {len(all_edges):,}")

    # ── Aggregate into weighted graph ────────────────────────────────────
    adj: dict[str, set[str]] = defaultdict(set)
    edge_weight: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for smirks_cls, true_cls in all_edges:
        adj[smirks_cls].add(true_cls)
        edge_weight[smirks_cls][true_cls] += 1

    # Convert to plain dicts
    adj = dict(adj)
    edge_weight = {k: dict(v) for k, v in edge_weight.items()}

    n_edges = sum(len(v) for v in adj.values())
    total_weight = sum(w for d in edge_weight.values() for w in d.values())
    print(f"  Unique FP edges (class pairs): {n_edges:,}")
    print(f"  Total FP weight: {total_weight:,}")

    # ── Compute optimal ordering ─────────────────────────────────────────
    print("Computing optimal ordering...")
    all_cls = set(r[0] for r in smirks_records if r[0])
    order = compute_optimal_order(all_cls, adj, edge_weight)

    stats = compute_stats(order, adj, edge_weight)
    print(f"\nFP elimination (on full graph):")
    print(f"  Total FPs:      {stats['total_fps']:>12,}")
    print(f"  Eliminated:     {stats['eliminated_fps']:>12,}")
    print(f"  Remaining:      {stats['remaining_fps']:>12,}")
    print(f"  Reduction:      {stats['reduction_pct']:>11.1f}%")

    # ── Write reordered SMIRKS DB ────────────────────────────────────────
    # Group entries by class
    entries_by_class: dict[str, list[dict]] = defaultdict(list)
    for e in db_entries:
        entries_by_class[e.get("class", "")].append(e)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for cls in order:
            for entry in entries_by_class.get(cls, []):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1
        # Append any classes not in the order
        for cls in sorted(entries_by_class.keys()):
            if cls not in set(order):
                for entry in entries_by_class[cls]:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"\nWrote {n_written} SMIRKS entries to {args.output}")

    # ── Save metadata ────────────────────────────────────────────────────
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta = {
        "n_classes": len(order),
        "n_smirks": n_written,
        "n_reactions_tested": len(reaction_data),
        "n_fp_edges": n_edges,
        "fp_stats": stats,
        "elapsed_seconds": round(elapsed, 1),
        "class_order": order,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {meta_path}")

    # ── Save FP graph for future use ─────────────────────────────────────
    fp_path = Path(args.output).with_suffix(".fp_graph.json")
    # Convert to serializable format
    fp_graph = {
        cls: {tc: w for tc, w in targets.items()}
        for cls, targets in edge_weight.items()
    }
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fp_graph, f, indent=2, ensure_ascii=False)
    print(f"FP graph saved to {fp_path}")


if __name__ == "__main__":
    main()
