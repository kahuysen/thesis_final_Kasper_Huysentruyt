"""Build an optimally-ordered SMIRKS database from a checkpoint file.

Because ``name_reaction`` returns the **first** matching SMIRKS, we can
eliminate almost all false positives simply by placing each class's SMIRKS
*after* the classes whose reactions it would incorrectly match.

Algorithm
---------
1. Extract SMIRKS from Phase-2 training results (checkpoint).
2. Build a directed FP graph from Phase-3 results: edge A → B means
   "A's SMIRKS fires correctly on B's reactions" (a false positive).
   For correct naming, B must appear **before** A so B's own SMIRKS
   matches first.
3. Condense the graph into its strongly-connected components (SCCs).
   The DAG of SCCs can be topologically sorted to eliminate all
   inter-SCC FPs.
4. Within each SCC (mutual FPs), use a greedy heuristic to minimise
   the weight of backward edges (minimum feedback arc set).
5. Write the ordered SMIRKS DB as JSONL (``{"name": ..., "smirks": ...}``),
   directly compatible with ``gen_rxn_insight.naming.name_reaction``.

Requires only the Python standard library + json.

Usage
-----
    python order_smirks_db.py \\
        --checkpoint gemini_smirks_checkpoint.json \\
        --output ordered_smirks_db.json \\
        [--use-refined]       # prefer Phase-4 refined SMIRKS where available
        [--mapping structured_mapping.json]  # optional: use class names
        [--stats]             # print FP elimination statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Graph utilities ──────────────────────────────────────────────────────────


def tarjan_sccs(adj: dict[str, set[str]]) -> list[list[str]]:
    """Find strongly-connected components (iterative Tarjan's algorithm).

    Returns a list of SCCs in reverse topological order (sinks first).
    """
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    # Collect all nodes
    nodes: set[str] = set(adj.keys())
    for v in adj:
        nodes.update(adj[v])

    # Iterative implementation to avoid recursion limit issues
    for root in nodes:
        if root in index:
            continue

        # DFS stack: (node, iterator_over_neighbours, is_initial_visit)
        call_stack: list[tuple[str, list[str], int]] = []
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        call_stack.append((root, list(adj.get(root, set())), 0))

        while call_stack:
            v, neighbours, ni = call_stack[-1]

            if ni < len(neighbours):
                # Advance the neighbour iterator
                call_stack[-1] = (v, neighbours, ni + 1)
                w = neighbours[ni]

                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append(
                        (w, list(adj.get(w, set())), 0)
                    )
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                # All neighbours processed
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


def topo_sort_sccs(
    sccs: list[list[str]],
    adj: dict[str, set[str]],
) -> list[list[str]]:
    """Topologically sort the DAG of SCCs (sources first).

    For naming, sources = classes that are NOT false-positively matched
    by others, so they should come first.  But our edge semantics are:
    A → B means "A fires on B" ⇒ B must come before A.

    So we want: nodes with **no incoming edges** in the condensed DAG
    should come **last** (they don't get matched by anyone else, so
    their position doesn't matter for FP elimination).  Conversely,
    nodes that many others fire on should come **first**.

    We reverse the condensed DAG and do a standard topological sort.
    """
    # Build SCC membership
    node_to_scc: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for n in scc:
            node_to_scc[n] = i

    # Build condensed DAG (edge A_scc → B_scc means A fires on B)
    # For ordering: B_scc must come before A_scc
    # Equivalently: edge B_scc ← A_scc in reverse graph, topo-sort reverse
    n_sccs = len(sccs)
    condensed_in: dict[int, set[int]] = defaultdict(set)  # reversed edges
    for a in adj:
        a_scc = node_to_scc.get(a)
        if a_scc is None:
            continue
        for b in adj[a]:
            b_scc = node_to_scc.get(b)
            if b_scc is None or b_scc == a_scc:
                continue
            # Original: A fires on B ⇒ B before A
            # In ordering terms: B_scc should precede A_scc
            # So A_scc depends on B_scc ⇒ edge A_scc → B_scc in dep graph
            condensed_in[a_scc].add(b_scc)

    # Kahn's algorithm on the dependency graph
    in_degree: dict[int, int] = defaultdict(int)
    out_adj: dict[int, set[int]] = defaultdict(set)
    for a_scc, deps in condensed_in.items():
        for b_scc in deps:
            out_adj[b_scc].add(a_scc)
            in_degree[a_scc] += 1

    queue: list[int] = []
    for i in range(n_sccs):
        if in_degree[i] == 0:
            queue.append(i)

    order: list[int] = []
    while queue:
        # Among nodes with in_degree 0, prefer those that are depended
        # on by many others (stabilise the sort)
        queue.sort(key=lambda x: -len(out_adj.get(x, set())))
        node = queue.pop(0)
        order.append(node)
        for succ in out_adj.get(node, set()):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # Handle cycles in the condensed DAG (shouldn't happen, but safety)
    if len(order) < n_sccs:
        remaining = [i for i in range(n_sccs) if i not in set(order)]
        order.extend(remaining)

    return [sccs[i] for i in order]


def greedy_order_within_scc(
    scc: list[str],
    edge_weight: dict[str, dict[str, int]],
) -> list[str]:
    """Order nodes within an SCC to minimise backward-edge FP weight.

    Greedy heuristic: iteratively pick the node with the highest
    (incoming_weight − outgoing_weight) among remaining nodes.
    That node benefits most from being placed early (many other
    classes fire on its reactions).
    """
    if len(scc) <= 1:
        return list(scc)

    scc_set = set(scc)

    # Compute initial in/out weights within the SCC
    in_w: dict[str, int] = defaultdict(int)
    out_w: dict[str, int] = defaultdict(int)
    for a in scc:
        for b, w in edge_weight.get(a, {}).items():
            if b in scc_set:
                out_w[a] += w
                in_w[b] += w

    remaining = set(scc)
    order: list[str] = []

    while remaining:
        # Pick node with max (in - out): it has many others firing on it,
        # so placing it early lets its own SMIRKS match first
        best = max(remaining, key=lambda n: in_w[n] - out_w[n])
        order.append(best)
        remaining.remove(best)

        # Update weights: edges involving `best` no longer matter
        for b, w in edge_weight.get(best, {}).items():
            if b in remaining:
                in_w[b] -= w
        for a in list(remaining):
            w = edge_weight.get(a, {}).get(best, 0)
            if w:
                out_w[a] -= w

    return order


# ── Main logic ───────────────────────────────────────────────────────────────


def load_checkpoint(path: str) -> dict:
    """Load checkpoint JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_smirks(
    ckpt: dict,
    use_refined: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Extract SMIRKS per class from checkpoint.

    Returns:
        ``{class_code: [{"name": ..., "smirks": ...}, ...]}``
    """
    train_results = ckpt.get("train_results", {})
    finetune_results = ckpt.get("finetune_results", {})

    result: dict[str, list[dict[str, str]]] = {}
    for cls, tr in train_results.items():
        if cls.startswith("CONFLICT:"):
            continue
        smirks_list = tr.get("smirks", [])
        if not smirks_list:
            continue

        # Optionally use refined SMIRKS
        if use_refined and cls in finetune_results:
            refined = finetune_results[cls].get("smirks_refined", [])
            if refined:
                smirks_list = refined

        name = tr.get("reaction_class", cls)
        result[cls] = [
            {"name": name, "smirks": s, "class": cls}
            for s in smirks_list
        ]

    return result


def build_fp_graph(
    ckpt: dict,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Build directed FP graph from Phase-3 results.

    Returns:
        ``(adjacency, edge_weights)`` where edge A → B means
        A's SMIRKS fires on B's reactions.
    """
    fp_results = ckpt.get("fp_results", {})
    adj: dict[str, set[str]] = defaultdict(set)
    weights: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for smirks_class, fps_list in fp_results.items():
        if smirks_class.startswith("CONFLICT:"):
            continue
        for fp_record in fps_list:
            true_class = fp_record["true_class"]
            if true_class.startswith("CONFLICT:"):
                continue
            adj[smirks_class].add(true_class)
            weights[smirks_class][true_class] += 1

    return dict(adj), {k: dict(v) for k, v in weights.items()}


def compute_optimal_order(
    all_classes: set[str],
    adj: dict[str, set[str]],
    edge_weight: dict[str, dict[str, int]],
) -> list[str]:
    """Compute an optimal class ordering to minimise FPs.

    Classes not involved in any FP edge are placed at the end
    (their position doesn't matter).
    """
    # Find SCCs
    sccs = tarjan_sccs(adj)

    # Topologically sort the SCC DAG
    sorted_sccs = topo_sort_sccs(sccs, adj)

    # Within each SCC, apply greedy min-FAS ordering
    ordered: list[str] = []
    seen: set[str] = set()
    for scc in sorted_sccs:
        if len(scc) == 1:
            ordered.append(scc[0])
        else:
            ordered.extend(greedy_order_within_scc(scc, edge_weight))
        seen.update(scc)

    # Append classes not in any FP edge (order doesn't matter)
    remaining = sorted(all_classes - seen)
    ordered.extend(remaining)

    return ordered


def compute_stats(
    order: list[str],
    adj: dict[str, set[str]],
    edge_weight: dict[str, dict[str, int]],
) -> dict[str, int]:
    """Compute FP statistics for a given ordering."""
    pos = {cls: i for i, cls in enumerate(order)}

    total_fps = 0
    eliminated_fps = 0
    remaining_fps = 0

    for a in edge_weight:
        for b, w in edge_weight[a].items():
            total_fps += w
            if a not in pos or b not in pos:
                remaining_fps += w
                continue
            if pos[b] < pos[a]:
                # b comes before a: b's SMIRKS matches first → FP eliminated
                eliminated_fps += w
            else:
                # a comes before b: a's SMIRKS matches b's rxn first → FP
                remaining_fps += w

    return {
        "total_fps": total_fps,
        "eliminated_fps": eliminated_fps,
        "remaining_fps": remaining_fps,
        "reduction_pct": (
            eliminated_fps / total_fps * 100 if total_fps > 0 else 100.0
        ),
    }


def write_ordered_db(
    order: list[str],
    smirks_by_class: dict[str, list[dict[str, str]]],
    output_path: str,
    include_class: bool = False,
) -> int:
    """Write the ordered SMIRKS DB as JSONL.

    Args:
        order: Class codes in optimal order.
        smirks_by_class: SMIRKS entries per class.
        output_path: Output file path.
        include_class: If True, include ``"class"`` field in output.

    Returns:
        Number of SMIRKS entries written.
    """
    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for cls in order:
            entries = smirks_by_class.get(cls, [])
            for entry in entries:
                record = {"name": entry["name"], "smirks": entry["smirks"]}
                if include_class:
                    record["class"] = entry["class"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    return n


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an optimally-ordered SMIRKS database from a checkpoint "
            "to minimise false positives in first-match naming."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to gemini_smirks_checkpoint.json",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path for the ordered SMIRKS DB.",
    )
    p.add_argument(
        "--mapping",
        default=None,
        help="Path to structured_mapping.json (optional, for richer names).",
    )
    p.add_argument(
        "--use-refined",
        action="store_true",
        help="Prefer Phase-4 refined SMIRKS where available.",
    )
    p.add_argument(
        "--include-class",
        action="store_true",
        help='Include "class" field in output JSONL records.',
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print FP elimination statistics.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)

    train_results = ckpt.get("train_results", {})
    fp_results = ckpt.get("fp_results", {})
    print(f"  Classes with SMIRKS: {sum(1 for r in train_results.values() if r.get('smirks'))}")
    print(f"  Classes with FPs: {sum(1 for v in fp_results.values() if v)}")

    # ── Extract SMIRKS ───────────────────────────────────────────────────
    smirks_by_class = extract_smirks(ckpt, use_refined=args.use_refined)
    total_smirks = sum(len(v) for v in smirks_by_class.values())
    print(f"  Total SMIRKS entries: {total_smirks}")

    # ── Build FP graph ───────────────────────────────────────────────────
    adj, edge_weight = build_fp_graph(ckpt)

    total_fps = sum(
        w for d in edge_weight.values() for w in d.values()
    )
    print(f"  Total FP instances: {total_fps:,}")

    # ── Compute optimal order ────────────────────────────────────────────
    all_classes = set(smirks_by_class.keys())
    print("Computing optimal ordering...")
    order = compute_optimal_order(all_classes, adj, edge_weight)

    # ── Statistics ───────────────────────────────────────────────────────
    if args.stats or True:  # always show summary
        stats = compute_stats(order, adj, edge_weight)
        print(f"\nFP elimination results:")
        print(f"  Total FPs:      {stats['total_fps']:>10,}")
        print(f"  Eliminated:     {stats['eliminated_fps']:>10,}")
        print(f"  Remaining:      {stats['remaining_fps']:>10,}")
        print(f"  Reduction:      {stats['reduction_pct']:>9.1f}%")

    # ── Write output ─────────────────────────────────────────────────────
    n_written = write_ordered_db(
        order, smirks_by_class, args.output,
        include_class=args.include_class,
    )
    print(f"\nWrote {n_written} SMIRKS entries to {args.output}")

    # ── Optional: write ordering metadata ────────────────────────────────
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta = {
        "n_classes": len(order),
        "n_smirks": n_written,
        "use_refined": args.use_refined,
        "fp_stats": stats if args.stats or True else None,
        "class_order": order,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Ordering metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
