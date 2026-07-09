"""
Compare two pipeline runs (selector vs. roundrobin).

Usage:
    python compare_runs.py                          # auto-picks the latest selector + roundrobin run
    python compare_runs.py runs/20260425_selector runs/20260425_roundrobin

SMILES are canonicalised via RDKit before comparison; equivalent SMILES
written differently match. Metrics (validity, IoU, role compliance) come
from eval/metrics.py so the same numbers appear here and in eval/run_eval.py.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from eval.metrics import (  # noqa: E402
    _canonical,
    _condition_set,
    evaluate,
    summarise,
)


RUNS_DIR = REPO / "runs"


# ── helpers ──────────────────────────────────────────────────────────────────

def latest_run(team: str) -> Path | None:
    candidates = sorted(RUNS_DIR.glob(f"*_{team}"), reverse=True)
    return candidates[0] if candidates else None


def load_json(run_dir: Path) -> dict | None:
    p = run_dir / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def parse_conversation(run_dir: Path) -> list[dict]:
    """Return list of {agent, text} dicts from conversation.txt."""
    p = run_dir / "conversation.txt"
    if not p.exists():
        return []
    messages = []
    current_agent = None
    current_lines: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^-{6,} \S+ \((\w+)\) -{6,}$", line)
        if m:
            if current_agent is not None:
                messages.append({"agent": current_agent, "text": "\n".join(current_lines).strip()})
            current_agent = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    if current_agent is not None:
        messages.append({"agent": current_agent, "text": "\n".join(current_lines).strip()})
    return messages


def agent_stats(messages: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for msg in messages:
        counts[msg["agent"]] = counts.get(msg["agent"], 0) + 1
    return counts


def smiles_set(result: dict | None, role: str) -> set[str]:
    """Canonicalised SMILES set; SMILES that don't parse are dropped."""
    if not result:
        return set()
    out: set[str] = set()
    for rxn in result.get("reactions", []):
        for item in rxn.get(role, []):
            c = _canonical(item.get("smiles"))
            if c:
                out.add(c)
    return out


def conditions_set(result: dict | None) -> set[tuple[str, str]]:
    """Set of (role, normalised_text) — same form metrics.py uses."""
    if not result:
        return set()
    return _condition_set(result)


def sep(char: str = "─", width: int = 70) -> str:
    return char * width


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) == 3:
        dir_a, dir_b = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        dir_a = latest_run("selector")
        dir_b = latest_run("roundrobin")

    if not dir_a or not dir_b:
        print("Could not find runs. Pass two run directories as arguments, or run both teams first.")
        sys.exit(1)

    label_a = dir_a.name
    label_b = dir_b.name

    msgs_a = parse_conversation(dir_a)
    msgs_b = parse_conversation(dir_b)
    json_a = load_json(dir_a)
    json_b = load_json(dir_b)

    print(sep("═"))
    print("  PIPELINE RUN COMPARISON")
    print(sep("═"))
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print()

    # ── Conversation stats ────────────────────────────────────────────────────
    print(sep())
    print("  CONVERSATION STATISTICS")
    print(sep())

    stats_a = agent_stats(msgs_a)
    stats_b = agent_stats(msgs_b)
    all_agents = sorted(set(stats_a) | set(stats_b))

    col = 28
    print(f"  {'Agent':<{col}} {'A':>6}  {'B':>6}")
    print(f"  {'-'*col} {'------':>6}  {'------':>6}")
    for agent in all_agents:
        a_val = stats_a.get(agent, 0)
        b_val = stats_b.get(agent, 0)
        diff = " ◄" if a_val != b_val else ""
        print(f"  {agent:<{col}} {a_val:>6}  {b_val:>6}{diff}")
    print(f"  {'TOTAL':<{col}} {len(msgs_a):>6}  {len(msgs_b):>6}")
    print()

    # ── Agent turn order ──────────────────────────────────────────────────────
    print(sep())
    print("  AGENT TURN ORDER")
    print(sep())
    order_a = [m["agent"] for m in msgs_a]
    order_b = [m["agent"] for m in msgs_b]
    max_len = max(len(order_a), len(order_b))
    print(f"  {'#':<4} {'A':<30} {'B':<30}")
    print(f"  {'-'*4} {'-'*30} {'-'*30}")
    for i in range(max_len):
        a_agent = order_a[i] if i < len(order_a) else "—"
        b_agent = order_b[i] if i < len(order_b) else "—"
        marker = "  ◄" if a_agent != b_agent else ""
        print(f"  {i+1:<4} {a_agent:<30} {b_agent:<30}{marker}")
    print()

    # ── JSON output comparison ────────────────────────────────────────────────
    print(sep())
    print("  JSON OUTPUT COMPARISON")
    print(sep())

    if not json_a:
        print(f"  [!] No valid result.json found in {label_a}")
    if not json_b:
        print(f"  [!] No valid result.json found in {label_b}")

    if json_a and json_b:
        rxns_a = json_a.get("reactions", [])
        rxns_b = json_b.get("reactions", [])
        print(f"  Reactions found:     A={len(rxns_a)}   B={len(rxns_b)}")

        reactants_a = smiles_set(json_a, "reactants")
        reactants_b = smiles_set(json_b, "reactants")
        products_a  = smiles_set(json_a, "products")
        products_b  = smiles_set(json_b, "products")
        conds_a     = conditions_set(json_a)
        conds_b     = conditions_set(json_b)

        print(f"  Unique reactant SMILES: A={len(reactants_a)}   B={len(reactants_b)}")
        print(f"  Unique product SMILES:  A={len(products_a)}   B={len(products_b)}")
        print(f"  Condition entries:      A={len(conds_a)}   B={len(conds_b)}")
        print()

        only_a_reactants = reactants_a - reactants_b
        only_b_reactants = reactants_b - reactants_a
        if only_a_reactants:
            print("  Reactant SMILES only in A:")
            for s in sorted(only_a_reactants):
                print(f"    {s}")
        if only_b_reactants:
            print("  Reactant SMILES only in B:")
            for s in sorted(only_b_reactants):
                print(f"    {s}")

        only_a_products = products_a - products_b
        only_b_products = products_b - products_a
        if only_a_products:
            print("  Product SMILES only in A:")
            for s in sorted(only_a_products):
                print(f"    {s}")
        if only_b_products:
            print("  Product SMILES only in B:")
            for s in sorted(only_b_products):
                print(f"    {s}")

        only_a_conds = conds_a - conds_b
        only_b_conds = conds_b - conds_a
        if only_a_conds:
            print("  Conditions only in A:")
            for role, text in sorted(only_a_conds):
                print(f"    {role}: {text}")
        if only_b_conds:
            print("  Conditions only in B:")
            for role, text in sorted(only_b_conds):
                print(f"    {role}: {text}")

        if not (only_a_reactants or only_b_reactants or only_a_products or
                only_b_products or only_a_conds or only_b_conds):
            print("  Reactants, products, and conditions are identical in both runs.")
    print()

    # ── Per-run metric summary (canonical-SMILES-aware) ──────────────────────
    print(sep())
    print("  METRIC SUMMARY")
    print(sep())
    for label, rec in (("A", json_a), ("B", json_b)):
        if rec is None:
            print(f"  {label}: (no result.json)")
            continue
        print(f"  {label}: {label_a if label == 'A' else label_b}")
        print(summarise(evaluate(rec)))
        print()

    # ── Full JSON side-by-side ────────────────────────────────────────────────
    print(sep())
    print("  FULL JSON OUTPUTS")
    print(sep())
    print(f"\n  ── A: {label_a} ──")
    print(json.dumps(json_a, indent=2, ensure_ascii=False) if json_a else "  (none)")
    print(f"\n  ── B: {label_b} ──")
    print(json.dumps(json_b, indent=2, ensure_ascii=False) if json_b else "  (none)")
    print()
    print(sep("═"))


if __name__ == "__main__":
    main()
