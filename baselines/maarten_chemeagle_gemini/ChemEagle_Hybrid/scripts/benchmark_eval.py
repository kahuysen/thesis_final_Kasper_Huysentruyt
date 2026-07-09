#!/usr/bin/env python3
"""
Benchmark evaluation script for ChemEagle.

Evaluates a prediction JSON against GT3.json using two criteria:
  - Soft-match: all reactant + product SMILES exactly match the ground truth
  - Hard-match: soft-match AND condition SMILES also match

Reports Precision, Recall, F1 (per image and overall), plus:
  - Average Tanimoto similarity (Morgan fingerprints)
  - Average GED (Graph Edit Distance, approximated via MCS)

Usage:
    python scripts/benchmark_eval.py \\
        --gt  input/Benchmark_chemeagle/benchmark/benchmark/Benchmark_kasper_GT3/GT3.json \\
        --pred output/GT3_cpu_azure_gpt5mini.json \\
        --output results.csv
"""

import argparse
import csv
import json
import re
import sys
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFMCS
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    _MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    _MORGAN_GEN = None
    print("WARNING: RDKit not found — SMILES comparison uses string equality, "
          "Tanimoto and GED will be skipped.", file=sys.stderr)


# ── SMILES utilities ──────────────────────────────────────────────────────────

def _is_null(value) -> bool:
    """True for JSON null, empty string, or the string literal 'None'/'null'."""
    if value is None:
        return True
    return str(value).strip().lower() in ('none', 'null', '')


# Tools occasionally emit `[C@@](H)` / `[C@](H)` for chiral carbons — RDKit
# rejects this. The hydrogen belongs inside the brackets: `[C@@H]`.
_BAD_CHIRAL_H = re.compile(r"\[([CN])(@@?)\]\(H\)")


def _repair_chiral_h(smi: str) -> str:
    return _BAD_CHIRAL_H.sub(r"[\1\2H]", smi)


def canonicalize(smi) -> Optional[str]:
    """Return canonical SMILES, or None if the value is missing/unparseable."""
    if _is_null(smi):
        return None
    smi = _repair_chiral_h(str(smi).strip())
    if not RDKIT_OK:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def loose_canonicalize(smi) -> Optional[str]:
    """
    Canonical SMILES for loose matching. Reconciles two harmless conventions
    that would otherwise cause false mismatches:

      * Bare ``*`` (wildcard, no isotope) used as a stand-in for H in some tool
        outputs is converted to an explicit H atom.
      * Explicit ``[H]`` atoms are then removed (made implicit) so that, e.g.,
        ``*N([1*])[2*]`` and ``[2*]N([1*])[H]`` both canonicalise to the same
        string.

    Isotope-labelled wildcards (``[1*]``, ``[2*]`` …) are kept distinct because
    they carry R-group identity.
    """
    if _is_null(smi):
        return None
    s = _repair_chiral_h(str(smi).strip())
    if not RDKIT_OK:
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 0:
            atom.SetAtomicNum(1)  # bare * → H
    try:
        mol = Chem.RemoveHs(editable.GetMol())
    except Exception:
        mol = editable.GetMol()
    return Chem.MolToSmiles(mol)


# Populated by main() from --loose CLI flag; swapped in by smiles_set /
# condition_smiles_set so the rest of the pipeline stays unchanged.
_CANONICALIZE = canonicalize


def tanimoto_similarity(smi1, smi2) -> Optional[float]:
    """Morgan fingerprint (r=2, 2048 bits) Tanimoto similarity."""
    if not RDKIT_OK or _is_null(smi1) or _is_null(smi2):
        return None
    m1 = Chem.MolFromSmiles(_repair_chiral_h(str(smi1)))
    m2 = Chem.MolFromSmiles(_repair_chiral_h(str(smi2)))
    if m1 is None or m2 is None:
        return None
    fp1 = _MORGAN_GEN.GetFingerprint(m1)
    fp2 = _MORGAN_GEN.GetFingerprint(m2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_ged(smi1, smi2) -> Optional[int]:
    """
    Approximate GED via MCS (Maximum Common Substructure):
        GED ≈ (n1 - nMCS) + (n2 - nMCS) + (e1 - eMCS) + (e2 - eMCS)
    where n = atom count, e = bond count.
    """
    if not RDKIT_OK or _is_null(smi1) or _is_null(smi2):
        return None
    m1 = Chem.MolFromSmiles(_repair_chiral_h(str(smi1)))
    m2 = Chem.MolFromSmiles(_repair_chiral_h(str(smi2)))
    if m1 is None or m2 is None:
        return None
    try:
        result = rdFMCS.FindMCS(
            [m1, m2],
            timeout=5,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        mcs_mol = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
        n_mcs = mcs_mol.GetNumAtoms() if mcs_mol else 0
        e_mcs = mcs_mol.GetNumBonds() if mcs_mol else 0
        ged = (
            (m1.GetNumAtoms() - n_mcs)
            + (m2.GetNumAtoms() - n_mcs)
            + (m1.GetNumBonds() - e_mcs)
            + (m2.GetNumBonds() - e_mcs)
        )
        return max(0, ged)
    except Exception:
        return None


def molecule_size(smi) -> int:
    """Topological size = atoms + bonds. Used as penalty for unmatched molecules."""
    if _is_null(smi) or not RDKIT_OK:
        return 0
    mol = Chem.MolFromSmiles(_repair_chiral_h(str(smi)))
    return (mol.GetNumAtoms() + mol.GetNumBonds()) if mol else 0


def compute_reaction_ged(pred_rxn: dict, gt_rxn: dict) -> Optional[float]:
    """
    Total GED cost for one (pred, gt) reaction pair using Hungarian matching.

    Steps:
      1. Collect predicted and GT molecule SMILES (reactants + products, skip nulls).
      2. Build an n_pred × n_gt pairwise GED cost matrix.
      3. Run the Hungarian algorithm (linear_sum_assignment) to find the minimum-cost
         assignment.
      4. Penalise every unmatched molecule by its full topological size (atoms + bonds).
      5. Return: matched_cost + unmatched_penalty  (avg total cost per reaction).
    """
    if not RDKIT_OK:
        return None

    pred_smiles = [m.get('smiles') for m in pred_rxn.get('reactants', []) + pred_rxn.get('products', [])]
    gt_smiles   = [m.get('smiles') for m in gt_rxn.get('reactants',   []) + gt_rxn.get('products',   [])]
    pred_smiles = [s for s in pred_smiles if not _is_null(s)]
    gt_smiles   = [s for s in gt_smiles   if not _is_null(s)]

    n_pred, n_gt = len(pred_smiles), len(gt_smiles)

    if n_pred == 0 and n_gt == 0:
        return 0.0
    if n_pred == 0:
        return float(sum(molecule_size(s) for s in gt_smiles))
    if n_gt == 0:
        return float(sum(molecule_size(s) for s in pred_smiles))

    # Build pairwise GED cost matrix
    cost = np.zeros((n_pred, n_gt))
    for i, ps in enumerate(pred_smiles):
        for j, gs in enumerate(gt_smiles):
            g = compute_ged(ps, gs)
            cost[i, j] = g if g is not None else (molecule_size(ps) + molecule_size(gs))

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_cost = float(cost[row_ind, col_ind].sum())

    # Penalty for unmatched molecules (hallucinated pred or missing gt)
    unmatched_pred = sum(molecule_size(pred_smiles[i]) for i in range(n_pred) if i not in set(row_ind))
    unmatched_gt   = sum(molecule_size(gt_smiles[j])   for j in range(n_gt)   if j not in set(col_ind))

    return matched_cost + unmatched_pred + unmatched_gt


def _safe_avg(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


# ── Reaction-level helpers ────────────────────────────────────────────────────

def smiles_set(molecules: list) -> frozenset:
    """Canonical SMILES set from a list of molecule dicts.

    Falls back to ``smiles_original`` when ``smiles`` is null or unparseable —
    pipeline post-processors can null out a SMILES they couldn't fix while
    leaving the molnextr-original under ``smiles_original``.
    """
    result = set()
    for m in molecules:
        c = _CANONICALIZE(m.get('smiles'))
        if c is None:
            c = _CANONICALIZE(m.get('smiles_original'))
        if c:
            result.add(c)
    return frozenset(result)


def condition_smiles_set(conditions: list) -> frozenset:
    """Canonical SMILES set from conditions — null/None entries are skipped."""
    result = set()
    for c in conditions:
        s = _CANONICALIZE(c.get('smiles'))
        if s is None:
            s = _CANONICALIZE(c.get('smiles_original'))
        if s:
            result.add(s)
    return frozenset(result)


def _get_conditions(rxn: dict) -> list:
    """Read conditions from either ``conditions`` (plural) or ``condition`` (singular).

    Some GT JSONs (notably NC_2017) use the singular key while predictions use
    the plural — without this shim, eval treats one side as empty.
    """
    return rxn.get('conditions') or rxn.get('condition') or []


# ── LLM-based condition comparison ───────────────────────────────────────────

_GEMINI_API_KEY: Optional[str] = None
_GEMINI_MODEL: str = "gemini-3-flash-preview"
_USE_LLM_MATCH: bool = False
_gemini_client = None
_condition_cache: dict = {}
_match_cache: dict = {}


def _tolerant_json_loads(text: str):
    """Parse JSON that flash sometimes mangles with trailing commas, single
    quotes, or fenced code blocks. Returns parsed object or None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip ```json fences
    import re as _re
    m = _re.search(r"```(?:json)?\s*(.*?)\s*```", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Repair: trailing commas before } or ]
    repaired = _re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(repaired)
    except Exception:
        pass
    # Last resort: ast.literal_eval (handles Python single quotes)
    try:
        import ast
        return ast.literal_eval(text)
    except Exception:
        return None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        import os as _os
        import sys as _sys
        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        if _GEMINI_API_KEY:
            _os.environ["GEMINI_API_KEY"] = _GEMINI_API_KEY
        from utils.llm_client import GeminiOpenAIShim
        _gemini_client = GeminiOpenAIShim()
    return _gemini_client


def conditions_match_llm(pred_conds: list, gt_conds: list) -> bool:
    """Ask Gemini whether two condition lists convey the same general information."""
    cache_key = (
        json.dumps(pred_conds, sort_keys=True),
        json.dumps(gt_conds, sort_keys=True),
    )
    if cache_key in _condition_cache:
        return _condition_cache[cache_key]

    if not pred_conds and not gt_conds:
        _condition_cache[cache_key] = True
        return True

    prompt = (
        "You are evaluating whether two sets of chemical reaction conditions convey "
        "the same general information (same solvents, reagents, catalysts, temperatures, etc.), "
        "even if the text wording, SMILES notation, or format differs slightly.\n\n"
        f"Predicted conditions:\n{json.dumps(pred_conds, indent=2)}\n\n"
        f"Ground truth conditions:\n{json.dumps(gt_conds, indent=2)}\n\n"
        "Respond with only YES or NO."
    )

    try:
        client = _get_gemini_client()
        response = client.chat.completions.create(
            model=_GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
            extra_body={"thinking_budget": 0},
        )
        content = response.choices[0].message.content or ""
        answer = content.strip().upper()
        result = answer.startswith("YES")
    except Exception as e:
        print(f"WARNING: LLM condition check failed ({e}), falling back to SMILES set comparison.",
              file=sys.stderr)
        result = (condition_smiles_set(pred_conds) == condition_smiles_set(gt_conds))

    _condition_cache[cache_key] = result
    return result


def _reaction_summary(rxn: dict) -> dict:
    """Compact view of a reaction for the matching prompt."""
    def _mols(key):
        return [
            {'smiles': m.get('smiles'), 'label': m.get('label')}
            for m in rxn.get(key, [])
        ]
    return {
        'reaction_id': rxn.get('reaction_id'),
        'reactants': _mols('reactants'),
        'products':  _mols('products'),
        'conditions': _get_conditions(rxn),
    }


def match_reactions_llm(pred_reactions: list, gt_reactions: list) -> list:
    """Ask Gemini to pair predicted reactions with GT reactions.

    Returns a list of (pred_idx, gt_idx) tuples — one-to-one, no duplicates.
    On failure, returns an empty list (caller should fall back).
    """
    if not pred_reactions or not gt_reactions:
        return []

    pred_summ = [_reaction_summary(r) for r in pred_reactions]
    gt_summ   = [_reaction_summary(r) for r in gt_reactions]

    cache_key = (
        json.dumps(pred_summ, sort_keys=True),
        json.dumps(gt_summ,   sort_keys=True),
    )
    if cache_key in _match_cache:
        return _match_cache[cache_key]

    prompt = (
        "You are matching predicted chemical reactions against ground-truth reactions. "
        "Decide which predicted reaction corresponds to which ground-truth reaction based "
        "on the chemistry (reactants + products). Two reactions correspond if they represent "
        "the same transformation, even when SMILES differ in stereochemistry, tautomer, "
        "wildcard conventions, or one side lists an extra spectator molecule.\n\n"
        f"Predicted reactions (indexed 0..{len(pred_summ) - 1}):\n"
        f"{json.dumps(pred_summ, indent=2)}\n\n"
        f"Ground truth reactions (indexed 0..{len(gt_summ) - 1}):\n"
        f"{json.dumps(gt_summ, indent=2)}\n\n"
        "Return a JSON object with a single key \"pairs\" whose value is a list of "
        "{\"pred\": i, \"gt\": j} objects — one-to-one (each index used at most once). "
        "Omit predictions or GT reactions that have no corresponding counterpart. "
        "Output JSON only."
    )

    try:
        client = _get_gemini_client()
        response = client.chat.completions.create(
            model=_GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _tolerant_json_loads(content)
        if parsed is None:
            raise ValueError(f"could not parse JSON from response (first 200 chars: {content[:200]!r})")
        raw_pairs = parsed.get('pairs', parsed) if isinstance(parsed, dict) else parsed
    except Exception as e:
        print(f"WARNING: LLM reaction matching failed ({e}) — falling back to exact bipartite.",
              file=sys.stderr)
        _match_cache[cache_key] = []
        return []

    n_p, n_g = len(pred_reactions), len(gt_reactions)
    used_p, used_g = set(), set()
    pairs = []
    if isinstance(raw_pairs, list):
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue
            pi, gi = item.get('pred'), item.get('gt')
            if not isinstance(pi, int) or not isinstance(gi, int):
                continue
            if pi < 0 or pi >= n_p or gi < 0 or gi >= n_g:
                continue
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            pairs.append((pi, gi))

    _match_cache[cache_key] = pairs
    print(f"pairs: {pairs}")
    return pairs


def is_soft_match(pred: dict, gt: dict) -> bool:
    """Soft-match: reactant set AND product set exactly match (canonical)."""
    return (
        smiles_set(pred.get('reactants', [])) == smiles_set(gt.get('reactants', []))
        and smiles_set(pred.get('products', [])) == smiles_set(gt.get('products', []))
    )


def is_hard_match(pred: dict, gt: dict) -> bool:
    """Hard-match: soft-match AND conditions convey the same information (LLM judge)."""
    if not is_soft_match(pred, gt):
        return False
    if _GEMINI_API_KEY:
        return conditions_match_llm(_get_conditions(pred), _get_conditions(gt))
    # Fallback when no API key: exact SMILES set comparison
    return (
        condition_smiles_set(_get_conditions(pred))
        == condition_smiles_set(_get_conditions(gt))
    )


def is_reactant_match(pred: dict, gt: dict) -> bool:
    """Reactant-only match: reactant SMILES sets match, ignoring products/conditions."""
    return smiles_set(pred.get('reactants', [])) == smiles_set(gt.get('reactants', []))


def is_product_match(pred: dict, gt: dict) -> bool:
    """Product-only match: product SMILES sets match, ignoring reactants/conditions."""
    return smiles_set(pred.get('products', [])) == smiles_set(gt.get('products', []))


# ── Bipartite matching ────────────────────────────────────────────────────────

def _augment(u, adj, match_gt, visited):
    """Augmenting path for maximum bipartite matching (DFS)."""
    for v in adj[u]:
        if v not in visited:
            visited.add(v)
            if match_gt[v] == -1 or _augment(match_gt[v], adj, match_gt, visited):
                match_gt[v] = u
                return True
    return False


def max_bipartite_matching(n_pred: int, n_gt: int, edges: list) -> list:
    """
    Maximum bipartite matching given a list of (pred_idx, gt_idx) edges.
    Returns list of (pred_idx, gt_idx) matched pairs.
    """
    adj = {i: [] for i in range(n_pred)}
    for pi, gi in edges:
        adj[pi].append(gi)

    match_gt = [-1] * n_gt
    for u in range(n_pred):
        _augment(u, adj, match_gt, set())

    return [(match_gt[v], v) for v in range(n_gt) if match_gt[v] != -1]


def _mol_tanimoto_score(pred: dict, gt: dict) -> float:
    """Average pairwise Tanimoto between all reactants+products in two reactions."""
    pred_mols = [m.get('smiles') for m in pred.get('reactants', []) + pred.get('products', [])]
    gt_mols   = [m.get('smiles') for m in gt.get('reactants',   []) + gt.get('products',   [])]
    scores = []
    for p in pred_mols:
        for g in gt_mols:
            t = tanimoto_similarity(p, g)
            if t is not None:
                scores.append(t)
    return _safe_avg(scores) or 0.0


def tanimoto_alignment(pred_reactions: list, gt_reactions: list) -> list:
    """
    Greedy bipartite alignment maximising Tanimoto similarity.
    Returns list of (pred_idx, gt_idx) pairs used for GED/Tanimoto reporting.
    """
    n_p, n_g = len(pred_reactions), len(gt_reactions)
    scores = [
        (i, j, _mol_tanimoto_score(pred_reactions[i], gt_reactions[j]))
        for i in range(n_p)
        for j in range(n_g)
    ]
    scores.sort(key=lambda x: -x[2])
    used_p, used_g = set(), set()
    pairs = []
    for pi, gi, _ in scores:
        if pi not in used_p and gi not in used_g:
            pairs.append((pi, gi))
            used_p.add(pi)
            used_g.add(gi)
    return pairs


# ── Per-image evaluation ──────────────────────────────────────────────────────

def evaluate_image(pred_reactions: list, gt_reactions: list) -> dict:
    n_pred = len(pred_reactions)
    n_gt   = len(gt_reactions)

    if _USE_LLM_MATCH and n_pred > 0 and n_gt > 0:
        # ── LLM-based one-to-one pairing (single batched Gemini call) ──
        soft_pairs = match_reactions_llm(pred_reactions, gt_reactions)
        if not soft_pairs:
            # API failure or empty result → fall back to exact bipartite
            soft_edges = [
                (pi, gi)
                for pi in range(n_pred)
                for gi in range(n_gt)
                if is_soft_match(pred_reactions[pi], gt_reactions[gi])
            ]
            soft_pairs = max_bipartite_matching(n_pred, n_gt, soft_edges)
        soft_tp = len(soft_pairs)

        # Hard-match: LLM-paired reactions whose conditions also match.
        hard_pairs = [
            (pi, gi) for pi, gi in soft_pairs
            if conditions_match_llm(
                _get_conditions(pred_reactions[pi]),
                _get_conditions(gt_reactions[gi]),
            )
        ]
        hard_tp = len(hard_pairs)

        # ── Maximum bipartite matching for reactant-only match ──
        reactant_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_reactant_match(pred_reactions[pi], gt_reactions[gi])
        ]
        reactant_tp = len(max_bipartite_matching(n_pred, n_gt, reactant_edges))

        # ── Maximum bipartite matching for product-only match ──
        product_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_product_match(pred_reactions[pi], gt_reactions[gi])
        ]
        product_tp = len(max_bipartite_matching(n_pred, n_gt, product_edges))

        aligned = soft_pairs  # use LLM pairing for Tanimoto/GED too
    else:
        # ── Maximum bipartite matching for soft-match ──
        soft_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_soft_match(pred_reactions[pi], gt_reactions[gi])
        ]
        soft_pairs = max_bipartite_matching(n_pred, n_gt, soft_edges)
        soft_tp = len(soft_pairs)

        # ── Maximum bipartite matching for hard-match ──
        hard_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_hard_match(pred_reactions[pi], gt_reactions[gi])
        ]
        hard_pairs = max_bipartite_matching(n_pred, n_gt, hard_edges)
        hard_tp = len(hard_pairs)

        # ── Maximum bipartite matching for reactant-only match ──
        reactant_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_reactant_match(pred_reactions[pi], gt_reactions[gi])
        ]
        reactant_tp = len(max_bipartite_matching(n_pred, n_gt, reactant_edges))

        # ── Maximum bipartite matching for product-only match ──
        product_edges = [
            (pi, gi)
            for pi in range(n_pred)
            for gi in range(n_gt)
            if is_product_match(pred_reactions[pi], gt_reactions[gi])
        ]
        product_tp = len(max_bipartite_matching(n_pred, n_gt, product_edges))

        aligned = tanimoto_alignment(pred_reactions, gt_reactions)

    def prf(tp, np_, ng):
        p  = tp / np_ if np_ > 0 else 0.0
        r  = tp / ng  if ng  > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    soft_p, soft_r, soft_f1 = prf(soft_tp, n_pred, n_gt)
    hard_p, hard_r, hard_f1 = prf(hard_tp, n_pred, n_gt)
    reactant_p, reactant_r, reactant_f1 = prf(reactant_tp, n_pred, n_gt)
    product_p,  product_r,  product_f1  = prf(product_tp,  n_pred, n_gt)

    # ── Tanimoto + GED on the chosen alignment ──
    # Tanimoto: molecule-level average over aligned pairs (positional)
    # GED: Hungarian-matched per-reaction total cost, averaged over reactions
    tan_scores, ged_scores = [], []
    for pi, gi in aligned:
        pred_mols = pred_reactions[pi].get('reactants', []) + pred_reactions[pi].get('products', [])
        gt_mols   = gt_reactions[gi].get('reactants',   []) + gt_reactions[gi].get('products',   [])
        for pm, gm in zip(pred_mols, gt_mols):
            t = tanimoto_similarity(pm.get('smiles'), gm.get('smiles'))
            if t is not None:
                tan_scores.append(t)
        rxn_ged = compute_reaction_ged(pred_reactions[pi], gt_reactions[gi])
        if rxn_ged is not None:
            ged_scores.append(rxn_ged)

    return {
        'n_gt':            n_gt,
        'n_pred':          n_pred,
        'soft_tp':         soft_tp,
        'soft_precision':  soft_p,
        'soft_recall':     soft_r,
        'soft_f1':         soft_f1,
        'hard_tp':            hard_tp,
        'hard_precision':     hard_p,
        'hard_recall':        hard_r,
        'hard_f1':            hard_f1,
        'reactant_tp':        reactant_tp,
        'reactant_precision': reactant_p,
        'reactant_recall':    reactant_r,
        'reactant_f1':        reactant_f1,
        'product_tp':         product_tp,
        'product_precision':  product_p,
        'product_recall':     product_r,
        'product_f1':         product_f1,
        'avg_tanimoto':       _safe_avg(tan_scores),
        'avg_ged':            _safe_avg(ged_scores),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ChemEagle predictions against GT3 benchmark'
    )
    parser.add_argument('--gt',     required=True,
                        help='Ground truth JSON (GT3.json — list indexed by position)')
    parser.add_argument('--pred',   required=True,
                        help='Prediction JSON (dict keyed by string index "0".."N")')
    parser.add_argument('--output', default='benchmark_results.csv',
                        help='Output CSV path (default: benchmark_results.csv)')
    parser.add_argument('--loose', action='store_true',
                        help='Loose SMILES matching: treat bare "*" as equivalent '
                             'to [H]/implicit H, so wildcard-vs-hydrogen '
                             'convention differences do not cause false '
                             'mismatches. Isotope-labelled R-groups stay distinct.')
    parser.add_argument('--gemini-key', default=None,
                        help='Gemini API key for LLM-based condition matching '
                             '(falls back to exact SMILES set comparison if not set). '
                             'Can also be set via GEMINI_API_KEY env var.')
    parser.add_argument('--judge-model', default='gemini-3-flash-preview',
                        help='Gemini model name used as the judge '
                             '(default: gemini-3-flash-preview).')
    parser.add_argument('--llm-match', action='store_true',
                        help='Use Gemini to pair predicted reactions with GT '
                             'reactions (one batched call per image). Replaces '
                             'the strict bipartite soft-match. Requires --gemini-key.')
    args = parser.parse_args()

    global _CANONICALIZE, _GEMINI_API_KEY, _GEMINI_MODEL, _USE_LLM_MATCH
    _CANONICALIZE = loose_canonicalize if args.loose else canonicalize
    if args.loose:
        print("Loose matching mode: bare '*' treated as [H]/implicit H.\n")

    import os as _os
    _GEMINI_API_KEY = args.gemini_key or _os.environ.get('GEMINI_API_KEY')
    _GEMINI_MODEL = args.judge_model
    if _GEMINI_API_KEY:
        print(f"Hard-match: using Gemini ({_GEMINI_MODEL}) as LLM judge for condition comparison.\n")
    else:
        print("Hard-match: no Gemini key — falling back to exact SMILES set comparison.\n")

    if args.llm_match:
        if not _GEMINI_API_KEY:
            parser.error("--llm-match requires --gemini-key or GEMINI_API_KEY")
        _USE_LLM_MATCH = True
        print(f"LLM reaction matching: enabled (single batched Gemini call per image, model={_GEMINI_MODEL}).\n")

    with open(args.gt,   encoding='utf-8') as f:
        gt_data   = json.load(f)   # list
    with open(args.pred, encoding='utf-8') as f:
        pred_data = json.load(f)   # dict keyed "0".."15"

    FIELDNAMES = [
        'image_index', 'file_name',
        'n_gt', 'n_pred', 'soft_tp', 'hard_tp', 'reactant_tp', 'product_tp',
        'soft_precision',     'soft_recall',     'soft_f1',
        'hard_precision',     'hard_recall',     'hard_f1',
        'reactant_precision', 'reactant_recall', 'reactant_f1',
        'product_precision',  'product_recall',  'product_f1',
        'avg_tanimoto', 'avg_ged',
    ]

    rows = []
    print(f"\n{'Idx':<4}  {'File':<55}  {'GT':>3} {'Pred':>4} "
          f"{'Soft P/R/F1':<20}  {'Hard P/R/F1':<20}  {'Tan':>5} {'GED':>5}")
    print(f"{'':4}  {'':55}  {'':3} {'':4}  "
          f"{'Reactant P/R/F1':<20}  {'Product P/R/F1':<20}")
    print('─' * 120)

    for idx, gt_entry in enumerate(gt_data):
        file_name    = gt_entry.get('file_name', f'image_{idx}')
        gt_reactions = gt_entry.get('reactions', [])
        pred_entry   = pred_data.get(str(idx), {})
        pred_reactions = pred_entry.get('reactions', [])

        m = evaluate_image(pred_reactions, gt_reactions)
        rows.append({'image_index': idx, 'file_name': file_name, **m})

        tan_str = f"{m['avg_tanimoto']:.3f}" if m['avg_tanimoto'] is not None else '  N/A'
        ged_str = f"{m['avg_ged']:.1f}"      if m['avg_ged']      is not None else '  N/A'
        print(
            f"{idx:<4}  {file_name:<55}  {m['n_gt']:>3} {m['n_pred']:>4}  "
            f"{m['soft_precision']:.3f}/{m['soft_recall']:.3f}/{m['soft_f1']:.3f}          "
            f"{m['hard_precision']:.3f}/{m['hard_recall']:.3f}/{m['hard_f1']:.3f}  "
            f"{tan_str:>5} {ged_str:>6}"
        )
        print(
            f"{'':4}  {'':55}  {'':3} {'':4}  "
            f"{m['reactant_precision']:.3f}/{m['reactant_recall']:.3f}/{m['reactant_f1']:.3f}          "
            f"{m['product_precision']:.3f}/{m['product_recall']:.3f}/{m['product_f1']:.3f}"
        )

    # ── Overall summary (macro-average) ──────────────────────────────────────
    def col_avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    total_gt   = sum(r['n_gt']   for r in rows)
    total_pred = sum(r['n_pred'] for r in rows)
    total_soft_tp     = sum(r['soft_tp']     for r in rows)
    total_hard_tp     = sum(r['hard_tp']     for r in rows)
    total_reactant_tp = sum(r['reactant_tp'] for r in rows)
    total_product_tp  = sum(r['product_tp']  for r in rows)

    # Micro-average P/R/F1
    def micro_prf(tp, np_, ng):
        p  = tp / np_ if np_ > 0 else 0.0
        r  = tp / ng  if ng  > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    micro_soft     = micro_prf(total_soft_tp,     total_pred, total_gt)
    micro_hard     = micro_prf(total_hard_tp,     total_pred, total_gt)
    micro_reactant = micro_prf(total_reactant_tp, total_pred, total_gt)
    micro_product  = micro_prf(total_product_tp,  total_pred, total_gt)

    summary = {
        'image_index':     'OVERALL',
        'file_name':       f'macro-avg over {len(rows)} images (micro P/R/F1)',
        'n_gt':            total_gt,
        'n_pred':          total_pred,
        'soft_tp':            total_soft_tp,
        'hard_tp':            total_hard_tp,
        'reactant_tp':        total_reactant_tp,
        'product_tp':         total_product_tp,
        'soft_precision':     micro_soft[0],
        'soft_recall':        micro_soft[1],
        'soft_f1':            micro_soft[2],
        'hard_precision':     micro_hard[0],
        'hard_recall':        micro_hard[1],
        'hard_f1':            micro_hard[2],
        'reactant_precision': micro_reactant[0],
        'reactant_recall':    micro_reactant[1],
        'reactant_f1':        micro_reactant[2],
        'product_precision':  micro_product[0],
        'product_recall':     micro_product[1],
        'product_f1':         micro_product[2],
        'avg_tanimoto':       col_avg('avg_tanimoto'),
        'avg_ged':            col_avg('avg_ged'),
    }
    rows.append(summary)

    print('─' * 120)
    print(f"\n=== OVERALL ({len(rows)-1} images) ===")
    print(f"  Reactions   GT={total_gt}  Predicted={total_pred}  "
          f"Soft-TP={total_soft_tp}  Hard-TP={total_hard_tp}  "
          f"Reactant-TP={total_reactant_tp}  Product-TP={total_product_tp}")
    print(f"  Soft-match      Precision={micro_soft[0]}  Recall={micro_soft[1]}  F1={micro_soft[2]}")
    print(f"  Hard-match      Precision={micro_hard[0]}  Recall={micro_hard[1]}  F1={micro_hard[2]}")
    print(f"  Reactant-match  Precision={micro_reactant[0]}  Recall={micro_reactant[1]}  F1={micro_reactant[2]}")
    print(f"  Product-match   Precision={micro_product[0]}  Recall={micro_product[1]}  F1={micro_product[2]}")
    print(f"  Avg Tanimoto = {summary['avg_tanimoto']}")
    print(f"  Avg GED      = {summary['avg_ged']}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}\n")


if __name__ == '__main__':
    main()
