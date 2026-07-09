"""Score pipeline outputs against ground-truth JSON.

Ground-truth schema (from the benchmark):
  {
    "file_name": "...",
    "reactions": [
      {
        "reaction_id": "1_1",
        "reactants":  [{"smiles": "...", "label": "..."}],
        "products":   [{"smiles": "...", "label": "..."}],
        "conditions": [{"role": "reagent|temperature|time|yield|...",
                        "text": "...", "smiles?": "...", "label?": "..."}]
      }, ...
    ]
  }

Our schema (FigureExtraction → Reaction): products, reactants, reagents,
conditions(temperature/time/...). We canonicalize SMILES and use SMILES set
overlap as the matching primitive — order-independent, label-independent.

Metrics:
  - reaction_match_recall: fraction of GT reactions that have at least one
    predicted reaction whose product-SMILES set matches.
  - product_smiles_recall:  micro-averaged recall over GT product SMILES.
  - product_smiles_precision: micro-averaged precision (predicted products
    not in GT count as false positives).
  - yield_match: among reactions that matched, fraction where the predicted
    yield (rounded to int %) equals the GT yield.
  - structure_validity: fraction of predicted SMILES that RDKit can parse.

Wildcards in GT (`*` for R-group placeholders) are normalized so that any
heavy atom in our prediction at that position is treated as a match.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger

from .schema import FigureExtraction

# Silence RDKit's per-parse error spam during eval.
RDLogger.DisableLog("rdApp.error")

# Normalize atom-mapped SMILES wildcards (`[1*]`, `[12*]`) to plain `*` —
# the GT in this benchmark uses mapped wildcards that RDKit refuses to
# parse, even though they're semantically identical to bare `*`.
_ATOM_MAPPED_WILDCARD = re.compile(r"\[(\d+)\*\]")


def _normalize_wildcards(smi: str) -> str:
    if not smi:
        return smi
    # Strip the atom map number off any `[N*]` so RDKit's parser is happy.
    return _ATOM_MAPPED_WILDCARD.sub("*", smi)


# ---------- SMILES normalization ----------

def canonical(smiles: str, *, ignore_stereo: bool = False) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if ignore_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def smiles_eq(a: str, b: str, *, ignore_stereo: bool = True) -> bool:
    """Compare two SMILES, treating wildcards (`*`, `[*]`, `[1*]`, `[2*]`, …)
    as equivalent before canonicalization."""
    if not a or not b:
        return False
    a = _normalize_wildcards(a)
    b = _normalize_wildcards(b)
    ca = canonical(a, ignore_stereo=ignore_stereo)
    cb = canonical(b, ignore_stereo=ignore_stereo)
    return ca is not None and cb is not None and ca == cb


# ---------- Yield extraction ----------

YIELD_RX = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


def extract_yield_pct(text: str) -> Optional[float]:
    if not text:
        return None
    m = YIELD_RX.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------- Per-image scoring ----------

@dataclass
class ImageScore:
    image: str
    gt_reactions: int = 0
    pred_reactions: int = 0
    matched_reactions: int = 0
    gt_products: int = 0
    pred_products: int = 0
    matched_products: int = 0
    yield_pairs: int = 0  # GT+pred with both yields available
    yield_correct: int = 0
    invalid_smiles: int = 0
    parsed_smiles: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def reaction_recall(self) -> float:
        return self.matched_reactions / self.gt_reactions if self.gt_reactions else 0.0

    @property
    def product_recall(self) -> float:
        return self.matched_products / self.gt_products if self.gt_products else 0.0

    @property
    def product_precision(self) -> float:
        return self.matched_products / self.pred_products if self.pred_products else 0.0

    @property
    def yield_accuracy(self) -> float:
        return self.yield_correct / self.yield_pairs if self.yield_pairs else 0.0

    @property
    def smiles_validity(self) -> float:
        return self.parsed_smiles / (self.parsed_smiles + self.invalid_smiles) if (self.parsed_smiles + self.invalid_smiles) else 1.0


def _gt_products(rxn: dict) -> list[str]:
    return [p.get("smiles", "") for p in rxn.get("products", []) if p.get("smiles")]


def _gt_yield(rxn: dict) -> Optional[float]:
    for c in rxn.get("conditions", []):
        if (c.get("role") or "").lower() == "yield":
            return extract_yield_pct(c.get("text") or "")
    return None


def _pred_products(rxn) -> list[str]:
    return [p.smiles for p in rxn.products if p.smiles]


def _gt_reactants(rxn: dict) -> list[str]:
    return [r.get("smiles", "") for r in rxn.get("reactants", []) if r.get("smiles")]


def _pred_reactants(rxn) -> list[str]:
    return [r.smiles for r in rxn.reactants if r.smiles]


def _set_overlap(a: list[str], b: list[str]) -> int:
    """Count how many SMILES in `a` are equal (canonically) to any in `b`."""
    return sum(1 for x in a if any(smiles_eq(x, y) for y in b))


def _pair_score(gt_rxn: dict, pr) -> tuple[int, int]:
    """How well does pred reaction `pr` match GT reaction `gt_rxn`?

    Returns (product_overlap, reactant_overlap). Higher is better, with
    products taking precedence — that's what we sort by."""
    p_overlap = _set_overlap(_gt_products(gt_rxn), _pred_products(pr))
    r_overlap = _set_overlap(_gt_reactants(gt_rxn), _pred_reactants(pr))
    return (p_overlap, r_overlap)


def score_image(pred: FigureExtraction, gt: dict, image_name: str) -> ImageScore:
    s = ImageScore(image=image_name)
    gt_rxns = gt.get("reactions", [])
    s.gt_reactions = len(gt_rxns)
    s.pred_reactions = len(pred.reactions)

    # Tally invalid SMILES in prediction.
    for rxn in pred.reactions:
        for sp in list(rxn.reactants) + list(rxn.products) + list(rxn.reagents):
            if not sp.smiles:
                continue
            if Chem.MolFromSmiles(sp.smiles) is None:
                s.invalid_smiles += 1
            else:
                s.parsed_smiles += 1

    # ----- one-to-one bipartite pairing -----
    # Build (gt_idx, pred_idx, product_overlap, reactant_overlap) for every
    # candidate pair, then greedily pick the highest-scoring pairs without
    # reusing indices. This handles tables where many rows share the same
    # product SMILES (we then pair on reactants), and also positional fall-back
    # via the original ordering when both signals are weak.
    candidates: list[tuple[tuple[int, int], int, int]] = []
    for gi, gt_rxn in enumerate(gt_rxns):
        gtp = _gt_products(gt_rxn)
        for pi, pr in enumerate(pred.reactions):
            p_o, r_o = _pair_score(gt_rxn, pr)
            if p_o == 0 and r_o == 0:
                continue
            candidates.append(((p_o, r_o), gi, pi))

    # Sort: highest product overlap first, then reactant overlap, then
    # closest-in-position (preserves natural ordering on ties).
    candidates.sort(key=lambda c: (-c[0][0], -c[0][1], abs(c[1] - c[2])))

    paired_gt: dict[int, int] = {}
    used_pred: set[int] = set()
    for score, gi, pi in candidates:
        if gi in paired_gt or pi in used_pred:
            continue
        paired_gt[gi] = pi
        used_pred.add(pi)

    # Tally per-product matches and reaction-level matches.
    for gi, gt_rxn in enumerate(gt_rxns):
        gtp = _gt_products(gt_rxn)
        s.gt_products += len(gtp)
        if gi not in paired_gt:
            continue
        pr = pred.reactions[paired_gt[gi]]
        p_o = _set_overlap(gtp, _pred_products(pr))
        s.matched_products += p_o
        # Reaction match = every GT product has a counterpart. (Empty GT
        # product list still counts as match if any reactants overlap.)
        if (gtp and p_o == len(gtp)) or (not gtp and _set_overlap(_gt_reactants(gt_rxn), _pred_reactants(pr))):
            s.matched_reactions += 1

        # Yield: only count when we have a reasonable pairing.
        gt_y = _gt_yield(gt_rxn)
        pred_y = next((p.yield_pct for p in pr.products if p.yield_pct is not None), None)
        if gt_y is not None and pred_y is not None:
            s.yield_pairs += 1
            if abs(int(round(gt_y)) - int(round(pred_y))) == 0:
                s.yield_correct += 1

    s.pred_products = sum(len(_pred_products(r)) for r in pred.reactions)
    return s


# ---------- Aggregation + report ----------

def aggregate(scores: list[ImageScore]) -> dict:
    if not scores:
        return {}
    sums = {
        "gt_reactions": sum(s.gt_reactions for s in scores),
        "pred_reactions": sum(s.pred_reactions for s in scores),
        "matched_reactions": sum(s.matched_reactions for s in scores),
        "gt_products": sum(s.gt_products for s in scores),
        "pred_products": sum(s.pred_products for s in scores),
        "matched_products": sum(s.matched_products for s in scores),
        "yield_pairs": sum(s.yield_pairs for s in scores),
        "yield_correct": sum(s.yield_correct for s in scores),
        "invalid_smiles": sum(s.invalid_smiles for s in scores),
        "parsed_smiles": sum(s.parsed_smiles for s in scores),
    }
    def safe(n, d):
        return n / d if d else 0.0
    return {
        **sums,
        "reaction_recall": safe(sums["matched_reactions"], sums["gt_reactions"]),
        "product_recall": safe(sums["matched_products"], sums["gt_products"]),
        "product_precision": safe(sums["matched_products"], sums["pred_products"]),
        "yield_accuracy": safe(sums["yield_correct"], sums["yield_pairs"]),
        "smiles_validity": safe(sums["parsed_smiles"], sums["parsed_smiles"] + sums["invalid_smiles"]),
        "n_images": len(scores),
    }


def evaluate_run(
    *,
    pred_dir: str | Path,
    gt_dir: str | Path,
    image_dir: str | Path,
) -> tuple[list[ImageScore], dict]:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    image_dir = Path(image_dir)

    scores: list[ImageScore] = []
    for img in sorted(image_dir.glob("*.png")):
        stem = img.stem
        pred_path = pred_dir / f"{stem}.json"
        gt_path = gt_dir / f"{stem}.json"
        if not pred_path.exists():
            scores.append(ImageScore(image=img.name, notes=["no prediction"]))
            continue
        if not gt_path.exists():
            scores.append(ImageScore(image=img.name, notes=["no ground truth"]))
            continue
        try:
            pred = FigureExtraction.model_validate_json(pred_path.read_text())
        except Exception as e:
            scores.append(ImageScore(image=img.name, notes=[f"prediction unparseable: {e}"]))
            continue
        gt = json.loads(gt_path.read_text())
        scores.append(score_image(pred, gt, img.name))
    return scores, aggregate(scores)


def format_report(scores: list[ImageScore], summary: dict) -> str:
    lines = []
    lines.append(f"{'image':<60}  {'rxn':>8}  {'prod':>10}  {'yld':>6}  {'valid':>6}")
    lines.append("-" * 100)
    for s in scores:
        if s.notes:
            lines.append(f"{s.image:<60}  {'—':>8}  {'—':>10}  {'—':>6}  {'—':>6}  {' / '.join(s.notes)}")
            continue
        rxn = f"{s.matched_reactions}/{s.gt_reactions}"
        prod = f"{s.matched_products}/{s.gt_products}"
        yld = f"{s.yield_correct}/{s.yield_pairs}" if s.yield_pairs else "—"
        valid = f"{s.smiles_validity * 100:.0f}%"
        lines.append(f"{s.image:<60}  {rxn:>8}  {prod:>10}  {yld:>6}  {valid:>6}")
    lines.append("-" * 100)
    if summary:
        lines.append(
            f"AGGREGATE  rxn-recall={summary['reaction_recall']:.2f}  "
            f"prod-recall={summary['product_recall']:.2f}  "
            f"prod-precision={summary['product_precision']:.2f}  "
            f"yield-accuracy={summary['yield_accuracy']:.2f}  "
            f"smiles-validity={summary['smiles_validity']:.2f}  "
            f"({summary['n_images']} images)"
        )
    return "\n".join(lines)
