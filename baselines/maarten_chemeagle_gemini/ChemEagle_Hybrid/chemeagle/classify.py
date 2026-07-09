"""Downstream classification: parse_pdf JSON -> CSV with Rxn-INSIGHT + text-LLM data.

Per-reaction pipeline:
  1. Build SMILES strings (2-part for Rxn-INSIGHT; the 3-part form goes in CSV).
  2. Skip Rxn-INSIGHT entirely if any SMILES has R-group placeholders
     (`*`, `[*]`, `[R...]`, `[Ar...]`) — rxnmapper can't handle them.
  3. Reaction(smi, smirks=user_db, solvent=..., reagent=..., catalyst=...)
       .get_reaction_info() -> NAME (from SMIRKS DB), CLASS, SOLVENT, REAGENT,
       CATALYST, BY-PRODUCTS, SCAFFOLD.
  4. If NAME == "OtherReaction" or empty, call get_detailed_template(
       radius=0, radius_products=1) and stash in the template column.
  5. Always also run the text-LLM (one Gemini call per page, batched across
     all reactions on the same page) to fill name/solvent/catalyst/reagent/
     procedure as supplementary columns.

Public API: `classify(parse_result, *, ...) -> list[dict]`, `write_csv(rows, path)`.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Any, Optional


CSV_COLUMNS = [
    # Provenance
    "pdf", "page", "figure_png", "reaction_id",
    # SMILES
    "reactants_smiles", "products_smiles", "reagents_smiles", "rxn_smiles",
    # Rxn-INSIGHT
    "ri_name", "ri_class",
    "ri_tier_1", "ri_tier_2", "ri_tier_3", "ri_tier_4", "ri_tier_5",
    "ri_solvent", "ri_reagent", "ri_catalyst",
    "ri_byproducts", "ri_scaffold", "ri_template_r0p1",
    # Text-LLM (supplementary)
    "text_name", "text_solvent", "text_catalyst", "text_reagent",
    "text_procedure", "text_workup", "text_purification", "text_analysis",
    # Audit
    "chemeagle_conditions",
    # Errors (empty when all stages succeed)
    "ri_error",
]

_FIG_NAME_RE = re.compile(r"^(.+)_image_(\d+)_(\d+)\.png$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\*|\[R[0-9a-zA-Z]*\]|\[Ar[0-9a-zA-Z]*\]")


# --- public API ----------------------------------------------------------------


def classify(
    parse_result: dict,
    *,
    smirks_db_path: str = "/data/mdobb/reaction_classification/data/smirks_db.json",
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
) -> list[dict]:
    """Run SMIRKS+Rxn-INSIGHT+text-LLM enrichment on a parse_pdf result dict.

    Returns a list of row dicts (one per reaction) ready for write_csv.
    """
    # Bring up Rxn-INSIGHT (rxnmapper, classifier) and the SMIRKS DB.
    Reaction, get_class_name = _import_rxn_insight()
    smirks_db = _load_smirks_db(smirks_db_path)

    # Bring up the Gemini client for the text-LLM stage.
    os.environ["CHEMEAGLE_BACKEND"] = "gemini"
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Pass api_key=... or export GEMINI_API_KEY."
        )

    # Flatten figures -> reactions, attaching provenance + page text.
    pdf_path = parse_result.get("pdf_path", "")
    page_text_by_num = {p["page"]: p.get("text", "") for p in parse_result.get("pages") or []}

    flat: list[dict] = []
    n_seen = 0
    n_drop_generic = 0
    n_drop_template = 0
    n_partial = 0
    for fig in parse_result.get("figures") or []:
        fname = fig.get("png", "") or ""
        meta = _FIG_NAME_RE.match(fname)
        page = int(meta.group(2)) if meta else 1
        result = fig.get("result") or {}
        fig_reactions = result.get("reactions") or []
        for rxn in fig_reactions:
            n_seen += 1
            r_smi, p_smi, rg_smi = _split_smiles(rxn)
            two_part = _two_part(r_smi, p_smi)
            three_part = _three_part(r_smi, p_smi, rg_smi)
            decision = _inclusion_decision(rxn, fig_reactions)
            if decision == "DROP_GENERIC":
                n_drop_generic += 1
                continue
            if decision == "DROP_TEMPLATE_PARENT":
                n_drop_template += 1
                continue
            # KEEP_FULL or KEEP_PARTIAL — both go in the CSV. KEEP_PARTIAL
            # has wildcards somewhere; Rxn-INSIGHT will be skipped for it
            # but the text-LLM still runs and the SMILES are preserved.
            is_partial = decision == "KEEP_PARTIAL"
            if is_partial:
                n_partial += 1
            flat.append({
                "raw": rxn,
                "pdf": pdf_path,
                "page": page,
                "figure_png": fname,
                "reaction_id": rxn.get("reaction_id", ""),
                "reactants_smiles": r_smi,
                "products_smiles": p_smi,
                "reagents_smiles": rg_smi,
                "rxn_smiles": three_part if rg_smi else two_part,
                "_two_part": two_part,
                "_skip_rxn_insight": is_partial,
                "_solvent_hint": _condition_text(rxn, "solvent"),
                "_reagent_hint": _condition_text(rxn, "reagent"),
                "_catalyst_hint": _condition_text(rxn, "catalyst"),
            })
    if n_drop_generic or n_drop_template:
        sys.stderr.write(
            f"chemeagle.classify: dropped {n_drop_generic + n_drop_template}/{n_seen} "
            f"({n_drop_generic} generic-scheme + {n_drop_template} template-parent). "
            f"Kept {len(flat)} ({n_partial} partial-info, {len(flat) - n_partial} fully concrete).\n"
        )

    # Stage A: page-batched text-LLM (runs on ALL kept rows).
    text_by_key = _run_text_llm(flat, page_text_by_num, model=model)

    # Stage A.5: try to complete partial-info rows from the LLM's
    # `completed_smiles` field. If the completion is parseable and free of
    # wildcards, lift the _skip_rxn_insight flag and update the SMILES so
    # Rxn-INSIGHT can run on the now-concrete reaction. Otherwise log a
    # warning and leave the row as-is.
    n_completed = 0
    n_unresolved = 0
    for r in flat:
        if not r["_skip_rxn_insight"]:
            continue
        text = text_by_key.get((r["page"], r["reaction_id"]), {}) or {}
        candidate = (text.get("completed_smiles") or "").strip()
        if not candidate:
            sys.stderr.write(
                f"chemeagle.classify: WARN reaction_id={r['reaction_id']!r} on "
                f"{r['figure_png']!r}: LLM returned no completion for partial-info "
                f"SMILES {r['_two_part']!r}\n"
            )
            n_unresolved += 1
            continue
        completed = _try_complete(candidate)
        if not completed:
            sys.stderr.write(
                f"chemeagle.classify: WARN reaction_id={r['reaction_id']!r} on "
                f"{r['figure_png']!r}: LLM returned non-parseable completion "
                f"{candidate!r} for SMILES {r['_two_part']!r}\n"
            )
            n_unresolved += 1
            continue
        # Apply the completion.
        new_r, new_p = completed
        r["reactants_smiles"] = new_r
        r["products_smiles"] = new_p
        r["_two_part"] = _two_part(new_r, new_p)
        # Reagents: keep whatever chemEAGLE produced. They rarely carry
        # wildcards in practice; if they do, _has_placeholder below will
        # leave _skip_rxn_insight True.
        if not _has_placeholder(r["reagents_smiles"]):
            r["rxn_smiles"] = (
                _three_part(new_r, new_p, r["reagents_smiles"])
                if r["reagents_smiles"] else r["_two_part"]
            )
            r["_skip_rxn_insight"] = False
            n_completed += 1
        else:
            sys.stderr.write(
                f"chemeagle.classify: WARN reaction_id={r['reaction_id']!r} on "
                f"{r['figure_png']!r}: completion succeeded for reactants/products "
                f"but reagents still contain wildcards {r['reagents_smiles']!r}\n"
            )
            r["rxn_smiles"] = r["_two_part"]
            n_unresolved += 1
    if n_completed or n_unresolved:
        sys.stderr.write(
            f"chemeagle.classify: completion stage: {n_completed} resolved, "
            f"{n_unresolved} unresolved (kept as-is).\n"
        )

    # Stage B: per-reaction Rxn-INSIGHT (skipped on partial-info rows).
    rows: list[dict] = []
    for r in flat:
        if r["_skip_rxn_insight"]:
            ri = {"_skip": "r-group placeholder"}
        else:
            ri = _run_rxn_insight(Reaction, r, smirks_db, get_class_name)
        text = text_by_key.get((r["page"], r["reaction_id"]), {})
        rows.append(_build_row(r, ri, text))
    return rows


def write_csv(rows: list[dict], path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# --- Rxn-INSIGHT loader --------------------------------------------------------


def _import_rxn_insight():
    """Returns (Reaction class, get_class_name fn). Raises with a clear error if
    RXN_INSIGHT_PATH is unset or the package can't be loaded."""
    rxn_insight_path = os.environ.get("RXN_INSIGHT_PATH")
    if not rxn_insight_path:
        raise RuntimeError(
            "RXN_INSIGHT_PATH is not set. Export the path to Gen-Rxn-INSIGHT/src "
            "(e.g., /data/mdobb/Rxn-INSIGHT/Gen-Rxn-INSIGHT/src)."
        )
    if rxn_insight_path not in sys.path:
        sys.path.insert(0, rxn_insight_path)
    try:
        from gen_rxn_insight.reaction import Reaction
        from gen_rxn_insight.naming import get_class_name
        return Reaction, get_class_name
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import gen_rxn_insight from {rxn_insight_path}: {e}"
        ) from e


def _resolve_class_name(code: str, get_class_name) -> dict:
    """SMIRKS DB returns dotted codes like '3.1.1.2.1'. Expand the code into
    its full tier hierarchy.

    Returns a dict:
      {"name": <deepest non-empty tier>, "tiers": {1: <tier_1>, 2: <tier_2>, ...}}

    On failure (no dot in code, lookup error, no readable tiers), returns
    {"name": code, "tiers": {}}.
    """
    fallback = {"name": code or "", "tiers": {}}
    if not code or "." not in code:
        return fallback
    try:
        tiers_raw = get_class_name(code, tier=None)
    except Exception:
        return fallback
    if not isinstance(tiers_raw, dict) or not tiers_raw:
        return fallback
    tiers: dict[int, str] = {}
    for k, v in tiers_raw.items():
        if not isinstance(k, str) or not k.startswith("tier_") or not v:
            continue
        try:
            n = int(k.split("_", 1)[1])
        except ValueError:
            continue
        tiers[n] = str(v)
    if not tiers:
        return fallback
    deepest = tiers[max(tiers)]
    return {"name": deepest, "tiers": tiers}


def _load_smirks_db(path: str):
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    if not os.path.isfile(path):
        raise FileNotFoundError(f"SMIRKS DB not found: {path}")
    db = pd.read_json(path, lines=True)
    if "smirks" not in db.columns or "name" not in db.columns:
        raise RuntimeError(
            f"SMIRKS DB at {path} must contain 'smirks' and 'name' columns; "
            f"got {list(db.columns)}"
        )
    if "nreact" not in db.columns:
        # Count reactant patterns: split on '>>' then on '.'. Matches the
        # heuristic used downstream by gen_rxn_insight.naming.name_reaction.
        db["nreact"] = db["smirks"].apply(
            lambda s: len(s.split(">>")[0].split(".")) if isinstance(s, str) else 0
        )

    # Pre-filter: drop templates RDKit can't compile. gen_rxn_insight.naming
    # iterates over the DB and calls AllChem.ReactionFromSmarts(s) without
    # catching parser errors, so one broken SMIRKS terminates the loop for
    # every reaction. Filtering once here makes name_reaction safe.
    RDLogger.DisableLog("rdApp.*")
    keep = []
    for s in db["smirks"]:
        if not isinstance(s, str):
            keep.append(False)
            continue
        try:
            rxn = AllChem.ReactionFromSmarts(s)
        except Exception:
            keep.append(False)
            continue
        keep.append(rxn is not None)
    n_dropped = sum(1 for k in keep if not k)
    if n_dropped:
        sys.stderr.write(
            f"chemeagle.classify: dropped {n_dropped} unparseable SMIRKS "
            f"out of {len(db)} (RDKit ChemicalReactionParser).\n"
        )
    db = db[keep].reset_index(drop=True)
    return db


# --- SMILES assembly -----------------------------------------------------------


def _split_smiles(reaction: dict) -> tuple[str, str, str]:
    reactants = ".".join(
        s for s in (m.get("smiles") or "" for m in reaction.get("reactants") or []) if s
    )
    products = ".".join(
        s for s in (m.get("smiles") or "" for m in reaction.get("products") or []) if s
    )
    reagents = ".".join(
        s for s in (
            c.get("smiles") or "" for c in reaction.get("conditions") or []
        ) if s
    )
    return reactants, products, reagents


def _two_part(reactants: str, products: str) -> str:
    return f"{reactants}>>{products}"


def _three_part(reactants: str, products: str, reagents: str) -> str:
    return f"{reactants}>{reagents}>{products}"


def _has_placeholder(smi: str) -> bool:
    return bool(smi) and bool(_PLACEHOLDER_RE.search(smi))


def _try_complete(candidate: str) -> Optional[tuple[str, str]]:
    """Validate an LLM-completed `reactants>>products` SMILES.

    Returns (reactants_canonical, products_canonical) when:
      - the candidate splits cleanly on `>>`
      - both sides have no R-group placeholders
      - every component parses with RDKit

    Otherwise returns None.
    """
    if not candidate or ">>" not in candidate:
        return None
    parts = candidate.split(">>")
    if len(parts) != 2:
        return None
    reactants_s, products_s = parts[0].strip(), parts[1].strip()
    if not reactants_s or not products_s:
        return None
    if _has_placeholder(reactants_s) or _has_placeholder(products_s):
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    out = []
    for side in (reactants_s, products_s):
        canon_components = []
        for comp in side.split("."):
            if not comp:
                continue
            mol = Chem.MolFromSmiles(comp)
            if mol is None:
                return None
            canon_components.append(Chem.MolToSmiles(mol))
        if not canon_components:
            return None
        out.append(".".join(canon_components))
    return out[0], out[1]


def _all_smiles_concat(reaction: dict) -> str:
    """Concatenate every SMILES on a reaction (reactants/products/conditions)."""
    parts = []
    for bucket in ("reactants", "products", "conditions"):
        for m in reaction.get(bucket) or []:
            s = m.get("smiles") if isinstance(m, dict) else None
            if s:
                parts.append(s)
    return ".".join(parts)


def _inclusion_decision(reaction: dict, fig_reactions: list[dict]) -> str:
    """Decide whether to keep a reaction in the CSV.

    Returns one of:
      - 'KEEP_FULL'              all SMILES are concrete -> Rxn-INSIGHT runs
      - 'KEEP_PARTIAL'           some wildcards but reaction is genuine -> skip Rxn-INSIGHT, keep row
      - 'DROP_GENERIC'           every reaction on this figure is wildcarded (cover/strategy scheme)
      - 'DROP_TEMPLATE_PARENT'   reaction_id ends in '_template' AND a concrete sibling exists on this figure
    """
    smi_self = _all_smiles_concat(reaction)
    has_ph = _has_placeholder(smi_self)
    if not has_ph:
        return "KEEP_FULL"

    all_ph = all(_has_placeholder(_all_smiles_concat(r)) for r in fig_reactions)
    if all_ph:
        return "DROP_GENERIC"

    rid = reaction.get("reaction_id") or ""
    if rid.endswith("_template"):
        has_concrete_sibling = any(
            (not (r.get("reaction_id") or "").endswith("_template"))
            and not _has_placeholder(_all_smiles_concat(r))
            for r in fig_reactions
        )
        if has_concrete_sibling:
            return "DROP_TEMPLATE_PARENT"

    return "KEEP_PARTIAL"


def _condition_text(reaction: dict, role: str) -> str:
    parts = []
    for c in reaction.get("conditions") or []:
        if (c.get("role") or "").lower() == role.lower():
            text = c.get("text") or c.get("smiles") or ""
            if text:
                parts.append(str(text))
    return ".".join(parts)


# --- Rxn-INSIGHT call ----------------------------------------------------------


def _run_rxn_insight(Reaction, r: dict, smirks_db, get_class_name) -> dict:
    smi = r["_two_part"]
    if not smi or smi == ">>":
        return {"_skip": "empty SMILES"}
    try:
        rxn = Reaction(
            smi,
            smirks=smirks_db,
            solvent=r["_solvent_hint"],
            reagent=r["_reagent_hint"],
            catalyst=r["_catalyst_hint"],
        )
        info = rxn.get_reaction_info()
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}

    raw_name = (info.get("NAME") or "").strip()
    # SMIRKS DB returns dotted class codes (e.g. '5.1.9.22'). Expand into the
    # full tier hierarchy via gen_rxn_insight.naming.get_class_name. Each tier
    # gets its own CSV column; `name` carries the deepest tier for convenience.
    tiers: dict[int, str] = {}
    if raw_name and raw_name != "OtherReaction":
        resolved = _resolve_class_name(raw_name, get_class_name)
        readable = resolved["name"]
        tiers = resolved["tiers"]
    else:
        readable = raw_name
    out: dict[str, Any] = {
        "name": readable,
        "tiers": tiers,
        "class": info.get("CLASS") or "",
        "solvent": _to_csv_cell(info.get("SOLVENT")),
        "reagent": _to_csv_cell(info.get("REAGENT")),
        "catalyst": _to_csv_cell(info.get("CATALYST")),
        "byproducts": _to_csv_cell(info.get("BY-PRODUCTS")),
        "scaffold": info.get("SCAFFOLD") or "",
        "template_r0p1": "",
    }
    # Fallback template when SMIRKS DB couldn't name the reaction.
    if not raw_name or raw_name == "OtherReaction":
        try:
            tmpl = rxn.get_detailed_template(radius=0, radius_products=1)
            out["template_r0p1"] = tmpl or ""
        except Exception as e:
            out["template_r0p1"] = ""
            out["_template_error"] = f"{type(e).__name__}: {e}"
    return out


def _to_csv_cell(v) -> str:
    if isinstance(v, list):
        return ".".join(str(x) for x in v if x)
    if v is None:
        return ""
    return str(v)


# --- text-LLM batched call -----------------------------------------------------


_TEXT_SYS_PROMPT = """You are a chemistry assistant. You will receive:
1. The text of one page of a chemistry paper.
2. A list of reactions extracted from figures on that page (each with a reaction_id, the reaction SMILES, and the conditions parsed directly from the figure).

For each reaction, infer from the page text:
- name: the named reaction if the text mentions one (e.g. "Suzuki coupling", "Sonogashira reaction"). null if the text doesn't name it.
- solvent: solvent(s) named in the text for this reaction. null if not stated.
- catalyst: catalyst(s) named in the text. null if not stated.
- reagent: any other reagent(s) named in the text. null if not stated.
- procedure: one short sentence summarizing the reaction procedure (temperature, time, atmosphere, e.g. "Stirred at 60 C in THF for 12 h under N2"). null if not described.
- workup: the post-reaction workup as described in the text (quench, extraction, wash, drying), e.g. "Quenched with sat. NH4Cl, extracted with EtOAc, washed with brine, dried over Na2SO4". null if not described.
- purification: the purification method as described, e.g. "Silica gel chromatography (hexane/EtOAc 9:1)" or "Recrystallization from EtOH". null if not described.
- analysis: characterization techniques and notable spectral data as described, e.g. "1H NMR, 13C NMR, HRMS (m/z 245.1178)" or "Confirmed by X-ray". null if not described.
- completed_smiles: ONLY if the reaction's SMILES contains R-group placeholders (`*`, `[*]`, `[1*]`, `[2*]`, `[R..]`, `[Ar..]`) AND you can confidently infer the actual substituent from sibling reactions on this page or from the page text. Return the completed reactants>>products SMILES (2-part, no reagents) with NO wildcards remaining. Use atom-by-atom substitution: if `*` in this reaction occupies the same chemical position as a concrete group `X` in a sibling reaction, replace `*` with `X`. If you cannot resolve every wildcard with high confidence, return null — do NOT guess. If the reaction's SMILES has no wildcards at all, return null.

Use only what is stated or strongly implied by the page text and sibling reactions. Don't invent. Always return one entry per reaction_id you were given, in the same order.

Respond with strict JSON, no prose:
{"reactions": [{"reaction_id": "...", "name": ..., "solvent": ..., "catalyst": ..., "reagent": ..., "procedure": ..., "workup": ..., "purification": ..., "analysis": ..., "completed_smiles": ...}, ...]}"""


def _run_text_llm(
    flat: list[dict],
    page_text_by_num: dict[int, str],
    *,
    model: str,
) -> dict[tuple[int, str], dict]:
    if not flat:
        return {}

    from utils.llm_client import get_client
    client = get_client()

    # Group reactions by page.
    by_page: dict[int, list[dict]] = {}
    for r in flat:
        by_page.setdefault(r["page"], []).append(r)

    # Build the full manuscript text once. Workup/purification/analysis often
    # live in the experimental section on later pages of the main paper (and
    # not on the page where the figure was cropped). Giving every per-page
    # batch the full manuscript lets the LLM pull from the experimental
    # section when answering questions about reactions on earlier pages.
    full_manuscript = "\n\n".join(
        f"[Page {p}]\n{page_text_by_num[p]}"
        for p in sorted(page_text_by_num)
        if page_text_by_num[p]
    )
    if len(full_manuscript) > 80_000:
        full_manuscript = full_manuscript[:80_000] + "\n... (manuscript truncated)"

    cache_name = None
    try:
        cache_name = client.create_cache(
            model=model,
            system_instruction=_TEXT_SYS_PROMPT,
            ttl_seconds=3600,
            display_name="chemeagle_classify_v1",
        )
    except Exception:
        cache_name = None

    out: dict[tuple[int, str], dict] = {}
    try:
        for page in sorted(by_page):
            page_reactions = by_page[page]
            user_msg = _build_user_msg(full_manuscript, page, page_reactions)
            messages: list[dict] = []
            extra_body: dict[str, Any] = {"thinking_budget": 0}
            if cache_name:
                extra_body["cached_content"] = cache_name
            else:
                messages.append({"role": "system", "content": _TEXT_SYS_PROMPT})
            messages.append({"role": "user", "content": user_msg})
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=4096,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content or "{}"
                parsed = json.loads(content)
            except Exception:
                # Leave entries empty for this page; downstream rows will have
                # blank text_* columns.
                continue
            for entry in parsed.get("reactions") or []:
                rid = entry.get("reaction_id", "")
                if not rid:
                    continue
                out[(page, rid)] = {
                    "name": _maybe_str(entry.get("name")),
                    "solvent": _maybe_str(entry.get("solvent")),
                    "catalyst": _maybe_str(entry.get("catalyst")),
                    "reagent": _maybe_str(entry.get("reagent")),
                    "procedure": _maybe_str(entry.get("procedure")),
                    "workup": _maybe_str(entry.get("workup")),
                    "purification": _maybe_str(entry.get("purification")),
                    "analysis": _maybe_str(entry.get("analysis")),
                    "completed_smiles": _maybe_str(entry.get("completed_smiles")),
                }
    finally:
        if cache_name:
            try:
                client.delete_cache(cache_name)
            except Exception:
                pass
    return out


def _maybe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ".".join(str(x) for x in v if x)
    return str(v).strip()


def _build_user_msg(full_manuscript: str, page: int, reactions: list[dict]) -> str:
    lines = []
    for r in reactions:
        cond_summary = "; ".join(
            f"{c.get('role') or '?'}={c.get('text') or c.get('smiles') or ''}"
            for c in (r["raw"].get("conditions") or [])
            if c.get("text") or c.get("smiles")
        )
        lines.append(
            f"- reaction_id={r['reaction_id']!r} smiles={r['rxn_smiles']!r} "
            f"figure_conditions=[{cond_summary}]"
        )
    return (
        "=== FULL MANUSCRIPT ===\n"
        f"{full_manuscript}\n"
        "=== END MANUSCRIPT ===\n\n"
        f"The following reactions were extracted from figures on Page {page}. "
        "Use the WHOLE manuscript above as context. The procedure/reagents/"
        "solvent/catalyst are usually near the figure on the same page. The "
        "workup, purification, and analysis (NMR/HRMS/etc.) are usually in "
        "the experimental section, often on later pages of the manuscript. "
        "Match each reaction to the corresponding compound number or product "
        "structure mentioned in the experimental write-up.\n\n"
        f"Reactions on Page {page}:\n" + "\n".join(lines) +
        "\n\nReturn the JSON described in the system prompt."
    )


# --- row assembly --------------------------------------------------------------


def _build_row(r: dict, ri: dict, text: dict) -> dict:
    tiers = ri.get("tiers") or {}
    return {
        "pdf": r["pdf"],
        "page": r["page"],
        "figure_png": r["figure_png"],
        "reaction_id": r["reaction_id"],
        "reactants_smiles": r["reactants_smiles"],
        "products_smiles": r["products_smiles"],
        "reagents_smiles": r["reagents_smiles"],
        "rxn_smiles": r["rxn_smiles"],
        "ri_name": ri.get("name", ""),
        "ri_class": ri.get("class", ""),
        "ri_tier_1": tiers.get(1, ""),
        "ri_tier_2": tiers.get(2, ""),
        "ri_tier_3": tiers.get(3, ""),
        "ri_tier_4": tiers.get(4, ""),
        "ri_tier_5": tiers.get(5, ""),
        "ri_solvent": ri.get("solvent", ""),
        "ri_reagent": ri.get("reagent", ""),
        "ri_catalyst": ri.get("catalyst", ""),
        "ri_byproducts": ri.get("byproducts", ""),
        "ri_scaffold": ri.get("scaffold", ""),
        "ri_template_r0p1": ri.get("template_r0p1", ""),
        "text_name": text.get("name", ""),
        "text_solvent": text.get("solvent", ""),
        "text_catalyst": text.get("catalyst", ""),
        "text_reagent": text.get("reagent", ""),
        "text_procedure": text.get("procedure", ""),
        "text_workup": text.get("workup", ""),
        "text_purification": text.get("purification", ""),
        "text_analysis": text.get("analysis", ""),
        "chemeagle_conditions": json.dumps(
            r["raw"].get("conditions") or [], ensure_ascii=False,
        ),
        "ri_error": ri.get("_error") or ri.get("_skip") or ri.get("_template_error") or "",
    }
