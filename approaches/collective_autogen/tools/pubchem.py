"""Chemical name → SMILES resolution.

Multi-source cascade (OPSIN → PubChem → CIR) plus an abbreviations lexicon
and a shorthand parser for substrate-table fragments. Tool name kept as
`lookup_compound` for back-compat with agent prompts and tool wiring.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .trace import trace

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "pubchem"

OPSIN_BASE = "https://opsin.ch.cam.ac.uk/opsin/"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CIR_BASE = "https://cactus.nci.nih.gov/chemical/structure"

# Common chemistry abbreviations → English/IUPAC name we can re-query with.
# Lower-cased keys; matched case-insensitively.
GROUP_LEXICON: dict[str, str] = {
    # Reagents / bases / solvents that PubChem doesn't always synonym-match.
    "lda": "lithium diisopropylamide",
    "lihmds": "lithium bis(trimethylsilyl)amide",
    "nahmds": "sodium bis(trimethylsilyl)amide",
    "khmds": "potassium bis(trimethylsilyl)amide",
    "tbaf": "tetrabutylammonium fluoride",
    "dipea": "N,N-diisopropylethylamine",
    "dbu": "1,8-diazabicyclo[5.4.0]undec-7-ene",
    "thf": "oxolane",
    "dmf": "N,N-dimethylformamide",
    "dmso": "dimethyl sulfoxide",
    "dcm": "dichloromethane",
    "dce": "1,2-dichloroethane",
    "etoh": "ethanol",
    "meoh": "methanol",
    "ipa": "propan-2-ol",
    "mecn": "acetonitrile",
    "tfa": "trifluoroacetic acid",
    "tfaa": "trifluoroacetic anhydride",
    "ddq": "2,3-dichloro-5,6-dicyano-1,4-benzoquinone",
    # Halogens / alkyl
    "f": "fluoro", "cl": "chloro", "br": "bromo", "i": "iodo",
    "me": "methyl", "et": "ethyl", "npr": "propyl", "ipr": "propan-2-yl",
    "nbu": "butyl", "sbu": "butan-2-yl", "ibu": "2-methylpropyl", "tbu": "tert-butyl",
    # Alkoxy / ester / protecting groups
    "ome": "methoxy", "meo": "methoxy", "oet": "ethoxy", "oipr": "propan-2-yloxy",
    "otbu": "tert-butoxy", "obn": "benzyloxy", "oph": "phenoxy", "ocf3": "trifluoromethoxy",
    "oac": "acetoxy", "opiv": "pivaloyloxy",
    "ots": "4-tolylsulfonyloxy", "oms": "methanesulfonyloxy", "otf": "trifluoromethanesulfonyloxy",
    # Nitrogen / carbonyl / EWG
    "no2": "nitro", "nh2": "amino", "nme2": "dimethylamino", "net2": "diethylamino",
    "cho": "formyl", "ac": "acetyl", "coch3": "acetyl", "bz": "benzoyl",
    "cf3": "trifluoromethyl", "cn": "cyano",
    "conh2": "carbamoyl", "conme2": "dimethylcarbamoyl",
    "co2me": "methoxycarbonyl", "co2et": "ethoxycarbonyl",
    "co2ipr": "propan-2-yloxycarbonyl", "co2tbu": "tert-butoxycarbonyl",
    "boc": "tert-butoxycarbonyl", "cbz": "benzyloxycarbonyl",
    "fmoc": "9-fluorenylmethoxycarbonyl",
    # Sulfur / sulfonyl
    "sme": "methylsulfanyl", "set": "ethylsulfanyl", "sph": "phenylsulfanyl",
    "so2me": "methanesulfonyl", "so2ph": "phenylsulfonyl", "so2cf3": "trifluoromethanesulfonyl",
    # Other
    "cf3o": "trifluoromethoxy", "n3": "azido",
    # Ring aliases
    "ph": "phenyl", "py": "pyridyl", "th": "thienyl", "fur": "furyl", "np": "naphthyl", "ind": "indolyl",
    # Functional group expressed as suffix
    "b(oh)2": "boronic acid",
}


# ---------- low-level HTTP helpers ----------

def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code != 404
    return True


def _http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Collective-autogen/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
       retry=retry_if_exception(_is_transient))
def _opsin_smiles(name: str) -> str | None:
    url = f"{OPSIN_BASE}{urllib.parse.quote(name, safe='')}.json"
    try:
        status, body = _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    data = json.loads(body)
    if data.get("status") != "SUCCESS":
        return None
    return data.get("smiles") or None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
       retry=retry_if_exception(_is_transient))
def _pubchem_smiles(name: str) -> str | None:
    url = (
        f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name, safe='')}"
        f"/property/SMILES,ConnectivitySMILES,IsomericSMILES,CanonicalSMILES/JSON"
    )
    try:
        _, body = _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    data = json.loads(body)
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    p = props[0]
    for key in ("IsomericSMILES", "SMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        if p.get(key):
            return p[key]
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
       retry=retry_if_exception(_is_transient))
def _cir_smiles(name: str) -> str | None:
    url = f"{CIR_BASE}/{urllib.parse.quote(name, safe='')}/smiles"
    try:
        _, body = _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    body = body.strip()
    return body or None


_RESOLVERS = (
    ("opsin", _opsin_smiles),
    ("pubchem", _pubchem_smiles),
    ("cir", _cir_smiles),
)


def _try_resolvers_for_queries(queries: list[str]) -> tuple[str | None, str, str]:
    """Cycle each resolver through every query before falling to the next.

    Order matters: queries that include an expanded abbreviation must be tried
    against the high-trust resolvers (OPSIN, PubChem) before CIR sees the
    original acronym (CIR matches loose synonyms and can return wrong
    compounds for ambiguous strings like "LDA").

    Returns (smiles, source, query_used).
    """
    for src_name, fn in _RESOLVERS:
        for q in queries:
            try:
                s = fn(q)
            except Exception:
                s = None
            if s:
                return s, src_name, q
    return None, "", ""


# ---------- shorthand-substituent parser ----------

# Ring family detection: a regex on the heteroaromatic / aromatic core code,
# its IUPAC family name, and the parent compound name we can prepend
# substituents to.
_RING_FAMILIES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^C6H\d+$"),  "phenyl",   "benzene"),
    (re.compile(r"^C10H\d+$"), "naphthyl", "naphthalene"),
    (re.compile(r"^C5H\d+N$"), "pyridyl",  "pyridine"),
    (re.compile(r"^C4H\d+S$"), "thienyl",  "thiophene"),
    (re.compile(r"^C4H\d+O$"), "furyl",    "furan"),
    (re.compile(r"^C8H\d+N$"), "indolyl",  "indole"),
]
_RING_ALIASES = {"Ph": "benzene", "Py": "pyridine", "Th": "thiophene", "Fur": "furan", "Np": "naphthalene", "Ind": "indole"}


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("−", "-").replace("–", "-").replace("—", "-")


def _ring_parent(ring_code: str) -> str | None:
    if ring_code in _RING_ALIASES:
        return _RING_ALIASES[ring_code]
    for pat, _fam, parent in _RING_FAMILIES:
        if pat.match(ring_code):
            return parent
    return None


def _multiplier(n: int, prefix_form: bool) -> str:
    """Return 'di'/'tri'/'tetra' for n>1 (prefix_form=True for parent-substituted
    naming, False for 'bis'/'tris' enclosing-parens form)."""
    if n <= 1:
        return ""
    if prefix_form:
        return {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}.get(n, f"{n}-")
    return {2: "bis", 3: "tris", 4: "tetrakis"}.get(n, f"{n}-kis")


def _shorthand_candidates(token: str) -> list[str]:
    """Generate IUPAC-name candidates for substrate-table fragments.

    Examples it handles:
        4-NO2C6H4              -> ["1-nitro-4-...benzene"-ish, "4-nitrophenyl"]
        3,5-(CF3)2C6H3         -> ["1,3-bis(trifluoromethyl)benzene", ...]
        4-B(OH)2C6H4           -> ["4-(boronic acid)benzene"-style]
    """
    t = _norm(token)
    pat = re.compile(
        r"(?P<pos>\d(?:,\d)*)-"
        r"(?P<grp>\([A-Za-z0-9]+(?:\([A-Za-z0-9]+\))?\)\d+|[A-Za-z0-9()]+)"
        r"(?P<ring>(?:C[0-9]+H[0-9]+[NOS]?)|(?:Ph|Py|Th|Fur|Np|Ind))"
    )
    m = pat.fullmatch(t)
    if not m:
        return []
    positions = [int(p) for p in m.group("pos").split(",")]
    grp_raw = m.group("grp")
    ring_code = m.group("ring")

    parent = _ring_parent(ring_code)
    if parent is None:
        return []

    # Strip explicit (X)n multiplier on the substituent.
    mult_match = re.fullmatch(r"\(([^()]+(?:\([^()]+\))?)\)(\d+)", grp_raw)
    grp_core = mult_match.group(1) if mult_match else grp_raw

    # Map shorthand → IUPAC name (lower-case lookup).
    group_name = GROUP_LEXICON.get(grp_core.lower(), grp_core)
    is_suffix = group_name in {"boronic acid"}

    candidates: list[str] = []
    n_pos = len(positions)
    if is_suffix:
        if n_pos == 1:
            candidates.append(f"{parent}-{positions[0]}-{group_name}")
        else:
            mult = _multiplier(n_pos, prefix_form=True)
            candidates.append(f"{parent}-{','.join(map(str, positions))}-{mult}{group_name}")
    else:
        # Form A: prefix substituent on the parent (e.g. 1,3-bis(trifluoromethyl)benzene)
        if n_pos == 1:
            candidates.append(f"{positions[0]}-{group_name}{parent[0:0]}{parent}")
            candidates.append(f"{positions[0]}-{group_name}-{parent}")
        else:
            mult = _multiplier(n_pos, prefix_form=False)
            candidates.append(f"{','.join(map(str, positions))}-{mult}({group_name}){parent}")
        # Form B: same group as a substituted -yl (less common but OPSIN handles it)
        if n_pos == 1:
            candidates.append(f"{positions[0]}-{group_name}phenyl" if parent == "benzene" else f"{positions[0]}-{group_name}-{parent}-yl")

    return candidates


# ---------- public tool ----------

def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())[:120]


@trace()
def lookup_compound(name: str) -> dict:
    """Resolve a chemical name to a SMILES via OPSIN → PubChem → CIR.

    Strategy:
    1. Try the input verbatim through the resolver chain.
    2. If it fails AND the input matches an entry in GROUP_LEXICON, retry with
       the expanded form.
    3. If it fails AND the input looks like a substrate-table shorthand
       (e.g. "4-NO2C6H4"), generate IUPAC-name candidates and try each.

    Returns:
        {smiles: str|None, source: str, cached: bool, error: str|None}.
        `source` is one of: "opsin", "pubchem", "cir", or "" on failure.
    """
    if not isinstance(name, str) or not name.strip():
        return {"smiles": None, "source": "", "cached": False, "error": "empty input"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_slugify(name)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            return cached
        except Exception:
            pass  # fall through to live fetch

    queries: list[str] = [name]
    expanded = GROUP_LEXICON.get(name.strip().lower())
    if expanded:
        queries.append(expanded)
    queries.extend(_shorthand_candidates(name))
    # de-dup while preserving order
    seen = set()
    queries = [q for q in queries if not (q in seen or seen.add(q))]

    smiles, source, query_used = _try_resolvers_for_queries(queries)

    result = {
        "smiles": smiles,
        "source": source,
        "query_used": query_used,
        "cached": False,
        "error": None if smiles else "not found",
    }
    if smiles:
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False))
        except Exception:
            pass
    return result


def _all_lexicon_keys() -> Iterable[str]:
    """Helper for tests."""
    return GROUP_LEXICON.keys()
