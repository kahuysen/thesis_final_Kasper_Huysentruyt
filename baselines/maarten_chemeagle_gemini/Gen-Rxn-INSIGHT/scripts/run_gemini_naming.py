"""
Name all 335K unique reaction templates using Gemini + chemistry ontology.

Run:
    python scripts/run_gemini_naming.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project src to path so gen_rxn_insight imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.name_templates_gemini import TemplateNamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gemini_naming.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY        = "AIza..."   # ← paste your key here

DATA_PATH      = r"C:\Users\mrodobbe\OneDrive - UGent\Documents\EPFL_Research\Reaction Classification\data\classification_database.parquet"
ONTOLOGY_PATH  = r"C:\Users\mrodobbe\OneDrive - UGent\Documents\EPFL_Research\Reaction Classification\dynamic_mapping.json"
CHECKPOINT     = r"C:\Users\mrodobbe\OneDrive - UGent\Documents\EPFL_Research\Reaction Classification\data\gemini_checkpoint.parquet"
OUTPUT         = r"C:\Users\mrodobbe\OneDrive - UGent\Documents\EPFL_Research\Reaction Classification\data\template_classes.parquet"
LABELED_OUTPUT = r"C:\Users\mrodobbe\OneDrive - UGent\Documents\EPFL_Research\Reaction Classification\data\classification_database_labeled.parquet"

N_JOBS             = 8      # parallel threads
REQUESTS_PER_MIN   = 1000   # match your Gemini quota (paid tier)
N_EXAMPLES         = 3      # unmapped reaction examples per prompt
SMOKE_TEST_N       = 10     # set to 0 to skip smoke test

# ── Load data ─────────────────────────────────────────────────────────────────

log.info("Loading data...")
df = pd.read_parquet(DATA_PATH)
log.info(f"  Reactions:        {len(df):,}")
log.info(f"  Unique templates: {df['TEMPLATE'].nunique():,}")

# ── Build example reactions map ───────────────────────────────────────────────
# For each template, collect up to 5 unmapped reaction SMILES.
# Gemini uses these to confirm its classification.

log.info("Building example reactions map...")
example_map: dict[str, list[str]] = (
    df.groupby("TEMPLATE")["SANITIZED_REACTION"]
    .apply(lambda s: s.head(5).tolist())
    .to_dict()
)

unique_templates = df["TEMPLATE"].unique().tolist()
log.info(f"  Templates to name: {len(unique_templates):,}")

# ── Init namer ────────────────────────────────────────────────────────────────

namer = TemplateNamer(
    api_key=API_KEY,
    ontology_path=ONTOLOGY_PATH,
    model="gemini-2.0-flash",
    two_stage=True,    # 2-stage: ~$130 total vs ~$1140 single-stage
    temperature=0.1,
)

# ── Smoke test ────────────────────────────────────────────────────────────────

if SMOKE_TEST_N > 0:
    log.info(f"Smoke test ({SMOKE_TEST_N} templates)...")
    ok = 0
    for t in unique_templates[:SMOKE_TEST_N]:
        r = namer.name_template(t, example_reactions=example_map.get(t))
        status = "OK" if r["in_ontology"] else "MISS"
        log.info(
            f"  [{status}] conf={r['confidence']:.2f}  "
            f"{(r['final_reaction_class'] or 'NONE')[:80]}"
        )
        if r["in_ontology"]:
            ok += 1
    log.info(f"Smoke test: {ok}/{SMOKE_TEST_N} in ontology.")
    if ok < SMOKE_TEST_N // 2:
        log.error("Too many misses in smoke test. Check API key and ontology path.")
        sys.exit(1)

# ── Full batch ────────────────────────────────────────────────────────────────

log.info("Starting full batch run...")
result_df = namer.name_templates_batch(
    templates=unique_templates,
    example_reactions_map=example_map,
    n_examples=N_EXAMPLES,
    n_jobs=N_JOBS,
    requests_per_minute=REQUESTS_PER_MIN,
    checkpoint_path=CHECKPOINT,
    checkpoint_every=500,
    progress=True,
)

result_df.to_parquet(OUTPUT, index=False)
log.info(f"Template classes saved → {OUTPUT}")

# ── Summary ───────────────────────────────────────────────────────────────────

total     = len(result_df)
in_ont    = result_df["in_ontology"].sum()
novel     = result_df["is_novel"].sum()
fuzzy     = result_df["fuzzy_match"].sum()
errors    = result_df["error"].notna().sum()
high_conf = (result_df["confidence"] >= 0.7).sum()

log.info("── Results ───────────────────────────────")
log.info(f"  Total templates:     {total:,}")
log.info(f"  In ontology:         {in_ont:,}  ({in_ont/total:.1%})")
log.info(f"  High confidence:     {high_conf:,}  ({high_conf/total:.1%})")
log.info(f"  Novel (no match):    {novel:,}  ({novel/total:.1%})")
log.info(f"  Fuzzy-matched:       {fuzzy:,}  ({fuzzy/total:.1%})")
log.info(f"  Errors:              {errors:,}  ({errors/total:.1%})")

# ── Join labels back to all reactions ─────────────────────────────────────────

log.info("Joining labels back to reactions...")
label_map = (
    result_df[result_df["in_ontology"]]
    .set_index("template")["final_reaction_class"]
    .to_dict()
)
df["REACTION_CLASS"] = df["TEMPLATE"].map(label_map)
df["CLASS_CONFIDENCE"] = df["TEMPLATE"].map(
    result_df.set_index("template")["confidence"].to_dict()
)

labeled_frac = df["REACTION_CLASS"].notna().mean()
log.info(f"  Reactions labeled:  {labeled_frac:.1%}  ({df['REACTION_CLASS'].notna().sum():,})")

df.to_parquet(LABELED_OUTPUT, index=False)
log.info(f"Labeled reactions saved → {LABELED_OUTPUT}")
