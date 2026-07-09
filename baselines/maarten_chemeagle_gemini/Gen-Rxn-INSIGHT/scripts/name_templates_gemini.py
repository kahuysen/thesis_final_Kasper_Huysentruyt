"""Gemini-powered reaction template classifier.

Names reaction templates (SMIRKS) by classifying them against a hierarchical
chemistry ontology (dynamic_mapping.json).  Works at the **template** level:
one LLM call per unique template (~335K calls instead of 1.7M per reaction).

Usage (CLI)
-----------
python name_templates_gemini.py \\
    --templates templates.parquet \\
    --ontology dynamic_mapping.json \\
    --output named_templates.parquet \\
    --api-key $GEMINI_API_KEY

Or import directly:
    from scripts.name_templates_gemini import TemplateNamer

Cost estimate (Gemini Flash 2.0 — paid tier, $0.10/1M input tokens)
---------------------------------------------------------------------
* Ontology block in system prompt:  ~135K tokens (8 907 entries × ~60 chars)
* Context caching cuts cached token cost to $0.025/1M (75% cheaper).
* Per-template query:               ~200 tokens (SMIRKS + 3 examples)
* Cached system prompt cost:        135K × $0.025/1M × 335K calls ≈ $1 130
* Query tokens:                     200 × $0.10/1M × 335K calls   ≈ $7
  ------------------------------------------------------------------
  Total ≈ $1 140   (without caching: ~$4 530)

For a much cheaper alternative, use the 2-stage flag (--two-stage):
  Stage 1 picks one of 68 broad sections; Stage 2 picks within that section.
  This reduces system-prompt tokens by ~98% (from 135K to ~2K per call).
  Two-stage estimated cost: ~$130 for 335K templates (2 calls each).

Dependencies
------------
    pip install google-genai pydantic tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Ontology helpers ──────────────────────────────────────────────────────────

def load_ontology(path: str | Path) -> dict:
    """Load and parse dynamic_mapping.json.

    Returns a dict with keys:
        all_classes     – all non-condition entries (8 907 strings)
        top_level       – 1-dot entries only (68 broad sections)
        sections        – dict mapping top-level key (e.g. "1.1") to that
                          section's non-condition entries
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, list[str]] = json.load(f)

    all_classes: list[str] = []
    top_level: list[str] = []
    sections: dict[str, list[str]] = {}

    for section_key, entries in raw.items():
        section_entries = [e for e in entries if "cond:" not in e]
        sections[section_key] = section_entries
        all_classes.extend(section_entries)

        # The first entry in each section list is the section header itself
        if section_entries:
            top_level.append(section_entries[0])

    return {
        "all_classes": all_classes,
        "top_level": top_level,
        "sections": sections,
        "raw": raw,
    }


def _section_key_from_class(class_str: str) -> str | None:
    """Extract the 2-part section key (e.g. '1.3') from a class string."""
    code = class_str.split(" ", 1)[0]
    parts = code.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return None


# ── Pydantic response models ───────────────────────────────────────────────────

class TemplateClassification(BaseModel):
    reasoning: str = Field(
        description=(
            "Step-by-step analysis: identify functional groups, bond changes, "
            "and the reaction mechanism of this template."
        )
    )
    reaction_center_description: str = Field(
        description=(
            "Brief plain-English description of the key bond formed or broken "
            "(e.g. 'amide bond formation from carboxylic acid + amine')."
        )
    )
    broad_class: str = Field(
        description=(
            "The top-level or second-level class that best fits this template "
            "(e.g. '1.3 Acylation of Nitrogen' or '1.3.1 Amide Bond Formation'). "
            "Copy VERBATIM from the ontology."
        )
    )
    final_reaction_class: str = Field(
        description=(
            "The MOST SPECIFIC class from the ontology that matches this template. "
            "Copy the class string VERBATIM including its numeric prefix."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in the classification (1.0 = certain, 0.0 = no idea).",
    )
    is_novel: bool = Field(
        description=(
            "True if no ontology class fits well. Set final_reaction_class to "
            "the closest match anyway, and flag this as novel."
        )
    )


class BroadClassification(BaseModel):
    """Stage-1 output for 2-stage classification."""
    reasoning: str = Field(description="Brief reasoning for broad class selection.")
    broad_class: str = Field(
        description=(
            "The broad section that best fits this template. "
            "Copy VERBATIM from the provided list (e.g. '1.3 Acylation of Nitrogen')."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max_per_minute
        self._window = 60.0
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop calls outside the window
                while self._calls and now - self._calls[0] >= self._window:
                    self._calls.popleft()
                if len(self._calls) < self._max:
                    self._calls.append(now)
                    return
                wait_time = self._window - (now - self._calls[0]) + 0.01
            time.sleep(wait_time)


# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FULL = """\
You are an expert organic chemist specialising in reaction classification.

Your task: given a reaction template in SMIRKS notation, assign it to the MOST \
SPECIFIC matching class from the ontology below.

== HOW TO READ A SMIRKS TEMPLATE ==
- Atoms with map numbers (e.g. [C:1]) are in the reaction centre.
- The left side shows reactant substructure; the right side shows product substructure.
- Atoms/bonds that appear only on one side represent groups added or lost.
- Stereo descriptors (@, @@, /, \\) indicate stereochemistry constraints.

== CLASSIFICATION ONTOLOGY ==
Classes are numbered hierarchically.  Higher specificity = more dots in the number.
You MUST copy the class string EXACTLY as written below (number + description).

{ontology_block}

== END ONTOLOGY ==

Instructions:
1. Identify what new bond is formed or what bond is broken.
2. Identify the functional groups on the reactant and product sides.
3. Match to the deepest (most specific) applicable class.
4. If example reactions are shown, use them to confirm the class.
5. If two classes fit equally well, choose the shallower (less specific) one and \
   lower confidence below 0.7.
6. Set is_novel=true ONLY if no class fits at all; still provide the closest match.
"""

_SYSTEM_PROMPT_STAGE1 = """\
You are an expert organic chemist.

Your task: assign a reaction template to ONE of the following broad reaction \
sections.  You will classify more specifically in a later step.

BROAD SECTIONS (copy VERBATIM):
{top_level_block}

Copy the chosen section string EXACTLY, including its numeric prefix.
"""

_SYSTEM_PROMPT_STAGE2 = """\
You are an expert organic chemist.

The reaction template has been broadly assigned to section "{broad_class}".
Now choose the MOST SPECIFIC class from that section:

AVAILABLE CLASSES (copy VERBATIM):
{section_block}

Instructions:
- Copy the class string EXACTLY including its numeric prefix.
- Choose the deepest (most specific) applicable class.
- If no class fits well, pick the closest and set is_novel=true.
"""


def _build_user_prompt(
    template: str,
    example_reactions: list[str] | None,
    n_examples: int,
) -> str:
    lines = [
        "Classify the following reaction template.",
        "",
        f"Template SMIRKS:  {template}",
    ]
    if example_reactions:
        n = min(len(example_reactions), n_examples)
        lines += ["", f"Example reactions ({n} shown, unmapped SMILES):"]
        for rxn in example_reactions[:n]:
            lines.append(f"  {rxn}")
    return "\n".join(lines)


# ── Main class ─────────────────────────────────────────────────────────────────

class TemplateNamer:
    """Names reaction templates via Gemini structured output.

    Args:
        api_key:       Gemini API key (falls back to GEMINI_API_KEY env var).
        model:         Gemini model ID.
        ontology_path: Path to dynamic_mapping.json.
        temperature:   Sampling temperature (0.1 recommended for classification).
        two_stage:     Use 2-stage classification (cheaper; see module docstring).
        use_cache:     Enable Gemini context caching for the system prompt
                       (supported on gemini-1.5-flash-001 / gemini-1.5-pro-001 /
                       gemini-2.0-flash-001).  Saves ~75% of system-prompt tokens.

    Example::

        namer = TemplateNamer(
            api_key="...",
            ontology_path="dynamic_mapping.json",
            two_stage=True,
        )
        result = namer.name_template(
            "[C:1](=[O:2])[OH:3].[NH2:4]>>[C:1](=[O:2])[NH:4]",
            example_reactions=["CC(=O)O.CCN>>CC(=O)NCC"],
        )
        print(result["final_reaction_class"])
        # → '1.3.1.1 Carboxylic Acid + Primary Amine -> Amide'
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        ontology_path: str | Path = "dynamic_mapping.json",
        temperature: float = 0.1,
        two_stage: bool = False,
        use_cache: bool = False,
    ) -> None:
        from google import genai
        from google.genai import types as gtypes

        self._genai = genai
        self._gtypes = gtypes

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.two_stage = two_stage

        ont = load_ontology(ontology_path)
        self._all_classes: list[str] = ont["all_classes"]
        self._top_level: list[str] = ont["top_level"]
        self._sections: dict[str, list[str]] = ont["sections"]
        self._class_set: set[str] = set(self._all_classes)

        # Build system prompts
        ontology_block = "\n".join(self._all_classes)
        self._system_prompt_full = _SYSTEM_PROMPT_FULL.format(
            ontology_block=ontology_block
        )
        top_level_block = "\n".join(self._top_level)
        self._system_prompt_stage1 = _SYSTEM_PROMPT_STAGE1.format(
            top_level_block=top_level_block
        )

        # Optionally create a context cache for the heavy single-stage prompt
        self._cache_name: str | None = None
        if use_cache and not two_stage:
            self._cache_name = self._create_cache()

    # ── Context caching ────────────────────────────────────────────────────────

    def _create_cache(self) -> str | None:
        """Create a Gemini context cache for the system prompt.

        Returns the cache name string, or None if caching fails (e.g.
        because the model does not support it).
        """
        try:
            cache = self.client.caches.create(
                model=self.model,
                config=self._gtypes.CreateCachedContentConfig(
                    system_instruction=self._system_prompt_full,
                    ttl="3600s",
                ),
            )
            logger.info(f"Context cache created: {cache.name}")
            return cache.name
        except Exception as exc:
            logger.warning(
                f"Context caching not available for model {self.model!r}: {exc}. "
                "Falling back to uncached mode."
            )
            return None

    def refresh_cache(self) -> None:
        """Renew the context cache TTL (call every ~50 minutes for long runs)."""
        if self._cache_name is None:
            return
        try:
            self.client.caches.update(
                name=self._cache_name,
                config=self._gtypes.UpdateCachedContentConfig(ttl="3600s"),
            )
            logger.info(f"Cache TTL renewed: {self._cache_name}")
        except Exception as exc:
            logger.warning(f"Cache renewal failed: {exc}. Recreating.")
            self._cache_name = self._create_cache()

    # ── Internal Gemini call ───────────────────────────────────────────────────

    def _call(
        self,
        system_instruction: str | None,
        user_prompt: str,
        response_schema,
        cached_content: str | None = None,
        retries: int = 3,
    ) -> Any:
        """Make one Gemini call with retries. Returns the parsed Pydantic object."""
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": self.temperature,
        }
        if cached_content:
            config_kwargs["cached_content"] = cached_content
        elif system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=self._gtypes.GenerateContentConfig(**config_kwargs),
                    contents=[user_prompt],
                )
                return response.parsed
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s."
                )
                time.sleep(wait)
        raise RuntimeError(f"All {retries} attempts failed") from last_exc

    # ── Single prediction (1-stage) ───────────────────────────────────────────

    def _name_single_stage(
        self,
        template: str,
        example_reactions: list[str] | None,
        n_examples: int,
        retries: int,
    ) -> dict[str, Any]:
        user_prompt = _build_user_prompt(template, example_reactions, n_examples)
        result: TemplateClassification = self._call(
            system_instruction=self._system_prompt_full,
            user_prompt=user_prompt,
            response_schema=TemplateClassification,
            cached_content=self._cache_name,
            retries=retries,
        )
        return {
            "final_reaction_class": result.final_reaction_class,
            "broad_class": result.broad_class,
            "reasoning": result.reasoning,
            "reaction_center_description": result.reaction_center_description,
            "confidence": result.confidence,
            "is_novel": result.is_novel,
        }

    # ── Single prediction (2-stage) ───────────────────────────────────────────

    def _name_two_stage(
        self,
        template: str,
        example_reactions: list[str] | None,
        n_examples: int,
        retries: int,
    ) -> dict[str, Any]:
        user_prompt = _build_user_prompt(template, example_reactions, n_examples)

        # Stage 1: broad section
        stage1: BroadClassification = self._call(
            system_instruction=self._system_prompt_stage1,
            user_prompt=user_prompt,
            response_schema=BroadClassification,
            retries=retries,
        )
        broad_class = stage1.broad_class

        # Find the section whose entries match the broad class
        section_key = _section_key_from_class(broad_class)
        section_entries = self._sections.get(section_key, [])
        if not section_entries:
            # Fallback: fuzzy-match broad class to find the right section
            matches = get_close_matches(broad_class, self._top_level, n=1, cutoff=0.5)
            if matches:
                broad_class = matches[0]
                section_key = _section_key_from_class(broad_class)
                section_entries = self._sections.get(section_key, [])

        if not section_entries:
            # No section found — degrade to the broad class itself
            return {
                "final_reaction_class": broad_class,
                "broad_class": broad_class,
                "reasoning": stage1.reasoning,
                "reaction_center_description": "",
                "confidence": stage1.confidence * 0.5,
                "is_novel": True,
            }

        # Stage 2: specific class within section
        section_block = "\n".join(section_entries)
        system_stage2 = _SYSTEM_PROMPT_STAGE2.format(
            broad_class=broad_class,
            section_block=section_block,
        )
        stage2: TemplateClassification = self._call(
            system_instruction=system_stage2,
            user_prompt=user_prompt,
            response_schema=TemplateClassification,
            retries=retries,
        )
        return {
            "final_reaction_class": stage2.final_reaction_class,
            "broad_class": broad_class,
            "reasoning": stage2.reasoning,
            "reaction_center_description": stage2.reaction_center_description,
            "confidence": min(stage1.confidence, stage2.confidence),
            "is_novel": stage2.is_novel,
        }

    # ── Public single prediction ───────────────────────────────────────────────

    def name_template(
        self,
        template: str,
        example_reactions: list[str] | None = None,
        n_examples: int = 3,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Classify a single template SMIRKS.

        Args:
            template:          Reaction template in SMIRKS notation.
            example_reactions: Optional list of unmapped reaction SMILES that
                               this template applies to (up to *n_examples* used).
            n_examples:        Max examples to include in the prompt.
            retries:           API call retries per stage.

        Returns:
            Dict with keys:

            * ``template``                   — input SMIRKS
            * ``final_reaction_class``        — chosen ontology class (verbatim)
            * ``broad_class``                 — chosen broad section
            * ``reasoning``                   — model's chain-of-thought
            * ``reaction_center_description`` — plain-English RC description
            * ``confidence``                  — float in [0, 1]
            * ``is_novel``                    — True if no class fits well
            * ``in_ontology``                 — True if class found in ontology
            * ``fuzzy_match``                 — True if validated via fuzzy match
            * ``error``                       — error string if call failed
        """
        try:
            if self.two_stage:
                raw = self._name_two_stage(template, example_reactions, n_examples, retries)
            else:
                raw = self._name_single_stage(template, example_reactions, n_examples, retries)
        except Exception as exc:
            logger.error(f"Classification failed for {template[:60]!r}: {exc}")
            return {
                "template": template,
                "final_reaction_class": None,
                "broad_class": None,
                "reasoning": None,
                "reaction_center_description": None,
                "confidence": 0.0,
                "is_novel": True,
                "in_ontology": False,
                "fuzzy_match": False,
                "error": str(exc),
            }

        # Post-validate: must match an ontology entry
        frc = raw["final_reaction_class"]
        in_ontology = frc in self._class_set
        fuzzy_match = False
        if frc and not in_ontology:
            matches = get_close_matches(frc, self._all_classes, n=1, cutoff=0.55)
            if matches:
                frc = matches[0]
                in_ontology = True
                fuzzy_match = True
                logger.debug(
                    f"Fuzzy-matched {raw['final_reaction_class']!r} -> {frc!r}"
                )

        return {
            "template": template,
            "final_reaction_class": frc,
            "broad_class": raw.get("broad_class"),
            "reasoning": raw.get("reasoning"),
            "reaction_center_description": raw.get("reaction_center_description"),
            "confidence": raw.get("confidence", 0.0),
            "is_novel": raw.get("is_novel", False),
            "in_ontology": in_ontology,
            "fuzzy_match": fuzzy_match,
            "error": None,
        }

    # ── Batch prediction ───────────────────────────────────────────────────────

    def name_templates_batch(
        self,
        templates: list[str] | pd.Series,
        example_reactions_map: dict[str, list[str]] | None = None,
        n_examples: int = 3,
        n_jobs: int = 4,
        requests_per_minute: int = 60,
        retries: int = 3,
        progress: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 500,
        cache_refresh_every: int = 2_500,
    ) -> pd.DataFrame:
        """Name a batch of unique templates.

        Args:
            templates:            List/Series of SMIRKS strings (unique templates).
            example_reactions_map: Optional dict mapping template SMIRKS ->
                                   list of unmapped reaction SMILES to include
                                   in the prompt.
            n_examples:           Max example reactions per prompt.
            n_jobs:               Parallel Gemini API threads.  Keep <= RPM/5 to
                                  avoid burst violations.
            requests_per_minute:  Rate cap matching your Gemini quota
                                  (free tier: 15; paid: 1000+).
            retries:              Per-call retry count.
            progress:             Show tqdm progress bar.
            checkpoint_path:      Parquet file to save intermediate results.
                                  Existing file is resumed automatically.
            checkpoint_every:     Save frequency (number of completions).
            cache_refresh_every:  Renew context cache every N completions
                                  (only relevant when use_cache=True).

        Returns:
            DataFrame with one row per template.  Columns: template,
            final_reaction_class, broad_class, reasoning,
            reaction_center_description, confidence, is_novel,
            in_ontology, fuzzy_match, error.
        """
        templates_list = list(templates)
        n = len(templates_list)

        results: list[dict[str, Any] | None] = [None] * n
        done_indices: set[int] = set()

        # Resume from checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = pd.read_parquet(checkpoint_path)
            ckpt_map = {row["template"]: row.to_dict() for _, row in ckpt.iterrows()}
            for i, t in enumerate(templates_list):
                if t in ckpt_map:
                    results[i] = ckpt_map[t]
                    done_indices.add(i)
            logger.info(f"Resumed {len(done_indices)}/{n} from checkpoint.")

        todo = [(i, templates_list[i]) for i in range(n) if i not in done_indices]
        rate_limiter = _RateLimiter(requests_per_minute)
        completed = 0
        lock = threading.Lock()

        def call_one(args: tuple[int, str]) -> tuple[int, dict[str, Any]]:
            idx, tmpl = args
            rate_limiter.acquire()
            examples = (example_reactions_map or {}).get(tmpl)
            result = self.name_template(tmpl, examples, n_examples, retries)
            return idx, result

        pbar = tqdm(
            total=len(todo), desc="Naming templates", disable=not progress
        )
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(call_one, item): item for item in todo}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

                with lock:
                    completed += 1
                    pbar.update(1)

                    # Checkpoint
                    if checkpoint_path and completed % checkpoint_every == 0:
                        done = [r for r in results if r is not None]
                        pd.DataFrame(done).to_parquet(checkpoint_path, index=False)
                        logger.info(
                            f"Checkpoint: {len(done)}/{n} saved to {checkpoint_path}"
                        )

                    # Refresh context cache TTL
                    if self._cache_name and completed % cache_refresh_every == 0:
                        self.refresh_cache()

        pbar.close()

        df = pd.DataFrame(results)
        if checkpoint_path:
            df.to_parquet(checkpoint_path, index=False)
        return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Name reaction templates using Gemini + ontology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--templates", required=True,
                   help="Parquet/CSV file with a 'TEMPLATE' column.")
    p.add_argument("--ontology", required=True,
                   help="Path to dynamic_mapping.json.")
    p.add_argument("--output", required=True,
                   help="Output parquet file path.")
    p.add_argument("--reactions-col", default=None,
                   help="Column with unmapped reactions for example enrichment "
                        "(optional).")
    p.add_argument("--template-col", default="TEMPLATE",
                   help="Column name containing template SMIRKS.")
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var).")
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--two-stage", action="store_true",
                   help="Use 2-stage classification (much cheaper; ~$130 vs $1 140).")
    p.add_argument("--use-cache", action="store_true",
                   help="Enable Gemini context caching (saves ~75%% of system-"
                        "prompt tokens; only on supported models).")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--rpm", type=int, default=60,
                   help="Requests per minute (match your Gemini quota).")
    p.add_argument("--n-examples", type=int, default=3,
                   help="Max example reactions per prompt.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--checkpoint", default=None,
                   help="Parquet checkpoint file (auto-resumed if exists).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load templates
    if args.templates.endswith(".csv"):
        df = pd.read_csv(args.templates)
    else:
        df = pd.read_parquet(args.templates)

    unique_templates = df[args.template_col].dropna().unique().tolist()
    logger.info(f"Unique templates to classify: {len(unique_templates)}")

    # Build example reactions map (up to 5 examples per template)
    example_map: dict[str, list[str]] | None = None
    if args.reactions_col and args.reactions_col in df.columns:
        example_map = (
            df.dropna(subset=[args.template_col, args.reactions_col])
            .groupby(args.template_col)[args.reactions_col]
            .apply(lambda s: s.head(5).tolist())
            .to_dict()
        )
        logger.info("Example reactions map built.")

    # Classify
    namer = TemplateNamer(
        api_key=args.api_key,
        model=args.model,
        ontology_path=args.ontology,
        temperature=args.temperature,
        two_stage=args.two_stage,
        use_cache=args.use_cache,
    )

    result_df = namer.name_templates_batch(
        templates=unique_templates,
        example_reactions_map=example_map,
        n_examples=args.n_examples,
        n_jobs=args.n_jobs,
        requests_per_minute=args.rpm,
        checkpoint_path=args.checkpoint,
        progress=True,
    )

    result_df.to_parquet(args.output, index=False)
    logger.info(f"Results saved to {args.output} ({len(result_df)} rows).")

    # Summary stats
    in_ont = result_df["in_ontology"].sum()
    novel = result_df["is_novel"].sum()
    fuzzy = result_df["fuzzy_match"].sum()
    errors = result_df["error"].notna().sum()
    logger.info(
        f"Summary: {in_ont}/{len(result_df)} in ontology, "
        f"{novel} novel, {fuzzy} fuzzy-matched, {errors} errors."
    )


if __name__ == "__main__":
    main()
