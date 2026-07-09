import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    SourceMatchTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import Image as AGImage
from autogen_core.models import ModelFamily, ModelInfo
from eval.usage import summarise_autogen_messages, write_usage_json

from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from pydantic import ValidationError

from schema import ReactionRecord, condition_role_enum, schema_json
from tools import (
    ScriptedMolRecAgent,
    StructuralVerifierAgent,
    apply_substituents,
    atom_balance_check,
    canonicalize_smiles,
    crop_image_region,
    derive_substituents,
    detect_label_coref,
    detect_molecules,
    enumerate_rgroups,
    lookup_compound,
    ocr_image,
    predict_molecule_smiles,
    set_trace_path,
    validate_reaction_json,
    validate_smiles,
)


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, **substitutions: str) -> str:
    text = (PROMPTS_DIR / f"{name}.txt").read_text()
    # Always-available substitutions; per-prompt overrides win.
    defaults = {
        "schema_json": schema_json(),
        "condition_role_enum": condition_role_enum(),
    }
    merged = {**defaults, **substitutions}
    for key, value in merged.items():
        text = text.replace(f"{{{key}}}", value)
    return text


_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(text: str) -> str | None:
    m = _JSON_BLOCK_RE.search(text)
    return m.group(1) if m else None


def try_validate(messages) -> tuple[ReactionRecord | None, list[dict], str | None]:
    """Return (record, errors, raw_text). record is None on failure."""
    ds_messages = [m for m in messages if m.source == "data_structure"]
    if not ds_messages:
        return None, [{"loc": [], "msg": "no data_structure message produced"}], None
    raw = str(ds_messages[-1].content)
    block = extract_json_block(raw)
    if block is None:
        return None, [{"loc": [], "msg": "no ```json block in last data_structure message"}], raw
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as e:
        return None, [{"loc": [], "msg": f"JSONDecodeError: {e}"}], block
    try:
        return ReactionRecord.model_validate(obj), [], block
    except ValidationError as e:
        errs = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
        return None, errs, block


SPECIALIST_NAMES = (
    "reaction_template_parser",
    "molecular_recognition",
    "rgroup_substitution",
    "condition_interpretation",
    "text_extraction",
)


def _msg_text(m) -> str:
    c = getattr(m, "content", None)
    if isinstance(c, list):
        return "\n".join(x for x in c if isinstance(x, str))
    return str(c) if c is not None else ""


_FIG_CLASS_RE = re.compile(
    r"\[FIGURE_CLASSIFICATION\](.*?)\[/FIGURE_CLASSIFICATION\]", re.DOTALL
)
_FIG_CLASS_KEYS = {
    "type": str,
    "confidence": str,
    "row_count": "int_or_null",
    "variant_count": "int_or_null",
    "substrate_column_varies": "bool",
    "text_rgroup_table_present": "bool",
    "notes": str,
}


def _coerce_fig_class_value(raw: str, kind):
    s = raw.strip()
    if kind == str:
        return s
    if s.lower() in ("null", "none", ""):
        return None
    if kind == "int_or_null":
        try:
            return int(s)
        except ValueError:
            return None
    if kind == "bool":
        return s.lower() == "true"
    return s


def _extract_figure_classification(messages) -> dict | None:
    """Parse the figure_classifier's [FIGURE_CLASSIFICATION] block into a dict.

    Returns None if no block is found. The web UI uses this to show the figure
    type + confidence next to the source image without re-parsing conversation.txt.
    """
    for m in messages:
        if getattr(m, "source", None) != "figure_classifier":
            continue
        match = _FIG_CLASS_RE.search(_msg_text(m))
        if not match:
            continue
        body = match.group(1)
        out: dict = {}
        for line in body.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            if key in _FIG_CLASS_KEYS:
                out[key] = _coerce_fig_class_value(value, _FIG_CLASS_KEYS[key])
        if out:
            return out
    return None


async def _tee_events(stream, events_path: Path):
    """Pass-through wrapper that writes one JSON line per streamed message.

    The web UI's WebSocket endpoint tails ``events.jsonl`` to surface live
    agent activity to the browser. We deliberately yield items unchanged so
    autogen's ``Console`` consumer behaves exactly as it did before.

    Each event has: ts, source (agent name or 'system'), type (class name),
    text (truncated to 4 KB), turn_index. Truncation keeps the file small
    even for long conversations; the full text is still in conversation.txt
    when the run finishes.
    """
    turn = 0
    async for item in stream:
        try:
            source = getattr(item, "source", None) or "system"
            content = getattr(item, "content", None)
            if isinstance(content, list):
                text = "\n".join(c for c in content if isinstance(c, str))
            elif content is None:
                text = ""
            else:
                text = str(content)
            if len(text) > 4096:
                text = text[:4096] + "…"
            record = {
                "ts": time.time(),
                "turn": turn,
                "source": source,
                "type": type(item).__name__,
                "text": text,
            }
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            turn += 1
        except Exception:
            # Streaming must never break the run.
            pass
        yield item


def _has_findings_or_optout(messages, agent: str) -> bool:
    """A specialist has 'spoken' if it produced a [FINDINGS …] block or, for
    rgroup_substitution, the 'No R-groups detected.' opt-out string."""
    for m in messages:
        if getattr(m, "source", None) != agent:
            continue
        text = _msg_text(m)
        if f"[FINDINGS {agent}]" in text:
            return True
        if agent == "rgroup_substitution" and "no r-groups detected" in text.lower():
            return True
    return False


def make_selector_func(max_critiques: int, has_verifier: bool = True):
    """Hard rules layered on top of the model-based selector.

    Returns the agent name to use, or None to defer to the model selector.
    """

    def selector_func(messages) -> str | None:
        if not messages:
            return "figure_classifier"

        last = messages[-1]
        last_src = getattr(last, "source", None)
        last_text = _msg_text(last)

        # After the classifier emits its [FIGURE_CLASSIFICATION] tag, hand off
        # to the reaction_template_parser. The classifier is a one-shot agent;
        # it never speaks again in the run.
        if last_src == "figure_classifier":
            return "reaction_template_parser"

        # Handoffs that must be deterministic. Verifier-specific branches are
        # disabled when --no-verifier is set.
        if has_verifier and last_src == "data_structure" and "```json" in last_text:
            # Structural check on the rendered products comes before the schema verifier.
            return "structural_verifier"

        if has_verifier:
            struct_critique_count = sum(
                1 for m in messages
                if getattr(m, "source", None) == "structural_verifier"
                and "[CRITIQUE" in _msg_text(m)
            )
            if last_src == "structural_verifier":
                if "[CRITIQUE" in last_text:
                    # Cap structural retries the same way schema critiques are capped.
                    if struct_critique_count >= max_critiques:
                        return "verifier"
                    return "data_structure"
                # [STRUCT_OK] (or any non-critique output) → schema audit next.
                return "verifier"

            critique_count = sum(
                1 for m in messages
                if getattr(m, "source", None) == "verifier" and "[CRITIQUE" in _msg_text(m)
            )

            if last_src == "verifier" and "[CRITIQUE" in last_text:
                # Cap convergence: after max_critiques cycles, force the verifier
                # back on. Its prompt mandates TERMINATE on the third audit.
                if critique_count >= max_critiques:
                    return "verifier"
                return "data_structure"

        # Evidence-gate: data_structure (and verifier) are ineligible until
        # every specialist has either contributed a [FINDINGS …] card or
        # opted out (only rgroup_substitution may opt out).
        if not all(_has_findings_or_optout(messages, name) for name in SPECIALIST_NAMES):
            # Defer to model, but the candidate_func below scopes the choice.
            return None

        # All specialists done. If data_structure hasn't spoken yet, force it.
        ds_spoken = any(getattr(m, "source", None) == "data_structure" for m in messages)
        if not ds_spoken:
            return "data_structure"
        return None  # let the model pick (rare path)

    return selector_func


def make_candidate_func(has_verifier: bool = True):
    """Restrict the candidate pool the model selector chooses from."""

    def candidate_func(messages) -> list[str]:
        terminal = ["data_structure"] + (
            ["structural_verifier", "verifier"] if has_verifier else []
        )
        all_agents = list(SPECIALIST_NAMES) + terminal
        if not messages:
            return ["figure_classifier"]

        classifier_done = any(
            getattr(m, "source", None) == "figure_classifier" for m in messages
        )
        if not classifier_done:
            return ["figure_classifier"]

        # Block data_structure / verifiers until all specialists have spoken.
        # figure_classifier is intentionally absent from all_agents so it never
        # gets re-elected after its single turn.
        if not all(_has_findings_or_optout(messages, n) for n in SPECIALIST_NAMES):
            return [n for n in all_agents if n in SPECIALIST_NAMES]
        return all_agents

    return candidate_func


def _make_chat_client(
    model_or_deployment: str,
    vision: bool,
    seed: int | None = 42,
    temperature: float | None = None,
):
    """Build a chat-completion client for the given model name.

    Routing rules (in order):
      1. Model name starts with `claude-` → Azure AI Foundry's Anthropic
         passthrough. Uses `AZURE_ANTHROPIC_ENDPOINT` /
         `AZURE_ANTHROPIC_API_KEY` and AutoGen's `AnthropicChatCompletionClient`
         pointed at the Azure base URL. The model name is the Azure
         deployment name (e.g. `claude-opus-4-7`).
      2. AZURE_OPENAI_ENDPOINT is set in the environment → Azure OpenAI.
         The model name is treated as the Azure deployment name.
      3. Otherwise → plain OpenAI.

    `vision` flips the `vision` capability bit on the synthetic ModelInfo we
    pass for non-canonical model names (gpt-5.4, gpt-5-mini deployments,
    Anthropic Claude deployments). Set True for the five image-reading
    specialists, False for data_structure / verifier.

    `seed` and `temperature` are forwarded to the underlying SDK.
    temperature defaults to None because gpt-5 reasoning deployments reject
    non-default values; pass an explicit value via `--temperature` for chat
    models. Seed is forwarded to OpenAI / Azure-OpenAI but not to the
    Anthropic client (the Anthropic SDK doesn't accept a seed parameter).
    """
    azure_endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    # Strip stray quote characters that sometimes sneak in via copy-paste.
    azure_endpoint = azure_endpoint.strip('"').strip("“”").rstrip("/")

    extra: dict = {}
    if seed is not None:
        extra["seed"] = seed
    if temperature is not None:
        extra["temperature"] = temperature

    # Anthropic via Azure AI Foundry: any deployment name beginning with
    # `claude-` (e.g. `claude-opus-4-7`, `claude-opus-4-6`). The Foundry
    # endpoint is a native-Anthropic-API passthrough at
    # `https://<resource>.services.ai.azure.com/anthropic/`, so we point
    # the standard AnthropicChatCompletionClient at that base_url.
    if model_or_deployment.startswith("claude-"):
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        api_key = os.environ.get("AZURE_ANTHROPIC_API_KEY")
        base_url = (os.environ.get("AZURE_ANTHROPIC_ENDPOINT") or "").strip()
        base_url = base_url.strip('"').strip("“”").rstrip("/")
        if not api_key or not base_url:
            raise RuntimeError(
                f"Claude model {model_or_deployment!r} requested but "
                "AZURE_ANTHROPIC_API_KEY / AZURE_ANTHROPIC_ENDPOINT are not "
                "set in the environment."
            )
        info = ModelInfo(
            vision=vision,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family=ModelFamily.CLAUDE_4_OPUS,
            multiple_system_messages=False,
        )
        # Anthropic SDK doesn't accept `seed`; forward only `temperature`
        # (and only if explicitly set).
        anth_extra: dict = {}
        if temperature is not None:
            anth_extra["temperature"] = temperature
        return AnthropicChatCompletionClient(
            model=model_or_deployment,
            api_key=api_key,
            base_url=base_url,
            model_info=info,
            max_retries=5,
            timeout=240.0,
            max_tokens=8192,
            **anth_extra,
        )

    if azure_endpoint:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # gpt-5.4 / gpt-5-mini etc aren't on autogen's known-model list, so
        # we declare capabilities explicitly via ModelInfo.
        info = ModelInfo(
            vision=vision,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family=ModelFamily.GPT_5,
            multiple_system_messages=False,
        )
        return AzureOpenAIChatCompletionClient(
            model=model_or_deployment,
            azure_endpoint=azure_endpoint,
            azure_deployment=model_or_deployment,
            api_version=api_version,
            api_key=api_key,
            model_info=info,
            max_retries=5,
            timeout=240.0,
            **extra,
        )
    return OpenAIChatCompletionClient(
        model=model_or_deployment, max_retries=5, timeout=240.0, **extra
    )


def _molnextr_prompt_block(enabled: bool) -> str:
    """Optional prompt fragment that turns on the bbox-detect → MolNexTR workflow.

    Empty string when --use-molnextr is off, so prompts that reference {molnextr_block}
    fall back to free generation seamlessly.
    """
    if not enabled:
        return ""
    return (
        "\nMOLECULE EXTRACTION WORKFLOW — MANDATORY (do this BEFORE writing any SMILES):\n"
        "1. Call detect_molecules(image_path) ONCE per image. It returns a list of\n"
        "   bboxes shaped [{x1, y1, x2, y2, w, h, category, ...}, ...] in absolute\n"
        "   pixel coordinates. The detector is trained for chemistry figures and finds\n"
        "   every drawn molecule.\n"
        "2. For EACH detected bbox, call predict_molecule_smiles(image_path,\n"
        "   bbox=[x1, y1, w, h]). MolNexTR returns the canonical SMILES for that one\n"
        "   structure, including `*` / `[1*]` / `[2*]` wildcards for R-group placeholders.\n"
        "3. Trust the model: when predict_molecule_smiles returns confidence >= 0.5, USE\n"
        "   that SMILES verbatim. Do NOT free-generate competing SMILES from your own\n"
        "   visual reading — the agents have done that historically and got heterocycles\n"
        "   wrong (e.g. 1,2,3-triazole vs 1,2,4-triazole).\n"
        "4. Only fall back to your own reading when confidence < 0.5 AND you can clearly\n"
        "   read the structure. Mark such cases uncertain=true.\n"
        "5. Map the labels you see in the image (1, 2a, 3b, ...) to the bboxes by\n"
        "   inspecting their location relative to label text.\n"
        "Skip detect_molecules only if the image is genuinely a single molecule.\n"
    )


def _coref_prompt_block(enabled: bool) -> str:
    """Optional prompt fragment activating ChemEagle's label↔structure coref tool.

    Empty string when --use-molnextr is off (the coref runner shares the same
    venv + checkpoint pipeline), so prompts that reference {coref_block} fall
    back to free reasoning seamlessly.
    """
    if not enabled:
        return ""
    return (
        "\nLABEL COREFERENCE (use when the figure has compound numbering):\n"
        "If the [FIGURE_CLASSIFICATION] tag is `substrate_scope_grid`, `mixed_library`,\n"
        "or `optimization_table` with `substrate_column_varies: true` — call\n"
        "detect_label_coref(image_path) ONCE before assigning labels to bboxes.\n"
        "It returns:\n"
        "  - bboxes: list of {x1,y1,x2,y2,w,h, category_id, text} where\n"
        "      category_id == 1 → molecule structure\n"
        "      category_id == 3 → identifier label (with OCR'd `text`, e.g. '1a')\n"
        "  - corefs: list of [label_idx, struct_idx] pairs linking labels to structures.\n"
        "Use the coref pairs to assign labels deterministically — when a coref pair\n"
        "matches a structure bbox you also got from detect_molecules, copy the label\n"
        "verbatim from the coref entry's `text` field. This avoids the recurring\n"
        "label-mismatch failures on substrate-scope grids.\n"
        "Skip detect_label_coref when the figure is a single reaction with no compound\n"
        "numbering (it adds latency without information there).\n"
    )


def _file_sha(path: Path, algo: str = "sha1") -> str | None:
    try:
        h = hashlib.new(algo)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def build_run_meta(
    args: argparse.Namespace,
    *,
    seed: int | None,
    image_path: str,
    started_at: datetime,
) -> dict:
    repo = Path(__file__).parent
    prompts_dir = repo / "prompts"
    prompt_hashes = {
        p.name: _file_sha(p, "sha1")
        for p in sorted(prompts_dir.glob("*.txt"))
    }
    azure_endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    azure_endpoint = azure_endpoint.strip('"').strip("“”").rstrip("/") or None

    return {
        "started_at": started_at.isoformat(),
        "image_path": str(Path(image_path).resolve()),
        "image_sha256": _file_sha(Path(image_path), "sha256"),
        "vision_model": args.vision_model,
        "mini_model": args.mini_model,
        "seed": seed,
        "temperature": args.temperature,
        "team": args.team,
        "max_messages": args.max_messages,
        "max_critiques": args.max_critiques,
        "max_validate_retries": args.max_validate_retries,
        "use_molnextr": args.use_molnextr,
        "no_scripted_molrec": args.no_scripted_molrec,
        "no_tools": args.no_tools,
        "no_verifier": args.no_verifier,
        "cli_args": vars(args),
        "prompt_hashes_sha1": prompt_hashes,
        "endpoint_mode": "azure" if azure_endpoint else "openai",
        "azure_endpoint": azure_endpoint,
        "azure_api_version": os.environ.get("AZURE_OPENAI_API_VERSION") if azure_endpoint else None,
        "git_commit": _git_commit(repo),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "autogen_agentchat": _pkg_version("autogen-agentchat"),
            "autogen_ext": _pkg_version("autogen-ext"),
            "autogen_core": _pkg_version("autogen-core"),
            "rdkit": _pkg_version("rdkit"),
            "openai": _pkg_version("openai"),
            "pydantic": _pkg_version("pydantic"),
        },
    }


def format_validation_feedback(errors: list[dict]) -> str:
    lines = [
        "VALIDATION ERRORS: the last JSON failed schema validation. data_structure must emit a corrected ReactionRecord, then the verifier audits it.",
    ]
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", [])) or "<root>"
        lines.append(f"- {loc}: {err.get('msg')}")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chemistry reaction image analyser")
    parser.add_argument("image_path", help="Path to the reaction image")
    parser.add_argument(
        "--team",
        choices=["selector", "roundrobin"],
        default="selector",
        help="Team type to use (default: selector)",
    )
    parser.add_argument(
        "--max-validate-retries",
        type=int,
        default=2,
        help="Max retries when data_structure produces an invalid ReactionRecord (default: 2)",
    )
    parser.add_argument(
        "--mini-model",
        default=None,
        help="Optional cheaper text-only model (e.g. gpt-5-mini) used by data_structure and verifier. Vision agents always use --vision-model.",
    )
    parser.add_argument(
        "--vision-model",
        default="gpt-5.4",
        help="Model used by the five vision specialists (default: gpt-5.4 — auto-routes to Azure when AZURE_OPENAI_ENDPOINT is set, else OpenAI).",
    )
    parser.add_argument(
        "--max-critiques",
        type=int,
        default=2,
        help="Maximum number of [CRITIQUE] cycles allowed before forcing termination (default: 2).",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=60,
        help="Hard cap on total messages in the team conversation (default: 60).",
    )
    parser.add_argument(
        "--use-molnextr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Register the MolNexTR image-to-SMILES tool with the molecular_recognition and "
             "rgroup_substitution agents (default: on). Pass --no-use-molnextr to disable. "
             "Requires .venv-molnextr/ + checkpoint at molnextr/molnextr_official.pth.",
    )
    parser.add_argument(
        "--no-scripted-molrec",
        action="store_true",
        help="When --use-molnextr is on, by default molecular_recognition is replaced by a "
             "deterministic scripted agent that emits one entry per detected bbox. Pass this "
             "flag to keep the LLM-driven agent (slower but more nuanced; under-counts on big "
             "substrate-scope grids).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to the LLM client for reproducibility (default: 42). Pass a "
             "negative value to disable.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature forwarded to the LLM client. Default: unset (use API "
             "default). gpt-5 reasoning deployments reject non-default values, so this is "
             "opt-in.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Construct every AssistantAgent with tools=[]. LLM-only baseline ablation; "
             "the scripted molecular_recognition agent is also disabled.",
    )
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="Drop the verifier from the team and terminate as soon as data_structure emits "
             "valid schema-passing JSON. Ablation: removes the audit loop.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override the auto-generated run directory. Used by the web UI backend to "
             "place outputs at a known path so it can stream tool_trace.jsonl etc. "
             "Default: runs/<YYYYMMDD_HHMMSS>_<team>/.",
    )
    args = parser.parse_args()
    seed_arg = args.seed if args.seed is not None and args.seed >= 0 else None

    image_path = args.image_path
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    file_name = Path(image_path).name
    abs_image_path = str(Path(image_path).resolve())
    # max_retries handles transient 429s / 5xx from OpenAI / Azure.
    vision_model_client = _make_chat_client(
        args.vision_model, vision=True, seed=seed_arg, temperature=args.temperature
    )
    # data_structure + verifier are text-only — route them to a cheaper model if requested.
    if args.mini_model:
        text_model_client = _make_chat_client(
            args.mini_model, vision=False, seed=seed_arg, temperature=args.temperature
        )
    else:
        text_model_client = vision_model_client
    # The selector itself only needs text; reuse text client.
    selector_model_client = text_model_client

    # Create run dir up front so tools / agents can stream traces and rendered
    # artefacts into it before the team starts running.
    started_at = datetime.now()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = Path(__file__).parent / "runs" / f"{started_at.strftime('%Y%m%d_%H%M%S')}_{args.team}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_trace_path(run_dir / "tool_trace.jsonl")

    if args.no_tools:
        rt_tools, mr_tools, rg_tools = [], [], []
        ci_tools: list = []
        te_tools: list = []
        ds_tools: list = []
        ver_tools: list = []
    else:
        rt_tools = [validate_smiles, canonicalize_smiles, crop_image_region]
        mr_tools = [canonicalize_smiles, validate_smiles, lookup_compound, crop_image_region]
        rg_tools = [derive_substituents, apply_substituents, enumerate_rgroups, canonicalize_smiles, validate_smiles]
        ci_tools = [validate_smiles, lookup_compound, crop_image_region]
        te_tools = [ocr_image, crop_image_region]
        ds_tools = [canonicalize_smiles, validate_reaction_json]
        ver_tools = [validate_reaction_json]
        if args.use_molnextr:
            rt_tools += [detect_molecules, predict_molecule_smiles]
            mr_tools += [detect_molecules, predict_molecule_smiles, detect_label_coref]
            rg_tools += [detect_molecules, predict_molecule_smiles]

    figure_classifier = AssistantAgent(
        "figure_classifier",
        description="Classifies the figure type (optimization_table / substrate_scope_grid / mixed_library / single_reaction) so downstream specialists can adapt their behaviour. Speaks ONCE at the start of the run.",
        model_client=vision_model_client,
        system_message=load_prompt("figure_classifier"),
        tools=[],
        reflect_on_tool_use=False,
        max_tool_iterations=1,
    )

    reaction_template_parser = AssistantAgent(
        "reaction_template_parser",
        description="Parses the overall reaction scheme from the chemistry image — reactants, products, arrow direction, and R-group wildcard SMILES. Speaks immediately after figure_classifier.",
        model_client=vision_model_client,
        system_message=load_prompt("reaction_template_parser", molnextr_block=_molnextr_prompt_block(args.use_molnextr)),
        tools=rt_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=8,
    )

    if args.use_molnextr and not args.no_scripted_molrec and not args.no_tools:
        molecular_recognition = ScriptedMolRecAgent(
            "molecular_recognition",
            image_path=abs_image_path,
        )
    else:
        molecular_recognition = AssistantAgent(
            "molecular_recognition",
            description="Identifies individual molecular structures and converts them to SMILES with stereochemistry. Depends on reaction_template_parser output.",
            model_client=vision_model_client,
            system_message=load_prompt(
                "molecular_recognition",
                molnextr_block=_molnextr_prompt_block(args.use_molnextr),
                coref_block=_coref_prompt_block(args.use_molnextr),
            ),
            tools=mr_tools,
            reflect_on_tool_use=True,
            max_tool_iterations=8,
        )

    rgroup_substitution = AssistantAgent(
        "rgroup_substitution",
        description="Enumerates all R-group variants by replacing wildcard placeholders. Depends on reaction_template_parser and molecular_recognition.",
        model_client=vision_model_client,
        system_message=load_prompt("rgroup_substitution", molnextr_block=_molnextr_prompt_block(args.use_molnextr)),
        tools=rg_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=5,
    )

    condition_interpretation = AssistantAgent(
        "condition_interpretation",
        description="Extracts all reaction conditions: temperature, time, solvent, catalyst, reagents, yield, atmosphere.",
        model_client=vision_model_client,
        system_message=load_prompt("condition_interpretation"),
        tools=ci_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=5,
    )

    text_extraction = AssistantAgent(
        "text_extraction",
        description="Extracts all remaining textual annotations, labels, captions, footnotes, and metadata from the image.",
        model_client=vision_model_client,
        system_message=load_prompt("text_extraction"),
        tools=te_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=3,
    )

    data_structure = AssistantAgent(
        "data_structure",
        description="Synthesises all specialist findings into a structured JSON record. Should be selected after every specialist has produced a [FINDINGS …] card.",
        model_client=text_model_client,
        system_message=load_prompt("data_structure", file_name=file_name),
        tools=ds_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=20,
    )

    verifier = AssistantAgent(
        "verifier",
        description="Audits the JSON produced by data_structure. Selected immediately after data_structure. Emits [CRITIQUE …] for schema errors, or TERMINATE when the record passes.",
        model_client=text_model_client,
        system_message=load_prompt("verifier"),
        tools=ver_tools,
        reflect_on_tool_use=True,
        max_tool_iterations=4,
    )

    structural_verifier = StructuralVerifierAgent(
        "structural_verifier",
        image_path=abs_image_path,
        run_dir=run_dir,
        model_client=vision_model_client,
        system_prompt=load_prompt("structural_verifier"),
    )

    has_verifier = not args.no_verifier
    if has_verifier:
        selector_prompt = """Select the next agent.

{roles}

Current conversation:
{history}

Workflow:
0. figure_classifier         — first turn ONLY; emits a [FIGURE_CLASSIFICATION] tag describing what kind of figure this is so the specialists can adapt their behaviour. Speaks once and never again.
1. reaction_template_parser  — speaks immediately after figure_classifier; identifies the overall scheme.
2. molecular_recognition     — after the template is known.
3. rgroup_substitution       — after molecules are identified (or it self-opts-out with "No R-groups detected.").
4. condition_interpretation  — extracts reaction conditions.
5. text_extraction           — captions, footnotes, references; runs in parallel with conditions.
6. data_structure            — selected ONLY after every specialist has either emitted a [FINDINGS …] card or explicitly opted out. Emits the JSON.
7. structural_verifier       — selected immediately after data_structure emits JSON. Renders the product SMILES with RDKit and compares against the original image. Emits [STRUCT_OK] on match or [CRITIQUE structural_verifier …] on mismatch.
8. verifier                  — selected after structural_verifier emits [STRUCT_OK]. Audits the JSON for schema correctness. Emits TERMINATE on success or [CRITIQUE …] on failure.

If the most recent message is a JSON block from data_structure, the next speaker MUST be structural_verifier.
If the most recent message is [STRUCT_OK] from structural_verifier, the next speaker MUST be verifier.
If the most recent message is a [CRITIQUE structural_verifier …] block, the next speaker MUST be data_structure.
If the most recent message is a [CRITIQUE …] block from the verifier, the next speaker MUST be data_structure.

Select exactly one agent from {participants}.
"""
    else:
        selector_prompt = """Select the next agent.

{roles}

Current conversation:
{history}

Workflow (no-verifier ablation):
0. figure_classifier         — first turn ONLY; emits a [FIGURE_CLASSIFICATION] tag. Speaks once.
1. reaction_template_parser  — speaks immediately after figure_classifier; identifies the overall scheme.
2. molecular_recognition     — after the template is known.
3. rgroup_substitution       — after molecules are identified (or it self-opts-out with "No R-groups detected.").
4. condition_interpretation  — extracts reaction conditions.
5. text_extraction           — captions, footnotes, references; runs in parallel with conditions.
6. data_structure            — selected ONLY after every specialist has either emitted a [FINDINGS …] card or explicitly opted out. Emits the final JSON; the run ends as soon as that JSON appears.

Select exactly one agent from {participants}.
"""

    agents = [
        figure_classifier,
        reaction_template_parser,
        molecular_recognition,
        rgroup_substitution,
        condition_interpretation,
        text_extraction,
        data_structure,
    ]
    if has_verifier:
        agents.append(structural_verifier)
        agents.append(verifier)
    if has_verifier:
        # TERMINATE only counts when emitted by the verifier.
        termination = (
            TextMentionTermination("TERMINATE") & SourceMatchTermination(["verifier"])
        ) | MaxMessageTermination(max_messages=args.max_messages)
    else:
        # No verifier: end as soon as data_structure emits a ```json block.
        termination = (
            TextMentionTermination("```json") & SourceMatchTermination(["data_structure"])
        ) | MaxMessageTermination(max_messages=args.max_messages)

    if args.team == "roundrobin":
        team = RoundRobinGroupChat(agents, termination_condition=termination)
    else:
        team = SelectorGroupChat(
            agents,
            model_client=selector_model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,
            selector_func=make_selector_func(args.max_critiques, has_verifier=has_verifier),
            candidate_func=make_candidate_func(has_verifier=has_verifier),
        )

    image = AGImage.from_file(Path(image_path))
    closing_clause = (
        "data_structure produces JSON, then the verifier audits the JSON and ends the run."
        if has_verifier
        else "data_structure produces the final JSON and the run ends."
    )
    task = MultiModalMessage(
        content=[
            (
                "Analyse this chemistry reaction image. First, figure_classifier "
                "inspects the image and emits a [FIGURE_CLASSIFICATION] tag describing "
                "the figure type. Each downstream specialist reads that tag and adapts "
                "its behaviour, then contributes one [FINDINGS ...] card per turn. "
                f"Finally {closing_clause}\n"
                f"image_path = {abs_image_path}\n"
                "Pass that exact path as the first argument to crop_image_region(image_path, [x,y,w,h]) "
                "if you need to zoom in on a region of the image."
            ),
            image,
        ],
        source="user",
    )

    run_meta = build_run_meta(
        args, seed=seed_arg, image_path=image_path, started_at=started_at
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_messages = []
    validation_log: list[dict] = []
    record = None
    errors: list[dict] = []
    raw: str | None = None
    attempts = 0
    crash_info: str | None = None
    events_path = run_dir / "events.jsonl"
    try:
        try:
            result = await Console(_tee_events(team.run_stream(task=task), events_path))
            all_messages.extend(result.messages)

            record, errors, raw = try_validate(all_messages)
            validation_log.append({"attempt": attempts, "valid": record is not None, "errors": errors})

            while record is None and attempts < args.max_validate_retries:
                attempts += 1
                print(f"\n[validation] attempt {attempts}: {len(errors)} error(s); re-prompting data_structure")
                feedback = TextMessage(content=format_validation_feedback(errors), source="user")
                result = await Console(_tee_events(team.run_stream(task=feedback), events_path))
                all_messages.extend(result.messages)
                record, errors, raw = try_validate(all_messages)
                validation_log.append({"attempt": attempts, "valid": record is not None, "errors": errors})
        except Exception as exc:
            import traceback
            crash_info = f"{type(exc).__name__}: {exc}\n\n" + traceback.format_exc()
            print(f"\n[run] crashed: {type(exc).__name__}: {exc}")
    finally:
        await vision_model_client.close()
        if text_model_client is not vision_model_client:
            await text_model_client.close()
        set_trace_path(None)

    # Save full conversation (across all retry attempts) — even on crash
    lines: list[str] = []
    for msg in all_messages:
        lines.append(f"---------- {msg.__class__.__name__} ({msg.source}) ----------")
        content = msg.content
        if isinstance(content, list):
            lines.append("\n".join(c for c in content if isinstance(c, str)))
        else:
            lines.append(str(content))
        lines.append("")
    if crash_info:
        lines.append("---------- CRASH ----------")
        lines.append(crash_info)
    (run_dir / "conversation.txt").write_text("\n".join(lines), encoding="utf-8")

    # Persist the figure_classifier's [FIGURE_CLASSIFICATION] block as structured JSON
    # so the web UI doesn't have to re-parse conversation.txt every time it loads a run.
    fc = _extract_figure_classification(all_messages)
    if fc is not None:
        (run_dir / "figure_classification.json").write_text(
            json.dumps(fc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    (run_dir / "validation_log.json").write_text(
        json.dumps(validation_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_usage_json(
        run_dir / "usage.json",
        summarise_autogen_messages(
            all_messages,
            model=args.vision_model + (f"+{args.mini_model}" if args.mini_model else ""),
        ),
    )

    if record is not None:
        (run_dir / "result.json").write_text(
            record.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
        )
        print(f"\n[validation] PASSED after {attempts} retry/retries")
    else:
        fallback = raw if raw is not None else ""
        (run_dir / "result.json").write_text(fallback, encoding="utf-8")
        print(f"\n[validation] FAILED after {attempts} retry/retries; raw output saved (see validation_log.json)")

    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
