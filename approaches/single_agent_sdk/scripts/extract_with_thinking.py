"""Run the Anthropic SDK agent on one image with extended + interleaved thinking ON.

Prints, in order:
  - Claude's thinking blocks (the model's internal deliberation)
  - Claude's text blocks (natural-language commentary it emits alongside tool calls)
  - Each tool call (name + short args preview) and its result summary
  - The final FigureExtraction (validated against the schema)

This is a one-image diagnostic tool — it is NOT used by the benchmark suite.
It lives outside `pipeline/` so the existing extractor stays untouched.

Usage:
    .venv/bin/python3 scripts/extract_with_thinking.py
    .venv/bin/python3 scripts/extract_with_thinking.py --image <path>
    .venv/bin/python3 scripts/extract_with_thinking.py --thinking-budget 8192

Notes:
  * Extended thinking adds the model's reasoning as `thinking` blocks. Token
    usage shows up under output_tokens (`thinking` tokens count as output).
  * Interleaved thinking (the beta header) lets Claude think AGAIN between
    every tool-use turn — crucial for an agent loop, otherwise thinking only
    runs on the very first turn. Available for Claude 4 series.
  * Thinking blocks must be passed back unchanged in the assistant message
    when the agent loop continues. `resp.content` already contains them in
    the right order, so appending it verbatim (as the existing extractor
    does) is the correct move.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from anthropic import Anthropic
from pydantic import ValidationError

from pipeline.config import build_client_for, current_provider, default_model
from pipeline.extractor import (
    SYSTEM_PROMPT,
    _build_tool_schemas,
    _image_block,
    _format_args_preview,
    _format_result_summary,
)
from pipeline.schema import FigureExtraction
from pipeline.tools import TOOL_DISPATCH


DEFAULT_IMAGE = ROOT / "corpus/Benchmark_kasper_GT3_Maarten/CEJ_2016.pdf_page001_picture_02_s0.74.png"
DEFAULT_MODEL = "claude-opus-4-7"
MAX_AGENT_STEPS = 16  # one image, no need for the full 64 of the benchmark


# ---------- pretty printing ----------

CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _hr(char: str = "─", width: int = 80) -> None:
    print(DIM + char * width + RESET)


def _section(title: str, color: str = CYAN) -> None:
    _hr()
    print(f"{color}{BOLD}{title}{RESET}")
    _hr()


def _print_thinking(text: str, step: int) -> None:
    _section(f"[step {step}] THINKING", MAGENTA)
    # Indent each line so it's visually distinct.
    for line in text.rstrip().splitlines():
        print(f"  {MAGENTA}{line}{RESET}")


def _print_assistant_text(text: str, step: int) -> None:
    _section(f"[step {step}] ASSISTANT", CYAN)
    for line in text.rstrip().splitlines():
        print(f"  {line}")


def _print_tool_call(n: int, name: str, args: dict, result: dict, ok: bool) -> None:
    badge = f"{GREEN}OK{RESET}" if ok else f"{RED}ERR{RESET}"
    print(f"  {YELLOW}#{n} {name}{RESET}  {_format_args_preview(name, args)}")
    print(f"      → {badge}  {_format_result_summary(name, result)}")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Claude model deployment name (Azure-Anthropic or "
                         "Anthropic direct). Must support extended thinking.")
    ap.add_argument("--thinking-budget", type=int, default=4096,
                    help="(Opus 4.6 / older API) Max tokens of thinking per "
                         "round when --thinking-mode=enabled.")
    ap.add_argument("--thinking-mode",
                    choices=["adaptive", "enabled"],
                    default="adaptive",
                    help="`adaptive` = Opus 4.7+ format; the model decides how "
                         "long to think per turn (controlled by --effort). "
                         "`enabled` = legacy format with a fixed budget, used "
                         "by Opus 4.6 and earlier.")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="high",
                    help="Effort level for adaptive thinking (Opus 4.7+).")
    ap.add_argument("--max-tokens", type=int, default=12288,
                    help="Total max_tokens for each response (must exceed "
                         "thinking budget + room for tool calls / text).")
    ap.add_argument("--no-interleaved", action="store_true",
                    help="Disable the interleaved-thinking beta header. Without "
                         "this header the model only thinks on the first turn.")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1
    if args.thinking_budget >= args.max_tokens:
        print(f"thinking-budget ({args.thinking_budget}) must be < max-tokens "
              f"({args.max_tokens})", file=sys.stderr)
        return 2

    # Pick whichever Anthropic-compatible provider this env is configured for
    # (azure-hosted Claude or direct Anthropic API). Same path the real
    # extractor takes via `current_provider()`.
    provider = current_provider()
    if provider not in ("azure", "anthropic"):
        # LLM_PROVIDER might be unset or pointing at gemini/openrouter; the
        # thinking test only makes sense against a Claude endpoint.
        provider = "azure"
    client: Anthropic = build_client_for(provider)

    tools = _build_tool_schemas()
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                _image_block(args.image),
                {
                    "type": "text",
                    "text": (
                        f"Extract every reaction in this figure (filename: {args.image.name}). "
                        "Validate each SMILES, then call "
                        "`submit_extraction` exactly once with the final result."
                    ),
                },
            ],
        }
    ]

    create_kwargs: dict = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "system": SYSTEM_PROMPT,
        "tools": tools,
    }
    if args.thinking_mode == "adaptive":
        # Opus 4.7+ API: model decides per-turn how long to think; we only
        # set the effort knob. `output_config.effort` is the official knob.
        create_kwargs["thinking"] = {"type": "adaptive"}
        create_kwargs["output_config"] = {"effort": args.effort}
    else:
        # Legacy API (Opus 4.6 and earlier): explicit token budget.
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": args.thinking_budget,
        }
    if not args.no_interleaved:
        # Interleaved thinking lets Claude think again BETWEEN tool calls.
        # Without it, the model only emits thinking blocks on the first turn,
        # which is useless for an agent loop.
        create_kwargs["extra_headers"] = {
            "anthropic-beta": "interleaved-thinking-2025-05-14",
        }

    print(f"{BOLD}Image:{RESET} {args.image.name}")
    print(f"{BOLD}Model:{RESET} {args.model}")
    if args.thinking_mode == "adaptive":
        print(f"{BOLD}Thinking mode:{RESET} adaptive (effort={args.effort})  "
              f"interleaved={'OFF' if args.no_interleaved else 'ON'}")
    else:
        print(f"{BOLD}Thinking mode:{RESET} enabled  "
              f"budget={args.thinking_budget} tokens/turn  "
              f"interleaved={'OFF' if args.no_interleaved else 'ON'}")
    print()

    t0 = time.time()
    submitted_dump = None
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    thinking_tokens_total = 0  # rough: counted as characters / 4 from thinking blocks

    try:
        for step in range(MAX_AGENT_STEPS):
            resp = client.messages.create(messages=messages, **create_kwargs)
            input_tokens += resp.usage.input_tokens
            output_tokens += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})

            # Walk every block in order so we surface thinking → commentary →
            # tool calls as the model actually emitted them. Print the
            # TOOL CALLS header once per step, just before the first tool
            # block in that step.
            tool_results: list[dict] = []
            tool_header_printed = False
            for block in resp.content:
                bt = getattr(block, "type", None)
                if bt == "thinking":
                    text = getattr(block, "thinking", "") or ""
                    sig = getattr(block, "signature", None)
                    if text.strip():
                        thinking_tokens_total += len(text) // 4  # rough estimate
                        _print_thinking(text, step + 1)
                    elif sig:
                        # Opus 4.7 "adaptive" thinking returns only an
                        # encrypted signature — the plaintext lives
                        # server-side and isn't surfaced to clients. The
                        # signature still has to be passed back unchanged
                        # for the next turn, which `messages.append(content)`
                        # already does.
                        _section(f"[step {step + 1}] THINKING (encrypted)", MAGENTA)
                        print(f"  {DIM}Opus 4.7 returns only a thought signature, not the "
                              f"plaintext reasoning.{RESET}")
                        print(f"  {DIM}signature[:60] = {sig[:60]}…{RESET}")
                elif bt == "redacted_thinking":
                    _section(f"[step {step + 1}] THINKING (redacted by API)", MAGENTA)
                    print(f"  {DIM}(content withheld — encrypted by safety filter){RESET}")
                elif bt == "text":
                    text = getattr(block, "text", "") or ""
                    if text.strip():
                        _print_assistant_text(text, step + 1)
                elif bt == "tool_use":
                    tool_calls += 1
                    name = block.name
                    tool_args = block.input or {}
                    if not tool_header_printed:
                        _section(f"[step {step + 1}] TOOL CALLS", YELLOW)
                        tool_header_printed = True
                    if name == "submit_extraction":
                        try:
                            validated = FigureExtraction.model_validate(tool_args)
                            submitted_dump = validated.model_dump()
                            result = {"ok": True, "accepted": True}
                        except ValidationError as e:
                            result = {"ok": False, "error": "schema validation failed",
                                      "detail": e.errors()[:3]}
                    elif name in TOOL_DISPATCH:
                        try:
                            result = TOOL_DISPATCH[name](**tool_args)
                        except Exception as exc:
                            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    else:
                        result = {"ok": False, "error": f"unknown tool {name}"}
                    _print_tool_call(tool_calls, name, tool_args, result,
                                     ok=bool(result.get("ok")))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                        "is_error": not result.get("ok", False),
                    })

            if resp.stop_reason != "tool_use":
                # Model stopped without calling a tool — either it submitted
                # already, or it gave up. Exit the loop.
                break

            if not tool_results:
                # No tool blocks despite stop_reason=tool_use — defensive break.
                break

            messages.append({"role": "user", "content": tool_results})
            if submitted_dump is not None:
                break
    except Exception as exc:
        _section("ERROR", RED)
        print(f"  {RED}{type(exc).__name__}: {exc}{RESET}")
        return 1

    elapsed = time.time() - t0

    _hr("═")
    print(f"{BOLD}Summary{RESET}")
    _hr("═")
    print(f"  steps (round-trips): {step + 1}")
    print(f"  tool calls:          {tool_calls}")
    print(f"  input tokens:        {input_tokens:,}")
    print(f"  output tokens:       {output_tokens:,}  "
          f"({DIM}includes ~{thinking_tokens_total:,} estimated thinking tokens{RESET})")
    print(f"  wall time:           {elapsed:.1f}s")
    if submitted_dump is not None:
        n_rxn = len(submitted_dump.get("reactions", []))
        print(f"  reactions submitted: {GREEN}{n_rxn}{RESET}")
    else:
        print(f"  reactions submitted: {RED}0 (no submit_extraction call){RESET}")
    return 0 if submitted_dump is not None else 1


if __name__ == "__main__":
    sys.exit(main())
