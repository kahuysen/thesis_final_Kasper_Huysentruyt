"""Single-agent baseline: one AssistantAgent with all tools, one prompt.

Same I/O contract as main.py — produces runs/{ts}_single/{conversation,result,validation_log}
so the eval harness scores it identically.

Usage:
    python single_agent.py input/example1.png
    python single_agent.py path/to/image.png --vision-model gpt-5.4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_agentchat.ui import Console
from autogen_core import Image as AGImage

from main import (
    _make_chat_client,
    extract_json_block,
    format_validation_feedback,
    load_prompt,
)
from eval.usage import summarise_autogen_messages, write_usage_json
from schema import ReactionRecord
from pydantic import ValidationError
from tools import (
    apply_substituents,
    canonicalize_smiles,
    crop_image_region,
    derive_substituents,
    detect_molecules,
    enumerate_rgroups,
    lookup_compound,
    predict_molecule_smiles,
    set_trace_path,
    validate_reaction_json,
    validate_smiles,
)


def _try_validate_single(messages) -> tuple[ReactionRecord | None, list[dict], str | None]:
    """Find the last ```json block emitted by the agent and validate it."""
    for msg in reversed(messages):
        if getattr(msg, "source", None) == "user":
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else (
            "\n".join(c for c in content if isinstance(c, str)) if isinstance(content, list) else str(content)
        )
        block = extract_json_block(text)
        if block is None:
            continue
        try:
            obj = json.loads(block)
        except json.JSONDecodeError as e:
            return None, [{"loc": [], "msg": f"JSONDecodeError: {e}"}], block
        try:
            return ReactionRecord.model_validate(obj), [], block
        except ValidationError as e:
            errs = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
            return None, errs, block
    return None, [{"loc": [], "msg": "no ```json block found in agent output"}], None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent chemistry image analyser")
    parser.add_argument("image_path")
    parser.add_argument("--vision-model", default="gpt-5.4",
                        help="Model to use (default gpt-5.4 — auto-routes to Azure when AZURE_OPENAI_ENDPOINT is set).")
    parser.add_argument("--max-validate-retries", type=int, default=2)
    parser.add_argument("--max-tool-iterations", type=int, default=40,
                        help="Per-turn cap on tool-call rounds. Single agent does everything, so set high.")
    args = parser.parse_args()

    image_path = args.image_path
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    file_name = Path(image_path).name
    abs_image_path = str(Path(image_path).resolve())

    model_client = _make_chat_client(args.vision_model, vision=True)

    tools = [
        detect_molecules,
        predict_molecule_smiles,
        crop_image_region,
        validate_smiles,
        canonicalize_smiles,
        lookup_compound,
        derive_substituents,
        apply_substituents,
        enumerate_rgroups,
        validate_reaction_json,
    ]

    agent = AssistantAgent(
        "single_agent",
        model_client=model_client,
        system_message=load_prompt("single_agent", file_name=file_name),
        tools=tools,
        reflect_on_tool_use=True,
        max_tool_iterations=args.max_tool_iterations,
    )

    image = AGImage.from_file(Path(image_path))
    task = MultiModalMessage(
        content=[
            (
                "Analyse this chemistry reaction image end-to-end and emit one ReactionRecord JSON.\n"
                f"image_path = {abs_image_path}\n"
                "Pass that exact path as the first argument to detect_molecules, predict_molecule_smiles, "
                "and crop_image_region."
            ),
            image,
        ],
        source="user",
    )

    run_dir = Path(__file__).parent / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_single"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_trace_path(run_dir / "tool_trace.jsonl")

    all_messages = []
    validation_log: list[dict] = []
    record: ReactionRecord | None = None
    errors: list[dict] = []
    raw: str | None = None
    attempts = 0
    crash_info: str | None = None

    try:
        try:
            result = await Console(agent.run_stream(task=task))
            all_messages.extend(result.messages)
            record, errors, raw = _try_validate_single(all_messages)
            validation_log.append({"attempt": attempts, "valid": record is not None, "errors": errors})

            while record is None and attempts < args.max_validate_retries:
                attempts += 1
                print(f"\n[validation] attempt {attempts}: {len(errors)} error(s); re-prompting agent")
                feedback = TextMessage(content=format_validation_feedback(errors), source="user")
                result = await Console(agent.run_stream(task=feedback))
                all_messages.extend(result.messages)
                record, errors, raw = _try_validate_single(all_messages)
                validation_log.append({"attempt": attempts, "valid": record is not None, "errors": errors})
        except Exception as exc:
            import traceback
            crash_info = f"{type(exc).__name__}: {exc}\n\n" + traceback.format_exc()
            print(f"\n[run] crashed: {type(exc).__name__}: {exc}")
    finally:
        await model_client.close()
        set_trace_path(None)

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
    (run_dir / "validation_log.json").write_text(
        json.dumps(validation_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_usage_json(
        run_dir / "usage.json",
        summarise_autogen_messages(all_messages, model=args.vision_model),
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
