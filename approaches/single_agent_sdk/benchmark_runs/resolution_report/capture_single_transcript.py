"""One-off diagnostic: run the Gemini-3-Flash agent on a single image
across all five resolution conditions and dump the full conversation
transcript per condition.

This is a parallel implementation of
`pipeline/extractor_gemini_parallel.py` that *also* serialises the
in-memory `history` list (model text parts, function calls, function
responses) to disk so that we can compare reasoning across resolutions
without touching the production extractor.

Output: transcripts/<condition>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/Users/kasperhuysentruyt/Documents/thesis/5.Code/Single_SDK_agent")
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from google import genai
from google.genai import types

import os
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Re-use the production system prompt, tool schemas, and dispatchers so
# the agent loop is functionally identical to the benchmark runs.
from pipeline.extractor import SYSTEM_PROMPT
from pipeline.extractor_gemini_parallel import _build_function_declarations
from pipeline.tools import TOOL_DISPATCH
from pipeline.schema import FigureExtraction
from pydantic import ValidationError

MODEL  = "gemini-3-flash-preview"
IMAGE  = "ACScat_2020.pdf_page002_table_01_s0.88.png"

CONDITIONS = [
    ("full",    ROOT / "corpus" / "Benchmark_kasper_GT3_Maarten"          / IMAGE),
    ("half",    ROOT / "corpus" / "Benchmark_kasper_GT3_Maarten_half"     / IMAGE),
    ("quarter", ROOT / "corpus" / "Benchmark_kasper_GT3_Maarten_quarter"  / IMAGE),
    ("eighth",  ROOT / "corpus" / "Benchmark_kasper_GT3_Maarten_eighth"   / IMAGE),
    ("tenth",   ROOT / "corpus" / "Benchmark_kasper_GT3_Maarten_tenth"    / IMAGE),
]

OUT_DIR = Path(__file__).resolve().parent / "transcripts"
OUT_DIR.mkdir(exist_ok=True)


def serialise_content(content) -> dict:
    """Convert a google.genai Content object to a plain JSON-safe dict."""
    out = {"role": content.role, "parts": []}
    for p in (content.parts or []):
        part: dict[str, Any] = {}
        if getattr(p, "text", None):
            part["text"] = p.text
        if getattr(p, "function_call", None):
            fc = p.function_call
            part["function_call"] = {
                "id": getattr(fc, "id", None),
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            }
        if getattr(p, "function_response", None):
            fr = p.function_response
            part["function_response"] = {
                "id": getattr(fr, "id", None),
                "name": fr.name,
                "response": dict(fr.response) if fr.response else {},
            }
        if getattr(p, "thought_signature", None):
            # Opaque to us — record only its presence and length.
            sig = p.thought_signature
            part["thought_signature_bytes"] = len(sig) if sig else 0
        if getattr(p, "inline_data", None):
            part["inline_data"] = {"mime_type": p.inline_data.mime_type,
                                   "data_bytes": len(p.inline_data.data or b"")}
        out["parts"].append(part)
    return out


def run_one(condition: str, image_path: Path) -> dict:
    image_bytes = image_path.read_bytes()
    user_text = (
        f"Extract every reaction in this figure (filename: {image_path.name}). "
        "Validate every SMILES — call `validate_smiles` for all of them in "
        "parallel in the same turn rather than one at a time, since the calls "
        "are independent. Once every SMILES is validated, call "
        "`submit_extraction` exactly once with the final result."
    )
    history: list = [
        types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            types.Part.from_text(text=user_text),
        ]),
    ]
    function_decls = _build_function_declarations()
    tools = [types.Tool(function_declarations=function_decls)]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY"),
        ),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=8192,
    )

    submitted = None
    tool_calls = 0
    input_tok = 0
    output_tok = 0
    t0 = time.monotonic()

    for step in range(8):
        resp = client.models.generate_content(model=MODEL, contents=history, config=config)
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            input_tok  += getattr(u, "prompt_token_count", 0) or 0
            output_tok += getattr(u, "candidates_token_count", 0) or 0
        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None:
            break
        history.append(cand.content)
        calls = [p.function_call for p in (cand.content.parts or []) if p.function_call]
        if not calls:
            break
        response_parts = []
        for fc in calls:
            tool_calls += 1
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            if name == "submit_extraction":
                try:
                    validated = FigureExtraction.model_validate(args)
                    submitted = validated.model_dump()
                    result = {"ok": True, "accepted": True}
                except ValidationError as e:
                    result = {"ok": False, "error": "schema validation failed",
                              "detail": e.errors()[:3]}
            elif name in TOOL_DISPATCH:
                try:
                    result = TOOL_DISPATCH[name](**args)
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                result = {"ok": False, "error": f"unknown tool {name}"}
            response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    id=getattr(fc, "id", None), name=name, response=result,
                ),
            ))
        history.append(types.Content(role="user", parts=response_parts))
        if submitted is not None:
            break

    elapsed = time.monotonic() - t0
    return {
        "condition": condition,
        "image": str(image_path),
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "tool_calls": tool_calls,
        "elapsed_s": round(elapsed, 2),
        "submitted_extraction": submitted,
        "history": [serialise_content(c) for c in history],
    }


def main() -> None:
    for cond, path in CONDITIONS:
        print(f"[{cond}]  {path.name}  size={path.stat().st_size}")
        record = run_one(cond, path)
        out_path = OUT_DIR / f"{cond}.json"
        out_path.write_text(json.dumps(record, indent=2, default=str))
        n_text = sum(1 for c in record["history"] for p in c["parts"] if "text" in p)
        n_fc   = sum(1 for c in record["history"] for p in c["parts"] if "function_call" in p)
        n_fr   = sum(1 for c in record["history"] for p in c["parts"] if "function_response" in p)
        sub = record["submitted_extraction"]
        n_rxn = len(sub.get("reactions", [])) if sub else 0
        print(f"   in={record['input_tokens']:6d}  out={record['output_tokens']:5d}  "
              f"tools={record['tool_calls']:3d}  steps_text={n_text} fc={n_fc} fr={n_fr}  "
              f"reactions_submitted={n_rxn}  wall={record['elapsed_s']}s")


if __name__ == "__main__":
    main()
