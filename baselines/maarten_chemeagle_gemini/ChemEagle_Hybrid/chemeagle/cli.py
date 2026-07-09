"""Command-line entry point for the chemeagle package.

Subcommands:
    chemeagle parse <pdf> -o out.json [...]
    chemeagle classify <parse_result.json> -o out.csv [...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chemeagle",
        description=(
            "Parse a chemical PDF into reactions JSON via the molnextr + Gemini "
            "hybrid backend, then optionally classify the reactions with "
            "Rxn-INSIGHT + a text-LLM enrichment pass."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # parse -----------------------------------------------------------------
    parse = sub.add_parser(
        "parse",
        help="Parse a single PDF and write the JSON result.",
    )
    parse.add_argument("pdf", help="Path to the input PDF")
    parse.add_argument(
        "-o", "--output", required=True,
        help="Path to write the aggregated JSON result",
    )
    parse.add_argument(
        "--out-dir", default=None,
        help="Where cropped figure PNGs are written (tempdir if omitted)",
    )
    parse.add_argument(
        "--keep-figures", action="store_true",
        help="If --out-dir was omitted, preserve the tempdir of cropped PNGs",
    )
    parse.add_argument(
        "--model", default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview)",
    )

    # classify --------------------------------------------------------------
    classify = sub.add_parser(
        "classify",
        help=(
            "Run Rxn-INSIGHT (with the user-supplied SMIRKS DB) plus a text-LLM "
            "enrichment pass on a parse_pdf JSON result. Writes a CSV."
        ),
    )
    classify.add_argument(
        "parse_json",
        help="Path to a parse_pdf JSON result (output of `chemeagle parse`)",
    )
    classify.add_argument(
        "-o", "--output", required=True,
        help="Path to write the output CSV",
    )
    classify.add_argument(
        "--smirks",
        default="/data/mdobb/reaction_classification/data/smirks_db.json",
        help=(
            "Path to the SMIRKS DB JSONL "
            "(default: /data/mdobb/reaction_classification/data/smirks_db.json)"
        ),
    )
    classify.add_argument(
        "--model", default="gemini-3-flash-preview",
        help="Gemini model used for the text-LLM pass (default: gemini-3-flash-preview)",
    )
    return p


def _cmd_parse(args: argparse.Namespace) -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        sys.stderr.write("error: GEMINI_API_KEY environment variable is not set.\n")
        return 2

    from chemeagle.pdf import parse_pdf, write_result

    result = parse_pdf(
        args.pdf,
        out_dir=args.out_dir,
        model=args.model,
        keep_figures=args.keep_figures,
    )
    write_result(result, args.output)

    n_figs = len(result.get("figures", []))
    n_rxns = sum(
        len((f.get("result") or {}).get("reactions") or [])
        for f in result["figures"]
    )
    usage = result.get("usage", {})
    sys.stdout.write(
        f"chemeagle: parsed {n_figs} figure(s), {n_rxns} reaction(s) total. "
        f"Tokens: prompt={usage.get('prompt_tokens', 0)} "
        f"completion={usage.get('completion_tokens', 0)}. "
        f"Wrote {args.output}\n"
    )
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    if not os.environ.get("RXN_INSIGHT_PATH"):
        sys.stderr.write(
            "error: RXN_INSIGHT_PATH is not set. Export the path to "
            "Gen-Rxn-INSIGHT/src (e.g., "
            "/data/mdobb/Rxn-INSIGHT/Gen-Rxn-INSIGHT/src).\n"
        )
        return 2
    if not os.environ.get("GEMINI_API_KEY"):
        sys.stderr.write("error: GEMINI_API_KEY environment variable is not set.\n")
        return 2
    if not os.path.isfile(args.parse_json):
        sys.stderr.write(f"error: parse_json not found: {args.parse_json}\n")
        return 2

    from chemeagle.classify import classify, write_csv

    with open(args.parse_json, "r", encoding="utf-8") as f:
        parse_result = json.load(f)
    rows = classify(
        parse_result,
        smirks_db_path=args.smirks,
        model=args.model,
    )
    write_csv(rows, args.output)

    n_named = sum(1 for r in rows if r.get("ri_name") and r["ri_name"] != "OtherReaction")
    n_template = sum(1 for r in rows if r.get("ri_template_r0p1"))
    n_skipped = sum(1 for r in rows if r.get("ri_error"))
    sys.stdout.write(
        f"chemeagle: classified {len(rows)} reaction(s). "
        f"Named={n_named}, template-only={n_template}, ri-skipped/error={n_skipped}. "
        f"Wrote {args.output}\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "parse":
        return _cmd_parse(args)
    if args.command == "classify":
        return _cmd_classify(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
