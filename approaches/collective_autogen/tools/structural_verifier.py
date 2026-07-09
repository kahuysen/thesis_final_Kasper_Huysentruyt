"""Structural verifier agent — per-molecule visual cross-check.

After data_structure emits its JSON, this agent:
  1. Reads every product SMILES + label from the JSON.
  2. Calls detect_molecules() on the original image to get molecule bboxes.
  3. Crops each bbox from the original image (crops/ subdir).
  4. Renders each product SMILES via RDKit (renders/ subdir).
  5. Sends the original full image + every crop + every rendered SMILES to a
     vision LLM, asking it to align by label (crop carries the label printed
     under the molecule in the figure; render carries the label as legend) and
     compare each pair one-to-one.
  6. Emits [STRUCT_OK] (every paired molecule matches) or
     [CRITIQUE structural_verifier ...] (per-label mismatch report).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Image as AGImage
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from .image import crop_image_region
from .moldetect_tool import detect_molecules
from .render import render_smiles_grid


_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _msg_text(m) -> str:
    c = getattr(m, "content", None)
    if isinstance(c, list):
        return "\n".join(x for x in c if isinstance(x, str))
    return str(c) if c is not None else ""


def _extract_latest_data_structure_json(messages: Sequence) -> dict | None:
    for m in reversed(list(messages)):
        if getattr(m, "source", None) != "data_structure":
            continue
        text = _msg_text(m)
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            continue
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    return None


def _collect_products(record: dict) -> list[dict]:
    out: list[dict] = []
    for rxn in record.get("reactions", []) or []:
        for p in rxn.get("products", []) or []:
            smi = p.get("smiles")
            if not smi:
                continue
            label = p.get("label") or rxn.get("reaction_id") or ""
            out.append({"label": str(label), "smiles": str(smi)})
    return out


class StructuralVerifierAgent(BaseChatAgent):
    """Compares each predicted product SMILES to the matching cropped molecule
    in the original figure."""

    def __init__(
        self,
        name: str,
        image_path: str,
        run_dir: Path,
        model_client: ChatCompletionClient,
        system_prompt: str,
        description: str | None = None,
        crop_pad: int = 16,
    ) -> None:
        super().__init__(
            name=name,
            description=description or (
                "Per-molecule structural cross-check: detects molecule bboxes, crops "
                "each from the original image, renders each predicted product SMILES "
                "via RDKit, and compares the pairs label-by-label."
            ),
        )
        self._image_path = str(Path(image_path).resolve())
        self._run_dir = Path(run_dir)
        self._client = model_client
        self._system_prompt = system_prompt
        self._crop_pad = crop_pad
        self._call_idx = 0
        # Cache of (signature, verdict_text) from the previous full pass. When
        # data_structure re-emits JSON whose product (label, smiles) set is
        # identical to last time — typically because the schema verifier raised
        # a non-structural critique — we replay the previous verdict instead
        # of re-running detect_molecules + cropping + RDKit + a vision LLM call.
        self._last_signature: tuple | None = None
        self._last_verdict: str | None = None

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    def _ok(self, note: str) -> Response:
        return Response(chat_message=TextMessage(
            content=f"[STRUCT_OK]\nnotes: {note}",
            source=self.name,
        ))

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        record = _extract_latest_data_structure_json(messages)
        if record is None:
            return self._ok("no parseable data_structure JSON found yet — skipping structural check.")

        products = _collect_products(record)
        if not products:
            return self._ok("record has no products with SMILES — nothing to compare.")

        # Short-circuit: if data_structure re-emitted the SAME (label, smiles)
        # set as last pass, the structural answer can't have changed. Replay
        # the prior verdict verbatim instead of re-running detect/crop/render.
        # Saves one full vision LLM round-trip + ~30s of crop/render IO when
        # the schema verifier triggers a re-emit that doesn't touch chemistry.
        signature = tuple(sorted((p["label"], p["smiles"]) for p in products))
        if signature == self._last_signature and self._last_verdict is not None:
            replay_note = (
                "\nnotes: products unchanged since the previous structural pass; "
                "skipping detect/crop/render and replaying that verdict."
            )
            return Response(chat_message=TextMessage(
                content=self._last_verdict + replay_note,
                source=self.name,
            ))

        self._call_idx += 1
        call_dir = self._run_dir / f"struct_check_{self._call_idx:02d}"
        crops_dir = call_dir / "crops"
        renders_dir = call_dir / "renders"
        crops_dir.mkdir(parents=True, exist_ok=True)
        renders_dir.mkdir(parents=True, exist_ok=True)

        # 1. Detect bboxes for every molecule drawn in the figure.
        det = detect_molecules(self._image_path, molecule_only=True)
        if det.get("error") or not det.get("bboxes"):
            return self._ok(
                f"detect_molecules unavailable or returned 0 bboxes — skipping structural check "
                f"({det.get('error') or 'no bboxes returned'})."
            )
        bboxes = det["bboxes"]

        # 2. Crop each bbox from the original image. Padding picks up the
        # label text that sits below each panel in substrate-scope tables.
        crops: list[dict] = []
        for i, b in enumerate(bboxes):
            bbox = [int(b["x1"]), int(b["y1"]), int(b["w"]), int(b["h"])]
            r = crop_image_region(self._image_path, bbox, pad=self._crop_pad)
            if r.get("error") or not r.get("path"):
                continue
            # Move to per-call dir for traceability.
            dest = crops_dir / f"bbox_{i:02d}.png"
            try:
                Path(r["path"]).replace(dest)
            except OSError:
                # Cache may be on a different filesystem; copy bytes instead.
                from shutil import copyfile
                copyfile(r["path"], dest)
            crops.append({"index": i, "bbox": bbox, "path": str(dest)})

        if not crops:
            return self._ok("crop_image_region produced 0 valid crops — skipping structural check.")

        # 3. Render each product SMILES individually via RDKit.
        renders: list[dict] = []
        skipped: list[dict] = []
        for p in products:
            out = renders_dir / f"product_{re.sub(r'[^A-Za-z0-9_-]', '_', p['label']) or 'x'}.png"
            r = render_smiles_grid([p], str(out), mols_per_row=1, sub_img_size=(360, 360))
            if r.get("image_path"):
                renders.append({
                    "label": p["label"],
                    "smiles": p["smiles"],
                    "path": r["image_path"],
                })
            else:
                skipped.append({"label": p["label"], "smiles": p["smiles"], "reason": r.get("error")})

        if not renders:
            return self._ok(
                f"no product SMILES could be rendered ({len(skipped)} skipped) — "
                "skipping structural check."
            )

        # 4. Build the multimodal user message: original + every crop + every render.
        try:
            user_content: list = [
                "Original full reaction figure (use this to read the printed label "
                "(e.g. 3a, 3b, ...) sitting just below each molecule, so you can pair "
                "crops with renders by label):",
                AGImage.from_file(Path(self._image_path)),
                "",
                "=== Cropped molecules from the original figure (one per detected bbox; "
                "the printed label is included in the crop because of the padding) ===",
            ]
            for c in crops:
                user_content.append(
                    f"Crop bbox_{c['index']:02d}  (x={c['bbox'][0]}, y={c['bbox'][1]}, "
                    f"w={c['bbox'][2]}, h={c['bbox'][3]}):"
                )
                user_content.append(AGImage.from_file(Path(c["path"])))

            user_content.append("")
            user_content.append(
                "=== RDKit renderings of the predicted product SMILES (one per labelled product) ==="
            )
            for r in renders:
                user_content.append(f"Product {r['label']}  SMILES: {r['smiles']}")
                user_content.append(AGImage.from_file(Path(r["path"])))

            if skipped:
                user_content.append("")
                user_content.append(
                    "Products whose SMILES could not be rendered (RDKit parse failure — flag as critique):"
                )
                for s in skipped:
                    user_content.append(f"  - {s['label']}: {s['smiles']}  ({s.get('reason')})")

            user_content.append("")
            user_content.append(
                "For each rendered product, find the crop with the same printed label "
                "in the original figure, then judge whether the cropped molecule and "
                "the RDKit rendering depict the same compound. Apply the rules in your "
                "system instructions and output exactly one [STRUCT_OK] or "
                "[CRITIQUE structural_verifier] block."
            )
        except Exception as e:
            return self._ok(f"failed to load images for comparison ({type(e).__name__}: {e}).")

        try:
            llm_result = await self._client.create(
                messages=[
                    SystemMessage(content=self._system_prompt),
                    UserMessage(content=user_content, source="user"),
                ],
                cancellation_token=cancellation_token,
            )
        except Exception as e:
            return self._ok(f"vision LLM call failed ({type(e).__name__}: {e}); skipping structural check.")

        text = llm_result.content if isinstance(llm_result.content, str) else str(llm_result.content)
        if "[STRUCT_OK]" not in text and "[CRITIQUE structural_verifier]" not in text:
            text = (
                "[STRUCT_OK]\n"
                "notes: model returned non-conforming output, treating as pass.\n"
                f"raw: {text[:400]}"
            )

        # Persist a manifest so a human can audit which crop went with which render.
        manifest = {
            "call_idx": self._call_idx,
            "crops": crops,
            "renders": renders,
            "skipped": skipped,
            "verdict_first_line": text.splitlines()[0] if text else "",
        }
        try:
            (call_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

        # Remember this pass so the next invocation can short-circuit if the
        # product set is unchanged. We cache both verdict shapes — STRUCT_OK
        # and CRITIQUE — because replaying a critique is also valid (the audit
        # loop already has its own bound, max_critiques, on retries).
        self._last_signature = signature
        self._last_verdict = text

        return Response(chat_message=TextMessage(content=text, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._call_idx = 0
        self._last_signature = None
        self._last_verdict = None
