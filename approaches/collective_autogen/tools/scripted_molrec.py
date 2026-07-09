"""Deterministic replacement for the molecular_recognition agent.

The LLM-driven `molecular_recognition` agent reliably under-counts on
substrate-scope figures: it calls `predict_molecule_smiles` on every bbox
returned by `detect_molecules`, then drops most of them when emitting the
[FINDINGS] card (gpt-4o has a strong prior toward summarising "similar"
variants under a single representative entry).

This agent skips the LLM for that step. On its turn it:
  1. Runs `detect_molecules` on the source image.
  2. Runs `predict_molecule_smiles` on each detected bbox.
  3. Builds a `[FINDINGS molecular_recognition]` card with one entry per
     bbox — no merging, no deduplication.
The downstream agents (rgroup_substitution, condition_interpretation,
data_structure) read the card normally.
"""
from __future__ import annotations

from typing import Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken


class ScriptedMolRecAgent(BaseChatAgent):
    """Emits a deterministic [FINDINGS molecular_recognition] block.

    Args:
        name:        agent name (must equal "molecular_recognition" so the
                     selector / candidate-gate logic continues to recognise
                     this slot in the team).
        image_path:  absolute path to the source image (passed in main.py).
        confidence_threshold: bboxes whose MolNexTR confidence falls below
            this are still emitted but flagged uncertain=true.
    """

    def __init__(
        self,
        name: str,
        image_path: str,
        description: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            name=name,
            description=description or (
                "Identifies individual molecular structures and converts them to SMILES. "
                "Emits one entry per detected bbox — no deduplication."
            ),
        )
        self._image_path = image_path
        self._threshold = confidence_threshold

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        # Lazy import so this module can be loaded without the heavy tool imports.
        from .moldetect_tool import detect_molecules
        from .molnextr_tool import predict_molecule_smiles

        det = detect_molecules(self._image_path, molecule_only=True)
        if det.get("error"):
            card = self._error_card(f"detect_molecules failed: {det['error']}")
            return Response(chat_message=TextMessage(content=card, source=self.name))

        bboxes = det.get("bboxes", []) or []
        entries = []
        for i, b in enumerate(bboxes):
            bbox = [int(b["x1"]), int(b["y1"]), int(b["w"]), int(b["h"])]
            r = predict_molecule_smiles(self._image_path, bbox=bbox)
            entries.append({
                "label": f"bbox_{i}",
                "bbox": bbox,
                "smiles": r.get("smiles"),
                "confidence": float(r.get("confidence") or 0.0),
                "error": r.get("error"),
            })

        card = self._build_card(entries)
        return Response(chat_message=TextMessage(content=card, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return None

    # -- helpers --------------------------------------------------------------

    def _error_card(self, msg: str) -> str:
        return (
            "[FINDINGS molecular_recognition]\n"
            "molecule_count: 0\n"
            f"notes: scripted-agent error: {msg}\n"
            "[/FINDINGS]"
        )

    def _build_card(self, entries: list[dict]) -> str:
        lines = [
            "[FINDINGS molecular_recognition]",
            f"# scripted agent — one entry per detected bbox, no deduplication",
            f"molecule_count: {len(entries)}",
        ]
        for e in entries:
            smi = e["smiles"]
            conf = e["confidence"]
            uncertain = (smi is None) or (conf < self._threshold)
            lines.extend([
                "molecule:",
                f"  label: {e['label']}",
                f"  canonical_smiles: {smi if smi else 'null'}",
                f"  bbox: {e['bbox']}",
                f"  confidence: {conf:.3f}",
                f"  stereochemistry: from MolNexTR",
                f"  uncertain: {'yes' if uncertain else 'no'}",
                f"  notes: {('error: ' + e['error']) if e.get('error') else 'extracted via MolNexTR per-bbox'}",
            ])
        lines.append("[/FINDINGS]")
        return "\n".join(lines)
