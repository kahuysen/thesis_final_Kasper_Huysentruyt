"""
DRFP + FAISS nearest-neighbour reaction template retriever.

Predicts reaction templates from **unmapped** reaction SMILES without any
atom-mapping step.  For each query reaction the k nearest neighbours in the
training index are retrieved (by Hamming distance on DRFP fingerprints) and
their templates are tried in order until one correctly reconstructs the
expected product.

Dependencies (not installed by default):
    pip install drfp faiss-cpu

Typical workflow::

    # --- build (once, offline) ---
    retriever = TemplateRetriever()
    retriever.build(df["SANITIZED_REACTION"], df["TEMPLATE_rr0rp0_ring1"])
    retriever.save("retriever_r0/")

    # --- inference ---
    retriever = TemplateRetriever.load("retriever_r0/")
    template = retriever.predict("CC(=O)O.CCN>>CC(=O)NCC", k=10)
    if template is None:
        template = rxnmapper_rxninsight_fallback(reaction)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from gen_rxn_insight.template import measure_template_accuracy

_DRFP_BITS = 2048


# ── Dependency helpers ────────────────────────────────────────────────────────

def _drfp():
    try:
        from drfp import DrfpEncoder
        return DrfpEncoder
    except ImportError:
        raise ImportError(
            "drfp is required for TemplateRetriever. "
            "Install it with: pip install drfp"
        )


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "faiss is required for TemplateRetriever. "
            "Install it with: pip install faiss-cpu"
        )


# ── Encoding ──────────────────────────────────────────────────────────────────

def _encode_chunk(args):
    """Worker function: encode one chunk of reactions (must be top-level for pickling)."""
    batch, n_bits = args
    from drfp import DrfpEncoder
    fps = DrfpEncoder.encode(batch, n_folded_length=n_bits)
    return np.packbits(np.array(fps, dtype=np.uint8), axis=1)


def _encode(
        reactions: List[str],
        n_bits: int = _DRFP_BITS,
        chunk_size: int = 2_000,
        progress: bool = False,
        n_jobs: int = -1,
) -> np.ndarray:
    """Return packed DRFP fingerprints as uint8 array of shape (n, n_bits//8).

    DRFP encodes the symmetric difference of circular substructures between
    reactants and products, so it works directly on unmapped SMILES.
    ``np.packbits`` converts the 0/1 array to the format expected by FAISS
    ``IndexBinaryFlat``.

    Encoding is parallelised across CPU cores (``n_jobs=-1`` → all cores).
    DRFP has no GPU path so the H100 is not useful here.
    """
    import multiprocessing as mp
    from joblib import Parallel, delayed

    chunks = [
        reactions[i: i + chunk_size]
        for i in range(0, len(reactions), chunk_size)
    ]
    args = [(chunk, n_bits) for chunk in chunks]

    if n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        n_jobs = mp.cpu_count()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_encode_chunk)(a)
        for a in tqdm(args, desc="Encoding DRFP fingerprints", disable=not progress)
    )

    return np.vstack(results)  # (n, n_bits // 8)


# ── Main class ────────────────────────────────────────────────────────────────

class TemplateRetriever:
    """DRFP + FAISS nearest-neighbour template retriever.

    Retrieves the k most similar reactions from a pre-built index and returns
    the first template that correctly reconstructs the expected product.

    Args:
        n_bits: DRFP fingerprint length in bits (default: 2048).

    Example:
        >>> retriever = TemplateRetriever()
        >>> retriever.build(df["SANITIZED_REACTION"], df["TEMPLATE_rr0rp0_ring1"])
        >>> retriever.save("retriever/")
        >>> retriever = TemplateRetriever.load("retriever/")
        >>> retriever.predict("CC(=O)O.CCN>>CC(=O)NCC", k=10)
    """

    def __init__(self, n_bits: int = _DRFP_BITS) -> None:
        self.n_bits = n_bits
        self._index = None
        self._templates: List[str] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
            self,
            reactions: Union[List[str], pd.Series],
            templates: Union[List[str], pd.Series],
            chunk_size: int = 5_000,
            n_jobs: int = 1,
            progress: bool = True,
    ) -> "TemplateRetriever":
        """Build the FAISS index from a validated (reaction, template) dataset.

        Args:
            reactions: Unmapped reaction SMILES (e.g. ``SANITIZED_REACTION``
                column).  Must be in the same order as *templates*.
            templates: Validated template SMIRKS corresponding to each
                reaction.  ``None`` / empty entries are stored as ``""`` and
                skipped at query time.
            chunk_size: Reactions per parallel encoding chunk.  Smaller values
                give finer progress granularity; larger values reduce overhead.
            n_jobs: CPU cores for DRFP encoding (``-1`` → all cores).
                DRFP has no GPU path; the H100 is not useful here.
            progress: Show a tqdm progress bar during encoding.

        Returns:
            ``self`` — allows chaining: ``retriever.build(...).save(...)``.
        """
        faiss = _faiss()

        reactions_list = list(reactions)
        self._templates = [t if t else "" for t in templates]

        fps_packed = _encode(reactions_list, self.n_bits, chunk_size, progress, n_jobs)

        self._index = faiss.IndexBinaryFlat(self.n_bits)
        self._index.add(fps_packed)

        return self

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, directory: Union[str, Path]) -> None:
        """Save the index and template list to *directory*.

        Creates three files:
        - ``index.faiss`` — FAISS binary index
        - ``templates.parquet`` — template strings aligned with index positions
        - ``config.json`` — fingerprint parameters
        """
        faiss = _faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index_binary(self._index, str(directory / "index.faiss"))
        pd.DataFrame({"TEMPLATE": self._templates}).to_parquet(
            directory / "templates.parquet", index=False
        )
        (directory / "config.json").write_text(
            json.dumps({"n_bits": self.n_bits})
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "TemplateRetriever":
        """Load a previously saved retriever from *directory*."""
        faiss = _faiss()
        directory = Path(directory)

        config = json.loads((directory / "config.json").read_text())
        retriever = cls(n_bits=config["n_bits"])
        retriever._index = faiss.read_index_binary(
            str(directory / "index.faiss")
        )
        retriever._templates = (
            pd.read_parquet(directory / "templates.parquet")["TEMPLATE"]
            .fillna("")
            .tolist()
        )
        return retriever

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
            self,
            reaction: str,
            k: int = 10,
    ) -> Optional[str]:
        """Predict a template for a single unmapped reaction SMILES.

        Retrieves the *k* nearest neighbours by Hamming distance on DRFP
        fingerprints, deduplicates templates, and returns the first one that
        correctly reconstructs the expected product via
        :func:`~gen_rxn_insight.template.measure_template_accuracy`.

        Args:
            reaction: Unmapped reaction SMILES (``reactants>>products``).
            k: Number of nearest neighbours to try (default: 10).

        Returns:
            Template SMIRKS string, or ``None`` if no neighbour's template
            gives the correct product.
        """
        DrfpEncoder = _drfp()
        fps = DrfpEncoder.encode([reaction], n_folded_length=self.n_bits)
        fp_packed = np.packbits(np.array(fps, dtype=np.uint8), axis=1)

        _, neighbor_ids = self._index.search(fp_packed, k)

        seen: set = set()
        for idx in neighbor_ids[0]:
            template = self._templates[idx]
            if not template or template in seen:
                continue
            seen.add(template)
            try:
                if measure_template_accuracy(reaction, template)["correct"]:
                    return template
            except Exception:
                continue

        return None

    def predict_batch(
            self,
            reactions: Union[List[str], pd.Series],
            k: int = 10,
            chunk_size: int = 5_000,
            n_jobs: int = 1,
            progress: bool = True,
    ) -> pd.DataFrame:
        """Predict templates for many unmapped reactions.

        DRFP encoding and FAISS search are fully batched for speed.
        Template validation is sequential per reaction (each call to
        :func:`~gen_rxn_insight.template.measure_template_accuracy` is
        independent and fast).

        Args:
            reactions: Unmapped reaction SMILES.
            k: Number of nearest neighbours to try per reaction.
            chunk_size: Reactions per DRFP encoding chunk.
            progress: Show tqdm progress bars.

        Returns:
            DataFrame with columns:

            * ``REACTION`` — input SMILES.
            * ``TEMPLATE`` — predicted template, or ``None`` if not found.
            * ``FOUND`` — ``True`` if a correct template was found.
            * ``NEIGHBOR_RANK`` — 0-based rank of the successful neighbour
              (``None`` if not found).
        """
        reactions_list = list(reactions)
        n = len(reactions_list)

        # Batch-encode and batch-search
        fps_packed = _encode(reactions_list, self.n_bits, chunk_size, progress, n_jobs)
        _, all_neighbor_ids = self._index.search(fps_packed, k)  # (n, k)

        # Per-reaction template validation
        out_templates = np.empty(n, dtype=object)
        out_found = np.zeros(n, dtype=bool)
        out_rank = np.empty(n, dtype=object)

        iterable = range(n)
        if progress:
            iterable = tqdm(iterable, desc="Validating templates")

        for i in iterable:
            rxn = reactions_list[i]
            seen: set = set()
            for rank, idx in enumerate(all_neighbor_ids[i]):
                template = self._templates[idx]
                if not template or template in seen:
                    continue
                seen.add(template)
                try:
                    if measure_template_accuracy(rxn, template)["correct"]:
                        out_templates[i] = template
                        out_found[i] = True
                        out_rank[i] = rank
                        break
                except Exception:
                    continue

        return pd.DataFrame({
            "REACTION": np.array(reactions_list, dtype=object),
            "TEMPLATE": out_templates,
            "FOUND": out_found,
            "NEIGHBOR_RANK": out_rank,
        })

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return (
            f"TemplateRetriever(n_reactions={len(self)}, n_bits={self.n_bits})"
        )
