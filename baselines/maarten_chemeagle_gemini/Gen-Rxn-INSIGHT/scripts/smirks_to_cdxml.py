#!/usr/bin/env python
"""Convert generalized SMIRKS patterns to CDXML (ChemDraw) reaction schemes.

Usage examples:

    # Single SMIRKS → CDXML file
    python smirks_to_cdxml.py --smirks "[N;H3;D0;+0:1].[#6;+0:2]-[C;H2;D2;+0:3]-[F,Cl,Br,I;H0;+0]>>[#6;+0:2]-[C;H2;D2;+0:3]-[N;H2;D1;+0:1]" --name "N-alkylation" --output reaction.cdxml

    # Batch: JSON file → directory of CDXML files
    python smirks_to_cdxml.py --input validated_smirks.json --output cdxml_schemes/

    # From Python:
    from smirks_to_cdxml import smirks_to_cdxml
    cdxml_str = smirks_to_cdxml(smirks, name="N-alkylation")
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOND_LENGTH_PT = 30.0  # CDXML bond length in points
RDKIT_BOND_LENGTH = 1.5  # approximate RDKit 2D bond length
SCALE = BOND_LENGTH_PT / RDKIT_BOND_LENGTH

FONT_SIZE = 10
PAGE_WIDTH = 770
PAGE_HEIGHT = 525

PLUS_GAP = 50  # horizontal space around "+" sign
ARROW_LENGTH = 60  # arrow shaft length
ARROW_PAD = 15  # space before/after arrow
FRAG_PAD = 15  # padding around each fragment

HALIDE_SET = {"F", "Cl", "Br", "I"}

ELEMENT_SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 33: "As",
    34: "Se", 35: "Br", 50: "Sn", 53: "I",
}
SYMBOL_TO_NUM = {v: k for k, v in ELEMENT_SYMBOLS.items()}


# ---------------------------------------------------------------------------
# SMARTS atom interpretation
# ---------------------------------------------------------------------------

def _parse_smarts_atom(atom) -> dict:
    """Extract display-relevant properties from an RDKit query atom.

    Returns a dict with keys:
        element     int | None     atomic number (6, 7, …) or None
        h_count     int | None     explicit H count from SMARTS
        charge      int            formal charge
        map_num     int            atom map number (0 = unmapped)
        aromatic    bool           lowercase element in SMARTS
        elem_list   list[str]      OR-list elements (e.g. ["F","Cl","Br","I"])
        label       str | None     pre-determined display label (e.g. "X")
    """
    smarts = atom.GetSmarts()
    info: dict = {
        "element": None,
        "h_count": None,
        "charge": 0,
        "map_num": atom.GetAtomMapNum(),
        "aromatic": False,
        "elem_list": [],
        "label": None,
    }

    # Strip outer brackets and map number
    inner = smarts.strip("[]")
    inner = re.sub(r":\d+$", "", inner)

    # --- OR-list detection (e.g. "F,Cl,Br,I;H0;+0") ---
    # OR-list elements appear before the first ; or & that follows the list
    or_match = re.match(r"^([A-Z][a-z]?(?:,[A-Z][a-z]?)+)", inner)
    if or_match:
        elements = or_match.group(1).split(",")
        info["elem_list"] = elements
        if set(elements) <= HALIDE_SET:
            info["label"] = "X"
        elif set(elements) == {"O", "S"}:
            info["label"] = "Y"
        else:
            info["label"] = ",".join(elements)
        # Shared H / charge after ;
        hm = re.search(r"[;&]H(\d+)", inner)
        if hm:
            info["h_count"] = int(hm.group(1))
        cm = re.search(r"[;&]\+(\d+)", inner)
        if cm:
            info["charge"] = int(cm.group(1))
        return info

    # --- Single element ---
    # Atomic number notation (#6, #7, …)
    num_match = re.search(r"#(\d+)", inner)
    if num_match:
        info["element"] = int(num_match.group(1))
    else:
        # Element symbol (C, c, N, n, Si, …)
        sym_match = re.match(r"^([A-Z][a-z]?|[cnops])", inner)
        if sym_match:
            sym = sym_match.group(1)
            if sym.islower():
                info["aromatic"] = True
                sym = sym.upper()
            if sym in SYMBOL_TO_NUM:
                info["element"] = SYMBOL_TO_NUM[sym]

    # H count
    hm = re.search(r"[&;]?H(\d+)", inner)
    if hm:
        info["h_count"] = int(hm.group(1))

    # Charge
    pos = re.search(r"\+(\d+)", inner)
    if pos:
        info["charge"] = int(pos.group(1))
    elif re.search(r"\+(?!\d)", inner):
        info["charge"] = 1
    neg = re.search(r"(?<!^)-(\d+)", inner)
    if neg:
        info["charge"] = -int(neg.group(1))

    return info


# ---------------------------------------------------------------------------
# Fragment: one connected molecular component
# ---------------------------------------------------------------------------

class Fragment:
    """A drawable molecular fragment derived from a SMARTS pattern."""

    def __init__(self, mol):
        self.mol = mol
        self.nodes: list[dict] = []   # display info per atom
        self.bonds: list[tuple] = []  # (begin_idx, end_idx, order)
        self.coords: list[tuple] = [] # (x, y) in CDXML points, origin-normalised
        self.width: float = 0.0
        self.height: float = 0.0

        # Writable copy with ring info initialised (needed for SMARTS mols)
        self._rwmol = Chem.RWMol(mol)
        Chem.FastFindRings(self._rwmol)

        self._interpret_atoms()
        self._extract_bonds()
        self._compute_coords()

    def _interpret_atoms(self):
        for atom in self.mol.GetAtoms():
            info = _parse_smarts_atom(atom)
            degree = atom.GetDegree()
            node: dict = {"idx": atom.GetIdx(), "map_num": info["map_num"]}

            if info["label"] is not None:
                node["type"] = "nickname"
                node["label"] = info["label"]
            elif info["element"] is not None:
                # Terminal generic carbon → R group
                if (info["element"] == 6
                        and info["h_count"] is None
                        and degree <= 1):
                    node["type"] = "nickname"
                    node["label"] = "R"
                else:
                    node["type"] = "element"
                    node["element"] = info["element"]
                    node["h_count"] = info["h_count"]
                    node["charge"] = info["charge"]
                    node["aromatic"] = info["aromatic"]
            else:
                node["type"] = "nickname"
                node["label"] = "*"

            # Keep parsed SMARTS info for definition generation
            node["atom_info"] = info
            self.nodes.append(node)

    def _extract_bonds(self):
        aromatic_idxs: set[int] = set()  # RDKit bond indices
        rdkit_to_my: dict[int, int] = {}

        for bond in self.mol.GetBonds():
            my_idx = len(self.bonds)
            rdkit_to_my[bond.GetIdx()] = my_idx
            bt = bond.GetBondType()

            if bt == Chem.rdchem.BondType.AROMATIC:
                order = 1  # placeholder — Kekulized below
                aromatic_idxs.add(bond.GetIdx())
            elif bt == Chem.rdchem.BondType.DOUBLE:
                order = 2
            elif bt == Chem.rdchem.BondType.TRIPLE:
                order = 3
            else:
                order = 1

            self.bonds.append(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order)
            )

        if not aromatic_idxs:
            return

        # Kekulize aromatic bonds: alternate double/single within each ring
        ring_info = self._rwmol.GetRingInfo()
        assigned: set[int] = set()  # my-indices already fixed

        for ring_bond_ids in ring_info.BondRings():
            ar_in_ring = [bi for bi in ring_bond_ids if bi in aromatic_idxs]
            if not ar_in_ring:
                continue
            for j, rbi in enumerate(ar_in_ring):
                mi = rdkit_to_my[rbi]
                if mi not in assigned:
                    b, e, _ = self.bonds[mi]
                    self.bonds[mi] = (b, e, 2 if j % 2 == 0 else 1)
                    assigned.add(mi)

    def _compute_coords(self):
        rdDepictor.Compute2DCoords(self._rwmol)
        conf = self._rwmol.GetConformer()

        raw = []
        for i in range(self._rwmol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            raw.append((p.x * SCALE, -p.y * SCALE))  # flip Y for CDXML

        if raw:
            min_x = min(c[0] for c in raw)
            min_y = min(c[1] for c in raw)
            raw = [(x - min_x, y - min_y) for x, y in raw]
            self.width = max(c[0] for c in raw) if len(raw) > 1 else 0.0
            self.height = max(c[1] for c in raw) if len(raw) > 1 else 0.0

        self.coords = raw


# ---------------------------------------------------------------------------
# R-group labelling across a whole reaction
# ---------------------------------------------------------------------------

def _assign_r_labels(reactant_frags: list[Fragment],
                     product_frags: list[Fragment]) -> None:
    """Replace generic "R" labels with numbered R1, R2, … when needed.

    Atoms with the same map number across fragments get the same label.
    If there is only one unique R group, the label stays "R".
    """
    # Collect (map_num, fragment, node_index) for all R nodes
    r_nodes: list[tuple[int, Fragment, int]] = []
    for frag in reactant_frags + product_frags:
        for i, node in enumerate(frag.nodes):
            if node.get("type") == "nickname" and node.get("label") == "R":
                r_nodes.append((node["map_num"], frag, i))

    # Group by map number
    map_nums = sorted({mn for mn, _, _ in r_nodes if mn > 0})
    # Unmapped R-groups each get their own label
    unmapped = [(mn, f, i) for mn, f, i in r_nodes if mn == 0]
    next_label = len(map_nums) + 1

    if len(map_nums) + len(unmapped) <= 1:
        return  # single R group, keep as "R"

    # Assign labels
    label_map: dict[int, str] = {}
    for idx, mn in enumerate(map_nums, 1):
        label_map[mn] = f"R{idx}" if len(map_nums) + len(unmapped) > 1 else "R"

    for mn, frag, i in r_nodes:
        if mn > 0 and mn in label_map:
            frag.nodes[i]["label"] = label_map[mn]
        elif mn == 0:
            frag.nodes[i]["label"] = f"R{next_label}"
            next_label += 1


# ---------------------------------------------------------------------------
# Nickname definitions (X = F, Cl, Br, I; R = alkyl; …)
# ---------------------------------------------------------------------------

def _describe_atom(info: dict) -> str:
    """Derive a human-readable definition from parsed SMARTS atom info."""
    if info["elem_list"]:
        return ", ".join(info["elem_list"])
    elem = info.get("element")
    if elem == 6:
        return "aryl" if info.get("aromatic") else "alkyl"
    if elem == 7:
        return "aryl N" if info.get("aromatic") else "amine"
    if elem == 8:
        return "O"
    if elem == 16:
        return "S"
    if elem is not None:
        return ELEMENT_SYMBOLS.get(elem, f"#{elem}")
    return "?"


def _collect_definitions(reactant_frags: list["Fragment"],
                         product_frags: list["Fragment"]) -> dict[str, str]:
    """Build {label: definition} for every nickname that appears.

    Only looks at reactant-side atoms to avoid duplicate definitions
    (products repeat the same labels).  Falls back to product-only
    labels if they introduce a new nickname (e.g. a by-product).
    """
    defs: dict[str, str] = {}
    seen_labels: set[str] = set()

    for frag in reactant_frags + product_frags:
        for node in frag.nodes:
            if node["type"] != "nickname":
                continue
            label = node["label"]
            if label in seen_labels:
                continue
            seen_labels.add(label)
            defs[label] = _describe_atom(node["atom_info"])
    return defs


# ---------------------------------------------------------------------------
# CDXML builder
# ---------------------------------------------------------------------------

class CDXMLBuilder:
    """Assemble a CDXML reaction scheme from Fragment objects."""

    def __init__(self):
        self.reactants: list[Fragment] = []
        self.products: list[Fragment] = []
        self.definitions: dict[str, str] = {}  # label → definition
        self._next_id = 1

    def _id(self) -> int:
        val = self._next_id
        self._next_id += 1
        return val

    @classmethod
    def from_smirks(cls, smirks: str) -> "CDXMLBuilder":
        rxn = AllChem.ReactionFromSmarts(smirks)
        if rxn is None:
            raise ValueError(f"RDKit could not parse SMIRKS: {smirks}")
        builder = cls()
        for i in range(rxn.GetNumReactantTemplates()):
            builder.reactants.append(Fragment(rxn.GetReactantTemplate(i)))
        for i in range(rxn.GetNumProductTemplates()):
            builder.products.append(Fragment(rxn.GetProductTemplate(i)))
        _assign_r_labels(builder.reactants, builder.products)
        builder.definitions = _collect_definitions(
            builder.reactants, builder.products
        )
        return builder

    # ---- public ----------------------------------------------------------

    def build(self, name: str = "") -> str:
        """Return a complete CDXML string."""
        # --- layout ---
        centre_y = PAGE_HEIGHT / 2
        x = FRAG_PAD

        reactant_layout: list[tuple[Fragment, float, float]] = []
        for i, frag in enumerate(self.reactants):
            y_off = centre_y - frag.height / 2
            reactant_layout.append((frag, x, y_off))
            x += frag.width + FRAG_PAD
            if i < len(self.reactants) - 1:
                x += PLUS_GAP

        arrow_tail = x + ARROW_PAD
        arrow_head = arrow_tail + ARROW_LENGTH
        x = arrow_head + ARROW_PAD + FRAG_PAD

        product_layout: list[tuple[Fragment, float, float]] = []
        for i, frag in enumerate(self.products):
            y_off = centre_y - frag.height / 2
            product_layout.append((frag, x, y_off))
            x += frag.width + FRAG_PAD
            if i < len(self.products) - 1:
                x += PLUS_GAP

        total_w = x + FRAG_PAD
        page_w = max(total_w, PAGE_WIDTH)

        # --- XML tree ---
        root = Element("CDXML")
        root.set("CreationProgram", "Rxn-INSIGHT")
        root.set("BondLength", f"{BOND_LENGTH_PT:.2f}")
        root.set("LabelFont", "3")
        root.set("LabelSize", str(FONT_SIZE))
        root.set("CaptionFont", "3")
        root.set("CaptionSize", str(FONT_SIZE))

        # colour table (indices 2, 3, … — 0 and 1 are reserved)
        ct = SubElement(root, "colortable")
        SubElement(ct, "color", r="1", g="1", b="1")   # 2 = white
        SubElement(ct, "color", r="0", g="0", b="0")   # 3 = black
        SubElement(ct, "color", r="1", g="0", b="0")   # 4 = red
        SubElement(ct, "color", r="0", g="0", b="1")   # 5 = blue

        ft = SubElement(root, "fonttable")
        SubElement(ft, "font", id="3", charset="iso-8859-1", name="Arial")

        page = SubElement(root, "page")
        page.set("BoundingBox", f"0 0 {page_w:.0f} {PAGE_HEIGHT}")
        page.set("Width", f"{page_w:.0f}")
        page.set("Height", str(PAGE_HEIGHT))
        page.set("HeightPages", "1")
        page.set("WidthPages", "1")

        # optional title
        if name:
            tid = self._id()
            t = SubElement(page, "t", id=str(tid))
            t.set("p", f"{page_w / 2:.0f} 40")
            t.set("Justification", "Center")
            s = SubElement(t, "s", font="3", size="12")
            s.set("color", "3")
            s.text = name

        # --- fragments, +, arrow ---
        reactant_ids: list[int] = []
        product_ids: list[int] = []
        plus_ids: list[int] = []

        for i, (frag, xo, yo) in enumerate(reactant_layout):
            fid = self._id()
            reactant_ids.append(fid)
            self._write_fragment(page, frag, fid, xo, yo)
            if i < len(reactant_layout) - 1:
                # "+" text between reactants
                plus_x = xo + frag.width + FRAG_PAD + PLUS_GAP / 2
                pid = self._id()
                plus_ids.append(pid)
                t = SubElement(page, "t", id=str(pid))
                t.set("p", f"{plus_x:.1f} {centre_y + 4:.1f}")
                t.set("Justification", "Center")
                s = SubElement(t, "s", font="3", size=str(FONT_SIZE))
                s.set("color", "3")
                s.text = "+"

        # arrow
        arrow_id = self._id()
        ar = SubElement(page, "arrow", id=str(arrow_id))
        ar.set("FillType", "None")
        ar.set("ArrowheadHead", "Full")
        ar.set("ArrowheadType", "Solid")
        ar.set("HeadSize", "1000")
        ar.set("ArrowheadCenterSize", "625")
        ar.set("ArrowheadWidth", "250")
        ar.set("Head3D", f"{arrow_head:.1f} {centre_y:.1f} 0")
        ar.set("Tail3D", f"{arrow_tail:.1f} {centre_y:.1f} 0")
        bb_top = centre_y - 5
        bb_bot = centre_y + 5
        ar.set("BoundingBox",
               f"{arrow_tail:.1f} {bb_top:.1f} {arrow_head:.1f} {bb_bot:.1f}")

        for i, (frag, xo, yo) in enumerate(product_layout):
            fid = self._id()
            product_ids.append(fid)
            self._write_fragment(page, frag, fid, xo, yo)
            if i < len(product_layout) - 1:
                plus_x = xo + frag.width + FRAG_PAD + PLUS_GAP / 2
                pid = self._id()
                plus_ids.append(pid)
                t = SubElement(page, "t", id=str(pid))
                t.set("p", f"{plus_x:.1f} {centre_y + 4:.1f}")
                t.set("Justification", "Center")
                s = SubElement(t, "s", font="3", size=str(FONT_SIZE))
                s.set("color", "3")
                s.text = "+"

        # scheme
        sch_id = self._id()
        scheme = SubElement(page, "scheme", id=str(sch_id))
        step_id = self._id()
        step = SubElement(scheme, "step", id=str(step_id))
        step.set("ReactionStepReactants",
                 " ".join(str(x) for x in reactant_ids))
        step.set("ReactionStepProducts",
                 " ".join(str(x) for x in product_ids))
        step.set("ReactionStepArrows", str(arrow_id))
        if plus_ids:
            step.set("ReactionStepPlusSigns",
                     " ".join(str(x) for x in plus_ids))

        # --- definitions below scheme ---
        if self.definitions:
            def_parts = [f"{lbl} = {desc}"
                         for lbl, desc in self.definitions.items()]
            def_text = ";  ".join(def_parts)
            dtid = self._id()
            dt = SubElement(page, "t", id=str(dtid))
            dt.set("p", f"{page_w / 2:.0f} {centre_y + 50:.0f}")
            dt.set("Justification", "Center")
            ds = SubElement(dt, "s", font="3", size=str(FONT_SIZE))
            ds.set("color", "3")
            ds.text = def_text

        # --- serialise ---
        raw = tostring(root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ")
        # Remove minidom's own xml declaration line, add our own + DOCTYPE
        lines = pretty.split("\n", 1)
        body = lines[1] if len(lines) > 1 else lines[0]
        header = (
            '<?xml version="1.0" encoding="UTF-8" ?>\n'
            '<!DOCTYPE CDXML SYSTEM '
            '"http://www.cambridgesoft.com/xml/cdxml.dtd">\n'
        )
        return header + body

    # ---- private ---------------------------------------------------------

    def _write_fragment(self, parent, frag: Fragment,
                        frag_id: int, x_off: float, y_off: float):
        """Write a <fragment> element with its nodes and bonds."""
        el = SubElement(parent, "fragment", id=str(frag_id))
        node_ids: dict[int, int] = {}  # atom_idx → CDXML id

        for node, (cx, cy) in zip(frag.nodes, frag.coords):
            nid = self._id()
            node_ids[node["idx"]] = nid
            px, py = x_off + cx, y_off + cy

            n = SubElement(el, "n", id=str(nid))
            n.set("p", f"{px:.1f} {py:.1f}")

            if node["type"] == "element":
                n.set("Element", str(node["element"]))
                if node.get("h_count") is not None:
                    n.set("NumHydrogens", str(node["h_count"]))
                charge = node.get("charge", 0)
                if charge != 0:
                    n.set("Charge", str(charge))
            else:
                # GenericNickname
                label = node["label"]
                n.set("NodeType", "GenericNickname")
                n.set("GenericNickname", label)
                t = SubElement(n, "t")
                t.set("p", f"{px - 3:.1f} {py + 4:.1f}")
                s = SubElement(t, "s", font="3", size=str(FONT_SIZE))
                s.set("color", "3")
                s.text = label

        for begin, end, order in frag.bonds:
            bid = self._id()
            b = SubElement(el, "b", id=str(bid))
            b.set("B", str(node_ids[begin]))
            b.set("E", str(node_ids[end]))
            if order != 1:
                b.set("Order", str(order))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smirks_to_cdxml(smirks: str, name: str = "") -> str:
    """Convert a SMIRKS pattern to a CDXML reaction scheme string.

    Parameters
    ----------
    smirks : str
        Generalized SMIRKS (e.g. from the validated SMIRKS pipeline).
    name : str, optional
        Reaction name displayed as a title above the scheme.

    Returns
    -------
    str
        CDXML XML content, ready to write to a .cdxml file.
    """
    builder = CDXMLBuilder.from_smirks(smirks)
    return builder.build(name=name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert generalized SMIRKS to CDXML reaction schemes"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smirks", help="Single SMIRKS string")
    src.add_argument("--input", help="JSON file with SMIRKS entries")
    ap.add_argument("--output", required=True,
                    help="Output .cdxml file (single) or directory (batch)")
    ap.add_argument("--name", default="",
                    help="Reaction name (single SMIRKS mode)")
    args = ap.parse_args()

    if args.smirks:
        xml = smirks_to_cdxml(args.smirks, name=args.name)
        Path(args.output).write_text(xml, encoding="utf-8")
        print(f"Written: {args.output}")
    else:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        entries: list[dict] = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, dict) and "smirks" in val:
                    entries.append(val)
                elif isinstance(val, list):
                    entries.extend(
                        e for e in val if isinstance(e, dict) and "smirks" in e
                    )

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for entry in entries:
            smirks = entry.get("smirks", "")
            name = entry.get("name", "")
            if not smirks:
                continue
            safe = re.sub(r"[^\w\-.]", "_", (name or "reaction")[:80])
            fname = safe + ".cdxml"
            xml = smirks_to_cdxml(smirks, name=name)
            (out / fname).write_text(xml, encoding="utf-8")
            written += 1
        print(f"Written {written} CDXML files to {out}/")


if __name__ == "__main__":
    main()
