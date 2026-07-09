"""
Stereochemistry-aware reaction template extraction module for Rxn-INSIGHT.

This module provides functions to extract reaction templates from chemical reactions
while preserving stereochemical information and handling complex ring structures.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional, Any
import numpy as np
import itertools


def mendeleev(n: int) -> str:
    """
    Returns the chemical symbol for an atomic number.
    """
    periodic_table = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
        97: "Bk",
        98: "Cf",
        99: "Es",
        100: "Fm",
        101: "Md",
        102: "No",
        103: "Lr",
        104: "Rf",
        105: "Db",
        106: "Sg",
        107: "Bh",
        108: "Hs",
        109: "Mt",
        110: "Ds",
        111: "Rg",
        112: "Cn",
        113: "Nh",
        114: "Fl",
        115: "Mc",
        116: "Lv",
        117: "Ts",
        118: "Og"
    }
    atomic_symbol = periodic_table.get(n)

    if atomic_symbol is None:
        raise ValueError(f"Atom number {n} is not yet supported!")

    return atomic_symbol


def get_sssr_of_atom(sssr: List[Tuple[int]], idx: int) -> Tuple[int, Optional[int]]:
    """
    Get ring membership information for an atom.

    Determines how many rings an atom is part of and the size of the
    smallest ring containing the atom.

    Args:
        sssr: Smallest set of smallest rings (list of tuples of atom indices).
        idx: Index of the atom to check.

    Returns:
        Tuple containing:
            - Number of rings the atom is part of
            - Size of the smallest ring containing the atom (None if not in any ring)
    """
    is_in = 0
    smallest = None
    for rs in sssr:
        if idx in rs:  # only update when atom is in this ring
            is_in += 1
            if smallest is None or len(rs) < smallest:
                smallest = len(rs)
    return is_in, smallest


def get_atomic_smarts(
        atom: Chem.Atom,
        sssr: List[Tuple[int]],
        sssr_count: int,
        mol: Optional[Chem.Mol] = None,
        atom_idx: Optional[int] = None,
        discover_order: Optional[Dict[int, int]] = None,
        include_ring_info: bool = True,
        relaxed: bool = False,
) -> str:
    """
    Generate SMARTS string for a single atom with stereochemistry parity correction.

    Creates a SMARTS representation of an atom including its properties such as
    symbol, charge, hydrogen count, degree, and optionally stereochemistry and
    ring membership information.

    Args:
        atom: RDKit atom object to convert to SMARTS.
        sssr: Smallest set of smallest rings for the molecule.
        sssr_count: Total number of rings in SSSR.
        mol: Parent molecule (required for stereochemistry correction).
        atom_idx: Index of this atom in the molecule (required for stereochemistry).
        discover_order: DFS discovery order mapping for parity computation.
        include_ring_info: Whether to include ring membership information.
        relaxed: If True, produce a minimal SMARTS with only element,
            aromaticity, and formal charge (for context atoms).

    Returns:
        SMARTS string representation of the atom.
    """
    smarts = "["
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic():
        symbol = symbol.lower()
    smarts = smarts + symbol + ";"

    if relaxed:
        # Context atoms: element + aromaticity + H count + formal charge.
        # H count is kept because it distinguishes e.g. nH from n, which is
        # critical for retro-correctness even on context atoms.
        hs = atom.GetTotalNumHs()
        smarts += f"H{hs};"
        chg = atom.GetFormalCharge()
        if chg >= 0:
            smarts += f"+{int(chg)}]"
        else:
            smarts += f"{chg}]"
        return smarts

    # Add hydrogen count
    hs = atom.GetTotalNumHs()
    smarts += f"H{hs};"

    # Add degree
    deg = atom.GetDegree()
    smarts += f"D{deg};"

    # Add formal charge
    chg = atom.GetFormalCharge()
    if chg >= 0:
        smarts += f"+{int(chg)};"
    else:
        smarts += f"{chg};"

    # Add tetrahedral stereochemistry if present
    try:
        has_cip = atom.HasProp('_CIPCode')
    except Exception:
        has_cip = False
    ch_tag = atom.GetChiralTag()

    if (has_cip or (ch_tag != Chem.ChiralType.CHI_UNSPECIFIED)) and (mol is not None) and (atom_idx is not None):
        chirality = atom.GetChiralTag()
        local_at = "@@" if chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CW else "@"

        # Apply parity correction based on DFS traversal order
        if discover_order is not None:
            rdkit_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
            try:
                traversal_sorted = sorted(rdkit_neighbors, key=lambda x: discover_order.get(x, x))
                pos = {n: i for i, n in enumerate(traversal_sorted)}
                perm = [pos[n] for n in rdkit_neighbors]
                visited_p = [False] * len(perm)
                parity = 0
                for i in range(len(perm)):
                    if visited_p[i]:
                        continue
                    cycle_len = 0
                    j = i
                    while not visited_p[j]:
                        visited_p[j] = True
                        j = perm[j]
                        cycle_len += 1
                    if cycle_len > 0:
                        parity += (cycle_len - 1)
                if parity % 2 == 1:
                    local_at = "@@" if local_at == "@" else "@"
            except Exception:
                pass
        smarts += local_at + ";"

    # Add ring information if requested
    if include_ring_info:
        if sssr_count == 0:
            smarts += "!R]"
        else:
            in_sssr, smallest_r = get_sssr_of_atom(sssr, atom.GetIdx())
            if in_sssr == 0:
                smarts += "!R]"
            else:
                smarts += f"R{in_sssr};r{smallest_r}]"
    else:
        smarts = smarts[:-1] + "]"  # Close without ring info

    return smarts


def get_bond_symbol(bond: Optional[Chem.Bond]) -> str:
    """
    Get SMARTS bond symbol from RDKit bond.

    Args:
        bond: RDKit bond object or None.

    Returns:
        SMARTS bond symbol (e.g., '=' for double bond, '#' for triple bond).
    """
    if bond is None:
        return ""
    bond_type = bond.GetBondTypeAsDouble()
    bond_map = {1.0: "-", 1.5: ":", 2.0: "=", 3.0: "#"}
    return bond_map.get(bond_type, "~")


def get_adjacent_double_bond_slash(
        mol: Chem.Mol,
        atom_idx: int,
        nb_idx: int,
        discover_order: Dict[int, int]
) -> Optional[str]:
    """
    Determine E/Z stereochemistry slash for bonds adjacent to stereo double bonds.

    If the bond atom_idx -> nb_idx is a single bond adjacent to a stereo double bond,
    returns '/' or '\\' to encode E/Z stereochemistry. This is crucial for preserving
    stereochemical information in SMARTS patterns.

    Args:
        mol: Molecule containing the bond.
        atom_idx: Index of the first atom in the bond.
        nb_idx: Index of the second atom in the bond.
        discover_order: DFS discovery order mapping.

    Returns:
        '/' or '\\' for stereochemical bonds, None otherwise.
    """
    atom = mol.GetAtomWithIdx(atom_idx)

    # Check if atom_idx is part of a stereo double bond
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
            continue

        # Get the other atom in the double bond
        db_partner_idx = bond.GetOtherAtomIdx(atom_idx)

        # Skip if nb_idx is the double bond partner itself
        if nb_idx == db_partner_idx:
            continue

        # Get the atoms that define stereochemistry
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        stereo_atoms = bond.GetStereoAtoms()

        if not stereo_atoms or len(stereo_atoms) != 2:
            continue

        a1, b1 = stereo_atoms  # a1 is bonded to begin, b1 is bonded to end

        # Check if nb_idx is one of the stereo atoms
        if nb_idx not in stereo_atoms:
            continue

        # For Z configuration: use opposite slashes (/ then \)
        # For E configuration: use same slashes (/ then /)

        if atom_idx == begin_idx and nb_idx == a1:
            # From begin atom to its substituent
            return "/"  # Always use / for the first substituent

        elif atom_idx == end_idx and nb_idx == b1:
            # From end atom to its substituent
            if stereo == Chem.BondStereo.STEREOZ:
                return "\\"  # Opposite slash for Z
            else:  # STEREOE
                return "/"  # Same slash for E

    # Also check if the neighbor is part of a stereo double bond
    neighbor = mol.GetAtomWithIdx(nb_idx)
    for bond in neighbor.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
            continue

        db_partner_idx = bond.GetOtherAtomIdx(nb_idx)
        if atom_idx == db_partner_idx:
            continue

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        stereo_atoms = bond.GetStereoAtoms()

        if not stereo_atoms or len(stereo_atoms) != 2:
            continue

        a1, b1 = stereo_atoms

        if atom_idx not in stereo_atoms:
            continue

        # We're going TO a double-bond atom FROM its stereo substituent
        if nb_idx == begin_idx and atom_idx == a1:
            # To begin from its substituent
            return "/"

        elif nb_idx == end_idx and atom_idx == b1:
            # To end from its substituent
            if stereo == Chem.BondStereo.STEREOZ:
                return "\\"
            else:  # STEREOE
                return "/"

    return None


def get_leaving_groups(
        mol: Chem.Mol,
        reaction_center_indices: Set[int],
        product_connections: Set[int]
) -> Set[int]:
    """
    Identify complete leaving groups connected to reaction center atoms.

    A leaving group is any connected component attached to a reaction center atom
    that is not present in the products. This function performs a breadth-first
    search to find all atoms that are part of leaving groups.

    Args:
        mol: Reactant molecule to analyze.
        reaction_center_indices: Set of reaction center atom indices in this molecule.
        product_connections: Set of atom mapping numbers that the reaction center
                           connects to in the product.

    Returns:
        Set of atom indices that are part of leaving groups.
    """
    leaving_group_atoms = set()

    for rc_idx in reaction_center_indices:
        rc_atom = mol.GetAtomWithIdx(rc_idx)

        # Check each neighbor of the reaction center atom
        for neighbor in rc_atom.GetNeighbors():
            nb_idx = neighbor.GetIdx()

            # Check if this neighbor has a mapping
            nb_has_mapping = neighbor.HasProp('molAtomMapNumber')

            # If the neighbor is not mapped, it's likely part of a leaving group
            # Include it and all connected non-mapped atoms
            if not nb_has_mapping:
                # BFS to find all connected non-mapped atoms (complete leaving group)
                queue = [nb_idx]
                visited = set()

                while queue:
                    current_idx = queue.pop(0)
                    if current_idx in visited:
                        continue

                    visited.add(current_idx)
                    leaving_group_atoms.add(current_idx)

                    current_atom = mol.GetAtomWithIdx(current_idx)
                    for next_neighbor in current_atom.GetNeighbors():
                        next_idx = next_neighbor.GetIdx()
                        # Don't go back to reaction center
                        if next_idx not in reaction_center_indices and next_idx not in visited:
                            # Only include non-mapped atoms in the leaving group
                            if not next_neighbor.HasProp('molAtomMapNumber'):
                                queue.append(next_idx)

    return leaving_group_atoms


def find_shortest_path_between_sets(
        mol: Chem.Mol,
        set1: Set[int],
        set2: Set[int]
) -> Optional[List[int]]:
    """
    Find the shortest path between any atom in set1 and any atom in set2.

    This function is used to connect disconnected reaction center components
    by finding the minimal atoms needed to form a connected subgraph.

    Args:
        mol: Molecule containing the atoms.
        set1: First set of atom indices.
        set2: Second set of atom indices.

    Returns:
        List of atom indices forming the shortest path, or None if no path exists.
    """
    if not set1 or not set2:
        return None

    # If sets overlap, they're already connected
    if set1 & set2:
        return []

    shortest_path = None
    shortest_length = float('inf')

    for idx1 in set1:
        for idx2 in set2:
            path = Chem.GetShortestPath(mol, idx1, idx2)
            if path and len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = list(path)

    return shortest_path


def connect_reaction_centers(
        mol: Chem.Mol,
        reaction_center_indices: Set[int],
        included_atoms: Set[int]
) -> Set[int]:
    """
    Ensure all reaction centers in the same molecule are connected.

    This function adds atoms along the shortest paths between disconnected
    reaction center groups to ensure they form a connected subgraph. This is
    important for creating valid SMARTS patterns that capture the full
    reaction transformation.

    Args:
        mol: Molecule containing the reaction centers.
        reaction_center_indices: Set of reaction center atom indices.
        included_atoms: Set of atoms already included in the template.

    Returns:
        Expanded set of atom indices ensuring all reaction centers are connected.
    """
    if len(reaction_center_indices) <= 1:
        return included_atoms

    # Find connected components among reaction centers
    rc_components = []
    visited = set()

    for rc_idx in reaction_center_indices:
        if rc_idx in visited:
            continue

        # BFS to find connected component
        component = set()
        queue = [rc_idx]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            # Only follow paths through included atoms
            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx in reaction_center_indices and nb_idx not in visited:
                    if nb_idx in included_atoms:
                        queue.append(nb_idx)

        rc_components.append(component)

    # Connect all components by finding shortest paths
    expanded_atoms = set(included_atoms)

    while len(rc_components) > 1:
        # Find the two closest components
        min_dist = float('inf')
        best_pair = (0, 1)
        best_path = None

        for i in range(len(rc_components)):
            for j in range(i + 1, len(rc_components)):
                path = find_shortest_path_between_sets(mol, rc_components[i], rc_components[j])
                if path and len(path) < min_dist:
                    min_dist = len(path)
                    best_pair = (i, j)
                    best_path = path

        if best_path:
            # Add the connecting path to included atoms
            expanded_atoms.update(best_path)

            # Merge the components
            i, j = best_pair
            rc_components[i] = rc_components[i] | rc_components[j]
            del rc_components[j]
        else:
            # No path found between components (shouldn't happen in a connected molecule)
            break

    return expanded_atoms


def expand_atoms_by_radius_with_leaving_groups(
        mol: Chem.Mol,
        center_atoms: Set[int],
        radius: int,
        atom_map_dict: Dict[int, int],
        include_leaving_groups: bool = True,
        connect_centers: bool = True,
        product_mol: Optional[Chem.Mol] = None,
        product_map_dict: Optional[Dict[int, int]] = None
) -> Set[int]:
    """
    Expand atom set by radius from reaction center, including leaving groups and stereo atoms.

    This function expands the template to include atoms within a specified radius
    of the reaction center, always includes complete leaving groups, and ensures
    that atoms defining stereochemistry are included when stereochemistry changes.

    Args:
        mol: Current molecule (reactant or product).
        center_atoms: Set of reaction center atom map numbers.
        radius: Expansion radius (number of bonds from reaction center).
        atom_map_dict: Map from atom map number to atom index in current molecule.
        include_leaving_groups: Whether to include leaving groups.
        connect_centers: Whether to ensure all reaction centers are connected.
        product_mol: Product molecule (for checking stereochemistry changes).
        product_map_dict: Map from atom map number to atom index in product.

    Returns:
        Set of atom indices to include in the template.
    """
    included = set()
    current_shell = set()

    # Start with reaction center atoms
    reaction_center_indices = set()
    for map_num in center_atoms:
        if map_num in atom_map_dict:
            idx = atom_map_dict[map_num]
            included.add(idx)
            current_shell.add(idx)
            reaction_center_indices.add(idx)

    # ALWAYS include complete leaving groups (regardless of radius)
    if include_leaving_groups:
        leaving_groups = get_leaving_groups(mol, reaction_center_indices, set())
        included.update(leaving_groups)

    # Collect stereo reference atoms separately so that they don't block the
    # radius expansion from passing through them.  They are merged into
    # `included` *after* the main expansion loop.
    stereo_extra: Set[int] = set()

    # Check for bonds that change to/from stereo double bonds
    for rc_idx in list(reaction_center_indices):
        atom = mol.GetAtomWithIdx(rc_idx)

        # Check each bond from this reaction center atom
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(rc_idx)
            other_atom = mol.GetAtomWithIdx(other_idx)

            # Check if the other atom is also in reaction center
            if other_atom.HasProp('molAtomMapNumber'):
                other_map = other_atom.GetIntProp('molAtomMapNumber')
                if other_map in center_atoms:
                    # This is a reaction center bond

                    # Check if this bond becomes a stereo double bond in product
                    if product_mol is not None and product_map_dict is not None:
                        rc_map = atom.GetIntProp('molAtomMapNumber')
                        if rc_map in product_map_dict and other_map in product_map_dict:
                            prod_idx1 = product_map_dict[rc_map]
                            prod_idx2 = product_map_dict[other_map]
                            prod_bond = product_mol.GetBondBetweenAtoms(prod_idx1, prod_idx2)

                            # If product bond is stereo double, include substituents
                            if prod_bond and prod_bond.GetBondTypeAsDouble() == 2.0:
                                if prod_bond.GetStereo() in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                                    # Include substituents on both sides
                                    for neigh in atom.GetNeighbors():
                                        if neigh.GetIdx() != other_idx:
                                            stereo_extra.add(neigh.GetIdx())
                                            break
                                    for neigh in other_atom.GetNeighbors():
                                        if neigh.GetIdx() != rc_idx:
                                            stereo_extra.add(neigh.GetIdx())
                                            break

    # Handle existing stereo double bonds
    for rc_idx in list(reaction_center_indices):
        atom = mol.GetAtomWithIdx(rc_idx)
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0 and bond.GetStereo() in (
            Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                stereo_atoms = bond.GetStereoAtoms()
                if stereo_atoms and len(stereo_atoms) == 2:
                    for sat in stereo_atoms:
                        if sat >= 0:
                            stereo_extra.add(sat)
                else:
                    # Fallback: include first neighbor on each side
                    a_idx = bond.GetBeginAtomIdx()
                    b_idx = bond.GetEndAtomIdx()
                    for neigh in mol.GetAtomWithIdx(a_idx).GetNeighbors():
                        nidx = neigh.GetIdx()
                        if nidx != b_idx:
                            stereo_extra.add(nidx)
                            break
                    for neigh in mol.GetAtomWithIdx(b_idx).GetNeighbors():
                        nidx = neigh.GetIdx()
                        if nidx != a_idx:
                            stereo_extra.add(nidx)
                            break

    # Expand by radius from reaction center atoms.
    # stereo_extra atoms are intentionally excluded from `included` here so
    # that the expansion can pass through them freely.
    for r in range(radius):
        next_shell = set()
        for atom_idx in current_shell:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx not in included:
                    next_shell.add(nb_idx)
                    included.add(nb_idx)
        current_shell = next_shell

    # Merge stereo reference atoms after the expansion so they are always
    # present in the template even when they lie outside the radius.
    included.update(stereo_extra)

    # Ensure all reaction centers are connected
    if connect_centers and len(reaction_center_indices) > 1:
        included = connect_reaction_centers(mol, reaction_center_indices, included)

    return included


def find_ring_closures(
        mol: Chem.Mol,
        component_atoms: List[int],
        start_atom: int
) -> Tuple[Dict[int, List[Tuple[int, str, int]]], Dict[Tuple[int, int], int], Dict[int, int]]:
    """
    Pre-compute ring closures in a molecular component using depth-first search.

    This function identifies all ring closures (back-edges in the DFS tree) and
    assigns them unique closure numbers for SMARTS representation.

    Args:
        mol: Molecule to analyze.
        component_atoms: List of atom indices in the component.
        start_atom: Starting atom index for DFS.

    Returns:
        Tuple containing:
            - atom_closures: Dict mapping atom index to list of (closure_num, bond_symbol, other_idx)
            - ring_closures: Dict mapping sorted edge (u,v) to closure number
            - discover_order: Dict mapping atom index to DFS discovery order
    """
    component_set = set(component_atoms)
    visited = set()
    parent = {}
    ring_closures = {}  # Maps sorted edge (u,v) -> closure number
    atom_closures = defaultdict(list)
    discover_order = {}
    next_closure = 1
    t = 0  # discovery time counter

    def dfs(atom_idx, parent_idx):
        nonlocal next_closure, t
        visited.add(atom_idx)
        parent[atom_idx] = parent_idx
        discover_order[atom_idx] = t
        t += 1

        atom = mol.GetAtomWithIdx(atom_idx)

        # Sort neighbors to ensure consistent traversal order
        neighbors = []
        for neighbor in atom.GetNeighbors():
            nb_idx = neighbor.GetIdx()
            if nb_idx not in component_set:
                continue
            if nb_idx == parent_idx:
                continue
            neighbors.append(nb_idx)

        # Sort neighbors consistently
        neighbors.sort(key=lambda x: (mol.GetAtomWithIdx(x).GetAtomicNum(), x))

        for nb_idx in neighbors:
            if nb_idx in visited:
                # Found a back-edge to an ancestor (closure)
                edge = tuple(sorted((atom_idx, nb_idx)))
                if edge not in ring_closures:
                    ring_closures[edge] = next_closure
                    bond = mol.GetBondBetweenAtoms(atom_idx, nb_idx)
                    bond_symbol = get_bond_symbol(bond)
                    atom_closures[atom_idx].append((next_closure, bond_symbol, nb_idx))
                    atom_closures[nb_idx].append((next_closure, bond_symbol, atom_idx))
                    next_closure += 1
            else:
                dfs(nb_idx, atom_idx)

    dfs(start_atom, -1)
    return atom_closures, ring_closures, discover_order


def build_component_with_rings(
        mol: Chem.Mol,
        component_atoms: List[int],
        sssr: List[Tuple[int]],
        sssr_count: int,
        idx_to_map: Dict[int, int],
        include_ring_info: bool = True,
        rc_indices: Optional[Set[int]] = None,
) -> str:
    """
    Build SMARTS string for a connected component with proper ring closure handling.

    This function constructs a SMARTS pattern for a connected molecular component,
    handling ring closures and preserving atom mappings.

    Args:
        mol: Molecule containing the component.
        component_atoms: List of atom indices in the component.
        sssr: Smallest set of smallest rings for the molecule.
        sssr_count: Total number of rings in SSSR.
        idx_to_map: Dictionary mapping atom index to atom map number.
        include_ring_info: Whether to include ring membership information.
        rc_indices: Set of reaction-center atom indices.  When provided,
            atoms *not* in this set get relaxed (minimal) SMARTS.

    Returns:
        SMARTS string representation of the component.
    """
    if not component_atoms:
        return ""

    component_set = set(component_atoms)

    # Choose starting atom: mapped atom with the lowest map number.
    # This makes the DFS traversal order deterministic, so that
    # e.g. [O:1]-[C:2] and [C:2]-[O:1] always produce the same string.
    mapped_in_component = [
        (idx_to_map[idx], idx)
        for idx in component_atoms
        if idx in idx_to_map
    ]
    if mapped_in_component:
        start_atom = min(mapped_in_component)[1]
    else:
        start_atom = min(component_atoms)

    # Pre-compute ring closures + discovery order
    atom_closures, ring_closures, discover_order = find_ring_closures(mol, component_atoms, start_atom)

    # Build SMARTS with pre-computed ring closures
    visited = set()
    smarts = build_atom_dfs_with_precomputed_rings(
        mol, start_atom, -1, component_set, visited,
        sssr, sssr_count, idx_to_map, atom_closures, ring_closures, discover_order,
        include_ring_info, rc_indices=rc_indices,
    )

    return smarts


def build_atom_dfs_with_precomputed_rings(
        mol: Chem.Mol,
        atom_idx: int,
        parent_idx: int,
        component_set: Set[int],
        visited: Set[int],
        sssr: List[Tuple[int]],
        sssr_count: int,
        idx_to_map: Dict[int, int],
        atom_closures: Dict[int, List[Tuple[int, str, int]]],
        ring_closures: Dict[Tuple[int, int], int],
        discover_order: Dict[int, int],
        include_ring_info: bool = True,
        rc_indices: Optional[Set[int]] = None,
) -> str:
    """
    Build SMARTS string using DFS with pre-computed ring closures.

    This is a recursive function that builds the SMARTS pattern by traversing
    the molecular graph in depth-first order, handling ring closures and
    preserving stereochemistry.

    Args:
        mol: Molecule being processed.
        atom_idx: Current atom index.
        parent_idx: Parent atom index in DFS tree.
        component_set: Set of atoms in the current component.
        visited: Set of visited atom indices.
        sssr: Smallest set of smallest rings.
        sssr_count: Number of rings in SSSR.
        idx_to_map: Mapping from atom index to map number.
        atom_closures: Pre-computed ring closures for each atom.
        ring_closures: Mapping of edges to closure numbers.
        discover_order: DFS discovery order for atoms.
        include_ring_info: Whether to include ring membership information.
        rc_indices: Set of reaction-center atom indices.  When provided,
            atoms *not* in this set get relaxed (minimal) SMARTS.

    Returns:
        SMARTS string for the subtree rooted at atom_idx.
    """
    if atom_idx in visited:
        return ""

    visited.add(atom_idx)
    atom = mol.GetAtomWithIdx(atom_idx)

    # Atom SMARTS
    use_relaxed = rc_indices is not None and atom_idx not in rc_indices
    atom_smarts = get_atomic_smarts(
        atom, sssr, sssr_count, mol=mol, atom_idx=atom_idx,
        discover_order=discover_order,
        include_ring_info=include_ring_info and not use_relaxed,
        relaxed=use_relaxed,
    )

    # Add mapping for ALL mapped atoms
    if atom_idx in idx_to_map:
        map_num = idx_to_map[atom_idx]
        atom_smarts = atom_smarts[:-1] + f":{map_num}]"

    result = atom_smarts

    # Handle ring closures using the pre-computed atom_closures
    if atom_idx in atom_closures:
        entries = sorted(atom_closures[atom_idx], key=lambda x: (x[0], x[2]))
        for closure_num, bond_symbol, other_idx in entries:
            if other_idx not in visited:
                # Opening digit - the other atom hasn't been visited yet
                result += str(closure_num % 10)
            elif other_idx != parent_idx:
                # Closing digit - the other atom was already visited (and it's not our parent)
                # Add bond symbol if needed
                if bond_symbol and bond_symbol != "-":
                    result += bond_symbol
                result += str(closure_num % 10)

    # Process unvisited neighbors (tree edges only)
    neighbors_tree = []

    for neighbor in atom.GetNeighbors():
        nb_idx = neighbor.GetIdx()
        if nb_idx not in component_set or nb_idx == parent_idx:
            continue

        # Only process unvisited neighbors as tree edges
        if nb_idx not in visited:
            bond = mol.GetBondBetweenAtoms(atom_idx, nb_idx)
            neighbors_tree.append((nb_idx, bond))

    # Sort tree neighbors for consistent ordering
    neighbors_tree.sort(key=lambda x: (mol.GetAtomWithIdx(x[0]).GetAtomicNum(), x[0]))

    # Helper to compute bond symbol and possibly slash for E/Z on adjacent single bonds
    def compute_bond_symbol_for_edge(bond, from_idx, to_idx):
        bond_symbol_local = get_bond_symbol(bond)
        if bond.GetBondTypeAsDouble() == 1.0:
            slash = get_adjacent_double_bond_slash(mol, from_idx, to_idx, discover_order)
            if slash is not None:
                bond_symbol_local = slash
            else:
                # For single bonds, default to "-" if no slash needed
                bond_symbol_local = "-" if bond_symbol_local == "" else bond_symbol_local
        return bond_symbol_local

    # Add unvisited neighbors (tree edges)
    if len(neighbors_tree) == 1:
        nb_idx, bond = neighbors_tree[0]
        bond_symbol = compute_bond_symbol_for_edge(bond, atom_idx, nb_idx)
        nb_smarts = build_atom_dfs_with_precomputed_rings(
            mol, nb_idx, atom_idx, component_set, visited,
            sssr, sssr_count, idx_to_map, atom_closures, ring_closures, discover_order,
            include_ring_info, rc_indices=rc_indices,
        )
        if nb_smarts:
            result += bond_symbol + nb_smarts
    elif len(neighbors_tree) > 1:
        for nb_idx, bond in neighbors_tree[:-1]:
            bond_symbol = compute_bond_symbol_for_edge(bond, atom_idx, nb_idx)
            nb_smarts = build_atom_dfs_with_precomputed_rings(
                mol, nb_idx, atom_idx, component_set, visited,
                sssr, sssr_count, idx_to_map, atom_closures, ring_closures, discover_order,
                include_ring_info, rc_indices=rc_indices,
            )
            if nb_smarts:
                result += f"({bond_symbol}{nb_smarts})"

        nb_idx, bond = neighbors_tree[-1]
        bond_symbol = compute_bond_symbol_for_edge(bond, atom_idx, nb_idx)
        nb_smarts = build_atom_dfs_with_precomputed_rings(
            mol, nb_idx, atom_idx, component_set, visited,
            sssr, sssr_count, idx_to_map, atom_closures, ring_closures, discover_order,
            include_ring_info, rc_indices=rc_indices,
        )
        if nb_smarts:
            result += bond_symbol + nb_smarts

    return result


def build_smarts_with_ring_closures(
        mol: Chem.Mol,
        atom_indices: Set[int],
        sssr: List[Tuple[int]],
        sssr_count: int,
        idx_to_map: Dict[int, int],
        include_ring_info: bool = True,
        rc_indices: Optional[Set[int]] = None,
) -> str:
    """
    Build SMARTS string from a set of atoms with proper ring closure handling.

    This function handles disconnected components by building each component
    separately and joining them with dots.

    Args:
        mol: Molecule containing the atoms.
        atom_indices: Set of atom indices to include.
        sssr: Smallest set of smallest rings.
        sssr_count: Number of rings in SSSR.
        idx_to_map: Mapping from atom index to map number.
        include_ring_info: Whether to include ring membership information.
        rc_indices: Set of reaction-center atom indices.  When provided,
            atoms *not* in this set get relaxed (minimal) SMARTS.

    Returns:
        Complete SMARTS string, with disconnected components joined by dots.
    """
    if not atom_indices:
        return ""

    atom_indices = set(atom_indices)
    visited = set()
    components = []

    # Find all connected components
    for start_idx in atom_indices:
        if start_idx in visited:
            continue

        # Build this connected component
        component_atoms = []
        stack = [start_idx]

        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            component_atoms.append(idx)

            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                nb_idx = neighbor.GetIdx()
                if nb_idx in atom_indices and nb_idx not in visited:
                    stack.append(nb_idx)

        # Build SMARTS for this component with proper ring closures
        if component_atoms:
            component_smarts = build_component_with_rings(
                mol, component_atoms, sssr, sssr_count, idx_to_map,
                include_ring_info, rc_indices=rc_indices,
            )
            components.append(component_smarts)

    # Join disconnected components with dots
    return ".".join(components)


def check_template_accuracy(
        rsmi: str,
        template: str
) -> bool:
    """
    Verify that a template correctly reproduces the expected product.

    Tests whether applying the template to the reactants produces the
    expected product. This is useful for validating template extraction.

    Args:
        rsmi: Reaction SMILES string (reactants>>products).
        template: SMIRKS template to test.

    Returns:
        True if template produces the correct product, False otherwise.

    Examples:
        >>> rsmi = "CC(C)Br.O>>CC(C)O"
        >>> template = "[C:1][Br:2].[O:3]>>[C:1][O:3]"
        >>> check_template_accuracy(rsmi, template)
        True
    """
    try:
        rxn = AllChem.ReactionFromSmarts(template)
    except ValueError:
        return False

    reactants = rsmi.split(">")[0].split(".")
    product = Chem.CanonSmiles(rsmi.split(">")[-1])
    react_tuple = tuple(Chem.MolFromSmiles(reactant) for reactant in reactants)
    all_tuples = list(itertools.permutations(react_tuple))

    pred_products = []
    for tup in all_tuples:
        try:
            pred_product = rxn.RunReactants(tup)
        except Exception:
            continue
        pred_products += pred_product

    if len(pred_products) == 0:
        return False
    else:
        for prod in pred_products:
            smi = Chem.MolToSmiles(prod[0])
            if smi == product:
                return True

    return False


def measure_template_accuracy(
        rsmi: str,
        template: str,
) -> Dict[str, Any]:
    """Measure how well a SMIRKS template reconstructs a reaction bidirectionally.

    **Forward check**: applies the template to the reactants (trying all
    permutations) and verifies the correct product is recovered.

    **Retro check**: reverses the template and applies it to the product,
    verifying the correct reactant set is recovered.  The retro check catches
    templates that accidentally produce the right product from the wrong
    mechanism (forward-correct but retro-wrong).

    Args:
        rsmi: Atom-mapped or unmapped reaction SMILES (``reactants>>product``).
        template: SMIRKS template to evaluate.

    Returns:
        Dictionary with keys:

        * ``correct`` (bool) — forward template reproduces the exact product.
        * ``applicable`` (bool) — forward template fires on the reactants.
        * ``n_outcomes`` (int) — distinct forward product SMILES generated.
        * ``retro_correct`` (bool) — reversed template applied to the product
          recovers the exact reactant set.
        * ``retro_applicable`` (bool) — reversed template fires on the product.

    Examples:
        >>> rsmi = "CC(C)Br.O>>CC(C)O"
        >>> template = "[C:1][Br:2].[O:3]>>[C:1][O:3]"
        >>> measure_template_accuracy(rsmi, template)
        {'correct': True, 'applicable': True, 'n_outcomes': 1,
         'retro_correct': True, 'retro_applicable': True}
    """
    result: Dict[str, Any] = {
        'correct': False, 'applicable': False, 'n_outcomes': 0,
        'retro_correct': False, 'retro_applicable': False,
    }

    try:
        rxn = AllChem.ReactionFromSmarts(template)
    except Exception:
        return result

    parts = rsmi.split(">")
    reactant_smiles = parts[0].split(".")
    try:
        product = Chem.CanonSmiles(parts[-1])
    except Exception:
        return result

    react_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in react_mols):
        return result

    # ── Forward check ─────────────────────────────────────────────────────────
    unique_products: Set[str] = set()
    for tup in itertools.permutations(react_mols):
        try:
            outcomes = rxn.RunReactants(tup)
        except Exception:
            continue
        for outcome in outcomes:
            try:
                smi = Chem.MolToSmiles(outcome[0])
                unique_products.add(smi)
            except Exception:
                continue

    result['applicable'] = len(unique_products) > 0
    result['n_outcomes'] = len(unique_products)
    result['correct'] = product in unique_products

    # ── Retro check ───────────────────────────────────────────────────────────
    try:
        t_parts = template.split(">>")
        retro_rxn = AllChem.ReactionFromSmarts(f"{t_parts[1]}>>{t_parts[0]}")

        # Canonical reactant set to match against
        reactant_canon: Set[str] = {
            Chem.CanonSmiles(s) for s in reactant_smiles
            if Chem.MolFromSmiles(s) is not None
        }
        # Non-isomeric fallback set — catches E/Z stereo flips that occur
        # when a SMIRKS template is reversed (DFS traversal order changes
        # the meaning of '/' and '\' markers).
        reactant_nostereo: Set[str] = {
            Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
            for s in reactant_smiles
            if Chem.MolFromSmiles(s) is not None
        }

        product_mol = Chem.MolFromSmiles(parts[-1])
        if product_mol is not None:
            retro_outcomes = retro_rxn.RunReactants((product_mol,))
            if retro_outcomes:
                result['retro_applicable'] = True
                for outcome in retro_outcomes:
                    try:
                        # Each molecule in outcome may itself be disconnected
                        recovered: Set[str] = set()
                        recovered_nostereo: Set[str] = set()
                        for mol in outcome:
                            for frag in Chem.MolToSmiles(mol).split("."):
                                canon = Chem.CanonSmiles(frag)
                                if canon:
                                    recovered.add(canon)
                                frag_mol = Chem.MolFromSmiles(frag)
                                if frag_mol is not None:
                                    recovered_nostereo.add(
                                        Chem.MolToSmiles(frag_mol, isomericSmiles=False)
                                    )
                        if recovered == reactant_canon:
                            result['retro_correct'] = True
                            break
                        if recovered_nostereo == reactant_nostereo:
                            result['retro_correct'] = True
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    return result
