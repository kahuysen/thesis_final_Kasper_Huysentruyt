from rdkit import Chem
from typing import List, Union
from rdkit.Chem.rdchem import Mol


def get_peroxide_formers() -> List[Mol]:
    """
    This function returns a list of peroxide forming moieties in SMARTS format
    numbered from most (1) to least (14) likely to form
    :return:
    """
    peroxide_substructures = [
        # 1) Ethers and acetals with alpha-hydrogen
        "[#6][#8][#6h]",
        # 2) Alkenes with allylic hydrogen
        "[C]=[C][Ch]",
        # 3) Chloroalkenes, fluoroalkenes
        "[C]=[C][Cl,F]",
        # 4) Vinyl halides, Vinyl esters, Vinyl ethers
        "[Ch2]=[C][#6]",
        # 5) Dienes
        "[C]=[C][C]=[C]",
        # 5bis) Dienes in an aromatic ring that can participate in Diels-Alder reaction
        "[!c]cccc[!c]",
        # 6) Vinylalkynes with alpha-hydrogen
        "[C,c]=[Ch][C]#[Ch]",
        # 7) Alkylalkynes with alpha-hydrogen
        "[Ch][C]#[Ch]",
        # 8) Alkylarenes with tertiary alpha-hydrogen
        "[#6][Ch]([#6])([c])",
        # 9) Alkanes and cycloalkenes with tertiary hydrogen
        "[#6][Ch]([#6])([#6])",
        # 10) Acrylates, methacrylates
        "[C]=[C][C](=O)[O]",
        # 11) Secondary alcohols
        "[#6][Ch]([Oh])[#6]",
        # 12) Ketones with alpha-hydrogen
        "[#6](=O)[#6h,#6H]",
        # 13) Aldehydes
        "[CX3H1](=O)[#6]",
        # 14) Ureas, amides, and lactams with alpha-hydrogen atom on a carbon attached to nitrogen
        "[Ch](=O)[Nh][Ch]",
        # 15) Alkenes with benzylic hydrogen
        "[Ch]c1ccccc1",
        # 16) peroxide
        "[OX2,OX1-][OX2,OX1-]",
        # 17) Diynes
        "[C]#[C][C]#[C]",
    ]
    all_patterns = []
    for substructure in peroxide_substructures:
        pattern = Chem.MolFromSmarts(substructure)
        all_patterns.append(pattern)

    return all_patterns


def check_peroxide_formation(mol: Mol, peroxide_patterns: Union[List[Mol], None] = None) -> int:
    """
    This function checks if a molecule contains a peroxidizable organic moiety and returns 1 if it is the case and 0
    if not.
    """

    # Return 1 if any peroxide substructure is found in the molecule, otherwise return 0
    if peroxide_patterns is None:
        peroxide_patterns = get_peroxide_formers()

    score = 0

    for pattern in peroxide_patterns:
        if mol.HasSubstructMatch(pattern):
            score = 1
            break  # Exit loop once a match is found

    return score


def batch_peroxide_formation_check(smiles: List[str]) -> List[int]:
    """
    This function performs a peroxide formation check for a list of SMILES identifiers.
    """
    peroxide_patterns = get_peroxide_formers()
    scores = []

    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        score = check_peroxide_formation(mol)
        scores.append(score)

    return scores


if __name__ == '__main__':
    # Example usage
    input_smiles = ["CC#C"]
    peroxide_score = batch_peroxide_formation_check(input_smiles)
    print(peroxide_score)
