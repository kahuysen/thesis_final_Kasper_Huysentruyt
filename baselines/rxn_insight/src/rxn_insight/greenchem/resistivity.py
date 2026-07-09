from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem.rdchem import Mol
from typing import List


def has_charged_group(mol: Mol) -> bool:
    """
    Check if a molecule has charged groups (atoms with non-zero formal charge).
    """
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return True
    return False


def get_num_carbons(mol: Mol) -> int:
    """
    This function returns the number of carbon atoms in a molecule.
    """
    c_pattern = Chem.MolFromSmarts("[#6]")
    return len(mol.GetSubstructMatches(c_pattern))


def check_resistivity(mol: Mol) -> int:
    """
    This function checks if a molecule contains certain polar groups to assess if a solvent is likely to accumulate
    static electricity that could cause a spark and be an ignition point. Return 1 if the molecule is likely to
    accumulate electrostatic electricity and 0 if otherwise.
    """
    if mol is None:
        return 1
    elif has_charged_group(mol):
        return 0
    else:
        functional_groups = {
            "alcohol": Fragments.fr_Al_OH(mol),
            "aldehyde": Fragments.fr_aldehyde(mol),
            "ketone": Fragments.fr_ketone(mol),
            "acid": Fragments.fr_COO2(mol),
            "ester": Fragments.fr_ester(mol),
            "nitrile": Fragments.fr_nitrile(mol),
            "amide": Fragments.fr_amide(mol),
            "tertiary amine": Fragments.fr_NH0(mol),
            "secondary amine": Fragments.fr_NH1(mol),
            "Primary amine": Fragments.fr_NH2(mol),
            "aromatic amine": Fragments.fr_Ar_NH(mol),
            "nitro": Fragments.fr_nitro(mol)
        }

        # add a delta to avoid diving by 0 if there are no carbon
        num_carbon = float(get_num_carbons(mol)) + 0.00001
        num_acid_groups = functional_groups["acid"]
        num_ester_groups = functional_groups["ester"]
        fg_count = sum(functional_groups.values())

        if num_acid_groups == fg_count and num_acid_groups / num_carbon <= 1 / 3:
            return 1

        # This ratio of ester groups per carbon atoms is deducted from the paper
        elif num_ester_groups == fg_count and num_ester_groups / num_carbon <= 1 / 9:
            return 1

        elif any(count > 0 for count in functional_groups.values()):
            return 0
        else:
            return 1


def batch_resistivity_check(smiles_list: List[str]) -> List[int]:
    """
    This function performs a resistivity check for a list of SMILES identifiers.
    """

    scores = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        score = check_resistivity(mol)
        scores.append(score)

    return scores


if __name__ == '__main__':
    # Example usage
    input_smiles = ["Cc1ccoc1", "CCO", "CC(=O)C", "CC(=O)O", "C#N", "C(=O)O", "CCC",
                    "c1ccc(cc1)[N+](=O)[O-]", "CCC(=O)O", "CCCCOC(=O)CCCCCCCCC(=O)OCCCC", "[NH4+]"]
    resistivity_score = batch_resistivity_check(input_smiles)
    print(resistivity_score)
