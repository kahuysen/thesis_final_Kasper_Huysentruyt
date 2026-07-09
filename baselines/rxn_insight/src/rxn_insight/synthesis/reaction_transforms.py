import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import RDConfig
import os
import sys
from typing import Union, Dict, List, Any
import numpy as np
from numpy import typing as npt
from rxn_insight.utils import get_fp, get_similarity
import pandas as pd
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import logging
logger = logging.getLogger(__name__)


def find_substructure_matches(mol, smarts, rdkit_transformation):

    """
    Find substructure matches for a given molecule.

    Parameters:
        mol (rdkit.Chem.Mol): The target RDKit molecule.
        smarts (list): List of SMARTS strings.
        rdkit_transformation (list): List of RDKit molecules corresponding to the SMARTS strings.

    Returns:
        dict: A dictionary of matching SMARTS strings and RDKit molecules.
    """

    matches = {}
    for i in range(len(smarts)):
        if mol.HasSubstructMatch(rdkit_transformation[i]):
            matches[smarts[i]] = rdkit_transformation[i]

    return matches


def get_reactants(
        product: str,
        smirks: str,
        mol: Union[Mol, None] = None
) -> Union[None, Dict[str, Union[str, List[str], float]]]:

    """
    Get reactants for a given product and SMIRKS.

    Parameters:
        product (str): The product SMILES string.
        smirks (str): The SMIRKS string describing the reaction.
        mol (rdkit.Chem.Mol, optional): Precomputed RDKit molecule of the product. Defaults to None.

    Returns:
        dict or None: A dictionary representing the reaction graph, or None if no reactants are found.
    """

    if mol is None:
        target = Chem.MolFromSmiles(product)
    else:
        target = mol

    sa_product = sascorer.calculateScore(target)
    retro_template = smirks.split(">>")[1] + ">>" + smirks.split(">>")[0]
    rxn = AllChem.ReactionFromSmarts(retro_template)
    product_tuple = tuple([target])
    try:
        reactant_mols = rxn.RunReactants(product_tuple)
    except ValueError:
        return None

    if len(reactant_mols) == 0:
        return None

    reactants = []
    sa_reactants = []
    for mol in reactant_mols[0]:
        reactant_mol = Chem.RemoveHs(mol)
        sa_reactants.append(sascorer.calculateScore(reactant_mol))
        reactants.append(Chem.MolToSmiles(reactant_mol))

    sa_reactant_score = sum(sa_reactants) / len(sa_reactants)
    sa_diff = sa_product - sa_reactant_score

    reaction_smiles = ".".join(sorted(reactants)) + ">>" + product
    graph = {
        "reactants": reactants,
        "product": product,
        "smirks": [smirks],
        "reaction_smiles": reaction_smiles,
        "sa_difference": sa_diff
    }

    return graph


def find_reaction_neighbors(
        reaction: str,
        relevant_reactions: List[Dict[str, str]],
        threshold: float,
        fingerprint: str = "maccs",
        similarity_metric: str = "jaccard",
        reaction_fingerprint: bool = True,
) -> Union[None, Dict[str, Any]]:

    """
    Find neighbors for a given reaction based on similarity.

    Parameters:
        reaction (str): Reaction SMILES string.
        relevant_reactions (list): List of reaction dictionaries.
        threshold (float): Minimum similarity threshold.
        fingerprint (str): Fingerprint type ("maccs" or "morgan"). Defaults to "maccs".
        similarity_metric (str): Metric for similarity calculation. Defaults to "jaccard".
        reaction_fingerprint (bool): Use concatenated reaction fingerprints if True. Defaults to True.

    Returns:
        dict or None: Dictionary with details of the closest reaction, or None if no match is found.
    """

    target_fp = get_fp(rxn=reaction, fp=fingerprint, concatenate=True)
    if not reaction_fingerprint:
        target_fp = target_fp[int(len(target_fp) / 2):]
    closest_doi = ""
    highest_similarity = -np.inf
    closest_reaction = ""
    reaction_class = ""
    reaction_subclass = ""
    reaction_type = ""
    for reaction_dict in relevant_reactions:
        if fingerprint.lower() == "maccs":
            fp = np.array(list(reaction_dict["maccs_fp"]), dtype=np.int64)
        elif fingerprint.lower() == "morgan":
            fp = np.array(list(reaction_dict["morgan_fp"]), dtype=np.int64)
        else:
            raise ValueError(f"Fingerprint choice {fingerprint} is not available. Choose maccs or morgan.")

        if not reaction_fingerprint:
            fp = fp[int(len(fp) / 2):]
        similarity = get_similarity(target_fp, fp, metric=similarity_metric)

        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_doi = reaction_dict["doi"]
            closest_reaction = reaction_dict["reaction_smiles"]
            reaction_class = reaction_dict["reaction_class"],
            reaction_subclass = reaction_dict["reaction_subclass"],
            reaction_type = reaction_dict["reaction_type"]

    if highest_similarity > threshold:
        return {
            "similarity": highest_similarity,
            "closest_reference": closest_doi,
            "closest_reaction": closest_reaction,
            "reaction_class": reaction_class,
            "reaction_subclass": reaction_subclass,
            "reaction_type": reaction_type
        }
    else:
        return None


def sort_related_reactions_by_distance(
        fp: npt.NDArray[int],
        relevant_reactions: List[Dict[str, str]],
        fingerprint_type: str = "morgan",
        reaction_fingerprint: bool = True,
) -> pd.DataFrame:

    """
    Sort reactions by similarity to a given fingerprint.

    Parameters:
        fp (numpy.ndarray): Target fingerprint array.
        relevant_reactions (list): List of reaction dictionaries.
        fingerprint_type (str): Type of fingerprint ("morgan" or "maccs"). Defaults to "morgan".
        reaction_fingerprint (bool): Use concatenated fingerprints if True. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame sorted by similarity.
    """

    analyzed_reactions = {
        "reaction_smiles": [],
        "doi": [],
        "title": [],
        "similarity": []
    }
    if not reaction_fingerprint:
        fp = fp[int(len(fp) / 2):]

    for reaction_dict in relevant_reactions:
        if fingerprint_type.lower() == "maccs":
            neighbor_fp = np.array(list(reaction_dict["maccs_fp"]), dtype=np.int64)
        elif fingerprint_type.lower() == "morgan":
            neighbor_fp = np.array(list(reaction_dict["morgan_fp"]), dtype=np.int64)
        else:
            raise ValueError(f"Fingerprint type {fingerprint_type} is not supported!")

        if not reaction_fingerprint:
            neighbor_fp = neighbor_fp[int(len(neighbor_fp) / 2):]

        similarity = get_similarity(fp, neighbor_fp)
        analyzed_reactions["reaction_smiles"].append(reaction_dict["reaction_smiles"])
        analyzed_reactions["doi"].append(reaction_dict["doi"])
        analyzed_reactions["title"].append(reaction_dict["title"])
        analyzed_reactions["similarity"].append(similarity)

    df = pd.DataFrame(analyzed_reactions).sort_values(by="similarity", ascending=False)
    return df


def find_forward_substructure_matches(
        mol: rdkit.Chem.Mol,
        smirks_dict: dict,
):

    """
    Find substructure matches in reactants of SMIRKS for a given molecule.

    Parameters:
        mol (rdkit.Chem.Mol): The target RDKit molecule.
        smirks_dict (dict): Dictionary of SMIRKS and corresponding SMARTS/RDKit transformations.

    Returns:
        dict: A dictionary of matching SMIRKS and their details.
    """

    smirks_list = list(smirks_dict.keys())
    matching_smirks = {}
    for smirks in smirks_list:
        rdkit_transformations = smirks_dict[smirks]["rd_smarts"]
        smarts_dict = {}
        found = False
        num_transformations = len(rdkit_transformations)
        if num_transformations > 3:
            continue
        for i in range(num_transformations):
            if mol.HasSubstructMatch(rdkit_transformations[i]):
                smarts_dict[smirks_dict[smirks]["smarts"][i]] = True
                found = True
            else:
                smarts_dict[smirks_dict[smirks]["smarts"][i]] = False
        if found:
            found_smirks = smirks_dict[smirks]
            found_smirks["matching"] = smarts_dict
            matching_smirks[smirks] = found_smirks

    return matching_smirks


def match_reactants(
        smiles: str,
        smarts_dict: dict,
        buyables: pd.DataFrame,
        price: int = 50,
):

    """
    Match potential reactants from a buyable compounds list.

    Parameters:
        smiles (str): Product SMILES string.
        smarts_dict (dict): SMARTS pattern dictionary.
        buyables (pandas.DataFrame): DataFrame of buyable compounds with "Molecule" and "Price" columns.
        price (int, optional): Maximum price for buyables. Defaults to 50.

    Returns:
        dict: Dictionary of potential reactants for each reactant slot.
    """

    buyable_selection = buyables[buyables["Price"] < price].copy()
    reactant_smarts = smarts_dict["matching"]
    reactant_dict = {}
    reactant_id = 1
    for smarts in reactant_smarts:
        if reactant_smarts[smarts]:
            reactant_dict[f"reactant_{reactant_id}"] = [smiles]
        else:
            reactant_dict[f"reactant_{reactant_id}"] = []
            idx = reactant_id - 1
            rd_smarts = smarts_dict['rd_smarts'][idx]
            for i in buyable_selection.index:
                smi = buyable_selection["Molecule"][i]
                if "." in smi:
                    continue
                try:
                    m = Chem.MolFromSmiles(smi)
                except Exception as e:
                    print(e)
                    continue
                if m.HasSubstructMatch(rd_smarts):
                    reactant_dict[f"reactant_{reactant_id}"].append(smi)
            pass
        reactant_id += 1

    return reactant_dict


def predict_products(
        reactant_dict: dict,
        smirks: str,
):

    """
    Predict products from reactants and a SMIRKS reaction.

    Parameters:
        reactant_dict (dict): Dictionary of reactant SMILES strings.
        smirks (str): SMIRKS string describing the reaction.

    Returns:
        tuple: Lists of predicted product SMILES and corresponding reaction SMILES.
    """

    rxn = AllChem.ReactionFromSmarts(smirks)
    num_reactants = len(reactant_dict.keys())

    if num_reactants == 1:
        m = Chem.MolFromSmiles(reactant_dict['reactant_1'][0])
        reactant_tuple = tuple([m])
        product = rxn.RunReactants(reactant_tuple)
        if len(product) == 0:
            return None
        else:
            mol = Chem.RemoveHs(product[0][0])
            return [Chem.MolToSmiles(mol)], [f"{reactant_dict['reactant_1']}>>{Chem.MolToSmiles(mol)}"]

    elif num_reactants == 2:
        products = []
        reactions = []
        for r1 in reactant_dict['reactant_1']:
            m1 = Chem.MolFromSmiles(r1)
            for r2 in reactant_dict['reactant_2']:
                m2 = Chem.MolFromSmiles(r2)
                reactant_tuple = tuple([m1, m2])
                product = rxn.RunReactants(reactant_tuple)
                if len(product) == 0:
                    continue
                else:
                    try:
                        product_mol = Chem.RemoveHs(product[0][0])
                        product_smi = Chem.MolToSmiles(product_mol)
                        products.append(product_smi)
                        reactions.append(f"{r1}.{r2}>>{product_smi}")
                    except KeyboardInterrupt:
                        raise
                    except:
                        continue
        return products, reactions

    elif num_reactants == 3:
        products = []
        reactions = []
        for r1 in reactant_dict['reactant_1']:
            m1 = Chem.MolFromSmiles(r1)
            for r2 in reactant_dict['reactant_2']:
                m2 = Chem.MolFromSmiles(r2)
                for r3 in reactant_dict['reactant_3']:
                    m3 = Chem.MolFromSmiles(r3)
                    reactant_tuple = tuple([m1, m2, m3])
                    product = rxn.RunReactants(reactant_tuple)
                    if len(product) == 0:
                        continue
                    else:
                        try:
                            for product_set in product:
                                product_mol = Chem.RemoveHs(product_set[0])
                                product_smi = Chem.MolToSmiles(product_mol)
                                products.append(product_smi)
                                reactions.append(f"{r1}.{r2}.{r3}>>{product_smi}")
                        except KeyboardInterrupt:
                            raise
                        except:
                            continue
        return products, reactions

    else:
        logging.warning(f"Cannot handle {num_reactants} reactants (max 3)")
        return [], []
