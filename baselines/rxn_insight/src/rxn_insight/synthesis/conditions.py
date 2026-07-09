from typing import Union, Dict, List
import numpy as np
from numpy import typing as npt
from rxn_insight.utils import get_similarity, get_fp
import pandas as pd
from rxn_insight.synthesis.utils import score_solvent, load_chem21_data


def score_conditions(
        fp: npt.NDArray[int],
        relevant_reactions: List[Dict[str, str]],
        similarity_weight: float = 1.0,
        fingerprint_type: str = "morgan",
        metric: str = "jaccard",
        reaction_fingerprint: bool = True,
        similarity_threshold: float = 0.0,
) -> pd.DataFrame:

    """
    Scores chemical reaction conditions based on similarity to a reference fingerprint.

    Args:
        fp (npt.NDArray[int]): The reference fingerprint as a numpy array.
        relevant_reactions (List[Dict[str, str]]): A list of reaction data, where each entry is a dictionary containing reaction details.
        similarity_weight (float): Weight applied to similarity scores. Default is 1.0.
        fingerprint_type (str): Type of fingerprint used ("morgan" or "maccs"). Default is "morgan".
        metric (str): Similarity metric to use (e.g., "jaccard"). Default is "jaccard".
        reaction_fingerprint (bool): Whether to use reaction fingerprints (True) or product fingerprints (False). Default is True.
        similarity_threshold (float): Minimum similarity score to include a reaction. Default is 0.0.

    Returns:
        pd.DataFrame: A DataFrame with analyzed reaction conditions sorted by similarity score.
    """

    analyzed_reactions = {
        "solvent": [],
        "solvent_smiles": [],
        "solvent_score": [],
        "catalyst": [],
        "reagent": [],
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
            raise ValueError(f"Fingerprint type {fp} is not supported!")

        if not reaction_fingerprint:
            neighbor_fp = neighbor_fp[int(len(neighbor_fp) / 2):]

        similarity = get_similarity(fp, neighbor_fp, metric=metric)

        if similarity < similarity_threshold:
            continue

        weighted_similarity = similarity * similarity_weight
        analyzed_reactions["solvent"].append(reaction_dict.get("solvent", "unknown"))
        analyzed_reactions["solvent_smiles"].append(reaction_dict.get("solvent_smiles", "unknown"))
        analyzed_reactions["solvent_score"].append(reaction_dict.get("solvent_score", "unknown"))
        analyzed_reactions["catalyst"].append(reaction_dict.get("catalyst", "unknown"))
        analyzed_reactions["reagent"].append(reaction_dict.get("reagent", "unknown"))
        analyzed_reactions["similarity"].append(weighted_similarity)

    df = pd.DataFrame(analyzed_reactions).sort_values(by="similarity", ascending=False)

    return df


def sort_conditions(
        conditions: pd.DataFrame,
        green_solvents: bool = False
) -> Dict[str, Union[str, float, Dict[str, Union[str, float]]]]:

    """
    Sorts reaction conditions by similarity and organizes results.

    Args:
        conditions (pd.DataFrame): DataFrame containing reaction conditions and similarity scores.
        green_solvents (bool): If True, prioritizes solvents marked as "Recommended". Default is False.

    Returns:
        Dict[str, Union[str, float, Dict[str, Union[str, float]]]]: Dictionary of sorted conditions by solvents, catalysts, and reagents.
    """

    all_weights = sum(conditions["similarity"].to_list())
    output_dict = {}
    if green_solvents:
        unique_solvents = list(set(conditions[conditions["solvent_score"] == "Recommended"]["solvent"].tolist()))
    else:
        unique_solvents = list(set(conditions["solvent"].tolist()))
    unique_catalysts = list(set(conditions["catalyst"].tolist()))
    unique_reagents = list(set(conditions["reagent"].tolist()))

    solvent_dict = {}
    for solvent in unique_solvents:
        if solvent == "unknown":
            continue
        solvent_idx = conditions[conditions["solvent"] == solvent].index[0]
        score = sum(conditions[conditions["solvent"] == solvent]["similarity"].tolist()) / all_weights
        solvent_dict[solvent] = {"smiles": conditions["solvent_smiles"][solvent_idx],
                                 "hazard_level": conditions["solvent_score"][solvent_idx],
                                 "similarity": score}
        if solvent == "solvent-free":
            solvent_dict[solvent]["hazard_level"] = "Recommended"

    sorted_solvent_dict = dict(sorted(solvent_dict.items(), key=lambda item: item[1]["similarity"], reverse=True))
    output_dict["solvent"] = sorted_solvent_dict

    catalyst_dict = {}
    for catalyst in unique_catalysts:
        if catalyst == "unknown":
            continue
        score = sum(conditions[conditions["catalyst"] == catalyst]["similarity"].tolist()) / all_weights
        catalyst_dict[catalyst] = score
    sorted_catalyst_dict = dict(sorted(catalyst_dict.items(), key=lambda item: item[1], reverse=True))
    output_dict["catalyst"] = sorted_catalyst_dict

    reagent_dict = {}
    for reagent in unique_reagents:
        if reagent == "unknown":
            continue
        score = sum(conditions[conditions["reagent"] == reagent]["similarity"].tolist()) / all_weights
        reagent_dict[reagent] = score
    sorted_reagent_dict = dict(sorted(reagent_dict.items(), key=lambda item: item[1], reverse=True))
    output_dict["reagent"] = sorted_reagent_dict

    return output_dict


def find_uspto_conditions(
        database: pd.DataFrame,
        smirks: str,
        chem21_data: Union[dict, None] = None,
        fingerprint_type: str = "morgan",
):
    """
    Retrieves reaction conditions for a given SMIRKS pattern from a database.

    Args:
        database (pd.DataFrame): DataFrame containing reaction data.
        smirks (str): SMIRKS pattern to search for.
        chem21_data (Union[dict, None]): Preloaded CHEM21 data for scoring solvents. If None, it will be loaded during execution.
        fingerprint_type (str): Type of fingerprint used for reaction analysis. Default is "morgan".

    Returns:
        List[Dict]: A list of dictionaries with analyzed reaction conditions.
    """

    df_smirks = database[database['TEMPLATE'] == smirks].copy().fillna('')
    results = organize_batch_conditions(df_smirks, chem21_data, fingerprint_type=fingerprint_type)

    return results


def organize_batch_conditions(
        database: pd.DataFrame,
        chem21_data: Union[dict, None] = None,
        fingerprint_type: str = "morgan",
):
    """
    Organizes reaction conditions from a database into a batch structure with solvent scoring.

    Args:
        database (pd.DataFrame): DataFrame containing reaction data with SMILES strings and conditions.
        chem21_data (Union[dict, None]): Preloaded CHEM21 data for scoring solvents. If None, it will be loaded during execution.
        fingerprint_type (str): Type of fingerprint used for reaction analysis. Default is "morgan".

    Returns:
        List[Dict]: A list of dictionaries containing organized reaction data.
    """

    if not chem21_data:
        chem21_data = load_chem21_data()

    results = []

    for i in database.index:
        rxn_smiles = database['REACTION'][i]
        uspto_solvent = database['SOLVENT'][i]
        solvent_scoring_dict = score_solvent(chem21_data=chem21_data, smiles=uspto_solvent)

        results_dict = {
            f'{fingerprint_type}_fp': ''.join(
                get_fp(rxn=rxn_smiles, fp=fingerprint_type, concatenate=True).astype(str)
            ),
            'solvent': solvent_scoring_dict['solvent_name'],
            'solvent_smiles': solvent_scoring_dict['solvent_smiles'],
            'solvent_score': solvent_scoring_dict['solvent_score'],
            'catalyst': database['CATALYST'][i],
            'reagent': database['REAGENT'][i]
        }

        results.append(results_dict)

    return results
