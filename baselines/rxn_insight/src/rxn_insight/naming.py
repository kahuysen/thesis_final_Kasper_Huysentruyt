import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


def name_reaction(rxn: str, smirks_db: pd.DataFrame) -> str:
    """Determines the name of the reaction from a database based on SMIRKS transformations.

    Args:
        rxn (str): The name of the reaction in Reaction SMILES format.
        smirks_db (pd.DataFrame): DataFrame containing SMIRKS patterns and corresponding reaction names.

    Returns:
        str: The name of the reaction, or 'OtherReaction' if no specific name can be determined.
    """
    reactants_smiles, products_smiles = rxn.split(">>")
    reactants = reactants_smiles.split(".")
    products = products_smiles.split(".")

    if (
            len(reactants) > 4 or len(products) > 4
    ):  # There are no templates for reactions with more than four reactants.
        return "OtherReaction"

    new_products = []  # Try to canonicalize SMILES

    for product in products:
        try:
            new_products.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(product), isomericSmiles=False)
            )
        except:
            new_products.append(product)

    num_reactants = len(reactants)
    # num_products = len(products)

    rxn_name = ""
    selected_rxns = smirks_db[smirks_db["nreact"] == num_reactants]
    react_tuple = tuple(Chem.MolFromSmiles(reactant) for reactant in reactants)

    if num_reactants == 1:
        all_tuples = [react_tuple]
    else:
        all_tuples = list(
            itertools.permutations(react_tuple)
        )  # RDKit does not permute reactants by itself

    # TODO: Further refine reactions by superclass

    for i in selected_rxns.index:  # Iterate over all reactants to find a match
        smirks = selected_rxns["smirks"][i]
        rxn = AllChem.ReactionFromSmarts(smirks)
        pred_products = []

        for tup in all_tuples:
            try:
                pred_product = rxn.RunReactants(tup)
            except Exception:
                continue
            pred_products += pred_product

        if len(pred_products) == 0:  # No products are found
            continue
        else:
            for prods in pred_products:
                try:
                    prod = Chem.MolToSmiles(prods[0], isomericSmiles=False)
                except Exception:
                    continue

                if (
                        prod in new_products
                ):  # Predicted product is in the real reaction
                    rxn_name = selected_rxns["name"][i].strip("{}")
                    return rxn_name
                else:
                    continue

    if rxn_name == "":
        rxn_name = "OtherReaction"

    return rxn_name
