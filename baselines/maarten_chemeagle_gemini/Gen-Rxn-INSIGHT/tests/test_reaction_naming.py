from gen_rxn_insight.naming import name_reactions_batch
from gen_rxn_insight.reaction import Reaction


def test_protection():
    """
    Checks whether the protection reaction is named correctly.
    """
    rxn_smiles = "N[C@H](CO)Cc1ccccc1.CC(C)(C)OC(=O)OC(=O)O>>CC(C)(C)OC(=O)N[C@H](CO)Cc1ccccc1"
    rxn = Reaction(rxn_smiles)
    rxn_info = rxn.get_reaction_info()

    assert rxn_info["NAME"] == 'Boc amine protection of primary amine'


def test_acylation():
    """
    Checks whether the protection reaction is named correctly.
    """
    rxn_smiles = "O=C(Cl)CCl.CN>>CNC(=O)CCl"
    rxn = Reaction(rxn_smiles)
    rxn_info = rxn.get_reaction_info()

    assert rxn_info["CLASS"] == 'Acylation'


def test_other_reaction():
    """
    Checks whether the protection reaction is named correctly.
    """
    rxn_smiles = "CC>>OCCCCCO"
    rxn = Reaction(rxn_smiles)
    rxn_info = rxn.get_reaction_info()

    assert rxn_info["NAME"] == 'OtherReaction'


def test_name_reactions_batch():
    """Checks that name_reactions_batch returns correct names for a mixed list."""
    reactions = [
        "CC(=O)Cl.CCO>>CC(=O)OCC",                            # ester acylation
        "CC>>OCCCCCO",                                          # no match
        "N[C@H](CO)Cc1ccccc1.CC(C)(C)OC(=O)OC(=O)O"
        ">>CC(C)(C)OC(=O)N[C@H](CO)Cc1ccccc1",                # Boc protection
    ]
    names = name_reactions_batch(reactions, n_jobs=1)

    assert len(names) == len(reactions)
    assert names[0] == "Schotten-Baumann to ester"
    assert names[1] == "OtherReaction"
    assert names[2] == "Boc amine protection of primary amine"
