from importlib import resources
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import RDConfig, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from typing import Union, List
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


class DrugTarget:

    def __init__(self, mol: Chem.rdchem.Mol):
        self.molecule = mol
        self.molecule = self.get_molecule()
        self.mw = self.get_mw()
        self.num_h_donors = self.get_h_donors()
        self.num_h_acceptors = self.get_h_acceptors()
        self.clogp = self.get_clogp()
        self.tpsa = self.get_tpsa()
        self.num_rings = self.get_num_rings()
        self.num_atoms = self.get_num_atoms()
        self.num_carbons = self.get_num_carbons()
        self.num_heteroatoms = self.get_num_heteroatoms()
        self.num_rotatable_bonds = self.get_num_rotatable_bonds()
        self.refractivity = self.get_refractivity()
        self.inchikey = Chem.MolToInchiKey(self.molecule)
        self.sa_score = self.get_sa_score()
        self.qed = QED.qed(self.molecule)
        self.brenk = self.get_brenk()
        self.pains = self.get_pains()

    def get_molecule(self):
        """
        This function returns an RDKit Mol object with hydrogens.
        """
        return Chem.AddHs(self.molecule)

    def get_mw(self):
        """
        This function returns the molecular weight of the molecule.
        """
        return Descriptors.MolWt(self.molecule)

    def get_h_donors(self):
        """
        This function returns the number of hydrogen bond donors.
        """
        return Descriptors.NumHDonors(self.molecule)

    def get_h_acceptors(self):
        """
        This function returns the number of hydrogen bond acceptors.
        """
        return Descriptors.NumHAcceptors(self.molecule)

    def get_clogp(self):
        """
        This function returns the octanol-water partitioning coefficient.
        """
        return Descriptors.MolLogP(self.molecule)

    def get_tpsa(self):
        """
        This function returns the Topological Polar Surface Area.
        """
        return Descriptors.TPSA(self.molecule)

    def get_num_carbons(self):
        """
        This function returns the number of carbon atoms.
        """
        c_pattern = Chem.MolFromSmarts("[#6]")
        return len(self.molecule.GetSubstructMatches(c_pattern))

    def get_num_heteroatoms(self):
        """
        This function returns the number of heteroatoms.
        """
        heavy_atoms = self.molecule.GetNumHeavyAtoms()
        num_hetero = heavy_atoms - self.num_carbons
        return num_hetero

    def get_num_atoms(self):
        """
        This function returns the number of atoms.
        """
        num_atoms = self.molecule.GetNumAtoms()
        return num_atoms

    def get_num_rings(self):
        """
        This function returns the number of rings.
        """
        ri = self.molecule.GetRingInfo()
        atom_rings = ri.AtomRings()

        return len(atom_rings)

    def get_num_rotatable_bonds(self):
        """
        This function returns the number of rotatable bonds.
        """
        return Descriptors.NumRotatableBonds(self.molecule)

    def get_refractivity(self):
        """
        This function returns the molar refractivity according to Crippen's method.
        """
        return Descriptors.MolMR(self.molecule)

    def get_sa_score(self) -> float:
        return sascorer.calculateScore(self.molecule)

    def get_brenk(self) -> int:
        """
        This function returns if there are any unwanted substructures according to the Brenk filter. 0 is good.
        """
        params_unwanted = FilterCatalogParams()
        params_unwanted.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog_unwanted = FilterCatalog(params_unwanted)
        flag = catalog_unwanted.HasMatch(self.molecule)
        return int(flag)

    def get_pains(self) -> int:
        """
        This function returns if there are any unwanted substructures according to the Brenk filter. 0 is good.
        """
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        catalog_pains = FilterCatalog(params_pains)
        flag = catalog_pains.HasMatch(self.molecule)
        return int(flag)


class Bioavailability:
    def __init__(self, df_result: pd.DataFrame):
        self.df_result = df_result
        self.df_result = self.get_all_properties()
        self.df_result = self.filter_compounds()

    def get_all_properties(self) -> pd.DataFrame:
        bad_rows = []
        for i in tqdm(self.df_result.index):
            m = self.df_result["mol"][i]
            try:
                mol = DrugTarget(m)
                self.df_result.loc[i, "qed"] = mol.qed
                self.df_result.loc[i, "mw"] = mol.mw
                self.df_result.loc[i, "tpsa"] = mol.tpsa
                self.df_result.loc[i, "hbond_donors"] = mol.num_h_donors
                self.df_result.loc[i, "hbond_acceptors"] = mol.num_h_acceptors
                self.df_result.loc[i, "clogp"] = mol.clogp
                self.df_result.loc[i, "num_rings"] = mol.num_rings
                self.df_result.loc[i, "num_atoms"] = mol.num_atoms
                self.df_result.loc[i, "num_carbons"] = mol.num_carbons
                self.df_result.loc[i, "num_heteroatoms"] = mol.num_heteroatoms
                self.df_result.loc[i, "num_rotatable_bonds"] = mol.num_rotatable_bonds
                self.df_result.loc[i, "refractivity"] = mol.refractivity
                self.df_result.loc[i, "brenk"] = mol.brenk
                self.df_result.loc[i, "pains"] = mol.pains
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print("Error!", self.df_result['smiles'][i], e)
                bad_rows.append(i)
                continue

        self.df_result = self.df_result.drop(bad_rows)

        return self.df_result

    def filter_compounds(self) -> pd.DataFrame:
        properties = self.df_result.copy()

        for i in tqdm(properties.index):
            properties.loc[i, "lipinski"] = calculate_lipinski(num_h_donors=properties["hbond_donors"][i],
                                                               num_h_acceptors=properties["hbond_acceptors"][i],
                                                               mw=properties["mw"][i],
                                                               logp=properties["logp_prediction"][i])
            properties.loc[i, "ghose"] = calculate_ghose(logp=properties["logp_prediction"][i],
                                                         refractivity=properties["refractivity"][i],
                                                         mw=properties["mw"][i],
                                                         num_atoms=properties["num_atoms"][i])
            properties.loc[i, "veber"] = calculate_veber(num_rotatable_bonds=properties["num_rotatable_bonds"][i],
                                                         tpsa=properties["tpsa"][i])
            properties.loc[i, "egan"] = calculate_egan(logp=properties["logp_prediction"][i],
                                                       tpsa=properties["tpsa"][i])
            properties.loc[i, "muegge"] = calculate_muegge(logp=properties["logp_prediction"][i],
                                                           num_rotatable_bonds=properties["num_rotatable_bonds"][i],
                                                           tpsa=properties["tpsa"][i],
                                                           num_h_donors=properties["hbond_donors"][i],
                                                           num_h_acceptors=properties["hbond_acceptors"][i],
                                                           mw=properties["mw"][i],
                                                           num_rings=properties["num_rings"][i],
                                                           num_carbons=properties["num_carbons"][i],
                                                           num_heteroatoms=properties["num_heteroatoms"][i])
            properties.loc[i, "cnsmpo"] = calculate_cnsmpo(logp=properties["logp_prediction"][i],
                                                           logd=properties["logd_prediction"][i],
                                                           mw=properties["mw"][i],
                                                           tpsa=properties["tpsa"][i],
                                                           num_h_donors=properties["hbond_donors"][i],
                                                           pka=999.0)

        return properties


def calculate_lipinski(num_h_donors, num_h_acceptors, mw, logp,
                       max_h_donors=5, max_h_acceptors=10, max_mw=500, max_logp=5):
    """
    This function checks whether Lipinski's rule of five is obeyed or not. Returns number of violations.
    """

    score = 0

    if num_h_donors > max_h_donors:
        score += 1

    if num_h_acceptors > max_h_acceptors:
        score += 1

    if mw > max_mw:
        score += 1

    if logp > max_logp:
        score += 1

    return score


def calculate_ghose(logp: float, refractivity: float, mw: float, num_atoms: float,
                    min_logp: float = -0.4, max_logp: float = 5.6,
                    min_refractivity: float = 40, max_refractivity: float = 130,
                    min_mw: float = 160, max_mw: float = 400,
                    min_num_atoms: float = 20, max_num_atoms: float = 70) -> int:
    """
    This function checks whether Ghose's filter is obeyed or not. Returns number of violations.
    """

    score = 0

    if not min_logp <= logp <= max_logp:
        score += 1

    if not min_refractivity <= refractivity <= max_refractivity:
        score += 1

    if not min_mw <= mw <= max_mw:
        score += 1

    if not min_num_atoms <= num_atoms <= max_num_atoms:
        score += 1

    return score


def calculate_egan(logp: float, tpsa: float,
                   max_logp: float = 5.88, max_tpsa: float = 131.6) -> int:
    """
    This function checks whether Egan's filter is obeyed or not. Returns number of violations.
    """

    score = 0

    if logp > max_logp:
        score += 1

    if tpsa > max_tpsa:
        score += 1

    return score


def calculate_veber(num_rotatable_bonds: float, tpsa: float,
                    max_num_rotatable_bonds: int = 10, max_tpsa: float = 140.0) -> int:
    """
    This function checks whether Veber's filter is obeyed or not. Returns number of violations.
    """

    score = 0

    if num_rotatable_bonds > max_num_rotatable_bonds:
        score += 1

    if tpsa > max_tpsa:
        score += 1

    return score


def calculate_muegge(logp: float, num_rotatable_bonds: int, tpsa: float, num_h_donors: int, num_h_acceptors: int,
                     mw: float, num_rings: int, num_carbons: int, num_heteroatoms: int,
                     min_logp: float = -2.0, max_logp: float = 5.0,
                     max_num_rotatable_bonds: int = 15, max_tpsa: float = 150.0,
                     max_num_h_donors: int = 5, max_num_h_acceptors: int = 10,
                     min_mw: float = 200.0, max_mw: float = 400.0, max_num_rings: int = 7,
                     min_num_carbons: int = 4, min_num_heteroatoms: int = 1) -> int:
    """
    This function checks whether Muegge's filter is obeyed or not. Returns number of violations.
    """
    score = 0

    if num_rotatable_bonds > max_num_rotatable_bonds:
        score += 1

    if tpsa > max_tpsa:
        score += 1

    if not min_logp <= logp <= max_logp:
        score += 1

    if num_h_donors > max_num_h_donors:
        score += 1

    if num_h_acceptors > max_num_h_acceptors:
        score += 1

    if not min_mw <= mw <= max_mw:
        score += 1

    if num_rings > max_num_rings:
        score += 1

    if num_carbons <= min_num_carbons:
        score += 1

    if num_heteroatoms <= min_num_heteroatoms:
        score += 1

    return score


def calculate_cnsmpo(logp: float, logd: float, mw: float, tpsa: float, num_h_donors: int, pka: float) -> float:
    """
    This function calculates the CNS-MPO score: https://doi.org/10.1021/cn100007x.
    """

    score = 0.0

    if logp < 3:
        score += 1
    elif logp > 5:
        score += 0
    else:
        score += -0.5 * logp + 2.5

    if logd < 2:
        score += 1
    elif logd > 4:
        score += 0
    else:
        score += -0.5 * logd + 2.0

    if mw < 360:
        score += 1
    elif mw > 500:
        score += 0
    else:
        score += -(1.0 / 140.0) * mw + (360.0 / 140.0)

    if tpsa < 20:
        score += 0
    elif 20 <= tpsa < 40:
        score += 0.05 * tpsa - 1.0
    elif 40 <= tpsa < 90:
        score += 1
    elif 90 <= tpsa < 120:
        score += (-1.0 / 30.0) * tpsa + 4.0
    else:
        score += 0

    if num_h_donors == 0:
        score += 1
    elif num_h_donors == 1:
        score += 0.75
    elif num_h_donors == 2:
        score += 0.5
    elif num_h_donors == 3:
        score += 0.25
    else:
        score += 0

    if pka < 8:
        score += 1
    elif pka > 10:
        score += 0
    else:
        score += -0.5 * pka + 5

    return score
