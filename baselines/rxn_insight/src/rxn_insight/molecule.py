"""
Module for handling molecular structures and properties in Rxn-INSIGHT.

This module provides the Molecule class for working with chemical structures,
analyzing their properties, retrieving information from PubChem, and searching
for reactions involving these molecules in databases.

Dependencies:
    - rdkit: Cheminformatics toolkit
    - numpy: Numerical operations
    - requests: HTTP requests for PubChem API
    - pandas: Data manipulation
"""

from tqdm import tqdm
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity
from rxn_insight.utils import *
# from rxn_insight.greenchem.ehs import fetch_single_cid
# from rxn_insight.greenchem.peroxide import check_peroxide_formation
# from rxn_insight.greenchem.resistivity import check_resistivity
import requests, json
import logging
logger = logging.getLogger(__name__)


class Molecule:
    """
    Class for handling and analyzing molecular structures.

    This class provides methods to work with chemical structures represented as SMILES,
    retrieve information from PubChem, calculate molecular similarities, identify
    functional groups and ring structures, and search for reactions involving the molecule.

    Attributes:
        mol: RDKit molecule object
        smiles (str): SMILES representation
        inchi (str): InChI identifier
        inchikey (str): InChIKey identifier
        trivial_name (str): Common name from PubChem (if available)
        iupac_name (str): IUPAC name from PubChem (if available)
        description (str): Description from PubChem (if available)
        cid (str): PubChem compound ID (if available)
        functional_groups: List of functional groups in the molecule
        rings: Ring structures in the molecule
        scaffold (str): Murcko scaffold of the molecule
        maccs_fp: MACCS fingerprint
        morgan_fp: Morgan fingerprint
        reactions: DataFrame of reactions involving this molecule

    Example:
        >>> import rxn_insight as ri
        >>> # Create a basic molecule
        >>> mol = ri.Molecule("c1ccccc1")
        >>> print(mol.smiles)
        "c1ccccc1"
        >>> # Create with PubChem data
        >>> mol_with_data = ri.Molecule("c1ccccc1", allow_pubchem=True)
        >>> print(mol_with_data.iupac_name)
        "benzene"
    """

    def __init__(self, smi: str, allow_pubchem: bool = False):
        """
        Initializes a Molecule object with the SMILES string of a molecule.

        Args:
            smi (str): A string containing the SMILES representation of the molecule.
            allow_pubchem (bool): Whether to fetch additional information from PubChem.
                Default is False to avoid unnecessary API calls.
        """
        self.mol = Chem.MolFromSmiles(smi)
        self.smiles = Chem.MolToSmiles(self.mol)
        self.inchi = Chem.MolToInchi(self.mol)
        self.inchikey = Chem.MolToInchiKey(self.mol)
        self.trivial_name = ""
        self.iupac_name = ""
        self.description = ""
        self.cid = ""
        self.ghs_status = "NA"
        self.ghs_labels = []
        self.ghs_pictograms = []
        self.signal = None
        self.hazard_statements = []
        self.precautionary_codes = []
        self.echa_summary = None
        self.peroxidizable = None
        self.accumulates_static_electricity = None
        self.functional_groups = None
        self.rings = tuple()
        self.scaffold = get_scaffold(self.mol)
        self.maccs_fp = maccs_fp(self.mol)
        self.morgan_fp = morgan_fp(self.mol)
        self.reactions = None

        if allow_pubchem:
            self.get_pubchem_information()

    def get_pubchem_information(self):
        """
        Retrieves information about the molecule from PubChem's REST API.

        This method fetches the IUPAC name, common name, description, and
        PubChem compound ID (CID) for the molecule. Results are stored in
        the corresponding attributes of the Molecule object.

        Note:
            This method requires an internet connection and might be slow
            for large or complex molecules. The PubChem servers may also
            have rate limits or be temporarily unavailable.
        """
        iupac_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{self.smiles}/property/IUPACName/txt"
        response = requests.get(iupac_url)
        if response.status_code == 200:
            self.iupac_name = response.text[:-1]
        info_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{self.smiles}/description/JSON"
        response = requests.get(info_url)
        if response.status_code != 200:
            logger.warning(f"Error fetching from PubChem: Status code {response.status_code}")
        else:
            data = response.json()
            infos = data['InformationList']['Information']
            for info in infos:
                if "CID" in info:
                    self.cid = info['CID']
                if 'Title' in info:
                    self.trivial_name = info['Title']
                if 'Description' in info:
                    self.description += info['Description']
                    self.description += " "

    # def get_ghs_labels(self):
    #     found_cid = False
    #     if self.cid == "":
    #         self.get_pubchem_information()
    #         if self.cid == "":
    #             found_cid = False
    #             logger.warning(f"Could not retrieve PubChem CID for {self.smiles}!")
    #         else:
    #             found_cid = True
    #     else:
    #         found_cid = True
    #
    #     if found_cid:
    #         ghs_results = fetch_single_cid(self.cid)
    #         if ghs_results['status'] == 'success':
    #             self.ghs_status = "success"
    #             self.hazard_statements = ghs_results['hazard_statements']
    #             self.ghs_pictograms = ghs_results['ghs_pictograms']
    #             self.signal = ghs_results['signal']
    #             self.precautionary_codes = ghs_results['precautionary_codes']
    #             self.echa_summary = ghs_results['echa_summary']
    #         else:
    #             self.ghs_status = "failed"
    #
    # def get_peroxide_info(self):
    #     self.peroxidizable = check_peroxide_formation(self.mol)
    #     return self.peroxidizable
    #
    # def get_resistivity_info(self):
    #     self.accumulates_static_electricity = check_resistivity(self.mol)
    #     return self.accumulates_static_electricity

    def calculate_similarity(self, smi: str) -> float:
        """
        Calculates the chemical similarity between this molecule and another one.

        This method uses Morgan fingerprints (ECFP4) and the Tanimoto coefficient
        to measure the structural similarity between molecules.

        Args:
            smi (str): SMILES string of the molecule to compare with.

        Returns:
            float: Tanimoto similarity value (0-1, where 1 is identical).

        Example:
            >>> import rxn_insight as ri
            >>> mol = ri.Molecule("c1ccccc1")  # Benzene
            >>> mol.calculate_similarity("c1ccccc1C")  # Toluene
            0.778
        """
        fpgen = GetMorganGenerator(radius=2, fpSize=2048)
        source_molecule = fpgen.GetFingerprint(self.mol)
        target_molecule = fpgen.GetFingerprint(Chem.MolFromSmiles(smi))
        sim = TanimotoSimilarity(source_molecule, target_molecule)
        return sim

    def search_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Searches for reactions involving the molecule as a product.

        This method looks through a reaction database to find reactions
        where this molecule appears as a product, using the InChIKey
        for exact matching.

        Args:
            df (pd.DataFrame): The DataFrame to search for reactions.

        Returns:
            pd.DataFrame: DataFrame containing matching reactions.

        Note:
            If the input DataFrame doesn't already have a 'PRODUCT' column
            with InChIKeys, one will be generated (which may take time for
            large datasets).
        """
        if "PRODUCT" in df.keys():
            dfc = df[df["PRODUCT"] == self.inchikey].copy()
        else:
            df["PRODUCT"] = ""
            for i in tqdm(df.index):
                try:
                    prod = df["REACTION"][i].split(">>")[1]
                    mol = Chem.MolFromSmiles(prod)
                    df["PRODUCT"][i] = Chem.MolToInchiKey(mol)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.warning(e)
                    continue
            dfc = df[df["PRODUCT"] == self.inchikey].copy()

        if "rxn_str_patt_fp" in dfc.keys():
            dfc = dfc.drop(columns=["rxn_str_patt_fp"])
        if "rxn_dif_patt_fp" in dfc.keys():
            dfc = dfc.drop(columns=["rxn_dif_patt_fp"])
        if "rxn_str_morgan_fp" in dfc.keys():
            dfc = dfc.drop(columns=["rxn_str_morgan_fp"])
        if "rxn_dif_morgan_fp" in dfc.keys():
            dfc = dfc.drop(columns=["rxn_dif_morgan_fp"])
        if "TAG" in dfc.keys():
            dfc = dfc.drop(columns=["TAG"])
        if "TAG2" in dfc.keys():
            dfc = dfc.drop(columns=["TAG2"])

        self.reactions = dfc

        return dfc

    def search_reactions_by_scaffold(
            self,
            df: pd.DataFrame,
            threshold: float = 0.5,
            max_return: int = 100,
            fp: str = "MACCS",
    ) -> pd.DataFrame:
        """
        Searches for reactions based on scaffold similarity.

        This method first finds reactions with products sharing the same
        molecular scaffold, then ranks them by fingerprint similarity.

        Args:
            df (pd.DataFrame): DataFrame containing reactions to search.
            threshold (float): Similarity threshold to apply (0-1).
                Default is 0.5.
            max_return (int): Maximum number of reactions to return.
                Default is 100.
            fp (str): Type of fingerprint to use for similarity calculation.
                Options are 'MACCS' or 'Morgan'. Default is 'MACCS'.

        Returns:
            pd.DataFrame: DataFrame of similar reactions, sorted by similarity.

        Raises:
            KeyError: If an unsupported fingerprint type is specified.
        """
        dfc = df[df["SCAFFOLD"] == self.scaffold].copy()
        if len(dfc.index) == 0:
            logger.warning("No products with the same scaffold found!")
            return None

        if fp.lower() == "maccs":
            fps = [
                maccs_fp(Chem.MolFromSmiles(r.split(">>")[1]))
                for r in tqdm(dfc["REACTION"].tolist(), desc="Making fingerprints...")
            ]
            dfc["SIMILARITY"] = [get_similarity(self.maccs_fp, fp) for fp in fps]
        elif fp.lower() == "morgan":
            fps = [
                morgan_fp(Chem.MolFromSmiles(r.split(">>")[1]))
                for r in tqdm(dfc["REACTION"].tolist(), desc="Making fingerprints...")
            ]
            dfc["SIMILARITY"] = [get_similarity(self.morgan_fp, fp) for fp in fps]
        else:
            raise KeyError(
                f"Fingerprint choice {fp} is not supported. Select MACCS or Morgan."
            )

        df_tag = dfc.sort_values(by="SIMILARITY", ascending=False).copy()
        df_tag["SOLVENT"].fillna("", inplace=True)
        df_tag["CATALYST"].fillna("", inplace=True)
        df_tag["REAGENT"].fillna("", inplace=True)
        max_similarity = df_tag["SIMILARITY"].max()
        df_tag = df_tag[df_tag["SIMILARITY"] > threshold].copy()
        logger.warning(
            f"Product found with similarity of {max_similarity:.3f}. This will be our best match."
        )
        df_return = df_tag.iloc[:max_return].copy()

        if "rxn_str_patt_fp" in df_return.keys():
            df_return = df_return.drop(columns=["rxn_str_patt_fp"])
        if "rxn_dif_patt_fp" in df_return.keys():
            df_return = df_return.drop(columns=["rxn_dif_patt_fp"])
        if "rxn_str_morgan_fp" in df_return.keys():
            df_return = df_return.drop(columns=["rxn_str_morgan_fp"])
        if "rxn_dif_morgan_fp" in df_return.keys():
            df_return = df_return.drop(columns=["rxn_dif_morgan_fp"])
        if "TAG" in df_return.keys():
            df_return = df_return.drop(columns=["TAG"])
        if "TAG2" in df_return.keys():
            df_return = df_return.drop(columns=["TAG2"])

        return df_return

    def get_functional_groups(self, df: pd.DataFrame = None) -> list[str]:
        """
        Identifies and returns the functional groups present in the molecule.

        This method analyzes the molecule structure to detect common functional
        groups using SMARTS patterns. The results are cached in the functional_groups
        attribute.

        Args:
            df (pd.DataFrame, optional): DataFrame containing functional group patterns.
                If None, the default patterns from the package will be loaded.

        Returns:
            list[str]: List of functional group names found in the molecule.
        """
        if df is None:
            from importlib import resources

            with resources.path(
                    f"{__package__}.data", "functional_groups.json"
            ) as path:
                df = pd.read_json(path, orient="records", lines=True)

        mol = self.mol
        atom_indices = np.array([atom.GetIdx() for atom in mol.GetAtoms()])
        fg = []
        visited_atoms: list[list[int]] = []
        for i in df.index:
            if len(np.in1d(visited_atoms, atom_indices)) != 0:
                if len(visited_atoms[np.in1d(visited_atoms, atom_indices)]) == len(
                        atom_indices
                ):
                    break
            sm = mol.GetSubstructMatches(Chem.MolFromSmarts(df["pattern"][i]))
            if len(sm) == 0:
                continue
            else:
                for m in sm:
                    matched_atoms = np.array(m)
                    if len(matched_atoms[np.in1d(matched_atoms, atom_indices)]) > 0:
                        if len(np.in1d(visited_atoms, matched_atoms)) == 0:
                            fg.append(df["name"][i])
                            visited_atoms = np.unique(
                                np.append(visited_atoms, matched_atoms)
                            )
                        elif len(
                                visited_atoms[np.in1d(visited_atoms, matched_atoms)]
                        ) != len(matched_atoms):
                            fg.append(df["name"][i])
                            visited_atoms = np.unique(
                                np.append(visited_atoms, matched_atoms)
                            )
                        else:
                            continue
                    else:
                        continue
        self.functional_groups = fg
        return fg

    def get_rings(self) -> list[str]:
        """
        Identifies and returns rings in the molecule.

        This method detects ring systems in the molecule and returns their
        SMILES representations. The results are cached in the rings attribute.

        Returns:
            list[str]: List of ring SMILES strings found in the molecule.

        Note:
            Returns an empty list if no rings are found or if an error occurs
            during ring detection.
        """
        mol = self.mol
        try:
            rs = get_ring_systems(mol, include_spiro=True)
        except:
            self.rings = []
            return []

        found_rings = []
        if len(rs) > 0:
            for k in range(len(rs)):
                found_rings.append(sanitize_ring(atom_remover(mol, [rs[k]])))

        self.rings = found_rings
        return found_rings

    def draw(self, size=(200, 200), filename: str = ""):
        if self.mol is not None:
            if filename != "":
                Chem.Draw.MolToFile(filename, size)
            else:
                return Chem.Draw.MolToImage(self.mol, size=size)


class Compound(Molecule):
    def __init__(self, name, allow_pubchem: bool = False, use_opsin: bool = True):
        self.name = name
        self.allow_pubchem = allow_pubchem
        self.use_opsin = use_opsin
        if self.use_opsin:
            self.read_from_opsin()
        else:
            self.read_from_pubchem()

    def read_from_pubchem(self):
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{self.name}/property/SMILES/txt"
        response = requests.get(smiles_url)
        if response.status_code == 200:
            try:
                smi = Chem.CanonSmiles(response.text[:-1])
                super().__init__(smi=smi, allow_pubchem=self.allow_pubchem)
            except:
                logger.warning("No valid SMILES found!")

    def read_from_opsin(self):
        smiles_url = f"https://opsin.ch.cam.ac.uk/opsin/{self.name}.json"
        reqdata = requests.get(smiles_url)
        jsondata = reqdata.json()
        del jsondata['cml']
        try:
            super().__init__(smi=jsondata['smiles'], allow_pubchem=self.allow_pubchem)
        except:
            logger.warning("No valid SMILES found!")
