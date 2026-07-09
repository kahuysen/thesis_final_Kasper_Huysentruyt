"""
This module provides tools for analyzing chemical reactions and converting reaction datasets into knowledge graphs.
It includes:
- ReactionAnalyzer: A class for analyzing chemical reactions and finding relevant templates and conditions.
- ReactionDatabase: A class for managing reaction databases and converting them to knowledge graphs.
"""

import numpy as np
from rxn_insight.reaction import Reaction
from rxn_insight.database import Database
from rxn_insight.utils import get_reaction_template, get_fp, maccs_fp, morgan_fp
from rxnmapper import RXNMapper
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from typing import Union
from networkx.classes.digraph import DiGraph
import networkx as nx
from multiprocessing import Pool, cpu_count

from rxn_insight.synthesis.utils import (
    get_reaction_subcategories,
    load_chem21_data,
    score_solvent
)
from rxn_insight.synthesis.knowledge_graph import KnowledgeGraph
from rxn_insight.synthesis.reaction_transforms import (
    find_substructure_matches,
    get_reactants,
    sort_related_reactions_by_distance
)
from rxn_insight.synthesis.conditions import (
    score_conditions,
    sort_conditions,
    find_uspto_conditions,
    organize_batch_conditions
)


class ReactionAnalyzer(Reaction):

    """
    A class for analyzing chemical reactions and finding relevant templates, references, and conditions.

    Attributes:
        reaction (str): Reaction SMILES string.
        graph (KnowledgeGraph): The reaction knowledge graph for querying templates and related reactions.
        rxn_mapper (RXNMapper, optional): Mapper for obtaining atom-mapped reactions.
        template (str): Reaction template derived from the mapped reaction.
        fingerprint (list): Reaction fingerprint for similarity comparisons.
    """

    def __init__(self, reaction: str, graph: KnowledgeGraph, matching_templates=None,
                 radius_reactants: int = 2, radius_products: int = 2,
                 rxn_mapper: Union[RXNMapper, None] = None, smirks: Union[pd.DataFrame, None] = None,
                 fg: Union[pd.DataFrame, None] = None, fingerprint_type: str = "morgan",
                 metric: str = "jaccard", reaction_fingerprint: bool = True):

        """
        Initializes the ReactionAnalyzer with the provided reaction and knowledge graph.

        Args:
            reaction (str): Reaction SMILES string.
            graph (KnowledgeGraph): Knowledge graph for querying reaction data.
            matching_templates (list, optional): List of pre-matched templates. Defaults to None.
            radius_reactants (int): Radius for generating templates around reactants. Defaults to 2.
            radius_products (int): Radius for generating templates around products. Defaults to 2.
            rxn_mapper (RXNMapper, optional): Mapper for atom-mapped reactions. Defaults to None.
            fingerprint_type (str): Type of fingerprint to use. Defaults to "morgan".
            metric (str): Similarity metric for fingerprints. Defaults to "jaccard".
            reaction_fingerprint (bool): Whether to use reaction fingerprints. Defaults to True.
        """

        super().__init__(
            reaction=reaction, rxn_mapper=rxn_mapper, smirks=smirks, fg=fg, classify=True, search_template=True
        )

        self.graph = graph
        self.rxn_mapper = rxn_mapper
        self.reaction = reaction
        self.reactants, self.products = self.reaction.split(">>")
        self.reactants = self.reactants.split(".")
        self.products = self.products.split(".")
        self.num_reactants = len(self.reactants)
        self.num_products = len(self.products)
        self.template = get_reaction_template(
            self.mapped_reaction,
            radius_reactants=radius_reactants,
            radius_products=radius_products
        )
        self.fingerprint_type = fingerprint_type
        self.reaction_info = self.get_reaction_info()
        self.matching_templates = matching_templates
        self.fingerprint = get_fp(self.reaction, fp=self.fingerprint_type, concatenate=True)
        self.metric = metric
        self.reaction_fingerprint = reaction_fingerprint

    def find_matching_templates(self):

        """
        Finds reaction templates that match the products of the reaction.

        Returns:
            list: A list of SMIRKS templates matching the reaction.
        """

        available_smarts, available_rdkit_transformations = self.graph.get_all_product_smarts()
        mol = Chem.MolFromSmiles(self.products[0])
        templates = find_substructure_matches(
            mol,
            available_smarts,
            available_rdkit_transformations
        )
        canonicalized_reactants = [Chem.CanonSmiles(reactant) for reactant in self.reactants]
        canonicalized_products = [Chem.CanonSmiles(product) for product in self.products]
        canonicalized_reaction = f"{'.'.join(sorted(canonicalized_reactants))}>>" \
                                 f"{'.'.join(sorted(canonicalized_products))}"
        matching_smirks = []
        for template in tqdm(templates, desc="Looping over all matching SMARTS...", position=0):
            matches = self.graph.look_up_smirks(template)
            for match in matches:
                smirks = match.get('smirks')
                num_smirks_reactants = len(smirks.split(">>")[0].split("."))
                num_smirks_products = len(smirks.split(">>")[1].split("."))
                if num_smirks_reactants != self.num_reactants:
                    continue
                elif num_smirks_products != self.num_products:
                    continue
                reactant_dict = get_reactants(canonicalized_products[0], smirks, mol=mol)
                if reactant_dict["reaction_smiles"] == canonicalized_reaction:
                    matching_smirks.append(smirks)
                elif smirks == self.template:
                    matching_smirks.append(smirks)
                else:
                    continue

        return matching_smirks

    def find_references(self):

        """
        Finds references to related reactions based on matching templates.

        Returns:
            pd.DataFrame: A DataFrame of related reactions sorted by similarity.
        """

        if self.matching_templates is None:
            self.matching_templates = self.find_matching_templates()

        all_dfs = []
        for template in self.matching_templates:
            related_reactions = self.graph.find_related_reactions(template)
            df_related = sort_related_reactions_by_distance(
                self.fingerprint,
                related_reactions,
                fingerprint_type=self.fingerprint_type,
                reaction_fingerprint=self.reaction_fingerprint,
            )
            all_dfs.append(df_related)

        all_references = pd.concat(all_dfs).reset_index(drop=True).sort_values(by="similarity", ascending=False)
        return all_references

    def find_conditions_by_tag(
            self,
            green_solvents: bool = False,
            similarity_weight: float = 1.0,
            similarity_threshold: float = 0.0,
            broad_search: bool = False,
    ):
        """
        Finds reaction conditions based on the broad tag (TAG2) of the reaction.

        This method searches for reactions with the same broad tag (similar reaction class
        and functional groups) and analyzes their conditions.

        Args:
            broad_search (bool): Use TAG2 instead`of TAG. Defaults to False.
            green_solvents (bool): Whether to prioritize green solvents. Defaults to False.
            similarity_weight (float): Weight for similarity-based scoring. Defaults to 1.0.
            similarity_threshold (float): Minimum similarity score for considering reactions. Defaults to 0.0.

        Returns:
            dict: Suggested conditions including solvents, reagents, and catalysts.
        """
        if broad_search:
            broad_tag = self.give_broad_tag()
            related_reactions = self.graph.find_reactions_by_tag(tag=broad_tag, broad_search=True)
        else:
            related_reactions = self.graph.find_reactions_by_tag(tag=self.tag, broad_search=False)

        if not related_reactions:
            return {"solvent": {}, "reagent": {}, "catalyst": {}}

        # Score the conditions based on similarity
        df_conditions = score_conditions(
            fp=self.fingerprint,
            relevant_reactions=related_reactions,
            similarity_weight=similarity_weight,
            fingerprint_type=self.fingerprint_type,
            metric=self.metric,
            reaction_fingerprint=self.reaction_fingerprint,
            similarity_threshold=similarity_threshold,
        )

        # Sort and organize the conditions
        sorted_conditions = sort_conditions(df_conditions, green_solvents)

        # Create dataframes for different condition types
        solvent_df = create_solvent_df(sorted_conditions)
        reagent_df = pd.DataFrame({
            "smiles": sorted_conditions["reagent"].keys(),
            "score": sorted_conditions["reagent"].values()
        })
        catalyst_df = pd.DataFrame({
            "smiles": sorted_conditions["catalyst"].keys(),
            "score": sorted_conditions["catalyst"].values()
        })

        # Normalize scores
        if len(reagent_df.index) > 1:
            reagent_df["score"] = reagent_df["score"].to_numpy() / np.sum(reagent_df["score"].to_numpy())

        if len(catalyst_df.index) > 1:
            catalyst_df["score"] = catalyst_df["score"].to_numpy() / np.sum(catalyst_df["score"].to_numpy())

        self.suggested_solvent = solvent_df
        self.suggested_reagent = reagent_df
        self.suggested_catalyst = catalyst_df

        return sorted_conditions

    def find_conditions_by_template(
            self,
            green_solvents: bool = False,
            database: Union[pd.DataFrame, None] = None,
            chem21_data: Union[dict, None] = None,
            batch_weight: float = 0.5,
            enrich: bool = False,
            search_extra_templates: bool = False,
            enrichment_limit: int = 1000,
            enrichment_weight: float = 0.1,
            enrichment_threshold: float = 0.3,
            similarity_threshold: float = 0.0,
            similarity_weight: float = 1.0,
            full_database_search: bool = False,
    ):

        """
        Finds reaction conditions, including solvents, reagents, and catalysts, based on matching templates.

        Args:
            green_solvents (bool): Whether to prioritize green solvents. Defaults to False.
            database (pd.DataFrame, optional): Reaction database for additional conditions. Defaults to None.
            chem21_data (dict, optional): Data for scoring solvents. Defaults to None.
            batch_weight (float): Weight for scoring batch reactions. Defaults to 0.5.
            enrich (bool): Whether to enrich the dataset with neighboring templates. Defaults to False.
            search_extra_templates (bool): Whether to search for additional templates in the database. Defaults to False.
            enrichment_limit (int): Maximum number of neighboring templates to consider during enrichment. Defaults to 1000.
            enrichment_weight (float): Weight for enrichment-based scoring. Defaults to 0.1.
            enrichment_threshold (float): Threshold for determining neighbors during enrichment. Defaults to 0.3.
            similarity_threshold (float): Minimum similarity score for considering reactions. Defaults to 0.0.
            similarity_weight (float): Weight for similarity-based scoring. Defaults to 1.0.
            full_database_search (bool): Whether to search the entire database for conditions. Defaults to False.

        Returns:
            dict: Suggested conditions including solvents, reagents, and catalysts.
        """

        if self.matching_templates is None:
            self.matching_templates = self.find_matching_templates()

        all_dfs = []
        for template in self.matching_templates:
            related_reactions = self.graph.find_conditions_by_template(template)
            df_conditions = score_conditions(
                fp=self.fingerprint,
                relevant_reactions=related_reactions,
                similarity_weight=similarity_weight,
                fingerprint_type=self.fingerprint_type,
                metric=self.metric,
                reaction_fingerprint=self.reaction_fingerprint,
                similarity_threshold=0.0,
            )
            all_dfs.append(df_conditions)

        if full_database_search:
            similar_reactions = self.graph.find_similar_reactions(
                fp=self.fingerprint,
                fingerprint_type=self.fingerprint_type,
                similarity_threshold=similarity_threshold,
                reaction_fingerprint=self.reaction_fingerprint,
                metric=self.metric
            )
            df_conditions = score_conditions(
                fp=self.fingerprint,
                relevant_reactions=similar_reactions,
                similarity_weight=similarity_weight,
                fingerprint_type=self.fingerprint_type,
                metric=self.metric,
                reaction_fingerprint=self.reaction_fingerprint,
                similarity_threshold=similarity_threshold,
            )
            all_dfs.append(df_conditions)

        if database is not None:
            all_uspto_templates = list(set(database['TEMPLATE'].tolist()))
            all_batch_dfs = []
            if enrich:
                enrichment_database = self.find_neighbors(
                    df=database,
                    max_return=enrichment_limit,
                    threshold=enrichment_threshold.imag,
                    fp=self.fingerprint_type
                )

                if enrichment_database is not None:
                    for template in self.matching_templates:
                        enrichment_database = enrichment_database[enrichment_database['TEMPLATE'] != template].copy()

                    enriched_reactions = organize_batch_conditions(
                        database=enrichment_database,
                        chem21_data=chem21_data,
                        fingerprint_type=self.fingerprint_type
                    )
                    df_enriched_batch_conditions = score_conditions(
                        self.fingerprint,
                        enriched_reactions,
                        similarity_weight=enrichment_weight,
                        fingerprint_type=self.fingerprint_type,
                        metric=self.metric,
                        reaction_fingerprint=self.reaction_fingerprint,
                    )

                    all_dfs += [df_enriched_batch_conditions]

            if search_extra_templates:
                for template in self.matching_templates:
                    if template in all_uspto_templates:
                        related_uspto_reactions = find_uspto_conditions(
                            database=database,
                            smirks=template,
                            chem21_data=chem21_data,
                            fingerprint_type=self.fingerprint_type
                        )
                        df_batch_conditions = score_conditions(
                            self.fingerprint,
                            related_uspto_reactions,
                            similarity_weight=batch_weight,
                            fingerprint_type=self.fingerprint_type,
                            metric=self.metric,
                            reaction_fingerprint=self.reaction_fingerprint,
                        )
                        all_batch_dfs.append(df_batch_conditions)

                    else:
                        continue

            all_dfs += all_batch_dfs

        if len(all_dfs) == 0:
            return {"solvent": {}, "reagent": {}, "catalyst": {}}
        else:
            all_conditions = pd.concat(all_dfs).reset_index(drop=True).sort_values(by="similarity", ascending=False)
        sorted_conditions = sort_conditions(all_conditions, green_solvents)

        solvent_df = create_solvent_df(sorted_conditions)
        reagent_df = pd.DataFrame({
            "smiles": sorted_conditions["reagent"].keys(),
            "score": sorted_conditions["reagent"].values()
        })
        catalyst_df = pd.DataFrame({
            "smiles": sorted_conditions["catalyst"].keys(),
            "score": sorted_conditions["catalyst"].values()
        })

        if len(reagent_df.index) > 1:
            reagent_df["score"] = reagent_df["score"].to_numpy() / np.sum(reagent_df["score"].to_numpy())

        if len(catalyst_df.index) > 1:
            catalyst_df["score"] = catalyst_df["score"].to_numpy() / np.sum(catalyst_df["score"].to_numpy())

        self.suggested_solvent = solvent_df
        self.suggested_reagent = reagent_df
        self.suggested_catalyst = catalyst_df

        return sorted_conditions


class ReactionDatabase(Database):

    """
    A class for managing reaction databases and converting them into knowledge graphs.

    Attributes:
        df (pd.DataFrame): The reaction database.
    """

    def __init__(
            self,
            df: Union[None, pd.DataFrame] = None
    ):

        """
        Initializes the ReactionDatabase with a DataFrame.

        Args:
            df (pd.DataFrame, optional): Reaction database. Defaults to None.
        """

        super().__init__(df=df)
        self.sanitize_database()

    def sanitize_database(self):

        """
        Cleans and prepares the reaction database by handling missing values and invalid reactions.
        """

        values = {
            "SOLVENT": "solvent-free",
            "CATALYST": "catalyst-free",
            "REAGENT": "reagent-free"
        }
        self.df = self.df.fillna(value=values)

        replace_values = {
            "SOLVENT": {"": "solvent-free"},
            "CATALYST": {"": "catalyst-free"},
            "REAGENT": {"": "reagent-free"},
        }

        self.df = self.df.replace(replace_values)

        bad_ids = []
        for i in self.df.index:
            t = self.df["TEMPLATE"][i]
            if ">>" not in t:
                bad_ids.append(i)

        self.df = self.df.drop(bad_ids)

    def convert_to_knowledge_graph(
            self,
            chunk_size: int = 10000,
            num_cpu: int = cpu_count(),
            include_smiles: bool = True,
    ) -> KnowledgeGraph:

        """
        Converts the reaction database into a knowledge graph.

        Args:
            chunk_size (int): Size of data chunks for parallel processing. Defaults to 10000.
            num_cpu (int): Number of CPUs for multiprocessing. Defaults to the system's CPU count.
            include_smiles (bool): Whether to explicitly include smiles in the graph. Defaults to True.
        Returns:
            KnowledgeGraph: The knowledge graph representation of the reaction database.
        """

        self.df["REF"] = self.df["REF"].fillna("not-reported")
        self.df["YIELD"] = self.df["YIELD"].fillna("not-reported")
        self.df = self.df.fillna("")

        all_items = sum((self.df[col].tolist() for col in ["SANITIZED_REACTION", "SOLVENT",
                                                           "CATALYST", "REAGENT", "REF"]), [])
        all_items += [item for sublist in self.df["REACTANTS"].tolist() for item in sublist.split(".")]
        all_items += [item for sublist in self.df["PRODUCTS"].tolist() for item in sublist.split(".")]
        all_items += ["catalyst-free", "reagent-free"]

        chemical_dict = {item: idx for idx, item in enumerate(set(all_items)) if item}
        yield_start_idx = max(list(chemical_dict.values())) + 1

        subclass_dict = get_reaction_subcategories()
        chem21 = load_chem21_data()

        # Create a solvent scoring dictionary
        all_solvents = {solvent: score_solvent(chem21, solvent) for solvent in set(self.df["SOLVENT"].tolist())}

        # Define the chunk size for splitting the DataFrame

        # Split the DataFrame into smaller chunks
        chunks = [self.df[i:i + chunk_size] for i in range(0, self.df.shape[0], chunk_size)]

        # Use multiprocessing to create subgraphs in parallel
        with Pool(num_cpu) as pool:
            subgraphs = list(tqdm(pool.imap(worker, [(chunk,
                                                      chemical_dict,
                                                      subclass_dict,
                                                      all_solvents,
                                                      yield_start_idx,
                                                      include_smiles) for chunk in chunks]),
                                  total=len(chunks), desc="Processing chunks"))

        # Merge all subgraphs into a single graph
        G_final = nx.DiGraph()
        for subgraph in subgraphs:
            G_final = nx.compose(G_final, subgraph)

        # Merge nodes with the same labels and handle other logic as required
        for i, node_data in G_final.nodes(data=True):
            if "Reaction" in node_data['labels']:
                # Find product nodes for this reaction
                product_nodes = [n for n in G_final.successors(i) if G_final.edges[(i, n)]['type'] == 'HAS_PRODUCT']
                for p_node in product_nodes:
                    # Find reactions where this product is a reactant
                    subsequent_reactions = [r for r in G_final.predecessors(p_node) if
                                            G_final.edges[(r, p_node)]['type'] == 'HAS_REACTANT']
                    for sr in subsequent_reactions:
                        G_final.add_edge(i, sr, type='PRECEDES_REACTION')

        # Convert the final graph to a KnowledgeGraph object
        kg = KnowledgeGraph(G_final)
        return kg


def create_subgraph_from_chunk(
        df: pd.DataFrame,
        chemical_dict: dict,
        subclass_dict: dict,
        all_solvents: dict,
        yield_start_idx: int,
        include_smiles: bool = True,
) -> DiGraph:

    """
    Creates a subgraph from a chunk of the reaction database.

    Args:
        df (pd.DataFrame): Chunk of the reaction database.
        chemical_dict (dict): Dictionary mapping chemicals to unique IDs.
        subclass_dict (dict): Dictionary of reaction subclasses.
        all_solvents (dict): Solvent scoring dictionary.
        yield_start_idx (int): Start index for yield nodes.
        include_smiles (bool): Whether to explicitly include smiles.

    Returns:
        DiGraph: A directed graph representing the reactions in the chunk.
    """

    G = nx.DiGraph()
    visited_nodes = []

    if "DOI" in list(df.keys()):
        doi = True
    else:
        doi = False

    for i, row in tqdm(df.iterrows()):
        rxn_idx = chemical_dict[row["SANITIZED_REACTION"]]
        if rxn_idx not in visited_nodes:
            subclass = subclass_dict.get(row["NAME"], row["NAME"])
            rxn_properties = {
                'smirks': row["TEMPLATE"],
                'product_smarts': row["TEMPLATE"].split(">>")[1],
                'reaction_subclass': subclass,
                'reaction_class': row["CLASS"],
                'reaction_type': row["NAME"],
                'maccs_fp': row["rxn_str_patt_fp"],
                'morgan_fp': row["rxn_str_morgan_fp"],
                'tag': row["TAG"],
                'tag2': row["TAG2"],
            }
            if include_smiles:
                rxn_properties['smiles'] = row["SANITIZED_REACTION"]

            G.add_node(rxn_idx, labels=["Reaction"], **rxn_properties)
            visited_nodes.append(rxn_idx)

        # Process reactants
        for reactant in row["REACTANTS"].split("."):
            r_idx = chemical_dict[reactant]
            if r_idx not in visited_nodes:
                maccs = get_molecular_fp(reactant, "MACCS")
                morgan = get_molecular_fp(reactant, "morgan")
                if include_smiles:
                    G.add_node(r_idx, labels=["Molecule"], smiles=reactant, maccs_fp=maccs, morgan_fp=morgan)
                else:
                    G.add_node(r_idx, labels=["Molecule"], maccs_fp=maccs, morgan_fp=morgan)

                visited_nodes.append(r_idx)
            G.add_edge(rxn_idx, r_idx, type='HAS_REACTANT')

        # Process products
        for product in row["PRODUCTS"].split("."):
            p_idx = chemical_dict[product]
            if p_idx not in visited_nodes:
                maccs = get_molecular_fp(product, "MACCS")
                morgan = get_molecular_fp(product, "morgan")
                if include_smiles:
                    G.add_node(p_idx, labels=["Molecule"], smiles=product, maccs_fp=maccs, morgan_fp=morgan)
                else:
                    G.add_node(p_idx, labels=["Molecule"], maccs_fp=maccs, morgan_fp=morgan)

                visited_nodes.append(p_idx)
            G.add_edge(rxn_idx, p_idx, type='HAS_PRODUCT')

        # Process solvent
        solvent = row["SOLVENT"]
        if solvent != "not-reported":
            s_idx = chemical_dict[solvent]
            if s_idx not in visited_nodes:
                sol_properties = {
                    'name': all_solvents[solvent]['solvent_name'],
                    'smiles': all_solvents[solvent]['solvent_smiles'],
                    'score': all_solvents[solvent]['solvent_score']
                }
                G.add_node(s_idx, labels=["Solvent"], **sol_properties)
                visited_nodes.append(s_idx)
            G.add_edge(rxn_idx, s_idx, type='USES_SOLVENT')

        # Process reagent
        reagent = row["REAGENT"] or "reagent-free"
        if reagent != "not-reported":
            re_idx = chemical_dict[reagent]
            if re_idx not in visited_nodes:
                G.add_node(re_idx, labels=["Reagent"], name=reagent)
                visited_nodes.append(re_idx)
            G.add_edge(rxn_idx, re_idx, type='USES_REAGENT')

        # Process catalyst
        catalyst = row["CATALYST"] or "catalyst-free"
        if catalyst != "not-reported":
            c_idx = chemical_dict[catalyst]
            if c_idx not in visited_nodes:
                G.add_node(c_idx, labels=["Catalyst"], name=catalyst)
                visited_nodes.append(c_idx)
            G.add_edge(rxn_idx, c_idx, type='USES_CATALYST')

        # Process reference
        ref = row["REF"] or "not-reported"
        if ref != "not-reported":
            ref_idx = chemical_dict[ref]
            if ref_idx not in visited_nodes:
                if doi:
                    ref_properties = {
                        'name': ref,
                        'doi': row["DOI"]
                    }
                else:
                    ref_properties = {
                        'name': ref
                    }
                G.add_node(ref_idx, labels=["Reference"], **ref_properties)
                visited_nodes.append(ref_idx)
            G.add_edge(ref_idx, rxn_idx, type='REPORTS_REACTION')

        reaction_yield = row["YIELD"] or "not-reported"
        if reaction_yield != "not-reported":
            y_idx = yield_start_idx + i
            yield_properties = {
                'value': reaction_yield,
                'ref': row["REF"]
            }
            G.add_node(y_idx, labels=["YIELD"], **yield_properties)
            G.add_edge(rxn_idx, y_idx, type='HAS_YIELD')

    for i, node_data in G.nodes(data=True):
        if "Reaction" in node_data['labels']:
            # Find product nodes for this reaction
            product_nodes = [n for n in G.successors(i) if G.edges[(i, n)]['type'] == 'HAS_PRODUCT']
            for p_node in product_nodes:
                # Find reactions where this product is a reactant
                subsequent_reactions = [r for r in G.predecessors(p_node) if
                                        G.edges[(r, p_node)]['type'] == 'HAS_REACTANT']
                for sr in subsequent_reactions:
                    G.add_edge(i, sr, type='PRECEDES_REACTION')

    return G


# Function to pass multiple arguments to create_subgraph_from_chunk
def worker(args):

    """
    Helper function for multiprocessing to create subgraphs.

    Args:
        args (tuple): Arguments for create_subgraph_from_chunk.

    Returns:
        DiGraph: A subgraph created from the chunk.
    """

    return create_subgraph_from_chunk(*args)


def create_solvent_df(conditions: dict):

    """
    Creates a DataFrame of suggested solvents based on conditions.

    Args:
        conditions (dict): Dictionary of suggested conditions.

    Returns:
        pd.DataFrame: DataFrame of solvents with their scores.
    """

    solvent_dict = {"name": [], "smiles": [], "hazard_level": [], "score": []}
    for solvent in conditions["solvent"]:
        solvent_dict["name"].append(solvent)
        solvent_dict["smiles"].append(conditions["solvent"][solvent]["smiles"])
        solvent_dict["hazard_level"].append(conditions["solvent"][solvent]["hazard_level"])
        solvent_dict["score"].append(conditions["solvent"][solvent]["similarity"])

    df_solvent = pd.DataFrame(solvent_dict)
    if len(df_solvent.index) > 1:
        df_solvent["score"] = df_solvent["score"].to_numpy() / np.sum(df_solvent["score"].to_numpy())

    return df_solvent


def get_molecular_fp(smi: str, fp: str) -> str:

    """
    Converts a SMILES into a bit-type fingerprint.

    Args:
        smi (str): SMILES identifier of a molecule.
        fp (str): Fingerprint type (MACCS or morgan).

    Returns:
        str: Fingerprint in string format.
    """
    m = Chem.MolFromSmiles(smi)
    if fp.lower() == "maccs":
        nfp = maccs_fp(m)
    else:
        nfp = morgan_fp(m)

    bfp = "".join(nfp.astype(str))
    return bfp
