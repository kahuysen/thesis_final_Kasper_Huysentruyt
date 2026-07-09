"""
This module provides a `KnowledgeGraph` class to handle reaction knowledge graphs.
The class supports various functionalities including retrieving reaction SMARTS,
counting reaction SMIRKS, finding related reactions, and storing the graph.

Dependencies:
    - rdkit
    - networkx
    - numpy
    - tqdm
    - multiprocessing
    - rxn_insight.synthesis.utils
    - rxn_insight.utils
"""

import logging
from rdkit import Chem
from collections import defaultdict, Counter
from rxn_insight.utils import get_similarity
from typing import Union
from networkx.classes.digraph import DiGraph
import networkx as nx
from tqdm import tqdm
import pickle
import numpy as np
import numpy.typing as npt


class KnowledgeGraph:

    """
    A class to manage and analyze reaction knowledge graphs.

    Attributes:
        graph (DiGraph): The directed graph representing reactions.
        n_edges (int): The number of edges in the graph.
        n_nodes (int): The number of nodes in the graph.
        node_types (list[str]): Unique types of nodes in the graph.
        edge_types (list[str]): Unique types of edges in the graph.
    """

    def __init__(self, graph: DiGraph):
        """
        Initializes a KnowledgeGraph instance.

        Args:
            graph (DiGraph): A directed graph.
        """
        self.graph = graph

        self.n_edges = self.graph.number_of_edges()
        self.n_nodes = self.graph.number_of_nodes()
        self.node_types = list(set(label for labels in nx.get_node_attributes(self.graph, 'labels').values()
                                   for label in labels))
        self.edge_types = list(set(labels for labels in nx.get_edge_attributes(self.graph, 'type').values()))

    def get_all_product_smarts(self):
        """
        Retrieves all unique product SMARTS and their RDKit objects.

        Returns:
            tuple: A tuple containing:
                - list[str]: Unique product SMARTS strings.
                - list[Chem.Mol]: Corresponding RDKit molecule objects.
        """
        all_smarts = []
        all_rdkit_objects = []

        for node, data in tqdm(self.graph.nodes(data=True)):
            if 'Reaction' in data.get('labels', []) and data.get('product_smarts') not in all_smarts:
                smarts = data.get('product_smarts')
                if smarts:
                    try:
                        rdkit_mol = Chem.MolFromSmarts(smarts)
                        if rdkit_mol:
                            all_rdkit_objects.append(rdkit_mol)
                            all_smarts.append(smarts)
                    except Exception as e:
                        logging.warning(f"Failed to process SMARTS {smarts}: {e}")

        return all_smarts, all_rdkit_objects

    def look_up_smirks(self, product_smarts):
        """
        Finds SMIRKS patterns and associated DOI nodes for a given product SMARTS.

        Args:
            product_smarts (str): The SMARTS pattern to look up.

        Returns:
            list[dict]: A list of dictionaries containing SMIRKS and paper counts.
        """
        # Dictionary to hold results
        results = defaultdict(list)

        # Iterate through all nodes and find Reaction nodes with the specified product_smarts
        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get('product_smarts') == product_smarts:
                #             print(node, data)
                # This is a matching Reaction node
                reaction_node = node
                smirks = data.get('smirks')

                # Find all DOI nodes connected to this Reaction node via in-edges
                doi_nodes = [
                    source for source, target, edge_data in self.graph.in_edges(reaction_node, data=True)
                    if edge_data.get('type') == 'REPORTS_REACTION' and (
                            'DOI' in self.graph.nodes[source].get('labels', []) or
                            'Reference' in self.graph.nodes[source].get('labels', [])
                    )
                ]

                # Store the result
                results[smirks] += doi_nodes

        # Convert results to list of dictionaries for easier use
        results_list = [{'smirks': smirks, 'paper_count': len(list(set(dois)))} for smirks, dois in results.items()]

        return results_list

    def get_smirks_count(self):
        """
        Counts the occurrences of each SMIRKS in the graph.

        Returns:
            dict: A dictionary where keys are SMIRKS strings and values are their counts.
        """
        # Extract all 'smirks' strings for nodes labeled as 'Reaction'
        smirks_list = [
            data.get('smirks')
            for _, data in self.graph.nodes(data=True)
            if 'Reaction' in data.get('labels', []) and data.get('smirks')
        ]

        # Count occurrences of each SMIRKS using Counter
        smirks_count = Counter(smirks_list)

        return dict(smirks_count)

    def find_similar_reactions(
            self,
            fp: npt.NDArray[int],
            fingerprint_type: str,
            similarity_threshold: float = 0.5,
            reaction_fingerprint: bool = True,
            metric: str = "jaccard",
    ):
        """
        Identifies reactions similar to a given fingerprint.

        Args:
            fp (npt.NDArray[int]): Query fingerprint array.
            fingerprint_type (str): Type of fingerprint ('morgan' or 'maccs').
            similarity_threshold (float): Minimum similarity score to include a reaction.
            reaction_fingerprint (bool): Whether to use the full reaction fingerprint.
            metric (str): Similarity metric to use.

        Returns:
            list[dict]: A list of dictionaries with reaction details.
        """

        if not reaction_fingerprint:
            fp = fp[int(len(fp) / 2):]

        results = []

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []):
                reaction_node = node
                morgan_fp = data.get('morgan_fp')
                maccs_fp = data.get('maccs_fp')

                if fingerprint_type.lower() == "morgan":
                    neighbor_fp = np.array(list(morgan_fp), dtype=np.int64)
                elif fingerprint_type.lower() == "maccs":
                    neighbor_fp = np.array(list(maccs_fp), dtype=np.int64)
                else:
                    raise ValueError(f"Fingerprint type {fingerprint_type} is not supported!")

                if not reaction_fingerprint:
                    neighbor_fp = neighbor_fp[int(len(neighbor_fp) / 2):]

                similarity = get_similarity(fp, neighbor_fp, metric=metric)

                if similarity < similarity_threshold:
                    continue

                result_dict = {
                    'morgan_fp': morgan_fp,
                    'maccs_fp': maccs_fp,
                }

                for source, target, edge_data in self.graph.out_edges(reaction_node, data=True):
                    if edge_data.get('type') == 'USES_SOLVENT' and 'Solvent' in self.graph.nodes[target].get(
                            'labels',
                            []
                    ):
                        solvent = self.graph.nodes[target]
                        solvent_name = solvent.get('name')
                        solvent_smiles = solvent.get('smiles')
                        solvent_score = solvent.get('score')
                        result_dict['solvent'] = solvent_name
                        result_dict['solvent_smiles'] = solvent_smiles
                        result_dict['solvent_score'] = solvent_score

                    elif edge_data.get('type') == 'USES_CATALYST' and 'Catalyst' in self.graph.nodes[target].get(
                            'labels', []):
                        catalyst = self.graph.nodes[target]
                        catalyst_name = catalyst.get('name')
                        result_dict['catalyst'] = catalyst_name

                    elif edge_data.get('type') == 'USES_REAGENT' and 'Reagent' in self.graph.nodes[target].get('labels',
                                                                                                               []):
                        reagent = self.graph.nodes[target]
                        reagent_name = reagent.get('name')
                        result_dict['reagent'] = reagent_name

                results.append(result_dict)

        return results

    def find_related_reactions(self, smirks: str):
        """
        Finds reactions related to a given SMIRKS pattern.

        Args:
            smirks (str): The SMIRKS pattern to search for.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains details of a related reaction, including:
                - 'reaction_smiles': The reaction's SMILES representation.
                - 'maccs_fp': The MACCS fingerprint of the reaction.
                - 'morgan_fp': The Morgan fingerprint of the reaction.
                - 'reaction_class': The reaction's class.
                - 'reaction_subclass': The reaction's subclass.
                - 'reaction_type': The type of the reaction.
                - 'doi': The DOI of the associated publication.
                - 'title': The title of the associated publication.
        """
        results = []

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get('smirks') == smirks:
                reaction_node = node
                rxn_smiles = data.get('smiles')
                maccs_fp = data.get('maccs_fp')
                morgan_fp = data.get('morgan_fp')
                reaction_class = data.get('reaction_class')
                reaction_subclass = data.get('reaction_subclass')
                reaction_type = data.get('reaction_type')

                for source, target, edge_data in self.graph.in_edges(reaction_node, data=True):

                    if edge_data.get('type') == 'REPORTS_REACTION' and (
                            'DOI' in self.graph.nodes[source].get('labels', [])
                            or 'Reference' in self.graph.nodes[source].get('labels', [])):
                        doi = self.graph.nodes[source].get('doi')
                        title = self.graph.nodes[source].get('title')
                        results.append(
                            {
                                'reaction_smiles': rxn_smiles,
                                'maccs_fp': maccs_fp,
                                'morgan_fp': morgan_fp,
                                'reaction_class': reaction_class,
                                'reaction_subclass': reaction_subclass,
                                'reaction_type': reaction_type,
                                'doi': doi,
                                'title': title
                            }
                        )
        return results

    def find_conditions_by_template(self, smirks: str):
        """
        Retrieves reaction conditions associated with a given SMIRKS pattern.

        Args:
            smirks (str): The SMIRKS pattern to search for.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains details of reaction conditions, including:
                - 'morgan_fp': The Morgan fingerprint of the reaction.
                - 'maccs_fp': The MACCS fingerprint of the reaction.
                - 'solvent': The name of the solvent used in the reaction (if applicable).
                - 'solvent_smiles': The SMILES representation of the solvent (if applicable).
                - 'solvent_score': The score associated with the solvent (if applicable).
                - 'catalyst': The name of the catalyst used in the reaction (if applicable).
                - 'reagent': The name of the reagent used in the reaction (if applicable).
        """
        results = []

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get('smirks') == smirks:
                reaction_node = node
                morgan_fp = data.get('morgan_fp')
                maccs_fp = data.get('maccs_fp')

                result_dict = {
                    'morgan_fp': morgan_fp,
                    'maccs_fp': maccs_fp,
                }

                for source, target, edge_data in self.graph.out_edges(reaction_node, data=True):
                    if edge_data.get('type') == 'USES_SOLVENT' and 'Solvent' in self.graph.nodes[target].get(
                            'labels',
                            []
                    ):
                        solvent = self.graph.nodes[target]
                        solvent_name = solvent.get('name')
                        solvent_smiles = solvent.get('smiles')
                        solvent_score = solvent.get('score')
                        result_dict['solvent'] = solvent_name
                        result_dict['solvent_smiles'] = solvent_smiles
                        result_dict['solvent_score'] = solvent_score

                    elif edge_data.get('type') == 'USES_CATALYST' and 'Catalyst' in self.graph.nodes[target].get(
                            'labels', []):
                        catalyst = self.graph.nodes[target]
                        catalyst_name = catalyst.get('name')
                        result_dict['catalyst'] = catalyst_name

                    elif edge_data.get('type') == 'USES_REAGENT' and 'Reagent' in self.graph.nodes[target].get('labels',
                                                                                                               []):
                        reagent = self.graph.nodes[target]
                        reagent_name = reagent.get('name')
                        result_dict['reagent'] = reagent_name

                results.append(result_dict)

        return results

    def find_reactions_by_tag(self, tag: str, broad_search: bool = False):
        """
        Finds reactions with a specific broad tag (TAG2).

        Args:
            tag (str): The broad tag to search for.
            broad_search (bool): Use TAG2 instead`of TAG. Defaults to False.

        Returns:
            list[dict]: A list of dictionaries containing reaction details and conditions.
        """
        results = []
        if broad_search:
            tag_type = "tag2"
        else:
            tag_type = "tag"

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get(tag_type) == tag:
                reaction_node = node
                morgan_fp = data.get('morgan_fp')
                maccs_fp = data.get('maccs_fp')
                reaction_class = data.get('reaction_class')
                reaction_subclass = data.get('reaction_subclass')
                reaction_type = data.get('reaction_type')
                smirks = data.get('smirks')
                rxn_smiles = data.get('smiles')

                result_dict = {
                    'morgan_fp': morgan_fp,
                    'maccs_fp': maccs_fp,
                    'reaction_class': reaction_class,
                    'reaction_subclass': reaction_subclass,
                    'reaction_type': reaction_type,
                    'smirks': smirks,
                    'reaction_smiles': rxn_smiles,
                }

                # Get conditions (solvent, catalyst, reagent)
                for source, target, edge_data in self.graph.out_edges(reaction_node, data=True):
                    if edge_data.get('type') == 'USES_SOLVENT' and 'Solvent' in self.graph.nodes[target].get('labels',
                                                                                                             []):
                        solvent = self.graph.nodes[target]
                        result_dict['solvent'] = solvent.get('name')
                        result_dict['solvent_smiles'] = solvent.get('smiles')
                        result_dict['solvent_score'] = solvent.get('score')

                    elif edge_data.get('type') == 'USES_CATALYST' and 'Catalyst' in self.graph.nodes[target].get(
                            'labels', []):
                        catalyst = self.graph.nodes[target]
                        result_dict['catalyst'] = catalyst.get('name')

                    elif edge_data.get('type') == 'USES_REAGENT' and 'Reagent' in self.graph.nodes[target].get('labels',
                                                                                                               []):
                        reagent = self.graph.nodes[target]
                        result_dict['reagent'] = reagent.get('name')

                results.append(result_dict)

        return results

    def get_available_reaction_subclasses(self):
        """
        Returns a list of all unique reaction subclasses (named reactions) in the knowledge graph.

        This is useful for discovering what named reactions are available in the database.

        Returns:
            list[str]: Sorted list of unique reaction subclasses.
        """
        subclasses = set()

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []):
                subclass = data.get('reaction_subclass')
                if subclass:
                    subclasses.add(subclass)

        return sorted(list(subclasses))

    def find_conditions_by_subclass(self, reaction_subclass: str):
        """
        Finds all reaction conditions for a specific reaction subclass (named reaction type).

        This is useful for finding common conditions used in named reactions like
        "Wittig Reaction", "Suzuki Coupling", etc.

        Args:
            reaction_subclass (str): The reaction subclass/named reaction to search for.
                                    Examples: "Wittig Reaction", "Suzuki Coupling", "Heck Reaction"

        Returns:
            list[dict]: A list of dictionaries containing reaction details and conditions
                       for all reactions of the specified subclass.
        """
        results = []

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get('reaction_subclass') == reaction_subclass:
                reaction_node = node
                morgan_fp = data.get('morgan_fp')
                maccs_fp = data.get('maccs_fp')
                reaction_class = data.get('reaction_class')
                reaction_type = data.get('reaction_type')  # Specific variant
                smirks = data.get('smirks')
                rxn_smiles = data.get('smiles')
                tag = data.get('tag')
                tag2 = data.get('tag2')

                result_dict = {
                    'morgan_fp': morgan_fp,
                    'maccs_fp': maccs_fp,
                    'reaction_class': reaction_class,
                    'reaction_subclass': reaction_subclass,
                    'reaction_type': reaction_type,  # Specific variant of the named reaction
                    'smirks': smirks,
                    'reaction_smiles': rxn_smiles,
                    'tag': tag,
                    'tag2': tag2,
                }

                # Get conditions (solvent, catalyst, reagent)
                for source, target, edge_data in self.graph.out_edges(reaction_node, data=True):
                    if edge_data.get('type') == 'USES_SOLVENT' and 'Solvent' in self.graph.nodes[target].get('labels',
                                                                                                             []):
                        solvent = self.graph.nodes[target]
                        result_dict['solvent'] = solvent.get('name')
                        result_dict['solvent_smiles'] = solvent.get('smiles')
                        result_dict['solvent_score'] = solvent.get('score')

                    elif edge_data.get('type') == 'USES_CATALYST' and 'Catalyst' in self.graph.nodes[target].get(
                            'labels', []):
                        catalyst = self.graph.nodes[target]
                        result_dict['catalyst'] = catalyst.get('name')

                    elif edge_data.get('type') == 'USES_REAGENT' and 'Reagent' in self.graph.nodes[target].get('labels',
                                                                                                               []):
                        reagent = self.graph.nodes[target]
                        result_dict['reagent'] = reagent.get('name')

                # Get reference/DOI information if available
                for source, target, edge_data in self.graph.in_edges(reaction_node, data=True):
                    if edge_data.get('type') == 'REPORTS_REACTION' and (
                            'DOI' in self.graph.nodes[source].get('labels', [])
                            or 'Reference' in self.graph.nodes[source].get('labels', [])):
                        result_dict['doi'] = self.graph.nodes[source].get('doi')
                        result_dict['title'] = self.graph.nodes[source].get('title')
                        break  # Just take the first reference for now

                results.append(result_dict)

        return results

    def store_graph(self, fname: str):
        """
        Saves the graph to a file in pickle format.

        Args:
            fname (str): The name of the file to save the graph.
        """
        with open(fname + ".gpickle", 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
