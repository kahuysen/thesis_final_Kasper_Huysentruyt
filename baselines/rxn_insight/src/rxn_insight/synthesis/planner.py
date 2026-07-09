from rxn_insight.synthesis.knowledge_graph import KnowledgeGraph
from rxn_insight.synthesis.utils import get_all_buyables, get_flow_ef
from rxn_insight.synthesis.reaction_transforms import (
    find_substructure_matches,
    get_reactants,
    find_reaction_neighbors
)
from rxn_insight.synthesis.analysis import ReactionAnalyzer

from rdkit import Chem
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import numpy as np
from rxnmapper import RXNMapper


class SynthesisTree:

    """
    A class for building and analyzing synthesis pathways for chemical compounds using knowledge graphs.

    Attributes:
        graph (KnowledgeGraph): The knowledge graph for synthesis.
        max_depth (int): Maximum depth of synthesis trees to explore.
        max_paths (int): Maximum number of synthesis paths to retain.
        similarity_threshold (float): Minimum similarity threshold for reaction matching.
        min_disconnection_score (float): Minimum score for acceptable disconnections.
        similarity_weight (float): Weight for similarity in scoring paths.
        ref_weight (float): Weight for reference counts in scoring paths.
        flow_weight (float): Weight for flow chemistry in scoring paths.
        accessibility_weight (float): Weight for accessibility in scoring paths.
        max_price (float): Maximum price for buyable reactants.
        fingerprint (str): Fingerprint method for similarity calculations.
        similarity_metric (str): Metric for similarity calculations.
        reaction_fingerprint (bool): Whether to use reaction fingerprints.
        prefer_flow (bool): Whether to prioritize flow chemistry reactions.
    """

    def __init__(
            self,
            graph: KnowledgeGraph,
            max_depth: int = 1,
            max_paths: int = 1,
            similarity_threshold: float = 0.5,
            min_disconnection_score: float = 10.0,
            similarity_weight: float = 20.0,
            ref_weight: float = 1.0,
            flow_weight: float = 20.0,
            accessibility_weight: float = 20.0,
            max_price: float = 50.0,
            fingerprint: str = "maccs",
            similarity_metric: str = "jaccard",
            reaction_fingerprint: bool = True,
            prefer_flow: bool = False,
    ):

        """
        Initialize the SynthesisTree class with parameters and load required data.

        Args:
            graph (KnowledgeGraph): The knowledge graph to use for synthesis.
            max_depth (int, optional): Maximum depth of synthesis trees to explore. Defaults to 1.
            max_paths (int, optional): Maximum number of synthesis paths to retain. Defaults to 1.
            similarity_threshold (float, optional): Minimum similarity threshold for reaction matching. Defaults to 0.5.
            min_disconnection_score (float, optional): Minimum score for acceptable disconnections. Defaults to 10.0.
            similarity_weight (float, optional): Weight for similarity in scoring synthesis paths. Defaults to 20.0.
            ref_weight (float, optional): Weight for reference counts in scoring synthesis paths. Defaults to 1.0.
            flow_weight (float, optional): Weight for flow chemistry in scoring synthesis paths. Defaults to 20.0.
            accessibility_weight (float, optional): Weight for accessibility in scoring synthesis paths. Defaults to 20.0.
            max_price (float, optional): Maximum price for buyable reactants. Defaults to 50.0.
            fingerprint (str, optional): Fingerprint method to use for similarity calculations. Defaults to "maccs".
            similarity_metric (str, optional): Metric for calculating similarity. Defaults to "jaccard".
            reaction_fingerprint (bool, optional): Whether to use reaction fingerprints for similarity. Defaults to True.
            prefer_flow (bool, optional): Whether to prioritize flow chemistry reactions. Defaults to False.
        """

        self.graph = graph
        self.buyables = get_all_buyables()
        self.flow_ef_dict = get_flow_ef()
        self.available_smarts, self.available_rdkit_transformations = self.graph.get_all_product_smarts()
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.similarity_threshold = similarity_threshold
        self.min_disconnection_score = min_disconnection_score
        self.similarity_weight = similarity_weight
        self.ref_weight = ref_weight
        self.accessibility_weight = accessibility_weight
        self.flow_weight = flow_weight
        self.max_price = max_price
        self.fingerprint = fingerprint
        self.similarity_metric = similarity_metric
        self.reaction_fingerprint = reaction_fingerprint
        self.prefer_flow = prefer_flow
        self.reaction_tree = None
        self.reaction_paths = None
        self.synthesis_tree = None

    def get_children(self, product):

        """
        Identify possible child reactions for a given product.

        Parameters:
            product (str): The product's SMILES representation.

        Returns:
            dict or None: A dictionary containing child reaction data or None if no reactions are found.
        """

        mol = Chem.MolFromSmiles(product)
        templates = find_substructure_matches(
            mol,
            self.available_smarts,
            self.available_rdkit_transformations
        )
        if len(templates) == 0:
            return None
        else:
            children = {}
            encountered_reactions = {}
            idx = 1

            for template in templates:
                matches = self.graph.look_up_smirks(template)
                for match in matches:
                    smirks = match.get('smirks')
                    try:
                        reactant_dict = get_reactants(product, smirks, mol=mol)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        continue

                    if reactant_dict:
                        are_buyable = []
                        price = 0
                        for reactant in reactant_dict["reactants"]:
                            if reactant not in self.buyables:
                                are_buyable.append(False)
                            else:
                                are_buyable.append(True)
                                price += self.buyables[reactant]

                        reactant_dict["buyable"] = all(are_buyable)
                        reactant_dict["buyable_pct"] = sum(are_buyable) / len(are_buyable)
                        reactant_dict["refs"] = match.get('paper_count', 0)
                        reactant_dict["price"] = price
                        reactant_dict["flow_ef"] = self.flow_ef_dict.get(smirks, 0)

                        reaction_smiles = reactant_dict["reaction_smiles"]
                        related_reactions = self.graph.find_related_reactions(smirks)

                        reaction_neighbors = find_reaction_neighbors(
                            reaction=reaction_smiles,
                            relevant_reactions=related_reactions,
                            threshold=self.similarity_threshold,
                            fingerprint=self.fingerprint,
                            reaction_fingerprint=self.reaction_fingerprint,
                            similarity_metric=self.similarity_metric
                        )

                        if reaction_neighbors is None:
                            continue
                        else:
                            reactant_dict["similarity"] = reaction_neighbors["similarity"]
                            reactant_dict["closest_reference"] = reaction_neighbors["closest_reference"]
                            reactant_dict["reaction_class"] = reaction_neighbors["reaction_class"][0]
                            reactant_dict["reaction_subclass"] = reaction_neighbors["reaction_subclass"][0]
                            reactant_dict["reaction_type"] = reaction_neighbors["reaction_type"]

                        if reaction_smiles not in encountered_reactions:
                            children[f"child_{idx}"] = reactant_dict
                            encountered_reactions[reaction_smiles] = f"child_{idx}"
                            idx += 1
                        else:
                            child = encountered_reactions[reaction_smiles]
                            children[child]["smirks"].append(reactant_dict["smirks"][0])
                            children[child]["refs"] += reactant_dict["refs"]

            return children

    def predict_single_step(
            self,
            product: str,
            include_buyable: bool = False
    ):

        """
        Predict single-step reactions for a given product and select the best disconnections.

        Parameters:
            product (str): The product's SMILES representation.
            include_buyable (bool, optional): Whether to include buyable reactants. Defaults to False.

        Returns:
            dict: Selected reaction paths with scores.
        """

        child_nodes = self.get_children(product)
        selected_paths = select_best_disconnection(
            all_paths=child_nodes,
            min_score=self.min_disconnection_score,
            include_buyable=include_buyable,
            max_paths=self.max_paths,
            similarity_weight=self.similarity_weight,
            ref_weight=self.ref_weight,
            accessibility_weight=self.accessibility_weight,
            flow_weight=self.flow_weight,
            prefer_flow=self.prefer_flow,
        )
        return selected_paths

    def make_synthesis_graph(
            self,
            product: str,
    ):

        """
        Create a synthesis graph with the initial product as the root node.

        Parameters:
            product (str): The product's SMILES representation.
        """

        graph = nx.DiGraph()
        product_properties = {"smiles": product, "price": self.buyables.get(product, None)}
        graph.add_node(product, labels=["Molecule"], **product_properties)
        self.synthesis_tree = graph

    def add_transformation_to_tree(
            self,
            child
    ):

        """
        Add a transformation to the synthesis tree.

        Parameters:
            child (dict): A dictionary containing details of the child reaction.
        """

        reaction_smiles = child.get('reaction_smiles')
        smirks = child.get('smirks')
        sa_difference = child.get('sa_difference')
        flow_ef = child.get('flow_ef')
        refs = int(child.get('refs'))
        buyable = child.get('buyable')
        score = child.get('score')
        if score:
            score = float(round(score, 2))

        if self.synthesis_tree.has_node(reaction_smiles):
            pass
        else:
            reaction_properties = {"smiles": reaction_smiles,
                                   "smirks": smirks,
                                   "sa_difference": sa_difference,
                                   "flow_ef": flow_ef,
                                   "refs": refs,
                                   "buyable": buyable,
                                   "score": score}
            self.synthesis_tree.add_node(reaction_smiles, labels=["Reaction"], **reaction_properties)

        reactants = child.get('reactants')
        for reactant in reactants:
            if not self.synthesis_tree.has_node(reactant):
                price = self.buyables.get(reactant, "Not for sale")
                if not isinstance(price, str):
                    price = float(price)
                reactant_properties = {"smiles": reactant,
                                       "price": price}
                self.synthesis_tree.add_node(reactant, labels=["Molecule"], **reactant_properties)

            if not self.synthesis_tree.has_edge(reactant, reaction_smiles):
                self.synthesis_tree.add_edge(reactant, reaction_smiles, type="IS_REACTANT")

        product = child.get('product')
        self.synthesis_tree.add_edge(reaction_smiles, product, type="HAS_PRODUCT")

    def construct_synthesis_tree(
            self,
            product,
            current_depth: int = 0
    ):

        """
        Construct a synthesis tree recursively starting from a target product.

        Parameters:
            product (str): The target product's SMILES representation.
            current_depth (int, optional): The current depth in the tree. Defaults to 0.
        """

        if not self.synthesis_tree:
            self.make_synthesis_graph(product)

        if current_depth >= self.max_depth or Chem.CanonSmiles(product) in self.buyables:
            return None

        children = self.get_children(product)

        if children:
            best_disconnections = select_best_disconnection(children,
                                                            self.min_disconnection_score,
                                                            similarity_weight=self.similarity_weight,
                                                            ref_weight=self.ref_weight,
                                                            max_paths=self.max_paths)
        else:
            best_disconnections = None

        if best_disconnections:
            for child_key, child_data in best_disconnections.items():
                self.add_transformation_to_tree(child_data)
                reactants = child_data.get('reactants')
                for reactant in reactants:
                    if self.buyables.get(reactant, self.max_price+1) < self.max_price:
                        continue
                    else:
                        self.construct_synthesis_tree(reactant, current_depth+1)

    def build_reaction_tree(
            self,
            product: str,
            current_depth: int = 0,
            previous_occurrences=None,
    ):
        """
        Recursively build a reaction tree up to a specified depth.

        Parameters:
            product (str): The target molecule's SMILES representation.
            current_depth (int, optional): Current depth of recursion. Defaults to 0.
            previous_occurrences (dict, optional): Tracks previously encountered nodes. Defaults to None.

        Returns:
            dict: A nested dictionary representing the reaction tree.
        """

        if current_depth >= self.max_depth or Chem.CanonSmiles(product) in self.buyables:
            return None

        tree = {
            "product": product,
            "children": []
        }
        if previous_occurrences is None:
            previous_occurrences = {}

        children = self.get_children(product)
        if children:
            if current_depth == 0:
                best_disconnections = select_best_disconnection(
                    children,
                    self.min_disconnection_score,
                    similarity_weight=self.similarity_weight,
                    ref_weight=self.ref_weight,
                    accessibility_weight=self.accessibility_weight
                )
            else:
                best_disconnections = select_best_disconnection(
                    children,
                    self.min_disconnection_score,
                    similarity_weight=self.similarity_weight,
                    ref_weight=self.ref_weight,
                    accessibility_weight=self.accessibility_weight,
                    max_paths=1,
                    include_buyable=True
                )
        else:
            best_disconnections = None

        if best_disconnections:
            for child_key, child_data in best_disconnections.items():
                if not child_data["buyable"]:
                    subtrees = {}
                    for reactant in child_data["reactants"]:
                        if reactant in self.buyables and self.buyables.get(reactant, self.max_price+1) < self.max_price:
                            subtrees[reactant] = None
                        elif reactant in previous_occurrences:
                            subtrees[reactant] = previous_occurrences[reactant]
                        else:
                            subtree = self.build_reaction_tree(reactant, current_depth + 1, previous_occurrences)
                            subtrees[reactant] = subtree
                            previous_occurrences[reactant] = subtree
                    child_tree = {
                        "details": child_data,
                        "children": subtrees
                    }
                else:
                    child_tree = {
                        "details": child_data,
                        "children": None
                    }

                tree["children"].append(child_tree)

        return tree

    def score_paths(
            self,
    ):

        """
        Evaluate and rank synthesis paths based on various scoring metrics.

        Returns:
            dict: A dictionary of scored and ranked synthesis paths.
        """

        tree = self.reaction_tree
        buyables = self.buyables
        all_paths = {}

        for idx, child in enumerate(tree["children"]):
            found_paths, starting_materials = get_all_paths(child)
            path_length = len(found_paths)
            prices = []
            buyable = []

            for mol in starting_materials:
                buyable.append(mol in buyables)
                prices.append(buyables.get(mol, 0))

            avg_similarity = sum([found_paths[d][0]['similarity'] for d in found_paths]) / path_length
            avg_reference = sum([found_paths[d][0]['refs'] for d in found_paths]) / path_length
            are_buyable = sum(buyable) / len(buyable)
            price = sum(prices)
            length_penalty = path_length * 10  # Arbitrary penalty per reaction step
            buyable_penalty = 100 * (1 - are_buyable)
            price_weight = price

            if price == 0:
                price_weight = 100

            similarity_weight = -20 * avg_similarity
            ref_weight = -3 * avg_reference  # Negative because higher ref count is better
            penalty = length_penalty + buyable_penalty + price_weight + ref_weight + similarity_weight

            all_paths[f"path_{idx}"] = {
                "penalty": round(penalty, 2),
                "path_length": path_length,
                "buyable": round(are_buyable, 2),
                "price": price,
                "reaction_smiles": [found_paths[d][0]['reaction_smiles'] for d in found_paths],
                "starting_materials": starting_materials,
                "route": found_paths
            }

        sorted_paths = sorted(
            all_paths.items(),
            key=lambda x: x[1]['penalty']
        )[:self.max_paths]  # Dict[str, Dict[str, Union[float, str]]]

        sorted_paths_dict = {}
        for path in sorted_paths:
            sorted_paths_dict[path[0]] = path[1]

        return sorted_paths_dict

    def plan_route(
            self,
            target_molecule: str,
    ):

        """
        Plan and score synthesis routes for the target molecule.

        Parameters:
            target_molecule (str): The target molecule's SMILES representation.

        Returns:
            dict: The top synthesis pathways.
        """

        # Build the reaction tree
        reaction_tree = self.build_reaction_tree(target_molecule)
        self.reaction_tree = reaction_tree
        best_paths = self.score_paths()
        self.reaction_paths = best_paths

        return best_paths

    def simulate_paths(
            self,
            product: str,
            num_simulations: int,
    ):

        """
        Simulate multiple synthesis routes and analyze their frequency of occurrence.

        Parameters:
            product (str): The target product's SMILES representation.
            num_simulations (int): Number of simulation iterations.

        Returns:
            dict: A dictionary containing simulated synthesis routes and statistics.
        """

        graph_dict = defaultdict(dict)
        path_counter = 1

        for _ in tqdm(range(num_simulations)):
            ng = nx.DiGraph()
            product_properties = {"smiles": product, "price": None}
            ng.add_node(product, labels=["Molecule"], **product_properties)
            found_graph, starting_materials = coin_transformation(
                self.synthesis_tree,
                ng.copy(),
                product,
                self.max_price,
                []
            )
            match = False

            for key in graph_dict.keys():
                if nx.utils.graphs_equal(graph_dict[key]["graph"], found_graph):
                    graph_dict[key]["count"] += 1
                    match = True
                    break

            if not match:
                idx = f"route_{path_counter}"
                path_counter += 1
                for_sale = []
                for material in starting_materials:
                    if material in self.buyables:
                        for_sale.append(1)
                    else:
                        for_sale.append(0)
                buyable = sum(for_sale) / len(for_sale)
                graph_dict[idx] = {"count": 1,
                                   "graph": found_graph,
                                   "starting_materials": starting_materials,
                                   "buyable": buyable}

        graph_dict = dict(sorted(graph_dict.items(), key=lambda item: -item[1]["count"]))
        print(f"{len(list(graph_dict.keys()))} synthesis routes are found!")

        return graph_dict


def coin_transformation(
        complete_graph: nx.DiGraph,
        new_graph: nx.DiGraph,
        node: str,
        max_price: float,
        starting_materials: list,
):

    """
    Recursively transform a node in a graph by simulating a probabilistic reaction path.

    Parameters:
    complete_graph (networkx.DiGraph): The complete reaction graph.
    new_graph (networkx.DiGraph): The subgraph to be updated with transformations.
    node (str): The current node to transform.
    max_price (float): Maximum allowed price for a reactant to be considered a starting material.
    starting_materials (list): List of starting material nodes identified so far.

    Returns:
    tuple: Updated graph (`new_graph`) and starting materials (`starting_materials`).
    """

    next_reactions = []
    scores = []
    for source, target, edge_data in complete_graph.in_edges(node, data=True):
        next_reactions.append(complete_graph.nodes[source].get('smiles'))
        scores.append(complete_graph.nodes[source].get('score'))

    if scores:
        scores = np.cumsum(np.array(scores) / np.sum(scores))
    else:
        starting_materials.append(node)
        return new_graph, starting_materials

    random_value = np.random.random()
    index = np.searchsorted(scores, random_value, side='right') - 1
    next_reaction = next_reactions[index]
    if new_graph.has_node(next_reaction):
        return new_graph, starting_materials

    reaction_info = complete_graph.nodes[next_reaction]
    new_graph.add_node(next_reaction, **reaction_info)
    new_graph.add_edge(next_reaction, node, type="HAS_PRODUCT")

    for source, target, edge_data in complete_graph.in_edges(next_reaction, data=True):
        reactant_info = complete_graph.nodes[source]
        new_graph.add_node(source, **reactant_info)
        new_graph.add_edge(source, target, type="IS_REACTANT")
        if not isinstance(reactant_info["price"], str) and reactant_info["price"] \
                and reactant_info["price"] < max_price:
            starting_materials.append(source)
        else:
            new_graph, starting_materials = coin_transformation(complete_graph, new_graph, source, max_price,
                                                                starting_materials)

    return new_graph, starting_materials


def get_all_paths(tree):

    """
    Extract all reaction pathways and starting materials from a reaction tree.

    Parameters:
    tree (dict): A nested dictionary representing the reaction tree.

    Returns:
    tuple: A dictionary of pathways (`all_paths`) and a list of starting materials (`starting_materials`).
    """

    def traverse(node, path_dict, materials):
        name = node["details"]["product"]
        if name not in path_dict:
            path_dict[name] = []
        path_dict[name].append(node["details"])
        children = node["children"]
        if children:
            for reactant in children:
                if children[reactant] is not None:
                    child_nodes = node["children"][reactant]["children"]

                    if isinstance(child_nodes, dict):
                        child_nodes = [child_nodes]

                    if child_nodes == []:
                        materials.append(reactant)
                    else:
                        for child_node in child_nodes:
                            path_dict, materials = traverse(child_node, path_dict, materials)
                else:
                    materials.append(reactant)
        else:
            materials += node["details"]["reactants"]
        return path_dict, materials

    starting_materials = []
    all_paths = {tree["details"]["product"]: []}
    all_paths, starting_materials = traverse(tree, all_paths, starting_materials)

    return all_paths, starting_materials


def select_best_disconnection(
        all_paths,
        min_score: float = 10.0,
        include_buyable: bool = False,
        max_paths: int = 10,
        similarity_weight: float = 20.0,
        ref_weight: float = 1.0,
        flow_weight: float = 20.0,
        accessibility_weight: float = 20.0,
        prefer_flow: bool = False,
):
    """
    Select the best reaction disconnections based on weighted criteria.

    Parameters:
        all_paths (dict): Dictionary of all possible pathways.
        min_score (float): Minimum score threshold for selecting pathways.
        include_buyable (bool): Whether to prioritize buyable compounds.
        max_paths (int): Maximum number of pathways to select.
        similarity_weight (float): Weight for reaction similarity in scoring.
        ref_weight (float): Weight for the number of references in scoring.
        flow_weight (float): Weight for environmental flow factors in scoring.
        accessibility_weight (float): Weight for synthetic accessibility in scoring.
        prefer_flow (bool): Whether to prioritize reactions with positive environmental flow.

    Returns:
        dict: Top pathways meeting the criteria.
    """

    def score_disconnection(
            path,
            similarity_weight: float = 20.0,
            ref_weight: float = 1.0,
            accessibility_weight: float = 20.0,
            flow_weight: float = 20.0,
    ):
        similarity_penalty = -similarity_weight * path['similarity']

        if include_buyable:
            buyable_penalty = -10 * path['buyable_pct']  # Reward if buyable
        else:
            buyable_penalty = 0

        if prefer_flow:
            flow_penalty = -flow_weight * path['flow_ef']  # Reward if positive EF
        else:
            flow_penalty = 0

        accessibility_penalty = -accessibility_weight * path['sa_difference']
        if path['sa_difference'] < 0:
            accessibility_penalty += 10

        ref_score = -ref_weight * min(10, path['refs'])  # Negative because higher ref count is better
        score = -(similarity_penalty + ref_score + buyable_penalty + accessibility_penalty + flow_penalty)
        path['score'] = score
        return -score

    sorted_paths = sorted(all_paths.items(), key=lambda x: score_disconnection(
        path=x[1],
        similarity_weight=similarity_weight,
        ref_weight=ref_weight,
        accessibility_weight=accessibility_weight,
        flow_weight=flow_weight
    ))

    return dict([path for path in sorted_paths if path[1]['score'] > min_score][:max_paths])


def match_solvents(
        route,
        knowledge_graph: KnowledgeGraph,
        green_solvents: bool = False
):

    """
    Match solvents to reaction routes based on knowledge graph and green chemistry considerations.

    Parameters:
    route (dict): The reaction route information.
    knowledge_graph (KnowledgeGraph): Knowledge graph for solvent-reaction matching.
    green_solvents (bool): Whether to prioritize environmentally friendly solvents.

    Returns:
    dict: Suggested solvents and the reactions they match.
    """

    solvent_dict = {}
    rxn_mapper = RXNMapper()

    for key in route["route"]:
        reaction = route["route"][key][0]["reaction_smiles"]
        matching_patterns = route["route"][key][0]["smirks"]
        rxn_analysis = ReactionAnalyzer(
            reaction,
            graph=knowledge_graph,
            matching_templates=matching_patterns,
            rxn_mapper=rxn_mapper
        )
        suggested_conditions = rxn_analysis.find_conditions(green_solvents=green_solvents)
        suggested_solvents = suggested_conditions["solvents"]
        for solvent in suggested_solvents:
            if solvent in solvent_dict:
                solvent_dict[solvent].append(reaction)
            else:
                solvent_dict[solvent] = [reaction]

    return solvent_dict
