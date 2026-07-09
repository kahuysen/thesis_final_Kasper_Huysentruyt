import pandas as pd
from rdkit import Chem
import logging
import numpy as np
from tqdm import tqdm

from rxn_insight.synthesis.utils import get_all_buyables
from rxn_insight.synthesis.reaction_transforms import (
    find_forward_substructure_matches,
    match_reactants,
    predict_products
)
from rxn_insight.utils import get_fp, get_similarity
logger = logging.getLogger(__name__)


class ForwardPredictor:
    """
    A class for predicting possible forward chemical reactions based on a knowledge graph of reactions.

    Attributes:
    knowledge_graph: A knowledge graph containing reaction data.
    graph: The graph structure extracted from the knowledge graph.
    smirks_dict: A dictionary of SMIRKS patterns mapped to reaction metadata.
    buyables: A DataFrame of commercially available molecules and their prices.
    """

    def __init__(self, knowledge_graph):

        """
        Initialize the ForwardPredictor with a knowledge graph.

        Parameters:
        knowledge_graph: A knowledge graph containing reaction data.
        """

        self.knowledge_graph = knowledge_graph
        self.graph = self.knowledge_graph.graph
        self.smirks_dict = self.get_all_reactant_smarts()
        buyables = get_all_buyables()
        self.buyables = pd.DataFrame({"Molecule": buyables.keys(), "Price": buyables.values()})

    def get_all_reactant_smarts(self):

        """
        Extract all unique reactant SMIRKS patterns from the knowledge graph.

        Returns:
        dict: A dictionary where keys are SMIRKS strings and values are reaction metadata.
        """

        smirks_dict = {}

        for node, data in self.graph.nodes(data=True):
            if 'Reaction' in data.get('labels', []) and data.get('smirks') not in smirks_dict:
                smirks = data.get('smirks')
                if smirks:
                    try:
                        rd_smarts = []
                        checked_smarts = []
                        reactant_smarts = smirks.split(">>")[0].split(".")
                        wrong = False
                        for smarts in reactant_smarts:
                            rdkit_mol = Chem.MolFromSmarts(smarts)
                            if rdkit_mol:
                                rd_smarts.append(rdkit_mol)
                                checked_smarts.append(smarts)
                            else:
                                wrong = True

                        if not wrong:
                            reaction_class = data.get('reaction_class')
                            reaction_smiles = data.get('reaction_smiles')
                            smirks_dict[smirks] = {"rd_smarts": rd_smarts,
                                                   "smarts": checked_smarts,
                                                   "reaction_class": reaction_class,
                                                   "reaction_smiles": reaction_smiles}
                    except Exception as e:
                        logging.warning(f"Failed to process SMARTS {smirks}: {e}")

        return smirks_dict

    def get_possible_reactions(self, mol):

        """
        Identify possible reactions for a given molecule.

        Parameters:
        mol (rdkit.Chem.Mol): The molecule to analyze.

        Returns:
        dict: A dictionary of potential SMIRKS patterns that match the molecule.
        """

        potential_smirks = find_forward_substructure_matches(mol, self.smirks_dict)
        return potential_smirks

    def find_matching_smirks_for_two_reactants(self, mol1, mol2):
        """
        Find SMIRKS patterns that match both input molecules as reactants.

        Parameters:
        mol1 (rdkit.Chem.Mol): First reactant molecule
        mol2 (rdkit.Chem.Mol): Second reactant molecule

        Returns:
        list: List of tuples (smirks, reactant_mapping) where reactant_mapping
              indicates which molecule matches which position
        """
        matching_smirks = []

        # Only consider SMIRKS patterns with exactly 2 reactants
        two_reactant_smirks = {
            smirks: data for smirks, data in self.smirks_dict.items()
            if len(data["rd_smarts"]) == 2
        }

        for smirks, data in two_reactant_smirks.items():
            rd_smarts = data["rd_smarts"]

            # Check if mol1 matches first position and mol2 matches second
            if mol1.HasSubstructMatch(rd_smarts[0]) and mol2.HasSubstructMatch(rd_smarts[1]):
                matching_smirks.append((smirks, {0: 0, 1: 1}))  # mol1->pos0, mol2->pos1

            # Check if mol2 matches first position and mol1 matches second
            elif mol2.HasSubstructMatch(rd_smarts[0]) and mol1.HasSubstructMatch(rd_smarts[1]):
                matching_smirks.append((smirks, {0: 1, 1: 0}))  # mol1->pos1, mol2->pos0

        return matching_smirks

    def predict_reaction_outcome(self, reactant1_smiles: str, reactant2_smiles: str,
                                 fingerprint_type: str = "morgan", similarity_metric: str = "jaccard"):
        """
        Predict the outcome of a reaction between two specific reactants.

        Parameters:
        reactant1_smiles (str): SMILES string of the first reactant
        reactant2_smiles (str): SMILES string of the second reactant
        fingerprint_type (str): Type of fingerprint to use for similarity ("morgan" or "maccs")
        similarity_metric (str): Similarity metric to use (default: "jaccard")

        Returns:
        tuple: (products, reactions, reaction_info) where:
            - products: list of predicted product SMILES
            - reactions: list of reaction SMILES
            - reaction_info: list of dictionaries with detailed reaction information
        """
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(reactant1_smiles)
        mol2 = Chem.MolFromSmiles(reactant2_smiles)

        if mol1 is None or mol2 is None:
            logging.error("Invalid SMILES provided")
            return [], [], []

        # Find matching SMIRKS patterns
        matching_patterns = self.find_matching_smirks_for_two_reactants(mol1, mol2)

        if not matching_patterns:
            logging.info("No matching reaction patterns found for these reactants")
            return [], [], []

        products = []
        reactions = []
        reaction_info = []

        for smirks, mapping in matching_patterns:
            # Prepare reactants in correct order based on mapping
            if mapping[0] == 0:  # mol1 is first, mol2 is second
                ordered_reactants = [reactant1_smiles, reactant2_smiles]
            else:  # mol2 is first, mol1 is second
                ordered_reactants = [reactant2_smiles, reactant1_smiles]

            # Create reactant dictionary for predict_products
            reactant_dict = {
                'reactant_1': [ordered_reactants[0]],
                'reactant_2': [ordered_reactants[1]]
            }

            # Predict products using the SMIRKS
            predicted_products, predicted_reactions = predict_products(reactant_dict, smirks)

            # Store results with additional information
            for prod, rxn in zip(predicted_products, predicted_reactions):
                if prod not in products:  # Avoid duplicates
                    products.append(prod)
                    reactions.append(rxn)

                    # Get reaction metadata
                    smirks_data = self.smirks_dict[smirks]

                    # Find the most similar reported reaction
                    related_reactions = self.knowledge_graph.find_related_reactions(smirks)

                    best_similarity = 0.0
                    best_reaction = None
                    best_doi = None
                    best_title = None

                    if related_reactions:
                        # Calculate fingerprint for the predicted reaction
                        pred_fp = get_fp(rxn, fp=fingerprint_type, concatenate=True)

                        for related in related_reactions:
                            # Get the stored fingerprint
                            if fingerprint_type.lower() == "morgan":
                                stored_fp = np.array(list(related["morgan_fp"]), dtype=np.int64)
                            else:  # maccs
                                stored_fp = np.array(list(related["maccs_fp"]), dtype=np.int64)

                            # Calculate similarity
                            similarity = get_similarity(pred_fp, stored_fp, metric=similarity_metric)

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_reaction = related["reaction_smiles"]
                                best_doi = related.get("doi", "")
                                best_title = related.get("title", "")

                    reaction_info.append({
                        'product': prod,
                        'reaction': rxn,
                        'smirks': smirks,
                        'reaction_class': smirks_data.get('reaction_class', 'Unknown'),
                        'most_similar_reaction': best_reaction,
                        'similarity': round(best_similarity, 3),
                        'reference_doi': best_doi,
                        'reference_title': best_title
                    })

        # Sort results by similarity (highest first)
        reaction_info = sorted(reaction_info, key=lambda x: x['similarity'], reverse=True)

        # Reorder products and reactions to match the sorted order
        sorted_products = [info['product'] for info in reaction_info]
        sorted_reactions = [info['reaction'] for info in reaction_info]

        return sorted_products, sorted_reactions, reaction_info

    def predict_possible_reactions(self, smiles: str, price: int = 50):

        """
        Predict possible products and reactions for a given molecule.

        Parameters:
        smiles (str): The SMILES string of the input molecule.
        price (int): The maximum price for commercially available reactants.

        Returns:
        tuple: A list of possible products and a list of possible reactions.
        """

        mol = Chem.MolFromSmiles(smiles)
        possible_smirks = self.get_possible_reactions(mol)
        possible_products = []
        possible_reactions = []

        for smirks in tqdm(possible_smirks):
            matched_reactants = match_reactants(smiles, possible_smirks[smirks], self.buyables, price=price)
            predicted_products, predicted_reactions = predict_products(matched_reactants, smirks)
            possible_products += predicted_products
            possible_reactions += predicted_reactions

        return possible_products, possible_reactions

    def _predict_base_forward_reaction(self, reactant_smiles_list: list,
                                       fingerprint_type: str = "morgan",
                                       similarity_metric: str = "jaccard",
                                       max_products: int = None):
        """
        Predict possible reactions and products for 1, 2, or 3 reactants.

        Parameters:
            reactant_smiles_list (list): List of SMILES strings for reactants (1-3 reactants)
            fingerprint_type (str): Type of fingerprint for similarity ("morgan" or "maccs")
            similarity_metric (str): Similarity metric to use (default: "jaccard")
            max_products (int): Maximum number of products to return (None for all)

        Returns:
            dict: Dictionary containing:
                - 'products': list of predicted product SMILES
                - 'reactions': list of reaction SMILES
                - 'reaction_info': list of detailed reaction information dicts
                - 'summary': summary statistics
        """
        import itertools
        from rdkit.Chem import AllChem

        # Validate input
        if not isinstance(reactant_smiles_list, list):
            reactant_smiles_list = [reactant_smiles_list]

        num_reactants = len(reactant_smiles_list)
        if num_reactants == 0 or num_reactants > 3:
            logging.error(f"Invalid number of reactants: {num_reactants}. Must be 1-3.")
            return {'products': [], 'reactions': [], 'reaction_info': [], 'summary': {}}

        # Convert SMILES to molecules
        reactant_mols = []
        for smiles in reactant_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.error(f"Invalid SMILES: {smiles}")
                return {'products': [], 'reactions': [], 'reaction_info': [], 'summary': {}}
            reactant_mols.append(mol)

        # Find matching SMIRKS patterns based on number of reactants
        matching_smirks = self._find_matching_smirks_patterns(reactant_mols, num_reactants)

        if not matching_smirks:
            logging.info(f"No matching reaction patterns found for {num_reactants} reactant(s)")
            return {'products': [], 'reactions': [], 'reaction_info': [], 'summary': {}}

        # Predict products for each matching SMIRKS
        all_products = []
        all_reactions = []
        all_reaction_info = []

        for smirks_data in matching_smirks:
            smirks = smirks_data['smirks']
            ordered_reactants = smirks_data['ordered_reactants']

            # Run the reaction
            try:
                rxn = AllChem.ReactionFromSmarts(smirks)
                reactant_tuple = tuple(Chem.MolFromSmiles(smi) for smi in ordered_reactants)
                products = rxn.RunReactants(reactant_tuple)

                if products:
                    for product_set in products:
                        # Get product SMILES
                        product_smiles_list = []
                        for prod_mol in product_set:
                            try:
                                prod_mol = Chem.RemoveHs(prod_mol)
                                prod_smiles = Chem.MolToSmiles(prod_mol)
                                product_smiles_list.append(prod_smiles)
                            except:
                                continue

                        if product_smiles_list:
                            product_smiles = '.'.join(product_smiles_list)
                            reaction_smiles = '.'.join(ordered_reactants) + '>>' + product_smiles

                            # Skip duplicates
                            if product_smiles not in all_products:
                                all_products.append(product_smiles)
                                all_reactions.append(reaction_smiles)

                                # Get additional information
                                info = self._get_reaction_info(
                                    reaction_smiles, smirks,
                                    fingerprint_type, similarity_metric
                                )
                                all_reaction_info.append(info)

            except Exception as e:
                logging.debug(f"Failed to run reaction with SMIRKS {smirks}: {e}")
                continue

        # Sort by similarity if available
        if all_reaction_info and 'similarity' in all_reaction_info[0]:
            sorted_indices = sorted(range(len(all_reaction_info)),
                                    key=lambda i: all_reaction_info[i]['similarity'],
                                    reverse=True)
            all_products = [all_products[i] for i in sorted_indices]
            all_reactions = [all_reactions[i] for i in sorted_indices]
            all_reaction_info = [all_reaction_info[i] for i in sorted_indices]

        # Apply max_products limit if specified
        if max_products and len(all_products) > max_products:
            all_products = all_products[:max_products]
            all_reactions = all_reactions[:max_products]
            all_reaction_info = all_reaction_info[:max_products]

        # Create summary
        summary = {
            'num_reactants': num_reactants,
            'num_matching_templates': len(matching_smirks),
            'num_predicted_products': len(all_products),
            'template_classes': list(set(info.get('reaction_class', 'Unknown')
                                         for info in all_reaction_info))
        }

        return {
            'products': all_products,
            'reactions': all_reactions,
            'reaction_info': all_reaction_info,
            'summary': summary
        }

    def _find_matching_smirks_patterns(self, reactant_mols: list, num_reactants: int):
        """
        Find SMIRKS patterns that match the given reactants.

        Parameters:
            reactant_mols (list): List of RDKit molecule objects
            num_reactants (int): Number of reactants (1, 2, or 3)

        Returns:
            list: List of dictionaries with matching SMIRKS and ordered reactants
        """
        import itertools

        matching_patterns = []

        # Filter SMIRKS by number of reactants
        filtered_smirks = {
            smirks: data for smirks, data in self.smirks_dict.items()
            if len(data["rd_smarts"]) == num_reactants
        }

        if num_reactants == 1:
            mol = reactant_mols[0]
            for smirks, data in filtered_smirks.items():
                if mol.HasSubstructMatch(data["rd_smarts"][0]):
                    matching_patterns.append({
                        'smirks': smirks,
                        'ordered_reactants': [Chem.MolToSmiles(mol)],
                        'reaction_class': data.get('reaction_class', 'Unknown')
                    })

        elif num_reactants == 2:
            mol1, mol2 = reactant_mols
            for smirks, data in filtered_smirks.items():
                rd_smarts = data["rd_smarts"]

                # Try both orderings
                if mol1.HasSubstructMatch(rd_smarts[0]) and mol2.HasSubstructMatch(rd_smarts[1]):
                    matching_patterns.append({
                        'smirks': smirks,
                        'ordered_reactants': [Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)],
                        'reaction_class': data.get('reaction_class', 'Unknown')
                    })
                elif mol2.HasSubstructMatch(rd_smarts[0]) and mol1.HasSubstructMatch(rd_smarts[1]):
                    matching_patterns.append({
                        'smirks': smirks,
                        'ordered_reactants': [Chem.MolToSmiles(mol2), Chem.MolToSmiles(mol1)],
                        'reaction_class': data.get('reaction_class', 'Unknown')
                    })

        elif num_reactants == 3:
            # For 3 reactants, we need to check all 6 permutations
            for smirks, data in filtered_smirks.items():
                rd_smarts = data["rd_smarts"]

                # Check all permutations of reactant ordering
                for perm in itertools.permutations(range(3)):
                    if all(reactant_mols[i].HasSubstructMatch(rd_smarts[perm[i]])
                           for i in range(3)):
                        ordered_smiles = [Chem.MolToSmiles(reactant_mols[perm[i]])
                                          for i in range(3)]
                        matching_patterns.append({
                            'smirks': smirks,
                            'ordered_reactants': ordered_smiles,
                            'reaction_class': data.get('reaction_class', 'Unknown')
                        })
                        break  # Found a match, no need to check other permutations

        return matching_patterns

    def _get_reaction_info(self, reaction_smiles: str, smirks: str,
                           fingerprint_type: str, similarity_metric: str):
        """
        Get detailed information about a predicted reaction.

        Parameters:
            reaction_smiles (str): The reaction SMILES
            smirks (str): The SMIRKS pattern used
            fingerprint_type (str): Type of fingerprint for similarity
            similarity_metric (str): Similarity metric to use

        Returns:
            dict: Detailed reaction information
        """
        import numpy as np

        info = {
            'reaction': reaction_smiles,
            'smirks': smirks,
            'reaction_class': self.smirks_dict[smirks].get('reaction_class', 'Unknown')
        }

        # Find similar reactions in knowledge graph
        try:
            related_reactions = self.knowledge_graph.find_related_reactions(smirks)

            if related_reactions:
                # Calculate fingerprint for the predicted reaction
                pred_fp = get_fp(reaction_smiles, fp=fingerprint_type, concatenate=True)

                best_similarity = 0.0
                best_reaction = None
                best_doi = None
                best_title = None

                for related in related_reactions:
                    # Get the stored fingerprint
                    if fingerprint_type.lower() == "morgan":
                        stored_fp = np.array(list(related["morgan_fp"]), dtype=np.int64)
                    else:  # maccs
                        stored_fp = np.array(list(related["maccs_fp"]), dtype=np.int64)

                    # Calculate similarity
                    similarity = get_similarity(pred_fp, stored_fp, metric=similarity_metric)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_reaction = related["reaction_smiles"]
                        best_doi = related.get("doi", "")
                        best_title = related.get("title", "")

                info.update({
                    'most_similar_reaction': best_reaction,
                    'similarity': round(best_similarity, 3),
                    'reference_doi': best_doi,
                    'reference_title': best_title
                })
        except Exception as e:
            logging.debug(f"Could not find related reactions: {e}")

        return info

    def predict_forward_reaction(self, reactant_smiles_list: list,
                                 allowed_atoms: dict = None,
                                 allow_other_atoms: bool = True,
                                 fingerprint_type: str = "morgan",
                                 similarity_metric: str = "jaccard",
                                 max_products: int = None,
                                 check_added_atoms: bool = True):
        """
        Predict possible reactions and products with atom-based filtering.

        Parameters:
            reactant_smiles_list (list): List of SMILES strings for reactants (1-3 reactants)
            allowed_atoms (dict): Dictionary specifying allowed added atoms. Format:
                - None: No filter, return all products
                - {}: Only reactions with NO added atoms
                - {'H': 0}: No H atoms added (H must not be added)
                - {'H': None}: Any amount of H atoms can be added (but H must be added)
                - {'H': 2}: Exactly 2 H atoms must be added
                - {'H': [0, 2]}: Either no H or exactly 2 H atoms added
                - {'H': [1, 2, 3]}: 1, 2, or 3 H atoms must be added
                - {'H': 2, 'O': 1}: Exactly 2 H and 1 O must be added
                - {'H': [0, 2], 'O': [0, 1]}: Combinations allowed
            allow_other_atoms (bool): Whether to allow atoms not specified in allowed_atoms
                - True: Other atoms can be added alongside specified atoms (default)
                - False: ONLY the specified atoms can be added (strict mode)
            fingerprint_type (str): Type of fingerprint for similarity ("morgan" or "maccs")
            similarity_metric (str): Similarity metric to use (default: "jaccard")
            max_products (int): Maximum number of products to return (None for all)
            check_added_atoms (bool): Whether to check added atoms (set False for speed)

        Returns:
            dict: Dictionary containing:
                - 'products': list of predicted product SMILES
                - 'reactions': list of reaction SMILES
                - 'reaction_info': list of detailed reaction information dicts
                - 'summary': summary statistics
                - 'filtered_out': dict of products filtered out with reasons (if filter applied)
        """

        # First, get all predictions without filtering
        base_result = self._predict_base_forward_reaction(
            reactant_smiles_list,
            fingerprint_type,
            similarity_metric,
            max_products=None  # Don't limit yet, we'll filter first
        )

        # If no filter specified or no products, return as is
        if allowed_atoms is None or not base_result['products'] or not check_added_atoms:
            if max_products and len(base_result['products']) > max_products:
                base_result['products'] = base_result['products'][:max_products]
                base_result['reactions'] = base_result['reactions'][:max_products]
                base_result['reaction_info'] = base_result['reaction_info'][:max_products]
            return base_result

        # Apply atom-based filtering
        filtered_products = []
        filtered_reactions = []
        filtered_info = []
        filtered_out = {}

        for idx, (product, reaction, info) in enumerate(zip(
                base_result['products'],
                base_result['reactions'],
                base_result['reaction_info']
        )):
            # Analyze the reaction to get added atoms
            added_atoms = self._get_added_atoms(reaction)

            # Check if this reaction passes the filter
            passes_filter, reason = self._check_atom_filter(added_atoms, allowed_atoms, allow_other_atoms)

            if passes_filter:
                filtered_products.append(product)
                filtered_reactions.append(reaction)
                # Add the added atoms info to the reaction info
                info['added_atoms'] = added_atoms
                filtered_info.append(info)
            else:
                filtered_out[reaction] = {
                    'product': product,
                    'added_atoms': added_atoms,
                    'filter_reason': reason
                }

        # Apply max_products limit after filtering
        if max_products and len(filtered_products) > max_products:
            filtered_products = filtered_products[:max_products]
            filtered_reactions = filtered_reactions[:max_products]
            filtered_info = filtered_info[:max_products]

        # Update summary
        summary = base_result['summary'].copy()
        summary['total_predicted'] = len(base_result['products'])
        summary['after_filter'] = len(filtered_products)
        summary['filtered_out'] = len(filtered_out)
        summary['filter_criteria'] = allowed_atoms
        summary['strict_mode'] = not allow_other_atoms if allowed_atoms is not None else None

        return {
            'products': filtered_products,
            'reactions': filtered_reactions,
            'reaction_info': filtered_info,
            'summary': summary,
            'filtered_out': filtered_out
        }

    def _get_added_atoms(self, reaction_smiles: str):
        """
        Analyze a reaction to determine which atoms were added.

        Parameters:
            reaction_smiles (str): Reaction SMILES string

        Returns:
            dict: Dictionary with atom symbols as keys and counts as values
                  Empty dict {} means no atoms were added
        """
        try:
            from rxn_insight import Reaction

            # Create Reaction object with hydrogen tracking
            rxn = Reaction(reaction_smiles, include_hydrogens=True)

            # Get the explanation which includes added atoms
            explanation = rxn.explain()

            # Parse added atoms from the explanation
            added_atoms_dict = {}

            if 'added_atoms' in explanation and explanation['added_atoms']:
                # The format is like: ["Atom types H,F,F, with atom indices 18,19,20"]
                for added_str in explanation['added_atoms']:
                    if 'Atom types' in added_str:
                        # Extract the atom symbols - everything between "Atom types" and "with atom indices"
                        atoms_part = added_str.split('Atom types')[1].split('with atom indices')[0].strip()
                        # Remove trailing comma and spaces
                        atoms_part = atoms_part.rstrip(', ')
                        atoms_list = atoms_part.split(',')

                        # Count each atom type
                        for atom in atoms_list:
                            atom = atom.strip()
                            if atom:
                                if atom not in added_atoms_dict:
                                    added_atoms_dict[atom] = 0
                                added_atoms_dict[atom] += 1

            return added_atoms_dict

        except Exception as e:
            logging.debug(f"Could not analyze added atoms for reaction {reaction_smiles}: {e}")
            # Fall back to comparing atom counts
            return self._get_added_atoms_fallback(reaction_smiles)

    def _get_added_atoms_fallback(self, reaction_smiles: str):
        """
        Fallback method to get added atoms by comparing atom counts.

        Parameters:
            reaction_smiles (str): Reaction SMILES string

        Returns:
            dict: Dictionary with atom symbols as keys and counts as values
                  Empty dict {} means no atoms were added
        """
        try:
            reactants_str, products_str = reaction_smiles.split('>>')

            # Count atoms in reactants
            reactant_atoms = {}
            for reactant_smiles in reactants_str.split('.'):
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    mol = Chem.AddHs(mol)  # Add implicit hydrogens
                    for atom in mol.GetAtoms():
                        symbol = atom.GetSymbol()
                        reactant_atoms[symbol] = reactant_atoms.get(symbol, 0) + 1

            # Count atoms in products
            product_atoms = {}
            for product_smiles in products_str.split('.'):
                mol = Chem.MolFromSmiles(product_smiles)
                if mol:
                    mol = Chem.AddHs(mol)  # Add implicit hydrogens
                    for atom in mol.GetAtoms():
                        symbol = atom.GetSymbol()
                        product_atoms[symbol] = product_atoms.get(symbol, 0) + 1

            # Calculate added atoms (products - reactants)
            added_atoms = {}
            for atom, count in product_atoms.items():
                reactant_count = reactant_atoms.get(atom, 0)
                if count > reactant_count:
                    added_atoms[atom] = count - reactant_count

            return added_atoms

        except Exception as e:
            logging.debug(f"Fallback atom counting failed: {e}")
            return {}

    def _check_atom_filter(self, added_atoms: dict, allowed_atoms: dict, allow_other_atoms: bool = True):
        """
        Check if the added atoms match the filter criteria.

        Parameters:
            added_atoms (dict): Dictionary of actually added atoms {symbol: count}
            allowed_atoms (dict): Filter specification
            allow_other_atoms (bool): Whether to allow atoms not specified in allowed_atoms

        Returns:
            tuple: (passes_filter: bool, reason: str)
        """
        # Special case: empty dict means NO atoms should be added
        if allowed_atoms == {}:
            if added_atoms == {}:
                return True, "No atoms added (as required)"
            else:
                return False, f"Atoms were added but none should be: {added_atoms}"

        # Check each specified atom type in the filter
        for atom_symbol, allowed_count in allowed_atoms.items():
            actual_count = added_atoms.get(atom_symbol, 0)

            if allowed_count is None:
                # Any amount of this atom is allowed (but must be > 0)
                if actual_count == 0:
                    return False, f"No {atom_symbol} atoms were added (but some required)"

            elif isinstance(allowed_count, int):
                # Exact count required (including 0)
                if actual_count != allowed_count:
                    return False, f"Expected exactly {allowed_count} {atom_symbol}, got {actual_count}"

            elif isinstance(allowed_count, (list, tuple)):
                # Count must be in the specified range
                if actual_count not in allowed_count:
                    return False, f"Expected {atom_symbol} count in {allowed_count}, got {actual_count}"

        # Strict mode: check if any unexpected atoms were added
        if not allow_other_atoms:
            for atom_symbol in added_atoms:
                if atom_symbol not in allowed_atoms:
                    return False, f"Unexpected atom {atom_symbol} was added (strict mode)"

        return True, "Passes filter"
