import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import euclidean
from rxn_insight.synthesis.analysis import ReactionAnalyzer
from rxn_insight.synthesis.utils import load_chem21_data, score_solvent, load_gsk_ssg_data, score_solvent_gsk
from rxn_insight.synthesis.knowledge_graph import KnowledgeGraph
from typing import Optional, Union


class SolventSelector:
    """
    A tool for intelligent solvent selection that combines reaction analysis
    with solvent similarity search and green chemistry considerations.
    """

    def __init__(
            self,
            knowledge_graph: KnowledgeGraph,
            solvent_property_df: Optional[pd.DataFrame] = None,
            include_temperatures: bool = False
    ):
        """
        Initialize the SolventSelector.

        Args:
            knowledge_graph: The reaction knowledge graph
            solvent_property_df: DataFrame with solvent properties including:
                - smiles, name, Tb, Volume, sig2, sig3, Hb_acc2, Hb_don2,
                  Hb_acc3, Hb_don3, cLogP, cRI
            include_temperatures: Whether to include reaction temperatures
        """
        self.kg = knowledge_graph
        if solvent_property_df is not None:
            self.solvent_df = solvent_property_df
        else:
            from importlib import resources
            with resources.path(f"rxn_insight.data", "ssg_coeffs.csv") as path:
                self.solvent_df = pd.read_csv(path, index_col=0)

        from importlib import resources
        with resources.path(f"rxn_insight.data", "hsp_coeffs.csv") as path:
            self.hsp_df = pd.read_csv(path, index_col=0)

        self.chem21_data = load_chem21_data()
        self.gsk_data = load_gsk_ssg_data()
        self.rxn_mapper = None  # Initialize if needed
        self.include_temperatures = include_temperatures
        self.temperature_data = load_solvent_temperatures()

    def retrieve_most_similar_solvent(self, smi: str, n_alternatives: int = 10, ssg: str = "chem21"):
        """
        Find the most similar solvents based on physicochemical properties.

        Args:
            smi: SMILES string of the reference solvent
            n_alternatives: Number of alternatives to return
            ssg: Solvent selection guide to use ("chem21" or "gsk")

        Returns:
            DataFrame with similar solvents ranked by similarity
        """
        smi = Chem.CanonSmiles(smi)

        # Check if solvent is in database
        if smi not in self.solvent_df["smiles"].values:
            print(f"Warning: Solvent {smi} not found in property database")
            return pd.DataFrame()

        solvent_idx = self.solvent_df[self.solvent_df["smiles"] == smi].index[0]
        solvent_coeff = self.solvent_df.loc[solvent_idx][
            ['Tb', 'Volume', 'sig2', 'sig3', 'Hb_acc2', 'Hb_don2',
             'Hb_acc3', 'Hb_don3', 'cLogP', 'cRI']
        ].to_numpy()

        dists = []
        for i in self.solvent_df.index:
            coeffs = self.solvent_df.loc[i][
                ['Tb', 'Volume', 'sig2', 'sig3', 'Hb_acc2', 'Hb_don2',
                 'Hb_acc3', 'Hb_don3', 'cLogP', 'cRI']
            ].to_numpy()
            dists.append(euclidean(solvent_coeff, coeffs))

        dist_df = self.solvent_df.copy()
        dist_df["similarity_score"] = dists

        # Add green chemistry scores based on selected SSG
        dist_df["green_score"] = dist_df["smiles"].apply(
            lambda x: self._get_green_score(x, ssg=ssg)
        )

        # Check which columns are available and ensure we have names
        columns_to_select = ["smiles", "similarity_score", "green_score"]
        if "name" in dist_df.columns:
            # Use existing name column
            result = dist_df[["smiles", "name", "similarity_score", "green_score"]] \
                         .sort_values(by="similarity_score") \
                         .iloc[1:n_alternatives + 1]  # Skip the first one (itself)
        else:
            # Look up names from other databases
            result = dist_df[columns_to_select] \
                         .sort_values(by="similarity_score") \
                         .iloc[1:n_alternatives + 1]  # Skip the first one (itself)
            result["name"] = result["smiles"].apply(self._get_solvent_name)
            # Reorder columns
            result = result[["smiles", "name", "similarity_score", "green_score"]]

        return result

    def retrieve_most_similar_solvent_hansen(self, smi: str, n_alternatives: int = 10, ssg: str = "chem21"):
        """
        Find the most similar solvents based on Hansen solubility parameters.

        Args:
            smi: SMILES string of the reference solvent
            n_alternatives: Number of alternatives to return
            ssg: Solvent selection guide to use ("chem21" or "gsk")

        Returns:
            DataFrame with similar solvents ranked by similarity
        """
        smi = Chem.CanonSmiles(smi)

        # Check if solvent is in database
        if smi not in self.hsp_df["smiles"].values:
            print(f"Warning: Solvent {smi} not found in Hansen database")
            return pd.DataFrame()

        solvent_idx = self.hsp_df[self.hsp_df["smiles"] == smi].index[0]
        solvent_coeff = self.hsp_df.loc[solvent_idx][
            ['hsp_d', 'hsp_p', 'hsp_h']
        ].to_numpy()

        dists = []
        for i in self.hsp_df.index:
            coeffs = self.hsp_df.loc[i][
                ['hsp_d', 'hsp_p', 'hsp_h']
            ].to_numpy()
            dists.append(euclidean(solvent_coeff, coeffs))

        dist_df = self.hsp_df.copy()
        dist_df["similarity_score"] = dists

        # Add green chemistry scores based on selected SSG
        dist_df["green_score"] = dist_df["smiles"].apply(
            lambda x: self._get_green_score(x, ssg=ssg)
        )

        # Check which columns are available
        columns_to_select = ["smiles", "similarity_score", "green_score"]
        if "name" in dist_df.columns:
            columns_to_select.insert(1, "name")

        # Sort by similarity and return top alternatives (excluding the reference)
        result = dist_df[columns_to_select] \
                     .sort_values(by="similarity_score") \
                     .iloc[1:n_alternatives + 1]  # Skip the first one (itself)

        # If name column wasn't present, look up names from other databases
        if "name" not in result.columns:
            result["name"] = result["smiles"].apply(self._get_solvent_name)

        # Ensure name column is in the correct position
        column_order = ["smiles", "name", "similarity_score", "green_score"]
        result = result[column_order]

        return result

    def _get_green_score(self, smiles: str, ssg: str = "chem21") -> Union[str, int]:
        """
        Get green chemistry score for a solvent.

        Args:
            smiles: SMILES string of the solvent
            ssg: Solvent selection guide ("chem21" or "gsk")

        Returns:
            For CHEM21: String ("Recommended", "Problematic", "Hazardous", "unknown")
            For GSK: Integer (1-10, where 10 is best)
        """
        if ssg.lower() == "gsk":
            score_dict = score_solvent_gsk(self.gsk_data, smiles)
            if score_dict['scores']['status'] == 'unknown':
                return 0  # Return 0 for unknown solvents in GSK
            return score_dict['scores']['total_score']
        else:  # default to chem21
            score_dict = score_solvent(self.chem21_data, smiles)
            return score_dict['solvent_score']

    def _get_solvent_name(self, smiles: str) -> str:
        """
        Get solvent name from SMILES by checking multiple databases.

        Args:
            smiles: SMILES string of the solvent

        Returns:
            Solvent name if found, otherwise returns the SMILES string
        """
        try:
            canonical_smiles = Chem.CanonSmiles(smiles)
        except:
            canonical_smiles = smiles

        # Check CHEM21 database
        if canonical_smiles in self.chem21_data['solvent_smiles']:
            return self.chem21_data['solvent_smiles'][canonical_smiles]

        # Check GSK database
        if canonical_smiles in self.gsk_data['solvent_smiles']:
            return self.gsk_data['solvent_smiles'][canonical_smiles]

        # Check main solvent property dataframe
        if hasattr(self, 'solvent_df') and 'name' in self.solvent_df.columns:
            match = self.solvent_df[self.solvent_df['smiles'] == canonical_smiles]
            if not match.empty:
                return match.iloc[0]['name']

        # Return SMILES if no name found
        return smiles

    def analyze_reaction_and_suggest_solvents(
            self,
            reaction_smiles: str,
            n_alternatives: int = 5,
            prefer_green: bool = False,
            similarity_threshold: float = 0.3,
            similarity_weight: float = 1.0,
            template_search: bool = False,
            broad_search: bool = False,
            ssg: str = "chem21",
            temperature: float = 298.15,
            ignore_most_popular: bool = True,
            distance_method: str = "descriptors",
    ):
        """
        Analyze a reaction, find the most popular solvent, and suggest alternatives.
        Three search modes are available: by tag (default), broad tag, and template.
        Setting the template search to True will overrule the tag mode even if broad_search is True.

        Args:
            ssg: Choose the solvent selection guide (options: "chem21", "gsk")
            temperature: Reaction temperature (default: 298.15 K)
            reaction_smiles: Reaction SMILES string
            n_alternatives: Number of alternative solvents to suggest
            prefer_green: Whether to prioritize green solvents (default False)
            similarity_threshold: Minimum similarity for condition matching
            similarity_weight: Weight for similarity-based scoring. Defaults to 1.0.
            template_search: Whether to search via matching templates (default False)
            broad_search: Whether to search via TAG2 (default False)
            ignore_most_popular: Whether to ignore most popular solvents and only search alternatives (default True)
            distance_method: Method for calculating solvent similarity ("descriptors" or "hsp")

        Returns:
            Dictionary with analysis results and solvent recommendations
        """
        # Analyze the reaction
        analyzer = ReactionAnalyzer(
            reaction_smiles,
            graph=self.kg,
            fingerprint_type="morgan"
        )

        if template_search:
            # Find conditions including solvents using the correct method
            conditions = analyzer.find_conditions_by_template(
                green_solvents=False,
                similarity_threshold=similarity_threshold,
            )
        else:
            # Find conditions using tag-based search
            conditions = analyzer.find_conditions_by_tag(
                green_solvents=False,
                similarity_weight=similarity_weight,
                similarity_threshold=similarity_threshold,
                broad_search=broad_search
            )

        if not conditions['solvent']:
            return {
                "status": "no_conditions_found",
                "message": "No reaction conditions found in the knowledge graph"
            }

        # Get the most popular solvent
        most_popular_solvent = None
        highest_score = 0
        message = ""

        for solvent, data in conditions['solvent'].items():
            if solvent == "solvent-free":
                continue
            elif self.include_temperatures:
                all_solvents = data['smiles'].split(".")
                not_liquid = False
                bps = []
                for s in all_solvents:
                    s = Chem.CanonSmiles(s)
                    if s not in self.temperature_data:
                        continue
                    else:
                        if temperature > self.temperature_data[s]["bp"] or temperature < self.temperature_data[s]["mp"]:
                            not_liquid = True
                            if not ignore_most_popular:
                                break
                            else:
                                bps.append(str(self.temperature_data[s]["bp"]))
                        else:
                            bps.append(str(self.temperature_data[s]["bp"]))

                if not_liquid and not ignore_most_popular:
                    continue
                else:
                    bp = ";".join(bps)

            if data['similarity'] > highest_score:
                highest_score = data['similarity']
                most_popular_solvent = {
                    'name': solvent,
                    'smiles': data['smiles'],
                    'score': data['similarity'],
                    'hazard_level': data['hazard_level'] if ssg == "chem21" else self._get_green_score(data['smiles'],
                                                                                                       ssg=ssg)
                }
                if self.include_temperatures:
                    most_popular_solvent["boiling_point"] = bp

        if not most_popular_solvent:
            return {
                "status": "no_solvent_found",
                "message": "No solvent recommendations found"
            }
        elif "solvent-free" in conditions['solvent']:
            if conditions["solvent"]["solvent-free"]["similarity"] > highest_score:
                message = "The highest similarity score is found for working solvent-free."

        # Find similar solvents
        if distance_method.lower() == "hsp" or distance_method.lower() == "hansen":
            alternatives = self.retrieve_most_similar_solvent_hansen(
                most_popular_solvent['smiles'],
                n_alternatives=n_alternatives * 5,  # Get more to filter
                ssg=ssg
            )
        else:
            alternatives = self.retrieve_most_similar_solvent(
                most_popular_solvent['smiles'],
                n_alternatives=n_alternatives * 5,  # Get more to filter
                ssg=ssg
            )

        if self.include_temperatures:
            alternatives_to_drop = []
            for i in alternatives.index:
                all_solvents = alternatives['smiles'][i].split(".")
                not_liquid = False
                bps = []
                for s in all_solvents:
                    s = Chem.CanonSmiles(s)
                    if s not in self.temperature_data:
                        continue
                    else:
                        if temperature > self.temperature_data[s]["bp"] or temperature < self.temperature_data[s]["mp"]:
                            not_liquid = True
                            break
                        else:
                            bps.append(str(self.temperature_data[s]["bp"]))

                if not_liquid:
                    alternatives_to_drop.append(i)
                else:
                    alternatives.loc[i, "boiling_point"] = ";".join(bps)

            alternatives = alternatives.drop(alternatives_to_drop)

        # Filter alternatives based on green chemistry preference
        if prefer_green and not alternatives.empty:
            if ssg.lower() == "gsk":
                # For GSK, prioritize scores >= 7 (Good to Best)
                green_alternatives = alternatives[
                    alternatives['green_score'] >= 7
                    ].head(n_alternatives)

                # If not enough green alternatives, add ones with scores 5-6
                if len(green_alternatives) < n_alternatives:
                    fair = alternatives[
                        (alternatives['green_score'] >= 5) & (alternatives['green_score'] < 7)
                        ].head(n_alternatives - len(green_alternatives))
                    alternatives = pd.concat([green_alternatives, fair])
                else:
                    alternatives = green_alternatives
            else:  # CHEM21
                # Prioritize "Recommended" solvents
                green_alternatives = alternatives[
                    alternatives['green_score'] == 'Recommended'
                    ].head(n_alternatives)

                # If not enough green alternatives, add "Problematic" ones
                if len(green_alternatives) < n_alternatives:
                    problematic = alternatives[
                        alternatives['green_score'] == 'Problematic'
                        ].head(n_alternatives - len(green_alternatives))
                    alternatives = pd.concat([green_alternatives, problematic])
                else:
                    alternatives = green_alternatives
        else:
            alternatives = alternatives.head(n_alternatives)

        # Check which alternatives have been used for similar reactions
        alternatives_with_precedent = []
        for _, alt in alternatives.iterrows():
            reactions_with_solvent = self._find_reactions_with_solvent(
                analyzer.tag if not broad_search else analyzer.give_broad_tag(),
                alt['smiles']
            )

            # Get name, using SMILES as fallback if name column doesn't exist
            alt_name = alt.get('name', alt['smiles'])

            alt_dict = {
                'name': alt_name,
                'smiles': alt['smiles'],
                'solvent_distance': alt['similarity_score'],
                'green_score': alt['green_score'],
                'precedent_count': len(reactions_with_solvent),
                'has_precedent': len(reactions_with_solvent) > 0
            }

            if self.include_temperatures and 'boiling_point' in alt:
                alt_dict['boiling_point'] = alt['boiling_point']

            alternatives_with_precedent.append(alt_dict)

        return {
            "status": "success",
            "message": message,
            "ssg_used": ssg.upper(),
            "reaction": {
                "smiles": reaction_smiles,
                "class": analyzer.reaction_info.get('CLASS', 'Unknown'),
                "name": analyzer.reaction_info.get('NAME', 'Unknown')
            },
            "most_popular_solvent": most_popular_solvent,
            "alternative_solvents": alternatives_with_precedent,
            "recommendation": self._generate_recommendation(
                most_popular_solvent,
                alternatives_with_precedent,
                ssg=ssg
            )
        }

    def _find_reactions_with_solvent(self, reaction_tag: str, solvent_smiles: str):
        """Find reactions with similar tag that use a specific solvent."""
        # Find reactions with the same tag
        similar_reactions = self.kg.find_reactions_by_tag(reaction_tag, broad_search=False)

        reactions_with_solvent = []
        for reaction in similar_reactions:
            if reaction.get('solvent_smiles') == solvent_smiles:
                reactions_with_solvent.append(reaction)

        return reactions_with_solvent

    def _generate_recommendation(self, popular_solvent, alternatives, ssg: str = "chem21"):
        """Generate a text recommendation based on the analysis."""
        if ssg.lower() == "gsk":
            # For GSK, green means score >= 7
            green_alternatives = [
                alt for alt in alternatives
                if alt['green_score'] >= 7 and alt['has_precedent']
            ]

            if green_alternatives:
                best_green = min(green_alternatives, key=lambda x: x['solvent_distance'])
                return (
                    f"Consider replacing {popular_solvent['name']} "
                    f"(GSK score: {popular_solvent['hazard_level']}/10) with {best_green['name']} "
                    f"(GSK score: {best_green['green_score']}/10, "
                    f"{best_green['precedent_count']} precedents). "
                    f"This alternative has similar properties (distance: "
                    f"{best_green['solvent_distance']:.3f}) and better sustainability profile."
                )
            else:
                return (
                    f"The most popular solvent {popular_solvent['name']} "
                    f"(GSK score: {popular_solvent['hazard_level']}/10) is currently used. "
                    f"Alternative solvents with similar properties have been identified, "
                    f"but may require experimental validation."
                )
        else:  # CHEM21
            green_alternatives = [
                alt for alt in alternatives
                if alt['green_score'] == 'Recommended' and alt['has_precedent']
            ]

            if green_alternatives:
                best_green = min(green_alternatives, key=lambda x: x['solvent_distance'])
                return (
                    f"Consider replacing {popular_solvent['name']} "
                    f"({popular_solvent['hazard_level']}) with {best_green['name']} "
                    f"(Recommended, {best_green['precedent_count']} precedents). "
                    f"This alternative has similar properties (distance: "
                    f"{best_green['solvent_distance']:.3f}) and better environmental profile."
                )
            else:
                return (
                    f"The most popular solvent {popular_solvent['name']} is currently used. "
                    f"Alternative solvents with similar properties have been identified, "
                    f"but may require experimental validation."
                )


def load_solvent_temperatures() -> dict:
    from importlib import resources
    with resources.path(f"rxn_insight.data", "solvent_temperatures.csv") as path:
        temp_df = pd.read_csv(path, index_col=0)

    temp_dict = {}
    for i in temp_df.index:
        temp_dict[temp_df["smiles"][i]] = {
            "bp": temp_df["bp"][i],
            "mp": temp_df["mp"][i],
        }

    return temp_dict
