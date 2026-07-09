import panel as pn
import anthropic
from panel.template import BootstrapTemplate
import rxn_insight as ri
from rxn_insight.utils import draw_chemical_reaction
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
import io
import base64
import os
import re
import time
from datetime import datetime
import pandas as pd
import numpy as np
from rxnmapper import RXNMapper

df_uspto = None
kg = None
rxn_mapper = None
database_loaded = False
kg_loaded = False
chat_bot_name = "Lucien"
client = None

DEFAULT_CONFIG = {
    # API settings
    "temperature": 0.7,
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 1000,

    # File paths
    "uspto_path": "/data/mdobb/ord/data/df_ord.gzip",
    "knowledge_graph_path": "/data/mdobb/FlowChemPy/src/flowchempy/data/flow_knowledge_graph.gpickle",

    # UI settings
    "port": 44429,

    # Retrosynthesis parameters
    "max_paths": 10,
    "max_depth": 1,
    "similarity_threshold": 0.0,
    "min_disconnection_score": 5.0,
    "similarity_weight": 200.0,
    "ref_weight": 1.0,
    "flow_weight": 0.0,
    "accessibility_weight": -1.0,
    "max_price": 200.0,

    # Analysis parameters
    "fingerprint_type": "morgan",
    "similarity_metric": "sokalsneath",
    "reaction_fingerprint": False,
    "prefer_flow": False
}

CONFIG = DEFAULT_CONFIG.copy()


def load_client(api_key: str):
    global df_uspto, kg, rxn_mapper, database_loaded, kg_loaded, client

    # Load the database using the configured path
    try:
        df_uspto = pd.read_parquet(CONFIG["uspto_path"])
        database_loaded = True
        print("USPTO database loaded successfully")
    except Exception as e:
        print(f"Error loading USPTO database: {str(e)}")
        database_loaded = False

    # Load the knowledge graph using the configured path
    try:
        kg = ri.load_graph_from_file(CONFIG["knowledge_graph_path"])
        kg_loaded = True
        print("Knowledge graph loaded successfully")
    except Exception as e:
        print(f"Error loading knowledge graph: {str(e)}")
        kg_loaded = False

    # Initialize RXNMapper
    try:
        rxn_mapper = RXNMapper()
        print("RXNMapper initialized successfully")
    except Exception as e:
        print(f"Error initializing RXNMapper: {str(e)}")
        rxn_mapper = None

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    return {
        "df_uspto": df_uspto,
        "kg": kg,
        "rxn_mapper": rxn_mapper,
        "database_loaded": database_loaded,
        "kg_loaded": kg_loaded,
        "client": client
    }


def setup_panel():
    css = """
    /* Import Lora font with multiple weights */
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700;900&display=swap');

    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }

    /* Special styling for first message */
    .first-message {
        font-size: 18px !important;
        font-family: 'Lora', Georgia, serif !important;
        font-weight: 500;
    }

    .message {
        margin-bottom: 15px;
        padding: 15px 20px;
        border-radius: 18px;
        max-width: 80%;
        position: relative;
        line-height: 1.6;
        font-family: 'Lora', Georgia, serif;
        font-size: 16px;
    }

    /* Fix for nested content inside messages */
    .message p {
        margin-bottom: 12px;
    }

    .message ul, .message ol {
        margin-left: 20px;
        margin-bottom: 12px;
    }

    .message li {
        margin-bottom: 8px;
        display: list-item;
    }

    /* Make the reaction image integrate better with message */
    .reaction-image {
        text-align: center;
        margin: 15px 0;
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    .message .reaction-image img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }

    .user-message {
        background-color: #e9e9eb;
        color: #000;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }

    .bot-message {
        background-color: #647253;
        color: white;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }

    .message-time {
        font-size: 0.7em;
        margin-top: 5px;
        opacity: 0.7;
    }

    .message-container {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    .thinking-indicator {
        font-style: italic;
        color: #888;
        margin-left: 10px;
    }

    .svg-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .chat-title {
        text-align: center;
        font-size: 26px;
        margin-bottom: 20px;
        color: #647253;
        font-weight: bold;
        font-family: 'Lora', Georgia, serif;
    }

    .input-container {
        display: flex;
        margin-top: 20px;
    }

    .input-field {
        flex-grow: 1;
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #ddd;
        outline: none;
    }

    .send-button {
        background-color: #647253;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        margin-left: 10px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .smiles-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    .reaction-image {
        text-align: center;
        margin: 15px 0;
    }

    .assistant-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #647253;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        margin-right: 10px;
        font-family: 'Lora', Georgia, serif;
        font-size: 22px;
        letter-spacing: -1px;
    }

    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #e9e9eb;
        color: #333;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-left: 10px;
    }

    .message-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }

    .input-with-button {
        display: flex;
        align-items: center;
        width: 100%;
    }

    .chemical-formula {
        font-family: monospace;
        background-color: rgba(255, 255, 255, 0.2);
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 0.95em;
    }

    .welcome-message {
        text-align: left;
        margin: 20px 0;
        padding: 20px 25px;
        background-color: #f5f7f5;
        border-radius: 10px;
        border-left: 4px solid #647253;
        font-family: 'Lora', Georgia, serif;
        color: #333;
        font-size: 16px;
    }

    .welcome-message h2 {
        font-family: 'Lora', Georgia, serif;
        color: #3A5A40;
        margin-bottom: 15px;
        font-weight: 700;
        font-size: 32px;
    }

    .welcome-message ul {
        margin-left: 15px;
        margin-bottom: 15px;
    }

    .welcome-message li {
        margin-bottom: 8px;
    }

    .code-block {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        overflow-x: auto;
        border-left: 3px solid #0b93f6;
    }

    .table-container {
        margin: 15px 0;
        overflow-x: auto;
    }

    .chem-table {
        width: 100%;
        border-collapse: collapse;
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .chem-table th, .chem-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }

    .chem-table th {
        background-color: #f5f7f5;
        font-weight: 600;
        color: #3A5A40;
    }

    .chem-table tr:hover {
        background-color: #f9f9f9;
    }

    /* Heatmap styling */
    .heatmap-cell {
        text-align: center;
        border: 1px solid white;
    }

    /* New style elements for better formatting */
    .highlight-text {
        font-weight: bold;
        color: #f0f0f0;
    }

    .emphasis-text {
        font-style: italic;
    }

    .chem-link {
        color: #a8d5ba;
        text-decoration: underline;
        word-break: break-all;
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
        color: #f0f0f0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 5px;
    }

    .reaction-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }

    .reaction-details {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .reaction-property {
        display: flex;
    }

    .property-label {
        font-weight: 600;
        min-width: 120px;
    }

    .property-value {
        flex: 1;
    }

    .reference-section {
        font-size: 0.9em;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px dashed rgba(255, 255, 255, 0.2);
    }
    """
    pn.extension(raw_css=[css])
    

def generate_reaction_plot(reaction_smiles):
    try:
        # Parse the reaction
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        
        # Get the image from RDKit
        img = Draw.ReactionToImage(rxn, subImgSize=(300, 200))
        
        # Convert to SVG
        buffer = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Convert to base64 for embedding
        data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create HTML image tag
        return f'<div class="reaction-image"><img src="data:image/png;base64,{data}" /></div>'
    except Exception as e:
        return f'<div class="error">Error generating reaction visualization: {str(e)}</div>'
    

def visualize_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f'<div class="error">Invalid SMILES string: {smiles}</div>'
        
        # Generate image
        img = Draw.MolToImage(mol, size=(300, 200))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'<div class="reaction-image"><img src="data:image/png;base64,{data}" /></div>'
    except Exception as e:
        return f'<div class="error">Error visualizing molecule: {str(e)}</div>'


def perform_retrosynthesis(smiles):
    try:
        # Create SynthesisTree object
        st = ri.SynthesisTree(
            graph=kg,
            max_paths=CONFIG["max_paths"],
            max_depth=CONFIG["max_depth"],
            similarity_threshold=CONFIG["similarity_threshold"],
            min_disconnection_score=CONFIG["min_disconnection_score"],
            similarity_weight=CONFIG["similarity_weight"],
            ref_weight=CONFIG["ref_weight"],
            flow_weight=CONFIG["flow_weight"],
            accessibility_weight=CONFIG["accessibility_weight"],
            max_price=CONFIG["max_price"],
            fingerprint=CONFIG["fingerprint_type"],
            similarity_metric=CONFIG["similarity_metric"],
            reaction_fingerprint=CONFIG["reaction_fingerprint"],
            prefer_flow=CONFIG["prefer_flow"],
        )
        
        # Perform retrosynthesis
        results = st.predict_single_step(smiles)
        
        if not results:
            return None, "No retrosynthetic disconnections found for this molecule."
        
        # Find the best disconnection (highest score)
        best_disconnection_id = max(results, key=lambda x: results[x]['score'])
        best_disconnection = results[best_disconnection_id]
        
        # Format results for Claude
        result_text = f"Target molecule: {smiles}\n\n"
        result_text += f"Found {len(results)} potential disconnections.\n\n"
        result_text += f"Best disconnection is {best_disconnection_id} with score {results[best_disconnection_id]['score']:.2f}\n\n"
        
        # First add the best disconnection
        result_text += f"Best Disconnection ({best_disconnection_id}):\n"
        result_text += f"Score: {results[best_disconnection_id]['score']:.2f}\n"
        result_text += f"Reactants: {', '.join(results[best_disconnection_id]['reactants'])}\n"
        result_text += f"Reaction class: {results[best_disconnection_id]['reaction_class']}\n"
        result_text += f"Reaction type: {results[best_disconnection_id]['reaction_type']}\n"
        result_text += f"Similarity: {results[best_disconnection_id]['similarity']:.2f}\n"
        result_text += f"Reference: {results[best_disconnection_id]['closest_reference']}\n\n"
        
        # Then add other top disconnections
        other_top_ids = [id for id in results if id != best_disconnection_id]
        other_top_ids = sorted(other_top_ids, key=lambda x: results[x]['score'], reverse=True)[:4]
        
        for i, child_id in enumerate(other_top_ids):
            child_data = results[child_id]
            result_text += f"Alternative {i+1} ({child_id}):\n"
            result_text += f"Score: {child_data['score']:.2f}\n"
            result_text += f"Reactants: {', '.join(child_data['reactants'])}\n"
            result_text += f"Reaction class: {child_data['reaction_class']}\n"
            result_text += f"Reaction type: {child_data['reaction_type']}\n"
            result_text += f"Similarity: {child_data['similarity']:.2f}\n"
            result_text += f"Reference: {child_data['closest_reference']}\n\n"
        
        # Generate reaction visualization
        reaction_viz = None
        try:
            if 'reaction_smiles' in best_disconnection:
                reaction_viz = generate_reaction_plot(best_disconnection['reaction_smiles'])
        except Exception as e:
            print(f"Error in visualization: {e}")
        
        return results, result_text, reaction_viz, best_disconnection_id
    except Exception as e:
        return None, f"Error performing retrosynthesis: {str(e)}", None, None
    

def analyze_reaction(reaction_smiles):
    try:
        # Create Reaction object
        rxn = ri.Reaction(reaction_smiles, rxn_mapper=rxn_mapper)
        
        # Get reaction information
        info = rxn.get_reaction_info()
        
        # Format results
        result_text = f"Reaction: {reaction_smiles}\n\n"
        result_text += f"Classification: {info['CLASS']}\n"
        result_text += f"Name: {info['NAME']}\n\n"
        
        result_text += f"Number of reactants: {info['N_REACTANTS']}\n"
        result_text += f"Number of products: {info['N_PRODUCTS']}\n\n"
        
        result_text += "Functional groups in reactants: " + ", ".join(info['FG_REACTANTS']) + "\n"
        result_text += "Functional groups in products: " + ", ".join(info['FG_PRODUCTS']) + "\n\n"
        
        if info['PARTICIPATING_RINGS_REACTANTS']:
            result_text += "Participating rings in reactants: " + ", ".join(info['PARTICIPATING_RINGS_REACTANTS']) + "\n"
        if info['PARTICIPATING_RINGS_PRODUCTS']:
            result_text += "Participating rings in products: " + ", ".join(info['PARTICIPATING_RINGS_PRODUCTS']) + "\n\n"
        
        if info['BY-PRODUCTS']:
            result_text += "Predicted by-products: " + ", ".join(info['BY-PRODUCTS']) + "\n\n"
        
        # Generate reaction visualization
        reaction_viz = generate_reaction_plot(reaction_smiles)
        
        return info, result_text, reaction_viz
    except Exception as e:
        return None, f"Error analyzing reaction: {str(e)}", None
    
    
def suggest_reaction_conditions(reaction_smiles):
    try:
        if df_uspto is None:
            return None, "Reaction database not loaded. Cannot suggest conditions."
            
        # Create Reaction object
        rxn = ri.Reaction(reaction_smiles)
        
        # Get suggested conditions
        conditions = rxn.suggest_conditions(df_uspto)
        print(conditions)
        
        # Format results
        result_text = f"Suggested conditions for: {reaction_smiles}\n\n"
        
        # Generate reaction visualization
        reaction_viz = generate_reaction_plot(reaction_smiles)
        
        # Format top solvents
        solvent_df = rxn.suggested_solvent
        solvent_html = "<div class='table-container'><table class='chem-table'>"
        solvent_html += "<tr><th>Solvent</th><th>Count</th></tr>"
        for i, row in solvent_df.head(5).iterrows():
            solvent_html += f"<tr><td>{row['NAME']}</td><td>{row['COUNT']}</td></tr>"
        solvent_html += "</table></div>"
        
        # Format top catalysts
        catalyst_df = rxn.suggested_catalyst
        catalyst_html = "<div class='table-container'><table class='chem-table'>"
        catalyst_html += "<tr><th>Catalyst</th><th>Count</th></tr>"
        for i, row in catalyst_df.head(5).iterrows():
            catalyst_html += f"<tr><td>{row['NAME']}</td><td>{row['COUNT']}</td></tr>"
        catalyst_html += "</table></div>"
        
        # Format top reagents
        reagent_df = rxn.suggested_reagent
        reagent_html = "<div class='table-container'><table class='chem-table'>"
        reagent_html += "<tr><th>Reagent</th><th>Count</th></tr>"
        for i, row in reagent_df.head(5).iterrows():
            reagent_html += f"<tr><td>{row['NAME']}</td><td>{row['COUNT']}</td></tr>"
        reagent_html += "</table></div>"
        
        result_text += f"Top recommended solvent: {conditions['Solvent']}\n"
        result_text += f"Top recommended catalyst: {conditions['Catalyst']}\n"
        result_text += f"Top recommended reagent: {conditions['Reagent']}\n\n"
        
        result_text += f"<h4>Top Solvents:</h4>{solvent_html}\n"
        result_text += f"<h4>Top Catalysts:</h4>{catalyst_html}\n"
        result_text += f"<h4>Top Reagents:</h4>{reagent_html}\n"
        
        return conditions, result_text, reaction_viz
    except Exception as e:
        return None, f"Error suggesting conditions: {str(e)}", None
    
    
def analyze_molecule(smiles):
    try:
        # Create Molecule object
        mol = ri.Molecule(smiles, allow_pubchem=True)
        
        # Get functional groups
        functional_groups = mol.get_functional_groups()
        
        # Get rings
        rings = mol.get_rings()
        
        # Get scaffold
        scaffold = mol.scaffold
        
        # Get description
        description = mol.description
        
        # Get IUPAC name
        iupac = mol.iupac_name
        
        # Get trivial name
        trivial = mol.trivial_name
        
        # Format results
        result_text = f"Analysis of molecule: {smiles}\n\n"
        
        if functional_groups:
            result_text += "Functional groups: " + ", ".join(functional_groups) + "\n\n"
        else:
            result_text += "No functional groups detected.\n\n"
        
        if rings:
            result_text += "Ring systems: " + ", ".join(rings) + "\n\n"
        else:
            result_text += "No ring systems detected.\n\n"
        
        if scaffold:
            result_text += f"Molecular scaffold: {scaffold}\n\n"
            result_text += "Scaffold visualization:\n"
            scaffold_viz = visualize_molecule(scaffold)
            result_text += scaffold_viz
            
        if description:
            result_text += f"\n\nDescription from PubChem: {description}\n\n"
            
        if iupac:
            result_text += f"\n\nIUPAC name: {iupac}\n\n"
            
        if trivial:
            result_text += f"\n\nTrivial name: {trivial}\n\n"
        
        # Generate molecule visualization
        mol_viz = visualize_molecule(smiles)
        
        return {
            "functional_groups": functional_groups,
            "rings": rings,
            "scaffold": scaffold
        }, result_text, mol_viz
    except Exception as e:
        return None, f"Error analyzing molecule: {str(e)}", None
    
    
def predict_forward_reactions(smiles):
    try:
        if not kg:
            return None, "Knowledge graph not loaded. Cannot predict forward reactions."
            
        # Create ForwardPredictor object
        predictor = ri.ForwardPredictor(kg)
        
        # Predict possible products
        possible_products, possible_reactions = predictor.predict_possible_reactions(smiles, price=50)
        
        if not possible_products:
            return None, "No forward reactions predicted for this molecule."
        
        # Format results
        result_text = f"Forward reaction predictions for: {smiles}\n\n"
        result_text += f"Found {len(possible_products)} potential products\n\n"
        
        # Add top 5 predictions
        for i, (product, reaction) in enumerate(zip(possible_products[:5], possible_reactions[:5])):
            result_text += f"Prediction {i+1}:\n"
            result_text += f"Product: {product}\n"
            result_text += f"Reaction: {reaction}\n"
            
            # Add visualization
            try:
                reaction_viz = generate_reaction_plot(reaction)
                result_text += reaction_viz + "\n"
            except:
                pass
        
        # Generate molecule visualization
        mol_viz = visualize_molecule(smiles)
        
        return {
            "products": possible_products,
            "reactions": possible_reactions
        }, result_text, mol_viz
    except Exception as e:
        return None, f"Error predicting forward reactions: {str(e)}", None
    

def calculate_molecular_similarity(smiles1, smiles2):
    try:
        # Create Molecule objects
        mol1 = ri.Molecule(smiles1)
        
        # Calculate similarity
        similarity = mol1.calculate_similarity(smiles2)
        
        # Format results
        result_text = f"Similarity between molecules:\n"
        result_text += f"Molecule 1: {smiles1}\n"
        result_text += f"Molecule 2: {smiles2}\n\n"
        result_text += f"Tanimoto similarity: {similarity:.3f}\n\n"
        
        # Generate visualizations
        mol1_viz = visualize_molecule(smiles1)
        mol2_viz = visualize_molecule(smiles2)
        
        result_text += "Molecule 1:\n" + mol1_viz + "\n"
        result_text += "Molecule 2:\n" + mol2_viz + "\n"
        
        return similarity, result_text
    except Exception as e:
        return None, f"Error calculating similarity: {str(e)}"
    
    
def extract_smiles(text):
    """Extract and validate SMILES strings from text input.
    Uses both pattern matching and RDKit validation."""
    
    # Check for common SMILES query patterns
    query_patterns = [
        r'(?:SMILES[:\s]+)([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)',
        r'(?:molecule[:\s]+)([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)',
        r'(?:structure[:\s]+)([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)',
        r'(?:synthesize[\s\w]+)([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)'
    ]
    
    for pattern in query_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                mol = Chem.MolFromSmiles(match)
                if mol is not None:
                    print(f"Found SMILES by query pattern: {match}")
                    return match
            except:
                continue
    
    # Try to find quoted strings that might be SMILES
    quoted_pattern = r'["\']([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)["\']'
    matches = re.findall(quoted_pattern, text)
    for match in matches:
        if len(match) >= 5:  # Minimum reasonable SMILES length
            try:
                mol = Chem.MolFromSmiles(match)
                if mol is not None:
                    print(f"Found SMILES in quotes: {match}")
                    return match
            except:
                continue
    
    # Last resort - look for any SMILES-like patterns
    # Look for strings with typical SMILES characteristics
    smiles_pattern = r'[A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]{5,}'
    matches = re.findall(smiles_pattern, text)
    
    # Sort by length (descending) to prioritize longer matches
    matches.sort(key=len, reverse=True)
    
    # Try to validate matches with RDKit
    for match in matches:
        # Skip common English words and patterns unlikely to be SMILES
        if match.lower() in ['smiles', 'molecule', 'structure', 'compound', 'synthesize', 'retrosynthesis']:
            continue
            
        try:
            mol = Chem.MolFromSmiles(match)
            if mol is not None:
                atom_count = mol.GetNumAtoms()
                # Ensure it's a reasonable molecule (at least 3 atoms)
                if atom_count >= 3:
                    print(f"Found SMILES by pattern: {match} (atoms: {atom_count})")
                    return match
        except Exception as e:
            print(f"Failed to parse potential SMILES: {match}, error: {str(e)}")
            continue
    
    print("No valid SMILES found in text")
    return None


def extract_reaction_smiles(text):
    """Extract a reaction SMILES from text."""
    
    # Look for reaction SMILES patterns (containing >>)
    pattern = r'([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+>>[A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            # Validate reaction SMILES
            rxn = AllChem.ReactionFromSmarts(match, useSmiles=True)
            if rxn is not None:
                print(f"Found reaction SMILES: {match}")
                return match
        except:
            continue
    
    # Look for quoted strings that might be reaction SMILES
    quoted_pattern = r'["\']([A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+>>[A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]+)["\']'
    matches = re.findall(quoted_pattern, text)
    for match in matches:
        try:
            rxn = AllChem.ReactionFromSmarts(match, useSmiles=True)
            if rxn is not None:
                print(f"Found reaction SMILES in quotes: {match}")
                return match
        except:
            continue
    
    return None


def extract_two_smiles(text):
    """Extract two SMILES strings for comparison."""
    
    # Extract all potential SMILES
    smiles_pattern = r'[A-Za-z0-9@\[\]\(\)\{\}/\\\.=#$%^&*+!,\-]{5,}'
    matches = re.findall(smiles_pattern, text)
    
    valid_smiles = []
    for match in matches:
        # Skip common English words and patterns unlikely to be SMILES
        if match.lower() in ['smiles', 'molecule', 'structure', 'compound', 'synthesize', 'retrosynthesis', 'compare', 'similarity']:
            continue
            
        try:
            mol = Chem.MolFromSmiles(match)
            if mol is not None:
                atom_count = mol.GetNumAtoms()
                # Ensure it's a reasonable molecule (at least 3 atoms)
                if atom_count >= 3:
                    valid_smiles.append(match)
                    if len(valid_smiles) == 2:
                        return valid_smiles
        except:
            continue
    
    return None if len(valid_smiles) < 2 else valid_smiles


def parse_query(text):
    """Determine the type of chemistry query and extract relevant information."""
    
    text_lower = text.lower()
    
    # Check for retrosynthesis query
    retro_keywords = [
        'retrosynthesis', 'how to make', 'how would you synthesize', 'how can i synthesize', 
        'synthesize this molecule', 'synthesis route', 'synthetic route', 'synthesis of', 
        'synthesize', 'synthesis for', 'synthesis path', 'route to', 'route for',
        'make this compound', 'prepare this', 'preparation of', 'how to synthesize',
        'help me synthesize', 'help me make'
    ]
    
    # First check for explicit retrosynthesis keywords
    if any(keyword in text_lower for keyword in retro_keywords):
        smiles = extract_smiles(text)
        if smiles:
            return 'retrosynthesis', smiles
            
    # Check for synthesis-related phrases
    synthesis_patterns = [
        r'synth\w+\s+.{0,20}for',  # synthesis, synthetic, etc. + for
        r'synth\w+\s+.{0,20}of',   # synthesis of...
        r'synth\w+\s+.{0,20}route', # synthesis route
        r'route\s+.{0,20}synth\w+', # route to synthesis
        r'mak\w+\s+.{0,20}compound', # make this compound
        r'prepar\w+\s+.{0,20}compound', # prepare this compound
        r'find\w+\s+.{0,20}synth\w+', # finding a synthesis
    ]
    
    if any(re.search(pattern, text_lower) for pattern in synthesis_patterns):
        smiles = extract_smiles(text)
        if smiles:
            return 'retrosynthesis', smiles
    
    # Check for reaction analysis query
    analysis_keywords = [
        'analyze this reaction', 'reaction analysis', 'classify this reaction', 'analyze reaction',
        'what type of reaction', 'identify this reaction', 'reaction classification',
        'name this reaction', 'characterize this reaction', 'explain this reaction',
        'what reaction is this', 'tell me about this reaction'
    ]
    if any(keyword in text_lower for keyword in analysis_keywords):
        reaction_smiles = extract_reaction_smiles(text)
        if reaction_smiles:
            return 'reaction_analysis', reaction_smiles
    
    # Check for condition suggestion query
    condition_keywords = [
        'suggest conditions', 'reaction conditions', 'best solvent', 'best catalyst', 'best reagent',
        'optimal conditions', 'recommend conditions', 'what conditions', 'suitable conditions',
        'good solvent', 'appropriate catalyst', 'which solvent', 'which catalyst', 'which reagent'
    ]
    condition_patterns = [
        r'condition\w*\s+.{0,20}for',   # conditions for...
        r'solvent\w*\s+.{0,20}for',     # solvents for... 
        r'catalyst\w*\s+.{0,20}for',    # catalysts for...
        r'reagent\w*\s+.{0,20}for',     # reagents for...
        r'how\s+.{0,20}perform',        # how to perform...
        r'run\s+.{0,20}reaction'        # run this reaction
    ]
    
    if any(keyword in text_lower for keyword in condition_keywords) or any(re.search(pattern, text_lower) for pattern in condition_patterns):
        reaction_smiles = extract_reaction_smiles(text)
        if reaction_smiles:
            return 'condition_suggestion', reaction_smiles
    
    # Check for molecule analysis query
    molecule_keywords = [
        'analyze this molecule', 'molecule analysis', 'analyze structure', 'functional groups',
        'properties of', 'characterize this molecule', 'identify functional groups',
        'analyze compound', 'molecular structure', 'tell me about this molecule',
        'structure of', 'analyze this compound', 'molecule properties'
    ]
    if any(keyword in text_lower for keyword in molecule_keywords):
        smiles = extract_smiles(text)
        if smiles:
            return 'molecule_analysis', smiles
    
    # Check for forward reaction prediction query
    forward_keywords = [
        'forward prediction', 'predict products', 'forward synthesis', 'possible reactions',
        'what products', 'what can I make', 'possible transformations', 'predict what happens',
        'reaction products', 'what will form', 'products of reaction', 'what can be made',
        'what is produced', 'what will this produce'
    ]
    if any(keyword in text_lower for keyword in forward_keywords):
        smiles = extract_smiles(text)
        if smiles:
            return 'forward_prediction', smiles
    
    # Check for similarity calculation query
    similarity_keywords = [
        'similarity', 'compare molecules', 'how similar', 'molecular similarity',
        'structural comparison', 'how different', 'compare structures',
        'resemblance between', 'alike are', 'similarity between', 'difference between'
    ]
    if any(keyword in text_lower for keyword in similarity_keywords):
        smiles_pair = extract_two_smiles(text)
        if smiles_pair:
            return 'similarity_calculation', smiles_pair
    
    # Context-based fallback detection for SMILES strings
    
    # First extract any SMILES or reaction SMILES
    smiles = extract_smiles(text)
    reaction_smiles = extract_reaction_smiles(text)
    
    # If query mentions "synthesis" or "route" but we didn't catch it above, prioritize retrosynthesis
    synthesis_words = ['synthesis', 'synthesize', 'synthesise', 'route', 'make', 'prepare']
    if any(word in text_lower for word in synthesis_words) and smiles:
        return 'retrosynthesis', smiles
    
    # If query contains "reaction" word with a SMILES (not reaction SMILES), guess forward prediction
    if 'reaction' in text_lower and smiles and not reaction_smiles:
        return 'forward_prediction', smiles
    
    # If a reaction SMILES is detected but query type is unclear, assume reaction analysis
    if reaction_smiles:
        # If asking about conditions, temperatures, etc.
        condition_hints = ['condition', 'solvent', 'catalyst', 'reagent', 'temperature', 'yield']
        if any(hint in text_lower for hint in condition_hints):
            return 'condition_suggestion', reaction_smiles
        else:
            return 'reaction_analysis', reaction_smiles
    
    # If a single SMILES is detected but query type is unclear, assume molecule analysis as last resort
    if smiles:
        return 'molecule_analysis', smiles
    
    # Default to general query
    return 'general', None


def clean_claude_formatting(text):
    """Converts Claude's markdown formatting to HTML for better display."""
    try:
        # First, handle code/SMILES formatting
        text = re.sub(r'`([^`]+)`', r'<span class="chemical-formula">\1</span>', text)
        
        # Format section headers
        text = re.sub(r'# ([^\n]+)', r'<div class="section-title">\1</div>', text)
        text = re.sub(r'## ([^\n]+)', r'<div class="subsection-title">\1</div>', text)
        
        # Handle bold and italic text
        text = re.sub(r'\*\*([^*]+)\*\*', r'<span class="highlight-text">\1</span>', text)
        text = re.sub(r'\*([^*]+)\*', r'<span class="emphasis-text">\1</span>', text)
        
        # CONVERT BULLET POINTS TO NORMAL TEXT
        # Replace bullet points with paragraph breaks instead
        text = re.sub(r'• ', r'', text)  # Remove bullet markers
        text = re.sub(r'- ', r'', text)  # Remove dash bullets
        
        # Basic paragraph formatting
        text = re.sub(r'\n\n', r'<br><br>', text)
        
        # SIMPLIFY LINK HANDLING
        # Completely remove any existing link tags to avoid nesting issues
        text = re.sub(r'<a[^>]*>([^<]*)</a>', r'\1', text)
        
        # Then create clean links for DOIs and URLs
        # For DOIs, process them once and correctly
        if "Reference:" in text:
            # Extract the reference section
            ref_parts = text.split("Reference:", 1)
            if len(ref_parts) > 1:
                main_content = ref_parts[0]
                ref_content = ref_parts[1]
                
                # Clean up DOI references
                ref_content = re.sub(r'doi\.org/([^\s<>"]+)', r'<a href="https://doi.org/\1" target="_blank" class="chem-link">doi.org/\1</a>', ref_content)
                
                # Put it back together
                text = main_content + "Reference: " + ref_content
        
        # Format reaction cards with simplified approach
        if 'Top Disconnection' in text or 'Recommended Conditions' in text:
            if '<div class="section-title">' in text:
                parts = text.split('<div class="section-title">', 1)
                if len(parts) > 1:
                    header = parts[0]
                    content = '<div class="section-title">' + parts[1]
                    text = f'{header}<div class="reaction-card">{content}</div>'
        
        return text
    except Exception as e:
        # Fallback - if any error occurs, return original text
        print(f"Error in clean_claude_formatting: {str(e)}")
        return text
    
    
def clean_input_for_claude(text):
    """Removes SVG/visualization content from text to reduce token usage."""
    # Remove entire reaction-image div containers and their contents
    text = re.sub(r'<div class="reaction-image">.*?</div>', '[VISUALIZATION]', text, flags=re.DOTALL)
    
    # Remove any remaining image tags
    text = re.sub(r'<img.*?/>', '[IMAGE]', text, flags=re.DOTALL)
    
    # Remove SVG elements
    text = re.sub(r'<svg.*?</svg>', '[SVG]', text, flags=re.DOTALL)
    
    return text


def process_with_claude(user_query, conversation_history):
    try:
        # Determine query type and extract relevant information
        query_type, extracted_data = parse_query(user_query)
        
        # Initialize variables
        results = None
        formatted_results = None
        visualization = None
        
        # Process the query based on its type
        if query_type == 'retrosynthesis' and extracted_data:
            smiles = extracted_data
            results, raw_results, visualization, best_disconnection_id = perform_retrosynthesis(smiles)
            # Add the raw results to the user query for Claude to process
            if raw_results:
                user_query += f"\n\n[RETROSYNTHESIS RESULTS]\n{raw_results}"
                
        elif query_type == 'reaction_analysis' and extracted_data:
            reaction_smiles = extracted_data
            results, formatted_results, visualization = analyze_reaction(reaction_smiles)
            # Add the formatted results to the user query for Claude to process
            if formatted_results:
                user_query += f"\n\n[REACTION ANALYSIS RESULTS]\n{formatted_results}"
                
        elif query_type == 'condition_suggestion' and extracted_data:
            reaction_smiles = extracted_data
            results, formatted_results, visualization = suggest_reaction_conditions(reaction_smiles)
            # Add the formatted results to the user query for Claude to process
            if formatted_results:
                user_query += f"\n\n[CONDITION SUGGESTION RESULTS]\n{formatted_results}"
                
        elif query_type == 'molecule_analysis' and extracted_data:
            smiles = extracted_data
            results, formatted_results, visualization = analyze_molecule(smiles)
            # Add the formatted results to the user query for Claude to process
            if formatted_results:
                user_query += f"\n\n[MOLECULE ANALYSIS RESULTS]\n{formatted_results}"
                
        elif query_type == 'forward_prediction' and extracted_data:
            smiles = extracted_data
            results, formatted_results, visualization = predict_forward_reactions(smiles)
            # Add the formatted results to the user query for Claude to process
            if formatted_results:
                user_query += f"\n\n[FORWARD REACTION PREDICTION RESULTS]\n{formatted_results}"
                
        elif query_type == 'similarity_calculation' and extracted_data:
            smiles1, smiles2 = extracted_data
            similarity, formatted_results = calculate_molecular_similarity(smiles1, smiles2)
            # Add the formatted results to the user query for Claude to process
            if formatted_results:
                user_query += f"\n\n[SIMILARITY CALCULATION RESULTS]\n{formatted_results}"
        
        # Format the conversation history for Claude
        messages = []
        for entry in conversation_history:
            role = "user" if entry["sender"] == "You" else "assistant"
            # Clean SVG/visualization content from the text before adding to messages
            cleaned_text = clean_input_for_claude(entry["text"])
            messages.append({"role": role, "content": cleaned_text})
        
        # Add the current query
        messages.append({"role": "user", "content": user_query})
        
        # Define system prompt based on query type
        if query_type == 'retrosynthesis' and results:
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in retrosynthesis and organic chemistry. 
            You are talking to a chemistry researcher who is using a retrosynthesis planning system.
            
            You've just analyzed their molecule and have retrosynthesis results to discuss. Be concise and organize your response clearly:
            Limit yourself to the information that is provided to you. Only make suggestions if you are very sure.
            # Retrosynthesis
            
            Begin with a clear title indicating this is retrosynthesis for the target molecule (include name if known).
            
            **Top Disconnection**:  
            - Summarize the main transformation (2-3 sentences)
            - Reactants needed
            - Reaction type
            
            **Key Advantages**:
            - List 2-3 reasons about why this approach is good
            - Include the confidence score and similarity if available
            
            **Alternatives**:
            - Note any promising alternative approaches
            - Why they might be considered
            
            **Reference**:
            For the reference, simply include "Reference: doi.org/[DOI number]" as plain text, without any special formatting.
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.

            """
        elif query_type == 'reaction_analysis':
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in reaction analysis and classification.
            You are talking to a chemistry researcher who wants to understand the details of a chemical reaction.
            
            You've just analyzed their reaction and have the detailed information. Organize your response as follows:
            
            # Reaction Analysis
            
            **Classification**: 
            - Reaction class and common name
            - Brief mechanistic insight
            
            **Transformation Details**:
            - Key functional group changes 
            - Bond formation/breaking
            
            **Structure Information**:
            - Ring systems or scaffolds involved
            - Expected by-products
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            """
        elif query_type == 'condition_suggestion':
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in optimizing reaction conditions.
            You are talking to a chemistry researcher who wants suggestions for conditions for their reaction.
            
            You've just analyzed similar reactions from the literature and have suggestions to share. Organize your response as follows:
            
            # Recommended Conditions
            
            **Optimal Conditions**:
            - Solvent: [recommended solvent] - brief justification
            - Catalyst: [recommended catalyst] - brief justification  
            - Reagent: [recommended reagent] - brief justification
            
            **Rationale**:
            - Why these conditions work well together
            - Key considerations for this reaction type
            
            **Alternatives**:
            - Other condition sets worth considering
            - When they might be preferred
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            """
        elif query_type == 'molecule_analysis':
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in molecular analysis.
            You are talking to a chemistry researcher who wants to understand the properties of a molecule.
            
            You've just analyzed their molecule and have detailed information. Organize your response as follows:
            
            # Molecular Analysis
            
            **Structure Overview**:
            - Molecular class and key features
            - General reactivity profile
            
            **Functional Groups**:
            - List each functional group
            - Brief note on each group's reactivity
            
            **Scaffold and Rings**:
            - Describe the core scaffold
            - Ring systems present
            
            **Key Properties**:
            - Structure-based insights
            - Potential synthetic considerations
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            """
        elif query_type == 'forward_prediction':
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in forward synthesis prediction.
            You are talking to a chemistry researcher who wants to know what products could be made from their molecule.
            
            You've just analyzed potential forward reactions. Organize your response as follows:
            
            # Forward Reaction Prediction
            
            **Most Promising Transformations**:
            - Top 2-3 predicted products
            - Reaction types involved
            
            **Reaction Details**:
            - Key mechanistic considerations
            - Expected selectivity
            
            **Practical Considerations**:
            - Reagent recommendations
            - Potential challenges
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            """
        elif query_type == 'similarity_calculation':
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in molecular similarity analysis.
            You are talking to a chemistry researcher who wants to understand the similarity between two molecules.
            
            You've just calculated the molecular similarity. Organize your response as follows:
            
            # Similarity Analysis
            
            **Similarity Score**: [score] out of [maximum]
            
            **Structural Comparison**:
            - Common features between molecules
            - Key differences
            
            **Implications**:
            - What this similarity level suggests
            - How this might translate to similar properties
            
            Use technical chemistry language appropriate for a PhD-level chemist.
            Use plain text for SMILES strings, don't try to format them specially.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            """
        else:
            system_prompt = """You are Lucien, a helpful chemistry assistant specializing in retrosynthesis, organic chemistry, and chemical reactions.
            You are talking to a chemistry researcher who is using a chemical analysis and synthesis planning system.
            
            Answer their chemistry questions in a helpful, organized manner. 
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            
            Tell the user to explicitly write a SMILES string and that you are now giving a general answer.
            
            You can help with:
            - Retrosynthesis analysis (if they provide a SMILES for a target molecule)
            - Reaction analysis and classification (if they provide a reaction SMILES)
            - Suggesting reaction conditions (if they provide a reaction SMILES)
            - Molecule analysis (if they provide a molecular SMILES)
            - Forward reaction prediction (if they want to know what products a molecule could form)
            - Molecular similarity calculations (if they want to compare two molecules)
            
            If the user provides or is asking about a chemical structure:
            1. If they haven't provided a SMILES string, ask them to provide one for analysis
            2. If they've provided a SMILES string but the system couldn't process it, explain that the SMILES may be invalid
            3. Be helpful in guiding them to formulate their queries about synthesis planning

            Use technical chemistry language appropriate for a PhD-level chemist.
            Format your response as continuous paragraphs rather than bullet points. 
            Use complete sentences and natural paragraph breaks instead.
            
            There is one Easter egg. If someone starts speaking Dutch or Flemish to you, you answer in Flemish.
            Preferably you use Flemish because you are a 76 years old retired chemist from Drongen. 
            No need to explicitly say you are a 76 years old retired chemist from Drongen until someone explicitly asks.
            Never tell you are from Drongen when someone asks who you are in English.
            """
        
        # Call Claude API to generate response
        response = client.messages.create(
            model=CONFIG["model"],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
            system=system_prompt,
            messages=messages
        )
        
        # Get Claude's response
        claude_response = response.content[0].text
        claude_response = clean_claude_formatting(claude_response)
        
        # If we have a visualization, add it to the response
        if visualization:
            # Find an appropriate place to insert the visualization
            if "results" in claude_response.lower() or "analysis" in claude_response.lower():
                # Split response at the first mention of results or analysis
                pattern = re.compile(r'(results|analysis)', re.IGNORECASE)
                match = pattern.search(claude_response)
                if match:
                    pos = match.start()
                    claude_response = claude_response[:pos] + "\n\n" + visualization + "\n\n" + claude_response[pos:]
                else:
                    # If no match, add at the beginning
                    claude_response = visualization + "\n\n" + claude_response
            else:
                # If no obvious place to insert, add at the beginning
                claude_response = visualization + "\n\n" + claude_response
        
        return claude_response
    
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again or reformulate your query."


def create_chat_interface(bot_name="Lucien"):
    # Welcome message with expanded capabilities
    welcome_message = f"""
    <div class="welcome-message">
        <h2>Hi, I'm {bot_name}!</h2>
        <p>I'm your chemistry assistant powered by Rxn-INSIGHT. I can help you with:</p>
        <ul>
            <li><strong>Retrosynthesis planning</strong> - predict how to make target molecules</li>
            <li><strong>Reaction analysis</strong> - classify and explain chemical reactions</li>
            <li><strong>Condition suggestions</strong> - recommend solvents, catalysts, and reagents</li>
            <li><strong>Molecule analysis</strong> - identify functional groups, rings, and scaffolds</li>
            <li><strong>Forward synthesis</strong> - predict products from a starting molecule</li>
            <li><strong>Molecular similarity</strong> - compare two molecules</li>
        </ul>
        <p>Try asking: <br>
        "How would you synthesize this molecule? CC(=O)Nc1ccc(O)cc1" <br>
        "Analyze this reaction: OB(O)c1ccccc1.Brc1ccccc1>>c1ccc(-c2ccccc2)cc1" <br>
        "Suggest conditions for: OB(O)c1ccccc1.Brc1ccccc1>>c1ccc(-c2ccccc2)cc1" <br>
        "Analyze this molecule: c1ccc(-c2ccccc2)cc1" <br>
        "Predict possible products from: c1ccccc1B(O)O" <br>
        "Compare the similarity between c1ccccc1 and Cc1ccccc1"</p>
    </div>
    """
    
    welcome_pane = pn.pane.HTML(welcome_message, css_classes=["welcome-container"])
    
    # Container for chat messages
    message_container = pn.Column(welcome_pane, css_classes=["message-container"], sizing_mode="stretch_width")
    
    # Input field and send button
    input_field = pn.widgets.TextAreaInput(placeholder="Ask me about chemistry...", 
                                          css_classes=["input-field"], sizing_mode="stretch_width", rows=2)
    
    send_button = pn.widgets.Button(name="Send", button_type="primary", css_classes=["send-button"])
    
    # Creating a conversation history list
    conversation_history = []
    
    # Function to add message to the chat
    def add_message(sender, text, time=None, is_first_message=False):
        if time is None:
            time = datetime.now().strftime("%H:%M")
        
        # Create message container with avatar
        try:
            if sender == "You":
                avatar = pn.pane.HTML(f'<div class="user-avatar">You</div>')
                message = pn.pane.HTML(
                    f'<div class="message user-message">{text}<div class="message-time">{time}</div></div>',
                    css_classes=["message-bubble"]
                )
                message_row = pn.Row(message, avatar, css_classes=["message-row"], sizing_mode="stretch_width")
            else:
                avatar = pn.pane.HTML(f'<div class="assistant-avatar">{bot_name[0]}</div>')
                formatted_text = clean_claude_formatting(text)
                # Apply special styling to first message from bot
                if is_first_message:
                    message = pn.pane.HTML(
                        f'<div class="message bot-message first-message">{formatted_text}<div class="message-time">{time}</div></div>',
                        css_classes=["message-bubble"]
                    )
                else:
                    message = pn.pane.HTML(
                        f'<div class="message bot-message">{formatted_text}<div class="message-time">{time}</div></div>',
                        css_classes=["message-bubble"]
                    )
                    
                message_row = pn.Row(avatar, message, css_classes=["message-row"], sizing_mode="stretch_width")
        except Exception as e:
            # Fallback plain message if any error occurs
            message_row = pn.pane.HTML(f'<div>{sender}: {text}</div>')
        
        # Add to message container
        message_container.append(message_row)
        
        # Update conversation history
        conversation_history.append({"sender": sender, "text": text, "time": time})
    
    # Function to handle sending a message
    def send_message(event):
        try:
            user_text = input_field.value.strip()
            if not user_text:
                return
            
            # Add user message to chat
            add_message("You", user_text)
            
            # Clear input field
            input_field.value = ""
            
            # Add simple "thinking" indicator text
            thinking = pn.pane.HTML(
                f'<div class="message-row"><div style="margin-left: 50px; color: #888; font-style: italic;">{bot_name} is thinking...</div></div>'
            )
            message_container.append(thinking)
            
            # Get bot response (with a slight delay to show the thinking indicator)
            time.sleep(0.5)  # Brief delay to show thinking indicator
            bot_response = process_with_claude(user_text, conversation_history)
            
            # Remove thinking indicator
            try:
                message_container.remove(thinking)
            except:
                # If removal fails, just continue
                pass
            
            # Add bot response
            add_message(bot_name, bot_response)
        except Exception as e:
            # Add error message if something goes wrong
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            add_message(bot_name, error_msg)
    
    # Link the button to the send_message function
    send_button.on_click(send_message)
    
    # Create the input container
    input_container = pn.Row(
        input_field, send_button, 
        css_classes=["input-with-button"], 
        sizing_mode="stretch_width"
    )
    
    # Add an initial message from the bot (marked as first message for special styling)
    add_message(bot_name, f"Hello! I'm {bot_name}, your chemistry assistant powered by Rxn-INSIGHT. How can I help you today?", is_first_message=True)
    
    # Layout the interface
    chat_interface = pn.Column(
        pn.pane.HTML(f'<div class="chat-title">{bot_name} - Chemistry Assistant</div>'),
        message_container,
        input_container,
        sizing_mode="stretch_width",
        css_classes=["chat-container"]
    )
    
    return chat_interface


class ChemicalAssistant:
    def __init__(self, api_key: str, bot_name: str = "Lucien", **kwargs):
        global CONFIG
        CONFIG.update(kwargs)

        setup_panel()
        load_client(api_key=api_key)
        self.bot_name = bot_name
        self.chat_interface = create_chat_interface(bot_name=bot_name)
        self.template = BootstrapTemplate(title=f"{self.bot_name} - Chemical Assistant")
        self.template.main.append(self.chat_interface)
        
    def start(self):
        pn.config.port = CONFIG["port"]
        self._server = self.template.show()
        return self._server

    def stop(self):
        # Proper cleanup of the server
        try:
            if hasattr(self, '_server') and self._server:
                self._server.stop()
                self._server = None
                return True
        except Exception as e:
            print(f"Error stopping server: {e}")
        return False

    def restart(self, **kwargs):
        # Stop current server
        self.stop()

        # Update configuration
        global CONFIG
        CONFIG.update(kwargs)

        # Recreate template
        pn.config.port = CONFIG["port"]
        self.template = BootstrapTemplate(title=f"{self.bot_name} - Chemical Assistant")
        self.template.main.append(self.chat_interface)

        # Start new server
        self._server = self.template.show()
        return self._server
    