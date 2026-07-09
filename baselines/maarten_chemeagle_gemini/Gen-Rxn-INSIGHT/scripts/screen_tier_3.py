import pandas as pd
from tqdm import tqdm
from collections import Counter
import gen_rxn_insight as ri
import json
tqdm.pandas()

def get_example_reaction(df: pd.DataFrame, template: str) -> str:
    dfc = df[df["TEMPLATE_rr0rp1_ring0"]==template].copy()
    series = dfc["SANITIZED_REACTION"].dropna()
    shortest_string = series.loc[series.str.len().idxmin()]
    return shortest_string

def screen_tier3(llm_class: str, df: pd.DataFrame, named_dict: dict):
    dfc = df[df["tier_3"]==llm_class].copy()
    idx = dfc.index[0]
    tier_2 = dfc["tier_2"][idx]
    top10_templates = Counter(dfc["TEMPLATE_rr0rp1_ring0"]).most_common(10)
    occs = sum([t[1] for t in top10_templates])
    coverage = occs / len(dfc.index)
    top10_examples = [get_example_reaction(dfc, t[0]) for t in top10_templates]
    name = f"{llm_class} - {named_dict[tier_2]}: {named_dict[llm_class]}"
    results_dict = {
        "reaction_class": name,
        "top10_templates": [t[0] for t in top10_templates],
        "top10_examples": top10_examples
    }
    return results_dict

def find_class(template: str, df: pd.DataFrame) -> str:
    dfc = df[df["TEMPLATE_rr0rp1_ring0"]==template].copy()
    c = Counter(dfc["tier_3"]).most_common(1)[0][0]
    return c

def add_new_template(df, template, human_template):
    with open("../data/gold_standard.json", "r") as f:
        gold_dict = json.load(f)
    
    c = find_class(template, df)
    if c in gold_dict:
        gold_dict[c].append(human_template)
    else:
        gold_dict[c] = [human_template]
    
    with open("../data/gold_standard.json", "w") as f:
        json.dump(gold_dict, f)
    
    return gold_dict

