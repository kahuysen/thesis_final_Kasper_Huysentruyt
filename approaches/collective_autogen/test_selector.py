"""
Quick test for SelectorGroupChat with the chemistry agents.
No image file required — passes a plain-text reaction description so you can
verify that the model-based speaker selection works without needing an API call
that processes an actual image.

Run:
    source .venv/bin/activate
    python test_selector.py
"""

import asyncio
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


PROMPTS_DIR = Path(__file__).parent / "prompts"

TASK = (
    "Analyse the following chemistry reaction description:\n\n"
    "Compound 1 (benzaldehyde, PhCHO) reacts with compound 2 (malononitrile, CH2(CN)2) "
    "in the presence of triethylamine (Et3N, 10 mol%) as catalyst in ethanol at room "
    "temperature for 2 hours to give compound 3 (2-benzylidenemalononitrile) in 92% yield. "
    "No R-groups are present.\n\n"
    "Each specialist should contribute their findings. "
    "The data_structure agent must synthesise everything into a JSON record and end with TERMINATE."
)

selector_prompt = """Select a chemistry specialist agent for the next analysis step.

{roles}

Current conversation context:
{history}

Workflow order (follow dependencies):
1. reaction_template_parser  — identifies the scheme first
2. molecular_recognition     — identifies molecules after the scheme is known
3. rgroup_substitution       — enumerates R-groups after molecules are identified
4. condition_interpretation  — extracts reaction conditions
5. text_extraction           — extracts remaining text and labels
6. data_structure            — synthesises everything into JSON (only after all five specialists above have contributed)

Select exactly one agent from {participants}.
"""


async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    reaction_template_parser = AssistantAgent(
        "reaction_template_parser",
        description="Parses the overall reaction scheme — reactants, products, arrow direction, and R-group wildcard SMILES. Should speak first.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "reaction_template_parser.txt").read_text(),
    )

    molecular_recognition = AssistantAgent(
        "molecular_recognition",
        description="Identifies individual molecular structures and converts them to SMILES with stereochemistry. Depends on reaction_template_parser output.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "molecular_recognition.txt").read_text(),
    )

    rgroup_substitution = AssistantAgent(
        "rgroup_substitution",
        description="Enumerates all R-group variants by replacing wildcard placeholders. Depends on reaction_template_parser and molecular_recognition.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "rgroup_substitution.txt").read_text(),
    )

    condition_interpretation = AssistantAgent(
        "condition_interpretation",
        description="Extracts all reaction conditions: temperature, time, solvent, catalyst, reagents, yield, atmosphere.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "condition_interpretation.txt").read_text(),
    )

    text_extraction = AssistantAgent(
        "text_extraction",
        description="Extracts all remaining textual annotations, labels, captions, footnotes, and metadata.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "text_extraction.txt").read_text(),
    )

    data_structure = AssistantAgent(
        "data_structure",
        description="Synthesises all specialist findings into a structured JSON record. Should be selected last, after all five specialist agents have contributed.",
        model_client=model_client,
        system_message=(PROMPTS_DIR / "data_structure.txt").read_text().replace(
            "{file_name}", "test_input.txt"
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=25)

    team = SelectorGroupChat(
        [
            reaction_template_parser,
            molecular_recognition,
            rgroup_substitution,
            condition_interpretation,
            text_extraction,
            data_structure,
        ],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
    )

    try:
        await Console(team.run_stream(task=TASK))
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
