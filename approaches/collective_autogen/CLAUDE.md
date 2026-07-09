# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# Run on an image
python main.py input/example1.png

# Run on any image path
python main.py /path/to/reaction_image.png
```

Requires `OPENAI_API_KEY` set in the environment (or a `.env` file loaded externally). The model used is `gpt-4o`.

## Architecture

This is a **multi-agent chemistry image analysis pipeline** built with AutoGen 0.7.5. It takes a reaction image and extracts structured JSON describing the chemical reactions depicted.

**Flow:**
1. `main.py` loads 6 `AssistantAgent` instances, each with a specialist prompt from `prompts/`
2. All agents share a single `OpenAIChatCompletionClient` (GPT-4o)
3. They run in a `RoundRobinGroupChat` — each agent receives the full conversation history and the original image
4. Termination is `MaxMessageTermination(13)` — enough for ~2 full passes through all 6 agents
5. The last message (always from `data_structure`) is parsed for a `json` code block and saved as `result.json`

**Agent roles (in order):**
1. `reaction_template_parser` — identifies the overall reaction scheme, produces SMILES with `*` wildcards for R-groups
2. `molecular_recognition` — recognises concrete molecular structures
3. `rgroup_substitution` — enumerates R-group variants
4. `condition_interpretation` — extracts reagents, solvents, catalyst, temperature, time, yield, atmosphere
5. `text_extraction` — extracts all text labels and metadata from the image
6. `data_structure` — synthesises all prior agent output into the final JSON (prompt receives `file_name` via substitution at runtime)

**Output per run** (`runs/{YYYYMMDD_HHMMSS}/`):
- `conversation.txt` — full agent dialogue
- `result.json` — structured extraction matching the schema in `prompts/data_structure.txt`

## Output JSON Schema

```json
{
  "file_name": "source.png",
  "reactions": [{
    "reaction_id": "0_N",
    "reactants": [{"smiles": "...", "label": "1"}],
    "conditions": [{"role": "reagent|solvent|catalyst|temperature|time|yield|atmosphere", "text": "...", "smiles": null}],
    "products": [{"smiles": "...", "label": "3"}],
    "additional_info": ["..."]
  }]
}
```

Each R-group variant becomes its own reaction entry. Uncertain SMILES get `" (uncertain)"` appended to their label.

## Key Extension Points

- **Change agents or order:** edit the list passed to `RoundRobinGroupChat` in `main.py`
- **Change termination:** adjust `MaxMessageTermination(13)` — 13 = 2 full rounds + 1 final synthesis
- **Modify agent behaviour:** edit the relevant `.txt` file in `prompts/`; `data_structure.txt` supports `{file_name}` substitution via `load_prompt()`
- **Switch model:** change `model="gpt-4o"` in `main.py`; `test2.py` shows an alternative two-agent pattern with `TextMentionTermination`
