// Mock extractions for the redesign prototype.
// Three datasets reflecting realistic ML-engineer corpus-building scenarios:
//   small (1 rxn) — a single tidy extraction
//   medium (3 rxn) — the actual run in 5 App/runs/13415d0010f8 (formylating transamidation)
//   large (8 rxn) — a substrate-scope figure: one reaction template, many products

const MOCK_SMALL = {
  source_image: "samples/figure1.png",
  figure_caption: "Figure 3. Pd-catalysed Suzuki–Miyaura coupling of aryl bromides with arylboronic acids.",
  extraction_notes: null,
  reactions: [
    {
      entry_id: "Fig 3",
      title: "Suzuki–Miyaura coupling — 4-bromoanisole + phenylboronic acid",
      reactants: [
        { label: "1a", name: "4-bromoanisole",       smiles: "COc1ccc(Br)cc1",  equiv: "1.0 equiv" },
        { label: "2a", name: "phenylboronic acid",   smiles: "OB(O)c1ccccc1",   equiv: "1.2 equiv" },
      ],
      reagents: [
        { label: null, name: "Pd(PPh3)4", smiles: null, role: "catalyst",  loading: "5 mol%" },
        { label: null, name: "K2CO3",     smiles: "[K+].[K+].O=C([O-])[O-]", role: "base", loading: "2.0 equiv" },
      ],
      conditions: { solvent: "1,4-dioxane / H2O (4:1)", temperature: "80 °C", time: "12 h", atmosphere: "Ar", other: null },
      products: [
        { label: "3a", name: "4-methoxybiphenyl", smiles: "COc1ccc(-c2ccccc2)cc1", yield_pct: 92, yield_note: null },
      ],
      notes: "Clean conversion; isolated by silica chromatography.",
      verified: true,
    },
  ],
};

const MOCK_MEDIUM = {
  source_image: "samples/figure1.png",
  figure_caption: "Scheme 1. Formylating transamidations promoted by boron-based catalysts.",
  extraction_notes:
    "Catalyst 1 drawn as 2,6-dichlorophenylboronic acid based on the Cl groups flanking the phenyl ring. DMF and HCONH2 serve dual roles as both formyl source and solvent in entries 1 and 3.",
  reactions: [
    {
      entry_id: "Nguyen 2012",
      title: "Formylating transamidation (Nguyen, 2012)",
      reactants: [
        { label: null, name: "amine", smiles: "[1*]N[2*]", equiv: null, role_note: "primary, secondary, anilines" },
      ],
      reagents: [
        { label: null, name: "B(OH)3", smiles: "OB(O)O", role: "catalyst", loading: "10 mol%" },
        { label: null, name: "H2O",    smiles: "O",      role: "additive", loading: "1–2 equiv" },
      ],
      conditions: { solvent: "DMF or HCONH2", temperature: "150 °C (DMF) or 25–150 °C (HCONH2)", time: null, atmosphere: null, other: null },
      products: [
        { label: null, name: "formamide", smiles: "[1*]N([2*])C=O", yield_pct: null, yield_note: "achiral scope" },
      ],
      notes: null,
      verified: false,
    },
    {
      entry_id: "Sheppard 2013",
      title: "Formylating transamidation (Sheppard, 2013)",
      reactants: [
        { label: null, name: "amine", smiles: "[1*]N[2*]", equiv: null },
      ],
      reagents: [
        { label: null, name: "B(OCH2CF3)3", smiles: "FC(F)(F)COB(OCC(F)(F)F)OCC(F)(F)F", role: "catalyst", loading: "2 equiv" },
        { label: null, name: "DMF",         smiles: "CN(C)C=O", role: "other", loading: null, role_note: "formyl source" },
      ],
      conditions: { solvent: "MeCN", temperature: "80 °C", time: "5 h", atmosphere: null, other: null },
      products: [
        { label: null, name: "formamide", smiles: "[1*]N([2*])C=O", yield_pct: null, yield_note: "primary amines, 1 chiral substrate" },
      ],
      notes: null,
      verified: true,
    },
    {
      entry_id: "This study",
      title: "Formylating transamidation (this study)",
      reactants: [
        { label: null, name: "amine", smiles: "[1*]N[2*]", equiv: null },
      ],
      reagents: [
        { label: "1", name: "2,6-dichlorophenylboronic acid", smiles: "OB(O)c1c(Cl)cccc1Cl", role: "catalyst", loading: "2.5–10 mol%" },
        { label: null, name: "AcOH",                          smiles: "CC(=O)O",             role: "additive", loading: "10–20 mol%" },
      ],
      conditions: { solvent: "DMF or HCONH2", temperature: "25–85 °C (DMF) or 45 °C (HCONH2)", time: null, atmosphere: null, other: null },
      products: [
        { label: null, name: "formamide", smiles: "[1*]N([2*])C=O", yield_pct: null, yield_note: "incl. α-aminoesters, chiral" },
      ],
      notes: null,
      verified: false,
    },
  ],
};

// Substrate-scope-style: one reaction template, 8 products with yields.
const MOCK_LARGE = {
  source_image: "samples/figure1.png",
  figure_caption: "Table 2. Substrate scope for the boron-catalysed N-formylation of secondary amines (catalyst 1, 5 mol%; HCONH2 as formyl source; 45 °C, 18 h).",
  extraction_notes: null,
  reactions: [
    { entry_id: "3a", title: "N-formyl morpholine",          reactants:[{label:"2a",name:"morpholine",                smiles:"C1COCCN1"}],         reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3a",name:"4-formylmorpholine",       smiles:"O=CN1CCOCC1",         yield_pct:94, yield_note:null}], notes:null, verified:true },
    { entry_id: "3b", title: "N-formyl piperidine",          reactants:[{label:"2b",name:"piperidine",                smiles:"C1CCNCC1"}],         reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3b",name:"1-formylpiperidine",        smiles:"O=CN1CCCCC1",         yield_pct:91, yield_note:null}], notes:null, verified:true },
    { entry_id: "3c", title: "N-formyl pyrrolidine",         reactants:[{label:"2c",name:"pyrrolidine",               smiles:"C1CCNC1"}],          reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3c",name:"1-formylpyrrolidine",       smiles:"O=CN1CCCC1",          yield_pct:88, yield_note:null}], notes:null, verified:true },
    { entry_id: "3d", title: "N-formyl N-methylaniline",     reactants:[{label:"2d",name:"N-methylaniline",           smiles:"CNc1ccccc1"}],       reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3d",name:"N-methyl-N-phenylformamide",smiles:"CN(C=O)c1ccccc1",     yield_pct:82, yield_note:null}], notes:null, verified:false },
    { entry_id: "3e", title: "N-formyl benzylamine",         reactants:[{label:"2e",name:"benzylamine",               smiles:"NCc1ccccc1"}],       reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3e",name:"N-benzylformamide",         smiles:"O=CNCc1ccccc1",       yield_pct:79, yield_note:null}], notes:null, verified:false },
    { entry_id: "3f", title: "N-formyl L-phenylalanine OMe", reactants:[{label:"2f",name:"L-Phe-OMe",                 smiles:"COC(=O)[C@@H](N)Cc1ccccc1"}], reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3f",name:"N-formyl-L-Phe-OMe",       smiles:"COC(=O)[C@@H](NC=O)Cc1ccccc1", yield_pct:71, yield_note:"99% ee"}], notes:"No racemisation observed.", verified:true },
    { entry_id: "3g", title: "N-formyl diethylamine",        reactants:[{label:"2g",name:"diethylamine",              smiles:"CCNCC"}],            reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3g",name:"N,N-diethylformamide",      smiles:"CCN(CC)C=O",          yield_pct:65, yield_note:null}], notes:null, verified:false },
    { entry_id: "3h", title: "N-formyl 4-chloroaniline",     reactants:[{label:"2h",name:"4-chloroaniline",           smiles:"Nc1ccc(Cl)cc1"}],    reagents:[{label:"1",name:"cat.",smiles:"OB(O)c1c(Cl)cccc1Cl",role:"catalyst",loading:"5 mol%"}], conditions:{solvent:"HCONH2",temperature:"45 °C",time:"18 h",atmosphere:null,other:null}, products:[{label:"3h",name:"N-(4-chlorophenyl)formamide",smiles:"O=CNc1ccc(Cl)cc1",   yield_pct:34, yield_note:"low conversion"}], notes:"Anilide bond competing.", verified:false },
  ],
};

// Structured Rxn-INSIGHT analysis (one entry per reaction in the active dataset).
// In the real backend this comes from /api/runs/:id/insight.
const MOCK_INSIGHT = {
  "Nguyen 2012":   { reaction_class: "C–N bond formation", named_reaction: null,                functional_groups: ["amine","carbonyl"], byproducts: ["H2O"],           rxn_class_id: "10.1.1", suggested_solvents: ["DMF","DMA"],     hazards: ["high temp"] },
  "Sheppard 2013": { reaction_class: "Transamidation",     named_reaction: null,                functional_groups: ["amine","amide"],    byproducts: ["B(OCH2CF3)2OH"], rxn_class_id: "10.1.4", suggested_solvents: ["MeCN","toluene"], hazards: ["pyrophoric reagent"] },
  "This study":    { reaction_class: "N-formylation",      named_reaction: "Bode N-formylation",functional_groups: ["amine","carbonyl"], byproducts: ["NH3"],           rxn_class_id: "10.1.7", suggested_solvents: ["DMF","HCONH2"],   hazards: ["—"] },
  "Fig 3":         { reaction_class: "C–C cross-coupling", named_reaction: "Suzuki–Miyaura",    functional_groups: ["aryl halide","arylboronic acid"], byproducts: ["KBr","K3BO3"], rxn_class_id: "3.1.1", suggested_solvents: ["dioxane/H2O","THF/H2O"], hazards: ["Pd-residue"] },
  // For substrate-scope entries, share one class
  ...Object.fromEntries(["3a","3b","3c","3d","3e","3f","3g","3h"].map(k => [k, {
    reaction_class: "N-formylation", named_reaction: "Bode N-formylation",
    functional_groups: ["amine","carbonyl"], byproducts: ["NH3"],
    rxn_class_id: "10.1.7", suggested_solvents: ["DMF","HCONH2"], hazards: ["—"]
  }])),
};

// Agent activity log (used by the collapsed drawer in both directions).
const MOCK_AGENT_LOG = [
  { n: 1, tool: "load_image",       summary: "samples/figure1.png · 1240×860", ok: true },
  { n: 2, tool: "describe_figure",  summary: "3 reactions identified · 1 inset (catalyst 1)", ok: true },
  { n: 3, tool: "extract_reaction", summary: "Nguyen 2012 — formylating transamidation",      ok: true },
  { n: 4, tool: "validate_smiles",  summary: "OB(O)O · ok",                                     ok: true },
  { n: 5, tool: "validate_smiles",  summary: "[1*]N([2*])C=O · ok",                             ok: true },
  { n: 6, tool: "extract_reaction", summary: "Sheppard 2013 — B(OCH2CF3)3 / DMF / MeCN, 80 °C",ok: true },
  { n: 7, tool: "validate_smiles",  summary: "FC(F)(F)COB(OCC(F)(F)F)OCC(F)(F)F · ok",          ok: true },
  { n: 8, tool: "extract_reaction", summary: "This study — catalyst 1 / AcOH",                  ok: true },
  { n: 9, tool: "resolve_inset",    summary: "‘1:’ → 2,6-dichlorophenylboronic acid",           ok: true },
  { n:10, tool: "validate_smiles",  summary: "OB(O)c1c(Cl)cccc1Cl · ok",                        ok: true },
  { n:11, tool: "submit_extraction",summary: "3 reactions · 5 reagents · 3 products",           ok: true },
];

const MOCK_MODELS = [
  { id: "claude-opus-4-7",         label: "Claude Opus 4.7",   provider: "anthropic", recommended: true,  context: "200k" },
  { id: "claude-sonnet-4-5",       label: "Claude Sonnet 4.5", provider: "anthropic", recommended: false, context: "200k" },
  { id: "gpt-5-vision",            label: "GPT-5 Vision",       provider: "openai",   recommended: false, context: "128k" },
  { id: "gemini-2.5-pro",          label: "Gemini 2.5 Pro",     provider: "google",   recommended: false, context: "1M"   },
  { id: "deepseek-vision-r2",      label: "DeepSeek VL2",       provider: "openrouter",recommended: false,context: "64k"  },
];

const MOCK_METADATA = { steps: 11, input_tokens: 18452, output_tokens: 3271, wall_time_s: 38.4 };

Object.assign(window, {
  MOCK_SMALL, MOCK_MEDIUM, MOCK_LARGE, MOCK_INSIGHT, MOCK_AGENT_LOG, MOCK_MODELS, MOCK_METADATA,
});
