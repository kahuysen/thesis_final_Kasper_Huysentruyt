// Build a Word .docx report on the resolution-ablation experiment.
// Academic language; designed to be handed to another writing assistant.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, PageOrientation, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageBreak,
} = require("docx");

const REPORT_DIR = "/Users/kasperhuysentruyt/Documents/thesis/5.Code/Single_SDK_agent/benchmark_runs/resolution_report";
const IMG_DIR    = path.join(REPORT_DIR, "showcase_resolutions");
const FIG_PATH   = path.join(REPORT_DIR, "fig_resolution_ablation.png");
const OUT        = path.join(REPORT_DIR, "resolution_ablation_report.docx");

// ---------- helpers ----------
const PAGE_W = 12240, PAGE_H = 15840;
const MARG = 1440;
const CONTENT_W = PAGE_W - 2 * MARG; // 9360

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, ...opts.run })],
  });
}

function H(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true })],
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    borders,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        children: [new TextRun({ text: String(text), bold: !!opts.bold, size: 20 })],
      }),
    ],
  });
}

function table(colWidths, rows, headerShade = "E6EEF7") {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: rows.map((r, i) =>
      new TableRow({
        children: r.map((c, j) => {
          const opts = typeof c === "object" && c !== null && !Array.isArray(c) ? c : { text: c };
          return cell(opts.text ?? c, {
            width: colWidths[j],
            shade: i === 0 ? headerShade : opts.shade,
            bold: i === 0 || opts.bold,
            align: j === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          });
        }),
      })
    ),
  });
}

function image(file, w, h, caption) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(path.join(IMG_DIR, file)),
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: file },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: caption, italics: true, size: 18 })],
    }),
  ];
}

// ---------- content ----------
const today = "2026-05-24";

const children = [];

// Title block
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({ text: "Effect of Input-Image Resolution on a Gemini-3-Flash Vision Agent", bold: true, size: 32 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({ text: "for Reaction Extraction from Journal Figures", bold: true, size: 32 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [
    new TextRun({ text: "Internal technical report — single-SDK agent (Gemini 3 Flash Preview)\n", size: 20 }),
    new TextRun({ text: `Author: Kasper Huysentruyt    ·    Date: ${today}    ·    Benchmark: GT3_Maarten (16 images)`, size: 20 }),
  ],
}));

// Abstract
children.push(H("Abstract", HeadingLevel.HEADING_1));
children.push(P(
  "A controlled ablation was conducted to quantify the effect of input-image down-sampling on the extraction quality of a Gemini-3-Flash-Preview vision agent applied to chemical figures. The same 16-image benchmark (GT3_Maarten) was processed at five input resolutions — full, one-half, one-quarter, one-eighth, and one-tenth of the original width and height — using an identical agent loop, identical system prompt, and identical reference ground truth. All runs were scored with the reference-based evaluation suite from the Collective-AutoGen project (Reactant IoU, Product IoU, Soft F1, Hard F1, Constituent F1, Partial F1 at Jaccard ≥ 0.5, and lenient condition coverage). Quality was found to be statistically flat from full to one-quarter resolution (Partial F1 = 0.79 → 0.78 → 0.78), to drop sharply at one-eighth resolution (Partial F1 = 0.25), and to collapse to near zero at one-tenth resolution (Partial F1 = 0.06), while schema-validity and SMILES-validity remained near-perfect (≥ 97 %) across all five conditions. Total token usage and cost were found to be largely insensitive to image resolution, because output tokens — dominated by the fixed JSON output schema — did not decrease and slightly increased at the lowest resolution. The results indicate (i) that the transition between robust extraction and collapse falls between one-quarter and one-eighth resolution and is not well described by a single per-image pixel-count threshold, (ii) that per-image variance is large near the transition and that repeat runs would be required to claim a sharp threshold, and (iii) that schema-based validity checks are not a sufficient quality signal for this pipeline."
));

// 1. Methodology
children.push(H("1. Methodology", HeadingLevel.HEADING_1));

children.push(H("1.1 Pipeline", HeadingLevel.HEADING_2));
children.push(P(
  "Extraction was performed with the single-SDK-agent pipeline described in the May 2026 thesis draft. The agent is a stateless, tool-using loop wrapped around the gemini-3-flash-preview model: each request submits the system instruction, the input image bytes, and a short user prompt, and the model is required to call an RDKit-backed validate_smiles tool followed by a Pydantic-validated submit_extraction terminator. No context caching, no conversation history across images, and no fine-tuned variant of the model were used; each image was processed in an independent API call, ruling out cross-image information leakage."
));

children.push(H("1.2 Benchmark and ground truth", HeadingLevel.HEADING_2));
children.push(P(
  "The benchmark is the GT3_Maarten 16-image set drawn from five chemistry journals (ACS Catalysis, Chemistry — A European Journal, Chemical Science, Green Chemistry, Nature Communications). Ground truth was constructed manually and is shared with all single-SDK-agent variants evaluated in the main thesis. Original image dimensions range from 663 × 393 pixels (smallest) to 1487 × 840 pixels (largest)."
));

children.push(H("1.3 Resolution conditions", HeadingLevel.HEADING_2));
children.push(P(
  "Five input conditions were prepared by Lanczos-resampling every image to a fraction of its original width and height: full (1/1), half (1/2 × 1/2 = 1/4 area), quarter (1/4 × 1/4 = 1/16 area), eighth (1/8 × 1/8 = 1/64 area), and tenth (1/10 × 1/10 = 1/100 area). The eighth condition was added a posteriori to better localise the transition between robust extraction and collapse, which was observed to lie somewhere between the quarter and the tenth conditions. Ground-truth files were copied unchanged into each condition; the batch driver was verified to enumerate only the *.png inputs, so ground-truth JSON files were never transmitted to the model."
));

children.push(H("1.4 Evaluation", HeadingLevel.HEADING_2));
children.push(P(
  "All four conditions were scored with scripts/eval_via_collective.py, which translates the pipeline's FigureExtraction schema into the Collective-AutoGen record schema and invokes the reference-based matcher in Collective_autogen/eval/metrics.py. Reported aggregates are macro-averages across the 16 images and are directly comparable with the rows already present in progress_April_data.xlsx. Token and wall-time accounting was collected from the per-image .meta.json files written by the pipeline; USD figures are projected from the published Google Standard-tier sticker for the Gemini 3 family ($0.30/M input, $2.50/M output)."
));

// 2. Showcase image
children.push(H("2. Showcase: degradation of a representative image", HeadingLevel.HEADING_1));
children.push(P(
  "Figure 1 shows a representative benchmark figure (ACScat_2020 page 002, table 01) at all five resolutions. The image contains 14 ground-truth reactions arranged in a scope-table layout. Visual legibility of bond connectivity and atom labels is preserved at half- and quarter-resolution; at one-eighth resolution the structures are heavily aliased and the agent failed to recover any reaction from this particular image; at one-tenth resolution the structures are reduced to indistinct shapes."
));

children.push(...image("01_full_756x646.png",    420, 360, "Figure 1a — Full resolution (756 × 646 px). Agent recovered 14/14 reactions."));
children.push(...image("02_half_378x323.png",    320, 274, "Figure 1b — Half resolution (378 × 323 px). Agent recovered 14/14 reactions."));
children.push(...image("03_quarter_189x161.png", 220, 188, "Figure 1c — Quarter resolution (189 × 161 px). Agent recovered 14/14 reactions."));
children.push(...image("03b_eighth_94x80.png",   140, 119, "Figure 1d — Eighth resolution (94 × 80 px). Agent recovered 0/14 reactions for this image."));
children.push(...image("04_tenth_75x64.png",     120, 102, "Figure 1e — Tenth resolution (75 × 64 px). Agent recovered 0/14 reactions."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 3. Quantitative results
children.push(H("3. Quantitative results", HeadingLevel.HEADING_1));
children.push(H("3.1 Reference-based quality metrics", HeadingLevel.HEADING_2));
children.push(P(
  "Table 1 reports the macro-averaged metrics produced by eval_via_collective.py across the 16 images for each resolution condition. The best value in each row is shown in bold."
));

const metricRows = [
  ["Metric",                       "Full",   "Half",   "Quarter", "Eighth", "Tenth"],
  ["Schema pass (of 16)",          "100 %",  "100 %",  "100 %",   "100 %",  "100 %"],
  ["SMILES validity",              "99 %",   "100 %",  "100 %",   "97 %",   "99 %"],
  ["Role-enum compliance",         "100 %",  "100 %",  "100 %",   "100 %",  "100 %"],
  ["Reactions predicted / gold",   "126/123","129/123","128/123", "129/123","126/123"],
  ["Reactant IoU",                 "0.57",   { text: "0.61", bold: true }, "0.56", "0.24", "0.15"],
  ["Product IoU",                  { text: "0.79", bold: true }, "0.75", "0.67", "0.27", "0.07"],
  ["Cond. recall (strict)",        "42 %",   { text: "45 %", bold: true }, "44 %", "22 %", "15 %"],
  ["Cond. recall (lenient)",       "81 %",   "82 %",   { text: "83 %", bold: true }, "38 %", "24 %"],
  ["Cond. precision (lenient)",    "70 %",   "70 %",   { text: "71 %", bold: true }, "31 %", "16 %"],
  ["Soft match F1",                "0.46",   { text: "0.51", bold: true }, "0.45", "0.13", "0.00"],
  ["Hard match F1",                "0.04",   "0.04",   "0.04",    "0.00",  "0.00"],
  ["Constituent F1",               "0.58",   "0.59",   { text: "0.63", bold: true }, "0.13", "0.00"],
  ["Partial F1 (Jaccard ≥ 0.5)",   { text: "0.79", bold: true }, "0.78", "0.78", "0.25", "0.06"],
];
children.push(table([2880, 1296, 1296, 1296, 1296, 1296], metricRows));
children.push(P("Table 1 — Reference-based metrics by input resolution. All values are macro-averages across the 16 GT3_Maarten images, scored by scripts/eval_via_collective.py against the Collective-AutoGen ground truth.", { run: { italics: true, size: 18 } }));

children.push(H("3.2 Token usage, wall-time and cost", HeadingLevel.HEADING_2));
children.push(P(
  "Table 2 reports the total input and output tokens, the summed wall-clock time, and the projected USD cost for each resolution condition. Token counts for the half-, quarter- and tenth-resolution runs were obtained from per-image .meta.json files; the full-resolution values were taken from the pre-existing Fig3_cost_quality row in progress_April_data.xlsx (the original full-resolution run did not emit per-image metadata)."
));
const costRows = [
  ["Quantity",                       "Full",  "Half",   "Quarter", "Eighth", "Tenth"],
  ["Input tokens (total)",           "≈ 416 000", "300 262", "306 392", "302 812", "390 311"],
  ["Output tokens (total)",          "≈ 40 000",  "41 201",  "44 251",  "44 908",  "47 515"],
  ["Total tokens",                   "464 000",   "341 463", "350 643", "347 720", "437 826"],
  ["Wall-time, sum (s)",             "855",       "784",     "762",     "1 019",   "1 086"],
  ["Tool calls (total)",             "—",         "240",     "233",     "207",     "251"],
  ["Cost (USD, $0.30 / $2.50 per M)","$0.23",     "$0.19",   "$0.20",   "$0.20",   "$0.24"],
  ["Cost (USD, $0.38 / $3.20 per M)","$0.28",     "$0.25",   "$0.26",   "$0.26",   "$0.30"],
];
children.push(table([2880, 1296, 1296, 1296, 1296, 1296], costRows));
children.push(P("Table 2 — Token consumption, wall-clock latency and projected USD cost per resolution condition. The second cost row uses the rate implied by the cost figure recorded for the full-resolution run in progress_April_data.xlsx; both rows are reported for transparency.", { run: { italics: true, size: 18 } }));

children.push(H("3.3 Graphical summary", HeadingLevel.HEADING_2));
children.push(P(
  "Figure 2 plots the reference-based quality metrics (panel a), the projected USD cost together with the summed wall-clock time (panel b), and the input and output token totals (panel c) against the linear scale factor of the input image. All x-axes are placed on a logarithmic scale to reflect the geometric spacing of the five conditions. The figure makes three patterns visually explicit. First, in panel (a), all four quality metrics remain on a plateau from the full-resolution condition through the quarter-resolution condition, drop sharply at the eighth-resolution condition, and reach a near-zero floor at the tenth-resolution condition; the Partial F1 trajectory is annotated for emphasis. The transition between the plateau and the floor falls entirely between the quarter and eighth conditions and is therefore narrower than the geometric spacing of the conditions tested. Second, in panel (b), neither the projected cost nor the summed wall-clock time decreases monotonically with image size: the half- and quarter-resolution conditions are local minima on both axes, the eighth-resolution condition is comparable in cost but slower, and the tenth-resolution condition is the most expensive in both cost and time. Third, in panel (c), the underlying reason for the cost behaviour in panel (b) is made explicit: input tokens fall by 28 % from full to half resolution and then remain on a plateau before rebounding at tenth resolution, while output tokens are monotonically non-decreasing across the entire sweep, rising from 40 000 at full resolution to 47 515 at tenth resolution. Because output tokens are priced approximately eight times higher than input tokens, the absence of a monotonic decrease in output volume — which is set by the fixed Pydantic JSON schema — neutralises the input-token savings."
));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 60 },
  children: [new ImageRun({
    type: "png",
    data: fs.readFileSync(FIG_PATH),
    transformation: { width: 680, height: 206 },
    altText: {
      title: "Resolution ablation — quality and cost",
      description: "Two-panel figure: (a) reference-based quality metrics vs scale factor; (b) projected cost and wall-clock time vs scale factor.",
      name: "fig_resolution_ablation.png",
    },
  })],
}));
children.push(P(
  "Figure 2 — Resolution ablation of the single-SDK Gemini-3-Flash agent on the GT3_Maarten 16-image benchmark. The x-axis of each panel is the linear scale factor applied to each side of the input images (log scale, inverted). The lower line of each x-tick label reports the corresponding effective dots-per-inch value, computed as the mean source DPI of the original PNGs (124.8, obtained from the 5 of 16 benchmark images carrying DPI metadata in their header — 96 dpi for two images and 144 dpi for three) multiplied by the scale factor and rounded to the nearest integer. (a) Reference-based quality metrics (Partial F1, Product IoU, Constituent F1, Soft F1); quality is essentially flat from full to quarter resolution (≈125 to ≈31 dpi), drops sharply at one-eighth (≈16 dpi), and collapses at one-tenth (≈12 dpi). (b) Projected USD cost (left axis, blue) and summed wall-clock time across the 16 images (right axis, orange); neither cost nor time decreases monotonically with image size, and the lowest-resolution condition is the most expensive in both quantities. (c) Total input tokens (left axis, blue) and total output tokens (right axis, orange) summed across the 16 images; input tokens fall and then rebound, while output tokens — fixed in volume by the Pydantic JSON output schema — are monotonically non-decreasing, which explains the cost behaviour in panel (b).",
  { run: { italics: true, size: 18 } }
));

children.push(H("3.4 Per-image collapse pattern at low resolution", HeadingLevel.HEADING_2));
children.push(P(
  "At one-eighth resolution, 5 of the 16 images yielded full or near-full reaction recall and 10 yielded zero correct reactions, with a single image producing a partial extraction (1 of 13). At one-tenth resolution, only 2 of the 16 images yielded any correctly extracted reaction. Per-image counts at both low-resolution conditions are reported in Table 3 together with the minimum image edge in pixels, for comparison with full-resolution recall. The two buckets — successful and collapsed — overlap substantially in minimum edge at the eighth condition: the worked images span 45 – 96 pixels of minimum edge, and the collapsed images span 49 – 105 pixels. The largest minimum edge among the eighth-resolution images (105 pixels, GC_2015 page 006 picture 01) yielded zero correct reactions, while the smallest (45 pixels, GC_2015 page 007 picture 02) yielded perfect recall. The minimum-edge hypothesis suggested by the tenth-resolution data alone is therefore not supported by the eighth-resolution data, and per-image variance — rather than a deterministic pixel-count threshold — appears to dominate near the transition."
));
const collapseRows = [
  ["Image (file stem)",                                  "Min. edge (1/8)", "Rec. (1/8)", "Min. edge (1/10)", "Rec. (1/10)", "Rec. (full)"],
  ["CEJ_2016.pdf_page002_table_02_s0.91",                "96", "14 / 14", "76", "0 / 14", "14 / 14"],
  ["CS_2016.pdf_page002_table_02_s0.88",                 "49", "7 / 7",   "39", "0 / 7",  "7 / 7"],
  ["GC_2015.pdf_page002_picture_02_s0.56",               "92", "7 / 7",   "74", "7 / 7",  "7 / 7"],
  ["GC_2015.pdf_page004_table_01_s0.81",                 "60", "6 / 6",   "48", "0 / 6",  "6 / 6"],
  ["GC_2015.pdf_page007_picture_02_s0.76",               "45", "5 / 5",   "36", "0 / 5",  "5 / 5"],
  ["NC_2017.pdf_page004_picture_01_s0.91",               "70", "1 / 13",  "56", "0 / 13", "9 / 13"],
  ["ACScat_2020.pdf_page002_table_01_s0.88",             "80", "0 / 14",  "64", "0 / 14", "14 / 14"],
  ["ACScat_2020.pdf_page003_picture_01_s0.93",           "70", "0 / 8",   "56", "0 / 8",  "7 / 8"],
  ["CEJ_2016.pdf_page001_picture_02_s0.74",              "65", "0 / 3",   "52", "0 / 3",  "3 / 3"],
  ["CEJ_2016.pdf_page004_table_02_s0.66",                "95", "0 / 10",  "76", "9 / 10", "9 / 10"],
  ["GC_2015.pdf_page002_table_01_s0.61",                 "75", "0 / 10",  "60", "0 / 10", "10 / 10"],
  ["GC_2015.pdf_page003_picture_03_s0.71",               "66", "0 / 4",   "53", "0 / 4",  "4 / 4"],
  ["GC_2015.pdf_page003_table_02_s0.78",                 "56", "0 / 5",   "44", "0 / 5",  "5 / 5"],
  ["GC_2015.pdf_page006_picture_01_s0.86",              "105", "0 / 9",   "84", "0 / 9",  "9 / 9"],
  ["GC_2015.pdf_page007_picture_03_s0.60",               "49", "0 / 5",   "39", "0 / 5",  "5 / 5"],
  ["NC_2017.pdf_page005_picture_01_s0.90",               "71", "0 / 3",   "56", "0 / 3",  "3 / 3"],
];
children.push(table([3500, 1120, 1180, 1260, 1180, 1120], collapseRows));
children.push(P("Table 3 — Per-image extraction recall at one-eighth and one-tenth resolution, with the corresponding full-resolution count for comparison. Rows are sorted by 1/8 outcome (full-recall images first, then partial, then collapsed). Minimum-edge columns are in pixels. Note in particular that the largest 1/8 minimum edge (GC_2015 page 006 picture 01, 105 px) yielded zero correct reactions while the smallest (GC_2015 page 007 picture 02, 45 px) yielded perfect recall.", { run: { italics: true, size: 18 } }));

// 4. Discussion
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H("4. Discussion", HeadingLevel.HEADING_1));

children.push(H("4.1 Plateau from full to one-quarter resolution", HeadingLevel.HEADING_2));
children.push(P(
  "Reducing the input image to one-quarter of its width and height (i.e. 1/16 of the original pixel area) did not produce a measurable drop in the headline Partial F1 metric (0.79 → 0.78). On certain sub-metrics — notably Soft F1 and Constituent F1 — the half- and quarter-resolution runs slightly exceeded the full-resolution run. Two non-exclusive explanations are advanced. First, the 16-image benchmark is small relative to the per-image stochasticity of an LLM-based extractor sampled at non-zero temperature, and per-resolution variance has not been characterised; differences below approximately 0.03–0.05 on Partial F1 should be treated as within run-to-run noise. Second, chemical-scheme images are unusually low-information per pixel — large white margins, high-contrast atom labels, and bond lines that remain several pixels wide even after fourfold down-sampling — so the practical legibility ceiling of the vision encoder is reached far below the original 700–1500-pixel dimensions used in the benchmark."
));

children.push(H("4.2 Sharp transition between one-quarter and one-eighth resolution", HeadingLevel.HEADING_2));
children.push(P(
  "Between the quarter and eighth conditions, Partial F1 dropped from 0.78 to 0.25, Product IoU from 0.67 to 0.27, and lenient condition recall from 83 % to 38 %. The transition is therefore concentrated in a narrow region of the resolution axis (a 2× linear / 4× area step) rather than being distributed across the sweep. The eighth condition itself exhibits highly bimodal per-image behaviour, with 5 of 16 images yielding full or near-full recall and 10 yielding zero correct reactions. This bimodality, taken together with the variance in per-image outcomes at constant resolution, suggests that the transition is not a sharp deterministic threshold but a stochastic regime in which marginal images flip between success and failure between calls."
));

children.push(H("4.3 Floor at one-tenth resolution and the validity-vs-correctness dissociation", HeadingLevel.HEADING_2));
children.push(P(
  "At one-tenth resolution, all reference-based F1 metrics fell to or near zero, while schema-validity and SMILES-validity remained at 99–100 %. The model continued to emit on average 7.9 reactions per image (126 across 16 images, vs. 123 in ground truth), every reaction passed the Pydantic schema validator, and every SMILES string was parseable by RDKit. Inspection of the predictions indicates that the model is producing chemically plausible but image-unrelated reactions. This dissociation between syntactic validity and semantic correctness is methodologically important: any quality-assurance scheme for this pipeline that relies solely on schema-pass or SMILES-validity checks would silently accept the one-tenth-resolution output as a successful extraction, despite a Partial F1 of 0.06. Reference-based evaluation against held-out ground truth is therefore not a sanity check but a necessary safeguard."
));

children.push(H("4.4 Insensitivity of token cost to image resolution", HeadingLevel.HEADING_2));
children.push(P(
  "Halving the input dimensions reduced total input tokens by 28 % relative to the full-resolution run, but successive halving (quarter, then eighth, then tenth) did not produce further savings. Output tokens, dominated by the fixed Pydantic JSON schema, did not decrease at any resolution; the lowest-resolution condition produced the largest output-token count (47 515). The projected USD cost is consequently flat-to-rising as resolution decreases: the half- and quarter-resolution conditions are weak local minima at ≈ $0.19 – $0.20, the eighth-resolution condition matches them on cost but is approximately 33 % slower in wall-clock time, and the tenth-resolution condition is the most expensive in both quantities at $0.24 and 1 086 s. Image down-sampling is therefore not an effective cost-reduction lever for this pipeline; the model spends additional reasoning effort on the degraded images that approximately cancels any savings from smaller image payloads."
));

children.push(H("4.5 Minimum-edge pixel count is not a reliable predictor of collapse", HeadingLevel.HEADING_2));
children.push(P(
  "An earlier inspection of the tenth-resolution data suggested that the minimum image edge in pixels might predict catastrophic collapse, with images above approximately 70 px succeeding and those below failing. The eighth-resolution data do not support this hypothesis. Among the 16 images at the eighth condition, the largest minimum edge (105 pixels, GC_2015 page 006 picture 01) yielded zero correct reactions, while the smallest (45 pixels, GC_2015 page 007 picture 02) yielded perfect recall (5/5). The full distributions of minimum edge for the successful and collapsed buckets overlap substantially (45 – 96 px and 49 – 105 px respectively). The pattern observed at the tenth condition is therefore better interpreted as a particular realisation of a high-variance stochastic regime than as evidence for a fixed pixel-count threshold. Establishing whether such a threshold exists — and, if so, what image features control it — would require repeat sampling at each resolution and a larger benchmark."
));

// 5. Conclusion
// ─────────────── 5. Reasoning comparison ───────────────
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H("5. Reasoning across resolutions on a single image", HeadingLevel.HEADING_1));

children.push(H("5.1 What constitutes \"reasoning\" in this configuration", HeadingLevel.HEADING_2));
children.push(P(
  "The Gemini agent loop used throughout this study is configured with tool_config set to ANY, which forces the model to invoke a tool on every turn and effectively suppresses free-text emission. Under the production configuration (Sections 3 and 4) the model therefore emits no natural-language commentary alongside its tool calls. Gemini 3 additionally returns opaque thought_signature byte sequences attached to each part of its response; these are internal continuity markers required by the API to preserve reasoning state across tool turns and are not human-readable. The externally visible reasoning trace in the production pipeline is therefore the sequence of tool calls the model chooses to issue — which SMILES strings it submits to validate_smiles, in what order, and what final dictionary it commits via submit_extraction — and that trace is analysed in Sections 5.3 to 5.5 below."
));
children.push(P(
  "The Gemini API does, however, expose a human-readable chain-of-thought when the generation configuration includes thinking_config with include_thoughts set to True. Under this setting, the response stream contains additional Part objects with the boolean field thought set to True and a text payload that carries the model's internal deliberation. The production extractor does not enable this option, but a separate diagnostic re-run with thinking_config enabled was performed for the same showcase image at all five resolutions and is reported in Section 5.6 below."
));

children.push(H("5.2 Procedure", HeadingLevel.HEADING_2));
children.push(P(
  "A standalone diagnostic script (benchmark_runs/resolution_report/capture_single_transcript.py) was implemented that mirrors the production extractor but, in addition to the structured extraction, persists the full in-memory history list to JSON. The script was run once per resolution condition on the showcase image (ACScat_2020 page 002, table 01) — a five-cell substrate scope of N-Boc-amino-acid + benzylamine peptide couplings, with 14 ground-truth reactions. The agent loop, system prompt, tool schemas, sampling temperature, and stop conditions are identical to those used in the benchmark runs reported in Sections 3 and 4. Per-condition transcripts are stored under transcripts/<condition>.json."
));

children.push(H("5.3 Convergent reasoning at full, half, and quarter resolution", HeadingLevel.HEADING_2));
children.push(P(
  "At all three plateau conditions the model issued exactly 12 validate_smiles calls and submitted exactly 14 reactions. The validated SMILES sets across the three conditions differ only in stereochemistry annotation: the chiral centre on the Boc-phenylalanine substrate appears as [C@H] at full resolution, [C@@H] at half resolution, and as an unspecified centre (C without stereodescriptor) at quarter resolution, with all other validated SMILES strings — the two main reactants (Boc-glycine, Boc-phenylalanine fragment), the benzylamine reagent, the four reported products, and the four solvent / atmosphere standards — being identical or canonicalisation-equivalent. The model correctly identifies the substrate-scope structure of the figure and constructs reactant–product pairs that match the ground-truth chemistry."
));

const snippetRowsHi = [
  ["Condition", "First 4 validated SMILES (verbatim, in call order)"],
  ["Full",     "CC(C)(C)OC(=O)NCC(=O)O   ·   CC(C)(C)OC(=O)N[C@H](Cc1ccccc1)C(=O)O   ·   NCc1ccccc1   ·   CC(C)(C)OC(=O)NCC(=O)NCc1ccccc1"],
  ["Half",     "CC(C)(C)OC(=O)N[C@@H]([1*])C(=O)O   ·   NCc1ccccc1   ·   CC(C)(C)OC(=O)N[C@@H]([1*])C(=O)NCc1ccccc1   ·   CC(C)(C)OC(=O)NCC(=O)O"],
  ["Quarter",  "CC(C)(C)OC(=O)NC([1*])C(=O)O   ·   NCc1ccccc1   ·   CC(C)(C)OC(=O)NC([1*])C(=O)NCc1ccccc1   ·   CC(C)(C)OC(=O)NCC(=O)O"],
];
children.push(table([1400, 7960], snippetRowsHi));
children.push(P("Table 4 — First four SMILES strings submitted to validate_smiles at each plateau condition. The Boc-amino-acid scaffold (CC(C)(C)OC(=O)N…C(=O)O), the benzylamine coupling partner (NCc1ccccc1), and the amide product are present in every condition; only the stereodescriptor on the α-carbon differs.", { run: { italics: true, size: 18 } }));

children.push(H("5.4 Chemistry shift at one-eighth and one-tenth resolution", HeadingLevel.HEADING_2));
children.push(P(
  "At the eighth-resolution condition, the model again issued 13 validate_smiles calls and submitted 14 reactions, but the chemistry it identified is no longer the Boc-amino-acid + benzylamine coupling of the ground truth. The model now reports a methyl-ester + propylamine amidation: the validated SMILES set contains COC(=O)CCc1ccccc1, COC(=O)C(C)Cc1ccccc1 and CCCN in place of the Boc-protected acid and benzylamine. The reaction class — formation of an amide from an activated acyl group and a primary amine — is preserved, but neither of the two ground-truth reactants is present in the prediction. At the tenth-resolution condition, the chemistry shifts again, this time to a tert-butyl-amide synthesis from an α-cyanoketone (e.g. [1*]C(=O)CC#N and CC(C)(C)N), with no overlap to either the ground truth or the eighth-condition prediction. Both lower-resolution conditions submit 14 schema-valid reactions with 100 % SMILES parseability."
));
children.push(P(
  "Table 5 quantifies the chemistry drift by counting, for each condition, the number of SMILES strings (validated or submitted) that contain six structural signatures: the two that characterise the ground-truth chemistry (Boc-carbamate scaffold and benzylamine), and four invented by the lower-resolution conditions (methyl ester, propylamine, tert-butylamine, and a cyano group). The two ground-truth signatures are present in tens of SMILES at the full, half, and quarter conditions and absent from the eighth and tenth conditions; the inverse holds for the invented signatures."
));
const featRows = [
  ["Structural signature (regex on canonical SMILES)", "Full", "Half", "Quarter", "Eighth", "Tenth"],
  ["Boc-carbamate scaffold  (ground truth)",            "6",   "6",   "34",      "0",      "0"],
  ["Benzylamine, NCc1ccccc1  (ground truth)",          "32",  "32",  "32",      "0",      "0"],
  ["Methyl ester, COC(=O)…  (invented at 1/8)",         "0",   "0",   "0",      "29",      "0"],
  ["Propylamine, CCCN  (invented at 1/8)",              "0",   "0",   "0",      "29",      "0"],
  ["tert-Butylamine, CC(C)(C)N  (invented at 1/10)",    "0",   "0",   "0",       "0",     "53"],
  ["Cyano group, #N  (invented at 1/10)",               "0",   "0",   "0",       "0",      "7"],
];
children.push(table([4080, 1056, 1056, 1056, 1056, 1056], featRows));
children.push(P("Table 5 — Occurrences of six structural signatures across all SMILES strings (validated or submitted) emitted by the model at each resolution. The first two rows are the signatures of the ground-truth chemistry (N-Boc-amino-acid + benzylamine peptide coupling); the remaining four are signatures invented by the lower-resolution conditions. The clean separation between the upper three conditions and the lower two indicates that the model identified the correct reaction class at full, half, and quarter resolution and fabricated alternative reaction classes at one-eighth and one-tenth.", { run: { italics: true, size: 18 } }));

children.push(H("5.5 Absence of expressed uncertainty", HeadingLevel.HEADING_2));
children.push(P(
  "At every resolution, including the two conditions at which the model identified entirely incorrect chemistry, no candidate response contained a text part — the model emitted zero natural-language tokens at any turn. Each condition committed 14 reactions through a single, terminal call to submit_extraction, with no preceding self-correction, no requests for clarification, and no diagnostic output indicating low confidence. The agent therefore offers no internally visible signal that would distinguish the convergent correct extractions at full–quarter resolution from the divergent hallucinated extractions at eighth–tenth resolution: both look identical from outside the loop, both pass schema validation, and both validate every SMILES string they emit. This observation reinforces the conclusion of Section 4.3 that schema-based and SMILES-validity checks are not sufficient quality signals for this pipeline; a comparison against held-out ground truth is the only operational way, given the present configuration, to detect that the lower-resolution conditions have produced confidently fabricated extractions."
));

children.push(P(
  "An additional observation merits explicit note. The tenth-resolution capture reported in this section submitted 14 schema-valid reactions, none of whose products match the ground-truth product SMILES under canonicalisation. The corresponding entry in the main benchmark run (Section 3.4) for the same image and the same resolution recorded 0 of 14 reactions recovered, but with a different set of fabricated SMILES. The two tenth-resolution executions therefore agree on the qualitative outcome — total mismatch with the ground truth — but produce different fabricated chemistries on each call, which is direct evidence that the per-image variance discussed in Section 4.5 extends to the structural content of the predictions and not only to the aggregate metrics."
));

// ─────────────── 5.6 Internal reasoning with include_thoughts ───────────────
children.push(H("5.6 Internal reasoning made visible with include_thoughts enabled", HeadingLevel.HEADING_2));

children.push(P(
  "A second diagnostic capture was performed using an otherwise identical script (capture_single_transcript_thinking.py) with thinking_config(include_thoughts=True, thinking_budget=8192) added to the GenerateContentConfig. The five per-condition transcripts are stored under transcripts_thinking/. The added thinking budget increased per-call output tokens by between approximately 150 and 1 500 (the model used far less of the 8 192-token budget than was made available), corresponding to a marginal cost increase of less than $0.01 per call. The number of validate_smiles calls per condition (13 at full / half / quarter / eighth, 21 at tenth) was unchanged or modestly elevated relative to the no-thinking capture; the final submission still contained 14 schema-valid reactions in every condition."
));

children.push(P(
  "Each condition produced two thought blocks: a first block in which the model identifies the figure and proposes SMILES strings for each compound, and a second block in which it assembles the per-entry table after the SMILES strings have been validated. The first block is the relevant artefact for understanding what chemistry the model believes it is reading. The structural identifications extracted from those first blocks are summarised in Table 6; verbatim excerpts from each first block are reproduced in Table 7."
));

const seenChemRows = [
  ["Resolution", "Substrate 1 (1a / 1b)", "Substrate 2", "Catalyst set", "Solvent panel"],
  ["Full",   "Boc-Gly-OH / Boc-L-Phe-OH",          "benzylamine",     "A, B, C, D (5 mol%)",            "toluene, PhF, (CH2Cl)2, CPME, DMF"],
  ["Half",   "Boc-Gly-OH / Boc-Phe-OH",            "benzylamine",     "A, B, C, D (5 mol%)",            "toluene, PhF, (CH2Cl)2, CPME, DMF"],
  ["Quarter","Boc-Gly-OH / Boc-Phe-OH",            "benzylamine",     "A, B, C, D (5 mol%)",            "toluene, PhF, (CH2Cl)2, CPME, DMF"],
  ["Eighth", "Boc-Gly-OH / Boc-Ala-OH",            "ethylenediamine", "A, B, C, D (5 mol%)",            "toluene, THF, DCE, CPME, DMF"],
  ["Tenth",  "benzamide / nonanamide",             "methylamine",     "I–VI (ammonium salts, 10 mol%)", "toluene, THF, CH3CN, i-PrOH, DMF"],
];
children.push(table([960, 2700, 1500, 2200, 2000], seenChemRows));
children.push(P("Table 6 — Compounds, catalysts, and solvents identified by the model in its first thought block at each resolution, extracted from transcripts_thinking/<condition>.json. The full, half, and quarter conditions all identify the ground-truth chemistry (Boc-amino-acid + benzylamine, with catalysts A–D and a five-solvent panel). The eighth condition retains the Boc scaffold and the catalyst labelling but substitutes ethylenediamine for benzylamine and alanine for phenylalanine. The tenth condition replaces the entire reaction class with an amide-to-N-methyl-amide transamidation using ammonium-salt catalysts.", { run: { italics: true, size: 18 } }));

// Verbatim excerpts table (one row per resolution).
function quoteRow(cond, q) {
  return [cond, { text: q }];
}
const quoteRows = [
  ["Resolution", "First thought block — opening identification (verbatim excerpt)"],
  ["Full",
    "“It seems we’re dealing with the coupling of Boc-protected amino acids with benzylamine. … We have two main ones: 1a, which is BocHN-CH2-CO2H (glycine derivative), and 1b, where R is Bn (benzyl) on the amino acid, i.e., BocHN-CH(Bn)-CO2H (a phenylalanine derivative). The latter has a stereocenter. I need to be careful with that.”"],
  ["Half",
    "“The core reaction is a coupling between a Boc-protected amino acid and benzylamine, forming an amide bond. … The phenylalanine derivative (1b) has a chiral center, and the drawing specifies a wedge. I have to be very careful here. … I’ll apply the Cahn-Ingold-Prelog (CIP) rules.”"],
  ["Quarter",
    "“The reaction seems to be a condensation of a Boc-protected amino acid derivative with benzylamine. … I see two primary substrates: 1a (R=H, Boc-Gly-OH) and 1b (R=Bn, Boc-Phe-OH). I need to make sure to account for the difference in R groups. I think the R group is not stereochemically explicit in the drawing, so I will take that into account.”"],
  ["Eighth",
    "“It looks like we’re optimizing a reaction involving the coupling of an N-Boc protected amino acid (Substrate 1) with ethylenediamine (Substrate 2) to form an amide product (Product 3). Substrate 1 can be either a glycine derivative (1a, R=H) or an alanine derivative (1b, R=Me). The ethylenediamine (2) is present as 1.5 equivalents, which is standard for coupling reactions.”"],
  ["Tenth",
    "“I’m looking at data from Table 1, which details the optimization of reaction conditions for transamidation of amides using methylamine. The general reaction involves an amide (1) reacting with methylamine (2) to produce an N-methylated amide (3). … Compound 1 is an amide, either benzamide (1a, NC(=O)c1ccccc1) or nonanamide (1b, CCCCCCCCC(N)=O). Methylamine (2, CN) is used in excess. The catalysts (10 mol%) are ammonium salts.”"],
];
children.push(table([960, 8400], quoteRows));
children.push(P("Table 7 — Verbatim opening excerpts of the first thought block at each resolution. The text was emitted by the model itself when thinking_config(include_thoughts=True) was active and is reproduced without paraphrase, with only inline ellipses inserted to elide chains of subordinate sentences. The excerpts make explicit that the chemistry the model believes it is reading is fully internalised — every condition produces a coherent, self-consistent identification of substrates, reagents, and conditions, even when, as at the eighth and tenth conditions, no element of that identification corresponds to the ground-truth figure.", { run: { italics: true, size: 18 } }));

children.push(P(
  "Three observations follow from Tables 6 and 7. First, the level of confidence and structural detail expressed in the thinking trace is invariant across resolutions: the model does not, at any scale, express uncertainty about what it is looking at, request clarification, or qualify its identification with hedging language. Second, the reasoning produced at the eighth and tenth conditions is internally consistent in the same sense that the reasoning at the full and quarter conditions is — equivalents, mol percentages, reaction times, and solvent panels are all assembled into a single coherent experimental table — but the underlying reactant identifications are, in those two conditions, hallucinated. Third, because the thinking trace is offered by the model with the same degree of fluency in both regimes, the trace itself cannot be used as a self-report of correctness; an external comparison against ground truth remains the only operational way to distinguish correct from confidently fabricated extractions in this pipeline."
));

children.push(P(
  "The methodological correction made in this subsection — that the Gemini API does in fact expose a human-readable chain-of-thought when include_thoughts is enabled, contrary to what was claimed earlier in this report based on the production extractor's behaviour — does not alter any of the quantitative results in Sections 3, 4, and 5.1 through 5.5. It does, however, mean that future executions of this pipeline could capture and persist the per-image reasoning trace at marginal additional cost, which may be useful for failure-mode auditing on the lower-resolution conditions or for any other case in which the structural extraction is suspected to disagree with the underlying image."
));

children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H("6. Conclusion", HeadingLevel.HEADING_1));
children.push(P(
  "On the GT3_Maarten 16-image benchmark, the Gemini-3-Flash-Preview vision agent tolerates an aggressive 4× downscaling of input images in each spatial dimension with no meaningful degradation of reference-based extraction quality (Partial F1 from 0.79 to 0.78). At an 8× downscaling the extraction quality drops sharply (Partial F1 = 0.25), and at a 10× downscaling it collapses to a near-zero floor (Partial F1 = 0.06). The transition between the plateau and the floor is narrower than the geometric spacing of the conditions tested and exhibits highly bimodal per-image behaviour at the intermediate eighth-resolution condition. The collapse is masked by schema-based validity signals, which remain near-perfect throughout the failure regime. Image down-sampling is not a useful cost-reduction strategy for this pipeline, because output-token volume — fixed by the JSON schema — dominates total cost and does not decrease with input size, while reasoning effort on degraded images partially offsets any savings on input tokens. The result motivates (i) reporting reference-based metrics rather than schema-pass rates in any future evaluation of the pipeline, (ii) characterising per-image variance over multiple repeat runs before claiming any deterministic resolution threshold, and (iii) treating image down-sampling as a quality-degradation lever rather than a cost-reduction lever in this pipeline."
));

// Appendix
children.push(H("Appendix A — Reproducibility", HeadingLevel.HEADING_1));
children.push(P("All four runs were executed from the Single_SDK_agent project tree with the following command template:"));
children.push(new Paragraph({
  spacing: { before: 60, after: 60 },
  children: [new TextRun({
    text:
      "LLM_PROVIDER=gemini GEMINI_MODEL=gemini-3-flash-preview \\\n" +
      "  .venv/bin/python3 cli.py benchmark \\\n" +
      "    corpus/Benchmark_kasper_GT3_Maarten[_half|_quarter|_eighth|_tenth] \\\n" +
      "    --out benchmark_runs/run_gemini3flash[_half|_quarter|_eighth|_tenth] \\\n" +
      "    --model gemini-3-flash-preview --concurrency 4",
    font: "Courier New", size: 18,
  })],
}));
children.push(P("Down-sampled corpora were produced by Lanczos resampling each PNG to ⌊W/k⌋ × ⌊H/k⌋ with k ∈ {2, 4, 8, 10} via the Pillow library (version 12.2.0); ground-truth JSON files and the manifest were copied unchanged. All scoring used scripts/eval_via_collective.py against Collective_autogen/eval/ground_truth/. The Gemini API was called via google-genai with automatic_function_calling disabled and tool_config set to ANY; sampling temperature was left at the SDK default. No seed was set and per-resolution variance was not measured."));

children.push(H("Appendix B — File locations", HeadingLevel.HEADING_1));
const fileRows = [
  ["Artefact", "Path (relative to Single_SDK_agent/)"],
  ["Full-resolution corpus",   "corpus/Benchmark_kasper_GT3_Maarten/"],
  ["Half-resolution corpus",   "corpus/Benchmark_kasper_GT3_Maarten_half/"],
  ["Quarter-resolution corpus","corpus/Benchmark_kasper_GT3_Maarten_quarter/"],
  ["Eighth-resolution corpus", "corpus/Benchmark_kasper_GT3_Maarten_eighth/"],
  ["Tenth-resolution corpus",  "corpus/Benchmark_kasper_GT3_Maarten_tenth/"],
  ["Full-resolution run",      "benchmark_runs/run_gemini3flash/"],
  ["Half-resolution run",      "benchmark_runs/run_gemini3flash_half/"],
  ["Quarter-resolution run",   "benchmark_runs/run_gemini3flash_quarter/"],
  ["Eighth-resolution run",    "benchmark_runs/run_gemini3flash_eighth/"],
  ["Tenth-resolution run",     "benchmark_runs/run_gemini3flash_tenth/"],
  ["Showcase image set",       "benchmark_runs/resolution_report/showcase_resolutions/"],
  ["Figure 2 (PNG)",           "benchmark_runs/resolution_report/fig_resolution_ablation.png"],
  ["Figure source script",     "benchmark_runs/resolution_report/build_figure.py"],
  ["Single-image transcripts (§5)", "benchmark_runs/resolution_report/transcripts/{full,half,quarter,eighth,tenth}.json"],
  ["Single-image thinking transcripts (§5.6)", "benchmark_runs/resolution_report/transcripts_thinking/{full,half,quarter,eighth,tenth}.json"],
  ["Transcript capture script (no thinking)",  "benchmark_runs/resolution_report/capture_single_transcript.py"],
  ["Transcript capture script (with thinking)", "benchmark_runs/resolution_report/capture_single_transcript_thinking.py"],
  ["This report",              "benchmark_runs/resolution_report/resolution_ablation_report.docx"],
];
children.push(table([3000, 6360], fileRows));

// ---------- build ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } }, // 11pt
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: "2E74B5" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARG, right: MARG, bottom: MARG, left: MARG },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT, `(${buf.length} bytes)`);
});
