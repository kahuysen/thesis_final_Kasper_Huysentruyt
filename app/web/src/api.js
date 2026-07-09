// Thin fetch helpers around the FastAPI backend exposed by 5 App/app/main.py.

export async function fetchHealth() {
  const r = await fetch('/api/health');
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function createRun(file) {
  const fd = new FormData();
  fd.append('image', file);
  const r = await fetch('/api/runs', { method: 'POST', body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`upload failed (${r.status}): ${text || r.statusText}`);
  }
  return r.json(); // { run_id, input, bytes }
}

export async function createPdfJob(file, { model, provider, modelSize } = {}) {
  const fd = new FormData();
  fd.append('image', file);
  const params = new URLSearchParams();
  if (modelSize) params.set('model_size', modelSize);
  if (model)     params.set('extract_model', model);
  if (provider)  params.set('extract_provider', provider);
  const qs = params.toString();
  const url = '/api/pdf_jobs' + (qs ? '?' + qs : '');
  const r = await fetch(url, { method: 'POST', body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`pdf upload failed (${r.status}): ${text || r.statusText}`);
  }
  return r.json(); // { job_id, kind: "pdf", bytes, pdf_name }
}

export function pdfEventsUrl(jobId) {
  return `/api/pdf_jobs/${jobId}/events`;
}

export function pdfFileUrl(jobId, name) {
  return `/api/pdf_jobs/${jobId}/file/${name}`;
}

export async function fetchPdfJob(jobId) {
  const r = await fetch(`/api/pdf_jobs/${jobId}`);
  if (!r.ok) throw new Error(`pdf job ${r.status}`);
  return r.json();
}

export async function submitPdfFigures(jobId, figureIds, { model, provider, folderName } = {}) {
  const r = await fetch(`/api/pdf_jobs/${jobId}/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      figure_ids: figureIds,
      ...(model ? { model } : {}),
      ...(provider ? { provider } : {}),
      ...(folderName ? { folder_name: folderName } : {}),
    }),
  });
  if (!r.ok) {
    const t = await r.text().catch(() => '');
    throw new Error(t || `submit ${r.status}`);
  }
  return r.json(); // { ok, pairs: [{figure_id, child_run_id}] }
}

export function eventsUrl(runId, { model, provider } = {}) {
  const p = new URLSearchParams();
  if (model)    p.set('model', model);
  if (provider) p.set('provider', provider);
  const qs = p.toString();
  return `/api/runs/${runId}/events${qs ? '?' + qs : ''}`;
}

export function fileUrl(runId, name) {
  return `/api/runs/${runId}/file/${name}`;
}

export async function fetchInsight(runId) {
  const r = await fetch(`/api/runs/${runId}/insight`, { method: 'POST' });
  const j = await r.json().catch(() => ({ ok: false, error: `bad json (${r.status})` }));
  if (!r.ok || !j.ok) {
    return { ok: false, error: j.error || `insight failed (${r.status})`, status: r.status };
  }
  return j; // { ok, analyses: [...], csv }
}

export async function fetchRuns() {
  const r = await fetch('/api/runs');
  if (!r.ok) throw new Error(`runs ${r.status}`);
  return r.json();
}

export async function fetchFolders() {
  const r = await fetch('/api/folders');
  if (!r.ok) throw new Error(`folders ${r.status}`);
  return r.json();
}

export async function createFolder(name) {
  const r = await fetch('/api/folders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) throw new Error((await r.text()) || `create folder ${r.status}`);
  return r.json();
}

export async function renameFolder(id, name) {
  const r = await fetch(`/api/folders/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) throw new Error((await r.text()) || `rename folder ${r.status}`);
  return r.json();
}

export async function deleteFolder(id) {
  const r = await fetch(`/api/folders/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error((await r.text()) || `delete folder ${r.status}`);
  return r.json();
}

export async function fetchSettings() {
  const r = await fetch('/api/settings');
  if (!r.ok) throw new Error(`settings ${r.status}`);
  return r.json();
}

export async function saveProviderSettings(provider, payload) {
  const r = await fetch(`/api/settings/providers/${provider}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error((await r.text()) || `save ${provider} ${r.status}`);
  return r.json();
}

export async function saveDefaultProvider(provider) {
  const r = await fetch('/api/settings/default_provider', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });
  if (!r.ok) throw new Error((await r.text()) || `default ${r.status}`);
  return r.json();
}

export async function moveRun(runId, folderId) {
  const r = await fetch(`/api/runs/${runId}/folder`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder_id: folderId || null }),
  });
  if (!r.ok) throw new Error((await r.text()) || `move run ${r.status}`);
  return r.json();
}

export async function fetchExtraction(runId) {
  const r = await fetch(`/api/runs/${runId}/file/extraction.json`);
  if (!r.ok) throw new Error(`extraction ${r.status}`);
  return r.json();
}

// Load the persisted Rxn-INSIGHT output for a past run, if any.
// Returns the array of analyses (same shape as POST /api/runs/{id}/insight),
// or null if the file isn't there.
export async function fetchInsightArtifact(runId) {
  const r = await fetch(`/api/runs/${runId}/file/insight.json`);
  if (!r.ok) return null;
  try { return await r.json(); } catch { return null; }
}

// Parse a semicolon-separated list like "amine; carbonyl" -> ["amine","carbonyl"].
function parseList(s) {
  if (s == null || s === '') return [];
  if (Array.isArray(s)) return s.filter(Boolean);
  return String(s).split(/[;,]\s*/).map(x => x.trim()).filter(Boolean);
}

// Backend insight result has shape { entry_id, ok, row?, error? } where `row`
// follows pipeline/rxn_insight.py INSIGHT_CSV_COLUMNS. Map onto the keys the
// Workbench's StructuredInsightPanel + DetailDrawer expect.
export function adaptInsight(analyses) {
  const out = {};
  for (const a of (analyses || [])) {
    const id = a.entry_id;
    if (id == null) continue;
    const row = a.row || {};
    out[id] = {
      ok: !!a.ok,
      error: a.error || null,
      reaction_class:    row.CLASS || null,
      named_reaction:    row.NAME || null,
      rxn_class_id:      row.TAG || row.TAG2 || null,
      functional_groups: [...parseList(row.FG_REACTANTS), ...parseList(row.FG_PRODUCTS)],
      byproducts:        parseList(row['BY-PRODUCTS']),
      template:          row.TEMPLATE || null,
      scaffold:          row.SCAFFOLD || null,
      suggested_solvents: [], // not provided by backend
      hazards:           [], // not provided by backend
    };
  }
  return out;
}

// Backend models list: [{ id, label, provider }]. The Workbench wants
// optional `context` and `recommended` flags; the model picker tolerates
// missing fields. We forward verbatim plus mark the default-provider's
// first entry as recommended.
export function adaptModels(models, defaultProvider) {
  if (!Array.isArray(models)) return [];
  let recommendedTaken = false;
  return models.map(m => {
    const recommended = !recommendedTaken && m.provider === defaultProvider;
    if (recommended) recommendedTaken = true;
    return { ...m, recommended };
  });
}
