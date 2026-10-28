const state = { catalog: null, components: [], variables: [], functions: [], functionText: '', presets: [], dragData: null, curves: {critical: [], caustic: []}, lensInfo: {r: [], kappa: [], magnification: []}, lensInfoImage: null, lensInfoPending: false, view: {xmin:-5, xmax:5, ymin:-5, ymax:5} };
const $ = (selector) => document.querySelector(selector);
const componentsEl = $('#components');
let latestImageData = null;
let latestImageDataUrl = null;
const FUNCTION_DEFAULT_ARGS = {
  'parameters.sample_normal': '0, 1',
  'parameters.sample_uniform': '0, 1',
  'parameters.sample_lognormal': '0, 1',
  'parameters.sample_exponential': '1',
  'parameters.sample_choice': '0, 1',
  'parameters.log10': '1e12',
  'parameters.halo_concentration_mass': '1e12,0.3',
  'parameters.galaxy_effective_radius_mass': '0.3,10.5,"early"',
  'parameters.halo_to_stellar_mass': '12,0.3',
  'parameters.log10_kpc_to_arcsec': '0.5,0.3',
};
const LENS_PARAMETER_SPECS = {
  pert: [[0,'z_fid'],[1,'gamma'],[4,'theta_gamma'],[7,'kappa']],
  clus3: [[0,'z_fid'],[1,'delta'],[4,'theta_delta']],
  mpole: [[0,'z_fid'],[1,'epsilon'],[4,'theta'],[5,'m'],[6,'n']],
  crline: [[0,'z_fid'],[1,'kappa_0'],[4,'theta_0'],[5,'epsilon']],
  point: [[0,'z_lens'],[1,'M_total'],[2,'x'],[3,'y']],
  sie: [[0,'z_lens'],[1,'sigma'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_core']],
  jaffe: [[0,'z_lens'],[1,'sigma'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_trunc'],[7,'r_core']],
  pow: [[0,'z_fid'],[1,'r_Ein'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'gamma']],
  powpot: [[0,'z_fid'],[1,'r_Ein'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'gamma']],
  sers: [[0,'z_lens'],[1,'M_total (Msun)'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_e'],[7,'n']],
  serspot: [[0,'z_lens'],[1,'M_total'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'r_e'],[7,'n']],
  hern: [[0,'z_lens'],[1,'M_total'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_b']],
  ahern: [[0,'z_lens'],[1,'M_total'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_b']],
  hernpot: [[0,'z_lens'],[1,'M_total'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'r_b']],
  nfw: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_vir']],
  anfw: [[0,'z_lens'],[1,'M_vir (Msun)'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_vir']],
  gaupot: [[0,'z_lens'],[1,'z_source/fid'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'sigma'],[7,'kappa_0']],
  nfwpot: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'c_vir']],
  gnfw: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_minus_2'],[7,'alpha']],
  gnfwpot: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'c_minus_2'],[7,'alpha']],
  tnfw: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_vir'],[7,'t']],
  tnfwpot: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'c_vir'],[7,'t']],
  acnfw: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_vir'],[7,'beta_core']],
  ein: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'c_E'],[7,'alpha_E']],
  einpot: [[0,'z_lens'],[1,'M_vir'],[2,'x'],[3,'y'],[4,'ellipticity_pot'],[5,'theta'],[6,'c_E'],[7,'alpha_E']],
  gals: [[0,'sigma_star'],[1,'x'],[2,'y'],[3,'r_trunc_star'],[4,'eta']]
};
const SOURCE_PARAMETER_SPECS = {
  gauss: [[0,'z_source'],[1,'flux'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'sigma']],
  sersic: [[0,'z_source'],[1,'flux'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_e'],[7,'n']],
  tophat: [[0,'z_source'],[1,'flux'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_0']],
  moffat: [[0,'z_source'],[1,'flux'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'alpha'],[7,'beta']],
  jaffe: [[0,'z_source'],[1,'flux'],[2,'x'],[3,'y'],[4,'ellipticity'],[5,'theta'],[6,'r_trunc'],[7,'r_core']],
  srcs: [[0,'z_source'],[1,'f_norm']],
  point: [[0,'z_source'],[1,'x'],[2,'y']]
};
const LUMINOUS_LENSES = new Set(['sers']);
let nextComponentId = 1;

function parameterSpecs(category, kind) {
  return (category === 'lens' ? LENS_PARAMETER_SPECS : SOURCE_PARAMETER_SPECS)[kind] || [];
}

async function boot() {
  state.catalog = await fetch('/api/catalog').then((response) => response.json());
  state.presets = await fetch('/api/presets').then((response) => response.json()).then((result) => result.presets || []);
  setupCanvasPages();
  $('#criticalOverlayToggle').disabled = true;
  $('#fullscreenCriticalOverlayToggle').disabled = true;
  $('#criticalOverlayToggle').insertAdjacentHTML('afterend', '<span id="criticalCurveStatus" class="curve-status">Run a simulation</span>');
  renderPalette();
  bindDropzone();
  $('#previewBtn').addEventListener('click', () => submit('/api/preview'));
  $('#runBtn').addEventListener('click', () => submit('/api/run'));
  $('#runMode').addEventListener('change', (event) => {
    const isBatch = event.target.value === 'batch';
    $('#batchNameWrap').hidden = !isBatch;
    $('#batchCountWrap').hidden = !isBatch;
  });
  $('#functionEditor').addEventListener('input', syncFunctionEditor);
  $('#clearFunctionsBtn').addEventListener('click', () => { $('#functionEditor').value = ''; syncFunctionEditor(); });
  renderFunctionEditor();
}
function setupCanvasPages() {
  const canvas = $('.canvas');
  canvas.insertAdjacentHTML('afterbegin', '<nav class="canvas-tabs" aria-label="Canvas views"><button class="canvas-tab active" data-canvas-page="editor" aria-selected="true">Components</button><button class="canvas-tab" data-canvas-page="functions" aria-selected="false">Functions</button><button class="canvas-tab" data-canvas-page="diagram" aria-selected="false">Lens info</button><button class="canvas-tab" data-canvas-page="presets" aria-selected="false">Presets</button><button class="canvas-tab" data-canvas-page="results" aria-selected="false">Past results</button></nav>');
  canvas.insertAdjacentHTML('beforeend', '<section id="componentDiagram" class="component-diagram" hidden aria-label="Lens position map"></section>');
  canvas.insertAdjacentHTML('beforeend', '<section id="lensProfiles" class="lens-profiles" hidden aria-label="Radial lens profiles"></section>');
  canvas.insertAdjacentHTML('beforeend', '<section id="presetPanel" class="preset-panel" hidden aria-label="Saved presets"><div class="preset-create"><label>Preset name<input id="presetName" type="text" placeholder="e.g. Compact group"></label><button id="savePresetBtn" class="button primary" type="button">Save input</button></div><div id="presetList" class="preset-list"></div></section>');
  canvas.insertAdjacentHTML('beforeend', '<section id="functionPanel" class="function-panel" hidden aria-label="Function definitions"><div class="function-editor-head"><div><strong>Scaling laws and variables</strong><small>Assignments use standard arithmetic, built-ins, <code>np.*</code>, or <code>!parameters.*</code>. Batch distributions can use <code>!parameters.sample_normal(mean, sigma)</code>, <code>sample_uniform(low, high)</code>, or the other sampling helpers below.</small></div><button id="clearFunctionsBtn" class="button secondary" type="button">Clear</button></div><textarea id="functionEditor" class="function-editor" spellcheck="false" aria-label="Function and variable definitions" placeholder="# Example&#10;mass = !parameters.sample_normal(1e12, 1e11)&#10;position = !parameters.sample_uniform(-1, 1)"></textarea><div id="functionLibrary" class="function-library"></div></section>');
  canvas.insertAdjacentHTML('beforeend', '<section id="resultsPanel" class="results-panel" hidden aria-label="Past simulation results"><div class="results-head"><div><strong>Compare past results</strong><small>Choose two results with matching dimensions to inspect their residual.</small></div><button id="refreshResultsBtn" class="button secondary" type="button">Refresh</button></div><div class="comparison-controls"><label>Result A<select id="resultA"></select></label><label>Result B<select id="resultB"></select></label><button id="compareBtn" class="button primary" type="button">Compare</button></div><div id="pastResultsList" class="past-results-list"></div><div id="comparisonPanel" class="comparison-panel" hidden></div></section>');
  renderPresets();
  $('#savePresetBtn').addEventListener('click', () => savePreset().catch((error) => { $('#output').textContent = error.message; $('#statusText').textContent = 'Needs attention'; }));
  $('#refreshResultsBtn').addEventListener('click', () => loadPastResults().catch((error) => {
    $('#pastResultsList').innerHTML = `<div class="comparison-error">${escapeHtml(error.message)}</div>`;
  }));
  $('#compareBtn').addEventListener('click', compareResults);
  canvas.querySelectorAll('[data-canvas-page]').forEach((button) => button.addEventListener('click', () => setCanvasPage(button.dataset.canvasPage)));
}
function setCanvasPage(page) {
  const diagram = $('#componentDiagram');
  const profiles = $('#lensProfiles');
  const editorVisible = page === 'editor';
  const diagramVisible = page === 'diagram';
  const presetsVisible = page === 'presets';
  const functionsVisible = page === 'functions';
  const resultsVisible = page === 'results';
  const pageHead = $('#canvasPageHead');
  $('#dropzone').hidden = !editorVisible;
  $('.canvas-actions').hidden = !editorVisible;
  pageHead.hidden = !editorVisible && !diagramVisible;
  pageHead.querySelector('span').textContent = diagramVisible ? 'Lens information' : 'Model canvas';
  pageHead.querySelector('small').textContent = diagramVisible ? 'diagnostics and radial profiles' : 'reorder components by dragging';
  diagram.hidden = !diagramVisible;
  profiles.hidden = !diagramVisible;
  $('#presetPanel').hidden = !presetsVisible;
  $('#functionPanel').hidden = !functionsVisible;
  $('#resultsPanel').hidden = !resultsVisible;
  if (functionsVisible) renderFunctionEditor();
  document.querySelectorAll('[data-canvas-page]').forEach((button) => { const active = button.dataset.canvasPage === page; button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); });
  if (page === 'diagram') { renderDiagram(); renderLensInfo(); }
  if (resultsVisible) loadPastResults().catch((error) => {
    $('#pastResultsList').innerHTML = `<div class="comparison-error">${escapeHtml(error.message)}</div>`;
  });
}
function legacyFunctionDefinitions(functions) { return (functions || []).map((item, index) => `$fn_${index + 1} = !${item.id}(${item.args || ''})`).join('\n'); }
function presetFunctionText(data) { const text = String(data.functionDefinitions ?? data.functionText ?? '').trim(); const legacy = legacyFunctionDefinitions(data.functions).trim(); if (!text) return legacy || (data.variables || []).map((item) => `${item.name} = ${item.value}`).join('\n'); if (!legacy) return text; return `${text}\n${legacy}`; }
function presetData() { syncComponentsFromDom(); syncFunctionEditor(); const functions = structuredClone(state.functions); const functionDefinitions = [state.functionText.trim(), legacyFunctionDefinitions(functions).trim()].filter(Boolean).join('\n'); return {config: Object.fromEntries([...document.querySelectorAll('[data-config]')].map((input) => [input.dataset.config, input.value])), components: structuredClone(state.components), variables: structuredClone(state.variables), functions, functionText: functionDefinitions, functionDefinitions}; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character])); }
function renderPresets() { const list = $('#presetList'); if (!list) return; const presets = state.presets; list.innerHTML = presets.length ? presets.map((preset, index) => { const data = preset.data || {}; const functionCount = presetFunctionText(data).split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith('#')).length; return `<article class="preset-item"><div><strong>${escapeHtml(preset.name)}</strong><small>${(data.components || []).length} component(s) · ${functionCount} function definition(s)</small></div><div class="preset-actions"><button type="button" data-preset-load="${index}" class="button secondary">Load</button><button type="button" data-preset-delete="${index}" class="icon-button" title="Delete preset">×</button></div></article>`; }).join('') : '<div class="preset-empty">No saved inputs yet.</div>'; list.querySelectorAll('[data-preset-load]').forEach((button) => button.addEventListener('click', () => loadPreset(Number(button.dataset.presetLoad)))); list.querySelectorAll('[data-preset-delete]').forEach((button) => button.addEventListener('click', () => deletePreset(Number(button.dataset.presetDelete)))); }
async function savePreset() { const name = $('#presetName').value.trim(); if (!name) { $('#presetName').focus(); return; } const response = await fetch('/api/presets', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, data:presetData()})}); const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Failed to save preset'); state.presets.unshift(result.preset); $('#presetName').value = ''; renderPresets(); }
function loadPreset(index) { const preset = state.presets[index]; if (!preset) return; const data = preset.data || {}; state.components = data.components || []; state.variables = data.variables || []; state.functions = data.functions || []; state.functionText = presetFunctionText(data); Object.entries(data.config || {}).forEach(([key, value]) => { const input = document.querySelector(`[data-config="${key}"]`); if (input) input.value = value; }); render(); renderFunctionEditor(); setCanvasPage('editor'); }
async function deletePreset(index) { const preset = state.presets[index]; if (!preset) return; const response = await fetch(`/api/presets/${encodeURIComponent(preset.id)}`, {method:'DELETE'}); if (!response.ok) { const result = await response.json(); throw new Error(result.error || 'Failed to delete preset'); } state.presets.splice(index, 1); renderPresets(); }
function curveMarkup(curves, xPosition, yPosition) { return (curves || []).map((segment) => `<polyline class="diagram-curve" points="${segment.map(([x, y]) => `${xPosition(x)},${yPosition(y)}`).join(' ')}"/>`).join(''); }
function renderDiagram() {
  const diagram = $('#componentDiagram');
  const width = 760;
  const height = 460;
  const margin = {left:58, right:24, top:24, bottom:48};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xPosition = (value) => margin.left + (Number(value) + 2) / 4 * plotWidth;
  const yPosition = (value) => margin.top + (2 - Number(value)) / 4 * plotHeight;
  const colors = {lens:'#111', source:'#d52b2b'};
  const ticks = [-2, -1, 0, 1, 2];
  const grid = ticks.map((tick) => `<line class="diagram-grid" x1="${xPosition(tick)}" y1="${margin.top}" x2="${xPosition(tick)}" y2="${height - margin.bottom}"/><line class="diagram-grid" x1="${margin.left}" y1="${yPosition(tick)}" x2="${width - margin.right}" y2="${yPosition(tick)}"/>`).join('');
  const labels = ticks.map((tick) => `<text class="diagram-label" x="${xPosition(tick)}" y="${height - margin.bottom + 20}" text-anchor="middle">${tick}</text><text class="diagram-label" x="${margin.left - 10}" y="${yPosition(tick) + 4}" text-anchor="end">${tick}</text>`).join('');
  const points = state.components.map((item) => { const [xIndex, yIndex] = item.category === 'source' && item.kind === 'point' ? [1, 2] : [2, 3]; const x = Number(item.parameters[xIndex]); const y = Number(item.parameters[yIndex]); if (!Number.isFinite(x) || !Number.isFinite(y)) return ''; const cx = xPosition(x); const cy = yPosition(y); const color = colors[item.category]; return `<g class="diagram-point" style="stroke:${color}" aria-label="${item.category} center"><line x1="${cx - 5}" y1="${cy}" x2="${cx + 5}" y2="${cy}"/><line x1="${cx}" y1="${cy - 5}" x2="${cx}" y2="${cy + 5}"/></g>`; }).join('');
  const caustics = curveMarkup(state.curves.caustic, xPosition, yPosition);
  const empty = state.components.length ? '' : '<div class="diagram-empty">Add components to see their positions here.</div>';
  diagram.innerHTML = `${empty}<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Component positions from minus two to two arcseconds"><rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" fill="#fbfcf8"/>${grid}<line class="diagram-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"/><line class="diagram-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"/>${labels}${caustics}${points}<text class="diagram-label" x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle">x (arcsec)</text><text class="diagram-label" transform="translate(15 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">y (arcsec)</text></svg><div class="diagram-legend"><span><i class="diagram-swatch" style="background:${colors.lens}"></i>lens center</span><span><i class="diagram-swatch" style="background:${colors.source}"></i>source center</span>${state.curves.caustic.length ? '<span><i class="diagram-swatch" style="background:#f1d447"></i>caustic</span>' : ''}</div>`;
}
function renderLensInfo() {
  const panel = $('#lensProfiles');
  if (!panel || panel.hidden) return;
  const lensInfo = state.lensInfo || {};
  const pending = state.lensInfoPending;
  const status = pending ? '<div id="lensInfoStatus" class="lens-info-status">Lens maps are still running...</div>' : state.lensInfoError ? `<div id="lensInfoStatus" class="lens-info-status">${escapeHtml(state.lensInfoError)}</div>` : '';
  const empty = !lensInfo.r?.length && !pending ? '<div class="lens-info-empty">Run a simulation to calculate lens information.</div>' : '';
  const image = state.lensInfoImage ? `<img class="lens-info-image" src="${state.lensInfoImage}?t=${Date.now()}" alt="Radial convergence and magnification profiles">` : '';
  panel.innerHTML = `${status}${empty}${image}<div class="diagram-legend"><span>Generated with writelens · coordinate along caustic major axis in arcsec · absolute magnification uses a logarithmic scale</span></div>`;
}
function renderPalette() {
  const addPalette = (target, items, category) => {
    target.innerHTML = items.map((item) => `<button class="palette-item" draggable="true" data-category="${category}" data-kind="${item.id}"><b>${item.label}</b><small>${item.help || item.id}</small></button>`).join('');
  };
  addPalette($('#lensPalette'), state.catalog.lensModels, 'lens');
  addPalette($('#sourcePalette'), state.catalog.sourceModels, 'source');
  document.querySelectorAll('.palette-item').forEach((item) => {
    item.addEventListener('dragstart', (event) => startDrag(event, {category:item.dataset.category, kind:item.dataset.kind}));
    item.addEventListener('dragend', clearDrag);
  });
}
function startDrag(event, data) {
  state.dragData = data;
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'copy';
    event.dataTransfer.setData('text/plain', JSON.stringify(data));
  }
}
function clearDrag() { state.dragData = null; }
function dragData(event) {
  let serialized = '';
  try { serialized = event.dataTransfer?.getData('text/plain') || ''; } catch (error) { serialized = ''; }
  if (serialized) {
    try { return JSON.parse(serialized); } catch (error) { return state.dragData; }
  }
  return state.dragData;
}
function bindDropzone() {
  const zone = $('#dropzone');
  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    zone.classList.remove('drag-over');
    const data = dragData(event);
    if (data?.category && data?.kind) addItem(data.category, data.kind);
    clearDrag();
  };
  zone.addEventListener('dragenter', (event) => event.preventDefault());
  zone.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', handleDrop);
  componentsEl.addEventListener('dragenter', (event) => event.preventDefault());
  componentsEl.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; });
  componentsEl.addEventListener('drop', handleDrop);
  componentsEl.addEventListener('click', (event) => { const removeButton = event.target.closest('.remove'); if (!removeButton) return; const card = removeButton.closest('.component:not(.dynamic-item)'); if (!card) return; event.preventDefault(); event.stopImmediatePropagation(); const cards = [...componentsEl.querySelectorAll('.component:not(.dynamic-item)')]; const removeIndex = cards.indexOf(card); syncComponentsFromDom(); if (removeIndex >= 0) state.components.splice(removeIndex, 1); render(); }, true);
}
function addItem(category, kind) {
  if (category === 'function') state.functions.push({id:kind, args:''});
  else if (category === 'variable') {
    const name = `var_${state.variables.length + 1}`;
    state.variables.push({name, value:'1'});
    state.functionText = `${state.functionText}${state.functionText ? '\n' : ''}${name} = 1`;
  }
  else { syncComponentsFromDom(); state.components.push({uid: `component_${nextComponentId++}`, category, kind, label: `${kind}_${state.components.length + 1}`, parameters: defaultParameters(category, kind), luminosity: category === 'source' ? '1e44' : undefined, massToLight: category === 'lens' && LUMINOUS_LENSES.has(kind) ? '4' : undefined, template: category === 'source' || LUMINOUS_LENSES.has(kind) ? 'elliptical' : undefined}); }
  render();
}
function defaultParameters(category, kind) {
  if (category === 'lens') {
    if (kind === 'point') return ['0.3','1e11','0','0'];
    if (kind === 'sie') return ['0.3','200','0','0','0.2','0','0.01','0'];
    if (kind === 'pow') return ['0.3','1','0','0','0.2','0','2','0'];
    if (kind === 'sers') return ['0.3','1e11','0','0','0.2','0','0.5','4'];
    if (kind === 'gaupot') return ['0.3','1.0','0','0','0.2','0','0.5','0.05'];
    return ['0.3','1e11','0','0','0.2','0','1','0'];
  }
  return kind === 'point' ? ['1.0','0','0','0','0','0','0','0'] : ['1.0','1','0','0','0.1','0','0.2','1'];
}
function functionArguments(functionId) { return FUNCTION_DEFAULT_ARGS[functionId] || '1'; }
function referenceFromDrop(data) {
  if (data.category === 'variable') {
    if (data.name) return `$${data.name}`;
    const name = `var_${state.variables.length + 1}`;
    state.variables.push({name, value:'1'});
    state.functionText = `${state.functionText}${state.functionText ? '\n' : ''}${name} = 1`;
    return `$${name}`;
  }
  if (data.category === 'function') {
    if (Number.isInteger(data.index)) return `$fn_${data.index + 1}`;
    state.functions.push({id:data.kind, args:functionArguments(data.kind)});
    return `$fn_${state.functions.length}`;
  }
  return null;
}
function dropReference(event, uid, parameter) {
  event.preventDefault();
  event.stopPropagation();
  const data = dragData(event);
  const reference = referenceFromDrop(data);
  if (!reference) { clearDrag(); return; }
  syncComponentsFromDom();
  const component = state.components.find((item) => item.uid === uid);
  if (component) component.parameters[parameter] = reference;
  clearDrag();
  render();
}
function render() {
  state.components.forEach((item) => { if (!item.uid) item.uid = `component_${nextComponentId++}`; });
  $('#emptyState').style.display = state.components.length ? 'none' : 'grid';
  componentsEl.innerHTML = state.components.map((item, index) => componentMarkup(item, index)).join('');
  componentsEl.querySelectorAll('.component').forEach((card) => {
    card.addEventListener('dragstart', (event) => { card.classList.add('dragging'); startDrag(event, {category:'component', uid:card.dataset.uid}); });
    card.addEventListener('dragend', () => { card.classList.remove('dragging'); clearDrag(); });
    card.addEventListener('dragover', (event) => { event.preventDefault(); const dragging = componentsEl.querySelector('.dragging'); if (dragging && dragging !== card) componentsEl.insertBefore(dragging, card); });
    card.querySelector('.remove').addEventListener('click', () => { const uid = card.dataset.uid; syncComponentsFromDom(); const removeIndex = state.components.findIndex((item) => item.uid === uid); if (removeIndex >= 0) state.components.splice(removeIndex, 1); render(); });
    card.querySelector('.component-label').addEventListener('input', (event) => state.components[index].label = event.target.value);
    const physicalInput = card.querySelector('[data-physical]');
    if (physicalInput) physicalInput.addEventListener('input', (event) => state.components[index][event.target.dataset.physical] = event.target.value);
    const templateInput = card.querySelector('[data-template]');
    if (templateInput) templateInput.addEventListener('change', (event) => state.components[index].template = event.target.value);
    const plotButton = card.querySelector('.spectrum-plot');
    if (plotButton) plotButton.addEventListener('click', () => {
      const redshiftText = card.querySelector('[data-parameter="0"]')?.value || '';
      const variableName = redshiftText.trim().replace(/^\$/, '');
      const variable = state.variables.find((item) => item.name === variableName);
      const resolvedText = variable ? variable.value : redshiftText;
      const redshift = Number.parseFloat(resolvedText);
      if (!Number.isFinite(redshift) || redshift < 0) {
        $('#output').textContent = `Spectrum plotting failed: resolve the source redshift to a non-negative number before plotting (received "${redshiftText}")`;
        $('#statusText').textContent = 'Needs attention';
        return;
      }
      return plotSpectrumWithFilter(templateInput.value, redshift).catch((error) => {
        $('#output').textContent = `Spectrum plotting failed: ${error.message}`;
        $('#statusText').textContent = 'Needs attention';
      });
    });
    card.querySelectorAll('[data-parameter]').forEach((input) => {
      input.addEventListener('input', (event) => state.components[index].parameters[Number(event.target.dataset.parameter)] = event.target.value);
      input.addEventListener('dragenter', (event) => event.preventDefault());
      input.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; });
      input.addEventListener('drop', (event) => dropReference(event, card.dataset.uid, Number(input.dataset.parameter)));
    });
  });
  const existing = document.querySelectorAll('.dynamic-item'); existing.forEach((item) => item.remove());
  if ($('#componentDiagram') && !$('#componentDiagram').hidden) renderDiagram();
}
function syncFunctionEditor() {
  const editor = $('#functionEditor');
  if (!editor) return;
  state.functionText = editor.value;
  const variables = [];
  for (const line of state.functionText.split(/\r?\n/)) {
    const match = line.match(/^\s*\$?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*(?:#.*)?$/);
    if (match && match[2]) variables.push({name: match[1], value: match[2]});
  }
  state.variables = variables;
}
function renderFunctionEditor() {
  const editor = $('#functionEditor');
  if (!editor) return;
  if (document.activeElement !== editor) editor.value = state.functionText || state.variables.map((item) => `${item.name} = ${item.value}`).join('\n');
  const library = $('#functionLibrary');
  if (!library || !state.catalog) return;
  library.innerHTML = `<div class="function-library-title">Built-in functions</div>${state.catalog.functions.map((item) => `<button type="button" class="function-snippet" data-snippet="${item.id}"><strong>${item.label}</strong><code>!${item.id}(${item.args.join(', ')})</code></button>`).join('')}`;
  library.querySelectorAll('[data-snippet]').forEach((button) => button.addEventListener('click', () => {
    const item = state.catalog.functions.find((entry) => entry.id === button.dataset.snippet);
    const line = `value_${state.variables.length + 1} = !${item.id}(${functionArguments(item.id)})`;
    const start = editor.selectionStart;
    const before = editor.value.slice(0, start);
    const after = editor.value.slice(editor.selectionEnd);
    editor.value = `${before}${before && !before.endsWith('\n') ? '\n' : ''}${line}${after}`;
    editor.focus();
    editor.selectionStart = editor.selectionEnd = before.length + line.length + (before && !before.endsWith('\n') ? 1 : 0);
    syncFunctionEditor();
  }));
}
function componentMarkup(item, index) { const models = item.category === 'lens' ? state.catalog.lensModels : state.catalog.sourceModels; const model = models.find((entry) => entry.id === item.kind); const physical = item.category === 'source' ? `<label>I-band luminosity (erg/s)<input data-physical="luminosity" value="${item.luminosity}"></label>` : LUMINOUS_LENSES.has(item.kind) ? `<label>M/L (solar)<input data-physical="massToLight" value="${item.massToLight}"></label>` : ''; const bright = item.category === 'source' || LUMINOUS_LENSES.has(item.kind); const spectrum = bright ? `<div class="spectrum-control"><label>Spectrum<select data-template>${state.catalog.spectralTemplates.map((template) => `<option value="${template.id}" ${template.id === item.template ? 'selected' : ''}>${template.label}</option>`).join('')}</select></label><button type="button" class="spectrum-plot" title="Plot selected spectrum">plot</button></div>` : ''; const fields = parameterSpecs(item.category, item.kind).map(([parameter, label]) => `<label>${label}<input data-parameter="${parameter}" value="${item.parameters[parameter]}"></label>`).join(''); return `<article class="component" draggable="true" data-index="${index}" data-uid="${item.uid}"><div class="component-top"><span class="drag-handle">⠿</span><div class="component-title"><span class="component-type">${item.category}</span><strong>${model?.label || item.kind}</strong></div><button class="remove">×</button></div><label class="name-field">Label<input class="component-label" value="${item.label}"></label><div class="parameter-grid">${physical}${spectrum}${fields}</div></article>`; }
function variableMarkup(item, index) { return `<article class="component dynamic-item variable-card" draggable="true" data-name="${item.name}"><div class="component-top"><span class="drag-handle">⠿</span><div class="component-title"><span class="component-type">variable</span><strong>$${item.name}</strong></div></div><div class="parameter-grid"><label>Name<input data-bind="variable:${index}:name" value="${item.name}"></label><label>Value<input data-bind="variable:${index}:value" value="${item.value}"></label></div></article>`; }
function functionMarkup(item, index) { const definition = state.catalog.functions.find((entry) => entry.id === item.id); const defaults = item.args || functionArguments(item.id); const argumentHelp = definition.args.map((argument, argumentIndex) => `${argument}: ${definition.argHelp[argumentIndex]}`).join(' · '); const galaxyType = item.id === 'parameters.galaxy_effective_radius_mass' ? `<label>Galaxy type<select data-function-option data-function-index="${index}"><option value="early" ${defaults.includes('"late"') ? '' : 'selected'}>Early-type</option><option value="late" ${defaults.includes('"late"') ? 'selected' : ''}>Late-type</option></select></label>` : ''; item.args = defaults; return `<article class="component dynamic-item function-card" draggable="true" data-index="${index}"><div class="component-top"><span class="drag-handle">⠿</span><div class="component-title"><span class="component-type">function → $fn_${index + 1}</span><strong>${definition.label}</strong></div><small>${definition.help}</small></div><label>Arguments<input data-bind="function:${index}:args" value="${defaults}"></label>${galaxyType}<small class="function-help">${argumentHelp}</small></article>`; }
function currentComponents() { return [...componentsEl.querySelectorAll('.component:not(.dynamic-item)')].map((card) => { const sourceIndex = Number(card.dataset.index); const source = state.components[sourceIndex]; const item = {...source, parameters:[...source.parameters]}; item.label = card.querySelector('.component-label').value; card.querySelectorAll('[data-parameter]').forEach((input) => { item.parameters[Number(input.dataset.parameter)] = input.value; }); const physicalInput = card.querySelector('[data-physical]'); if (physicalInput) item[physicalInput.dataset.physical] = physicalInput.value; const templateInput = card.querySelector('[data-template]'); if (templateInput) item.template = templateInput.value; return item; }); }
function syncComponentsFromDom() { if (!state.components.length || !componentsEl.querySelector('.component:not(.dynamic-item)')) return; state.components = currentComponents(); }
function payload() { syncComponentsFromDom(); syncFunctionEditor(); const components = state.components; const runMode = $('#runMode')?.value || 'single'; return { config: Object.fromEntries([...document.querySelectorAll('[data-config]')].map((input) => [input.dataset.config, input.value])), lenses: components.filter((item) => item.category === 'lens'), sources: components.filter((item) => item.category === 'source'), variables: state.variables, functions: state.functions, functionText: state.functionText, runMode, batchName: String($('#batchName')?.value || '').trim(), batchCount: Number($('#batchCount')?.value || 1) }; }
async function renderImages(images) { const latestBox = $('#latestImageBox'); const latestImage = $('#latestImage'); const history = $('#imageHistory'); history.innerHTML = ''; latestImageData = null; latestImageDataUrl = null; if (!images.length) { latestBox.hidden = true; return; } latestBox.hidden = false; latestImage.src = `${images[0].image}?t=${Date.now()}`; latestImageDataUrl = images[0].data || null; if (latestImageDataUrl) latestImageData = await fetch(`${latestImageDataUrl}?t=${Date.now()}`, {cache:'no-store'}).then((response) => response.json()); for (const [index, entry] of images.slice(1, 3).entries()) { const figure = document.createElement('figure'); const image = document.createElement('img'); image.src = `${entry.image}?t=${Date.now()}-${index}`; image.alt = `${index + 1} previous noisy I-band simulation`; const caption = document.createElement('figcaption'); caption.textContent = `${index + 1} previous`; figure.append(image, caption); history.append(figure); } }
function drawFullscreen() { if (!latestImageData) return; const canvas = $('#fullscreenCanvas'); const q = 10 ** Number($('#stretchSlider').value); $('#stretchValue').value = q; $('#stretchValue').textContent = q >= 100 ? q.toPrecision(4) : q.toFixed(2); canvas.width = latestImageData.width; canvas.height = latestImageData.height; const context = canvas.getContext('2d'); context.imageSmoothingEnabled = false; const pixels = context.createImageData(canvas.width, canvas.height); const denominator = Math.asinh(q); latestImageData.values.flat().forEach((value, index) => { const stretched = Math.max(0, Math.min(1, Math.asinh(Math.max(0, value) * q) / denominator)); const channel = Math.round(stretched * 255); const offset = index * 4; pixels.data[offset] = channel; pixels.data[offset + 1] = channel; pixels.data[offset + 2] = channel; pixels.data[offset + 3] = 255; }); context.putImageData(pixels, 0, 0); }
async function loadPastResults() {
  const response = await fetch('/api/results', {cache:'no-store'});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Failed to load past results');
  const results = result.results || [];
  const options = results.map((entry) => `<option value="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</option>`).join('');
  $('#resultA').innerHTML = options;
  $('#resultB').innerHTML = options;
  if (results.length > 1) $('#resultB').selectedIndex = 1;
  $('#compareBtn').disabled = results.length < 2;
  $('#pastResultsList').innerHTML = results.length ? results.map((entry) => `<figure class="past-result"><img src="${entry.image}?t=${entry.created}" alt="Past result ${escapeHtml(entry.name)}"><figcaption>${escapeHtml(entry.name)}</figcaption></figure>`).join('') : '<div class="preset-empty">No completed simulations yet.</div>';
}
async function compareResults() {
  const resultA = $('#resultA').value;
  const resultB = $('#resultB').value;
  const comparisonPanel = $('#comparisonPanel');
  if (!resultA || !resultB || resultA === resultB) {
    comparisonPanel.hidden = false;
    comparisonPanel.innerHTML = '<div class="comparison-error">Choose two different past results.</div>';
    return;
  }
  comparisonPanel.hidden = false;
  comparisonPanel.innerHTML = '<div class="comparison-loading">Building comparison...</div>';
  try {
    const response = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({resultA, resultB})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Comparison failed');
    const comparison = result.comparison;
    comparisonPanel.innerHTML = `<div class="comparison-grid"><figure><img src="${comparison.image}?t=${Date.now()}" alt="${escapeHtml(resultA)}"><figcaption>${escapeHtml(resultA)}</figcaption></figure><figure><img src="${comparison.imageB}?t=${Date.now()}" alt="${escapeHtml(resultB)}"><figcaption>${escapeHtml(resultB)}</figcaption></figure><figure><img src="${comparison.residual}?t=${Date.now()}" alt="Residual image"><figcaption>Residual (A − B)</figcaption></figure></div><small class="comparison-stat">Maximum absolute residual: ${Number(comparison.maxAbsoluteResidual).toPrecision(5)}</small>`;
  } catch (error) {
    comparisonPanel.innerHTML = `<div class="comparison-error">${escapeHtml(error.message)}</div>`;
  }
}
async function plotSpectrum(template) { const dialog = $('#spectrumDialog'); const title = state.catalog.spectralTemplates.find((item) => item.id === template)?.label || template; $('#spectrumTitle').textContent = title; const result = await fetch(`/api/spectrum/${encodeURIComponent(template)}`).then((response) => response.json()); const canvas = $('#spectrumCanvas'); const context = canvas.getContext('2d'); context.clearRect(0, 0, canvas.width, canvas.height); const padding = {top:20,right:22,bottom:62,left:68}; const points = result.wavelengths.map((wavelength, index) => [wavelength, result.fluxes[index]]).filter(([wavelength, flux]) => wavelength > 0 && flux > 0 && Number.isFinite(wavelength) && Number.isFinite(flux)); const wavelengths = points.map(([wavelength]) => wavelength); const fluxes = points.map(([, flux]) => flux); const minWavelength = Math.min(...wavelengths); const maxWavelength = Math.max(...wavelengths); const minFlux = Math.min(...fluxes); const maxFlux = Math.max(...fluxes); const logMinWavelength = Math.log10(minWavelength); const logMaxWavelength = Math.log10(maxWavelength); const logMinFlux = Math.log10(minFlux); const logMaxFlux = Math.log10(maxFlux); const plotWidth = canvas.width - padding.left - padding.right; const plotHeight = canvas.height - padding.top - padding.bottom; const xPosition = (value) => padding.left + (Math.log10(value) - logMinWavelength) / (logMaxWavelength - logMinWavelength) * plotWidth; const yPosition = (value) => padding.top + (logMaxFlux - Math.log10(value)) / (logMaxFlux - logMinFlux) * plotHeight; context.font = "11px 'DM Mono', monospace"; context.textAlign = 'center'; context.textBaseline = 'top'; for (let tick = Math.ceil(logMinWavelength); tick <= Math.floor(logMaxWavelength); tick += 1) { const x = xPosition(10 ** tick); context.strokeStyle = '#31483b'; context.lineWidth = 1; context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, canvas.height - padding.bottom); context.stroke(); context.fillStyle = '#9fb3a7'; context.fillText(`1e${tick}`, x, canvas.height - padding.bottom + 9); } context.textAlign = 'right'; context.textBaseline = 'middle'; for (let tick = Math.ceil(logMinFlux); tick <= Math.floor(logMaxFlux); tick += 1) { const y = yPosition(10 ** tick); context.strokeStyle = '#31483b'; context.beginPath(); context.moveTo(padding.left, y); context.lineTo(canvas.width - padding.right, y); context.stroke(); context.fillStyle = '#9fb3a7'; context.fillText(`1e${tick}`, padding.left - 8, y); } context.strokeStyle = '#71857a'; context.beginPath(); context.moveTo(padding.left, padding.top); context.lineTo(padding.left, canvas.height - padding.bottom); context.lineTo(canvas.width - padding.right, canvas.height - padding.bottom); context.stroke(); context.strokeStyle = '#ffb84d'; context.lineWidth = 2.5; context.lineJoin = 'round'; context.beginPath(); points.forEach(([wavelength, flux], index) => { const x = xPosition(wavelength); const y = yPosition(flux); if (index === 0) context.moveTo(x, y); else context.lineTo(x, y); }); context.stroke(); context.fillStyle = '#d6ebdc'; context.textAlign = 'center'; context.textBaseline = 'bottom'; context.font = "12px 'DM Mono', monospace"; context.fillText('Wavelength (Angstrom, log scale)', padding.left + plotWidth / 2, canvas.height - 8); context.save(); context.translate(15, padding.top + plotHeight / 2); context.rotate(-Math.PI / 2); context.fillText('Flux density (arbitrary units, log scale)', 0, 0); context.restore(); dialog.showModal(); }
function bindImageViewer() { const overlay = $('#fullscreenOverlay'); $('#fullscreenBtn').addEventListener('click', async () => { if (!latestImageData && latestImageDataUrl) latestImageData = await fetch(`${latestImageDataUrl}?t=${Date.now()}`, {cache:'no-store'}).then((response) => response.json()); overlay.hidden = false; drawFullscreen(); }); $('#stretchSlider').addEventListener('input', drawFullscreen); $('#closeFullscreenBtn').addEventListener('click', () => { overlay.hidden = true; }); overlay.addEventListener('click', (event) => { if (event.target === overlay) overlay.hidden = true; }); $('#closeSpectrumBtn').addEventListener('click', () => $('#spectrumDialog').close()); }
async function submit(endpoint) {
  const output = $('#output');
  const status = $('#statusText');
  const progress = $('#runProgress');
  const running = endpoint.endsWith('run');
  status.textContent = running ? 'Running' : 'Checking';
  progress.hidden = !running;
  progress.value = 0;
  output.textContent = 'Working...';
  try {
    const response = await fetch(endpoint, {method:'POST', cache:'no-store', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    if (!running) {
      output.textContent = `${result.runs} run(s)\n\n${result.input}`;
      status.textContent = 'Ready';
      return;
    }
    await pollRunJob(result.jobId, output, status);
  } catch (error) {
    output.textContent = error.message;
    status.textContent = 'Needs attention';
  }
}
async function pollRunJob(jobId, output, status) {
  const progress = $('#runProgress');
  for (;;) {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}?t=${Date.now()}`, {cache:'no-store'});
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Run job was not found');
    if (job.images?.length) {
      await renderImages(job.images);
      output.textContent = `${job.completed}/${job.total} run(s) complete\n${job.phase}`;
    } else {
      output.textContent = job.phase || 'Starting...';
    }
    progress.max = job.total || 1;
    progress.value = job.completed || 0;
    if (job.status === 'complete') {
      output.textContent = `${job.completed} run(s) completed\n\n${job.phase}`;
      status.textContent = `Complete · ${job.phase}`;
      progress.value = progress.max;
      progress.hidden = true;
      return;
    }
    if (job.status === 'error') {
      progress.hidden = true;
      throw new Error(job.error || 'Run failed');
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}
async function plotSpectrumWithFilter(template, redshift = 0) {
  const dialog = $('#spectrumDialog');
  const title = state.catalog.spectralTemplates.find((item) => item.id === template)?.label || template;
  $('#spectrumTitle').textContent = `${title} (z=${redshift})`;
  const response = await fetch(`/api/spectrum/${encodeURIComponent(template)}?z=${encodeURIComponent(redshift)}`, {cache:'no-store'});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Spectrum request failed (${response.status})`);
  const canvas = $('#spectrumCanvas');
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  const padding = {top:20,right:22,bottom:62,left:68};
  const points = result.wavelengths.map((wavelength, index) => [wavelength, result.fluxes[index]]).filter(([wavelength, flux]) => wavelength > 0 && flux > 0 && Number.isFinite(wavelength) && Number.isFinite(flux));
  const filterPoints = result.filterWavelengths.map((wavelength, index) => [wavelength, result.transmission[index]]).filter(([wavelength, transmission]) => wavelength > 0 && transmission > 0 && Number.isFinite(wavelength) && Number.isFinite(transmission));
  if (!points.length || !filterPoints.length) throw new Error('The spectrum or VIS filter returned no plottable data');
  const maxFlux = Math.max(...points.map(([, flux]) => flux));
  const maxTransmission = Math.max(...filterPoints.map(([, transmission]) => transmission));
  const logMinWavelength = Math.log10(Math.min(...points.map(([wavelength]) => wavelength)));
  const logMaxWavelength = Math.log10(Math.max(...points.map(([wavelength]) => wavelength)));
  const plotWidth = canvas.width - padding.left - padding.right;
  const plotHeight = canvas.height - padding.top - padding.bottom;
  const xPosition = (value) => padding.left + (Math.log10(value) - logMinWavelength) / (logMaxWavelength - logMinWavelength) * plotWidth;
  const yPosition = (value) => padding.top + (0 - Math.log10(value)) / 4 * plotHeight;
  context.font = "11px 'DM Mono', monospace";
  context.textAlign = 'center';
  context.textBaseline = 'top';
  for (let tick = Math.ceil(logMinWavelength); tick <= Math.floor(logMaxWavelength); tick += 1) {
    const x = xPosition(10 ** tick);
    context.strokeStyle = '#31483b';
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, canvas.height - padding.bottom);
    context.stroke();
    context.fillStyle = '#9fb3a7';
    context.fillText(`1e${tick}`, x, canvas.height - padding.bottom + 9);
  }
  context.textAlign = 'right';
  context.textBaseline = 'middle';
  for (let tick = -4; tick <= 0; tick += 1) {
    const y = yPosition(10 ** tick);
    context.strokeStyle = '#31483b';
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(canvas.width - padding.right, y);
    context.stroke();
    context.fillStyle = '#9fb3a7';
    context.fillText(`1e${tick}`, padding.left - 8, y);
  }
  context.strokeStyle = '#71857a';
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, canvas.height - padding.bottom);
  context.lineTo(canvas.width - padding.right, canvas.height - padding.bottom);
  context.stroke();
  const drawCurve = (curve, peak, color, lineWidth) => {
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.lineJoin = 'round';
    context.beginPath();
    curve.forEach(([wavelength, value], index) => {
      const x = xPosition(wavelength);
      const y = yPosition(value / peak);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  };
  drawCurve(points, maxFlux, '#ffb84d', 2.5);
  drawCurve(filterPoints, maxTransmission, '#62d9e8', 2);
  context.fillStyle = '#d6ebdc';
  context.textAlign = 'center';
  context.textBaseline = 'bottom';
  context.font = "12px 'DM Mono', monospace";
  context.fillText('Wavelength (Angstrom, log scale)', padding.left + plotWidth / 2, canvas.height - 8);
  context.save();
  context.translate(15, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText('Normalized flux / transmission (log scale)', 0, 0);
  context.restore();
  context.fillStyle = '#ffb84d';
  context.textAlign = 'left';
  context.fillText('spectrum', padding.left + 12, padding.top + 14);
  context.fillStyle = '#62d9e8';
  context.fillText('Euclid VIS transmission', padding.left + 92, padding.top + 14);
  dialog.showModal();
}
function drawCurveSegments(context, curves, width, height, bounds) { const xPosition = (value) => (Number(value) - bounds.xmin) / (bounds.xmax - bounds.xmin) * width; const yPosition = (value) => (bounds.ymax - Number(value)) / (bounds.ymax - bounds.ymin) * height; context.strokeStyle = '#d52b2b'; context.lineWidth = Math.max(1, width / 500); context.lineJoin = 'round'; (curves || []).forEach((segment) => { context.beginPath(); segment.forEach(([x, y], index) => { if (index === 0) context.moveTo(xPosition(x), yPosition(y)); else context.lineTo(xPosition(x), yPosition(y)); }); context.stroke(); }); }
function setCurveStatus(pending, error = '') { const status = $('#criticalCurveStatus'); const ready = !pending && state.curves.critical.length > 0; $('#criticalOverlayToggle').disabled = pending || !ready; $('#fullscreenCriticalOverlayToggle').disabled = pending || !ready; clearInterval(state.curveStatusTimer); if (pending) { state.curveStartedAt = Date.now(); const update = () => { if (status) status.textContent = `Processing critical curves with glafic... ${Math.floor((Date.now() - state.curveStartedAt) / 1000)}s`; }; update(); state.curveStatusTimer = setInterval(update, 1000); } else if (status) status.textContent = error ? `Curve generation failed: ${error}` : ready ? 'Critical curves ready' : 'No critical curves found'; }
function setLensInfoStatus(pending, error = '') { state.lensInfoPending = pending; state.lensInfoError = error; if (pending) { state.lensInfoStartedAt = Date.now(); } renderLensInfo(); }
function drawImageOverlay() { const canvas = $('#imageOverlayCanvas'); if (!canvas || !latestImageData) return; const overlayScale = 4; canvas.width = latestImageData.width * overlayScale; canvas.height = latestImageData.height * overlayScale; const context = canvas.getContext('2d'); context.clearRect(0, 0, canvas.width, canvas.height); if ($('#criticalOverlayToggle').checked) drawCurveSegments(context, state.curves.critical, canvas.width, canvas.height, state.view); }
async function pollCurves(url, token) { try { if (token !== state.curvePollToken) return; if (Date.now() - (state.curveStartedAt || Date.now()) > 130000) { setCurveStatus(false, 'generation did not finish within 130 seconds'); setLensInfoStatus(false, 'Lens-map generation did not finish within 130 seconds.'); return; } const response = await fetch(`${url}?t=${Date.now()}`, {cache:'no-store'}); if (token !== state.curvePollToken) return; if (!response.ok) { setTimeout(() => pollCurves(url, token), 250); return; } const diagnostics = await response.json(); if (!diagnostics.lensInfo || !diagnostics.critical || !diagnostics.caustic) { setTimeout(() => pollCurves(url, token), 250); return; } state.curves = diagnostics; state.lensInfo = diagnostics.lensInfo; state.lensInfoImage = diagnostics.lensInfoImage || (diagnostics.error ? null : url.replace(/web_curves\.json(?:\?.*)?$/, 'lens_info.png')); setCurveStatus(false, diagnostics.error || ''); setLensInfoStatus(false, diagnostics.error || ''); drawImageOverlay(); renderDiagram(); renderLensInfo(); } catch (error) { if (token === state.curvePollToken) setTimeout(() => pollCurves(url, token), 500); } }
const baseRenderImages = renderImages;
renderImages = async function(images) { if (!images.length) return; state.curvePollToken = (state.curvePollToken || 0) + 1; const token = state.curvePollToken; state.curves = images[0]?.curves || {critical: [], caustic: []}; state.lensInfo = images[0]?.lensInfo || {r: [], kappa: [], magnification: []}; state.lensInfoImage = images[0]?.lensInfoImage || null; const diagnosticsPending = images[0]?.diagnosticsPending ?? !(state.curves.critical.length > 0 || state.curves.caustic.length > 0 || state.lensInfo.r.length > 0); const diagnosticsUrl = images[0]?.diagnosticsUrl || images[0]?.curvesUrl || images[0]?.image?.replace(/ie_band\.png(?:\?.*)?$/, 'web_curves.json'); state.view = Object.fromEntries([...document.querySelectorAll('[data-config]')].map((input) => [input.dataset.config, Number(input.value)])); setCurveStatus(diagnosticsPending); setLensInfoStatus(diagnosticsPending); await baseRenderImages(images); drawImageOverlay(); renderDiagram(); renderLensInfo(); if (diagnosticsPending && diagnosticsUrl) pollCurves(diagnosticsUrl, token); };
const baseDrawFullscreen = drawFullscreen;
drawFullscreen = function() { baseDrawFullscreen(); if (!latestImageData) return; const imageCanvas = $('#fullscreenCanvas'); let curveCanvas = $('#fullscreenCurveCanvas'); if (!curveCanvas) { curveCanvas = document.createElement('canvas'); curveCanvas.id = 'fullscreenCurveCanvas'; curveCanvas.style.position = 'absolute'; curveCanvas.style.pointerEvents = 'none'; curveCanvas.style.zIndex = '1'; $('#fullscreenOverlay').append(curveCanvas); } curveCanvas.width = latestImageData.width * 4; curveCanvas.height = latestImageData.height * 4; curveCanvas.style.left = `${imageCanvas.offsetLeft}px`; curveCanvas.style.top = `${imageCanvas.offsetTop}px`; curveCanvas.style.width = `${imageCanvas.clientWidth}px`; curveCanvas.style.height = `${imageCanvas.clientHeight}px`; const context = curveCanvas.getContext('2d'); context.clearRect(0, 0, curveCanvas.width, curveCanvas.height); if ($('#fullscreenCriticalOverlayToggle')?.checked) drawCurveSegments(context, state.curves.critical, curveCanvas.width, curveCanvas.height, state.view); };
boot().then(bindImageViewer).then(() => { $('#criticalOverlayToggle').addEventListener('change', drawImageOverlay); $('#fullscreenCriticalOverlayToggle').addEventListener('change', drawFullscreen); }).catch((error) => { $('#output').textContent = error.message; $('#statusText').textContent = 'Offline'; });
