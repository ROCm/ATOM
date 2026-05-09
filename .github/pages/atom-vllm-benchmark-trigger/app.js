const state = {
  config: null,
  families: [],
  filteredFamilies: [],
  activeFamily: null,
  selectedTpSizes: [],
  selections: [],
  hidePopoverTimer: null,
};

const els = {};

function q(id) {
  return document.getElementById(id);
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function updateOutput(text, isError = false) {
  els.dispatchOutput.textContent = text;
  els.dispatchOutput.style.borderColor = isError ? "var(--accent-red)" : "var(--border)";
}

function setActiveFamily(family, anchorRect = null) {
  state.activeFamily = family;
  state.selectedTpSizes =
    family.supported_tp_sizes.length === 1 ? [family.supported_tp_sizes[0]] : [];
  renderFamilyList();
  renderTpPopover(anchorRect);
}

function hidePopoverSoon() {
  clearTimeout(state.hidePopoverTimer);
  state.hidePopoverTimer = setTimeout(() => {
    els.tpPopover.hidden = true;
  }, 120);
}

function keepPopoverOpen() {
  clearTimeout(state.hidePopoverTimer);
}

function renderFamilyList() {
  els.familyList.innerHTML = "";
  for (const family of state.filteredFamilies) {
    const item = document.createElement("div");
    item.className = "family-item";
    if (state.activeFamily?.family === family.family) {
      item.classList.add("active");
    }
    item.innerHTML = `
      <div class="family-title">${family.family}</div>
      <div class="family-meta">Supported TP sizes: ${family.supported_tp_sizes.join(", ")}</div>
    `;
    item.addEventListener("mouseenter", () => {
      setActiveFamily(family, item.getBoundingClientRect());
    });
    item.addEventListener("mouseleave", hidePopoverSoon);
    item.addEventListener("click", () => {
      setActiveFamily(family, item.getBoundingClientRect());
    });
    els.familyList.appendChild(item);
  }
}

function renderTpPopover(anchorRect = null) {
  if (!state.activeFamily) {
    els.tpPopover.hidden = true;
    return;
  }

  keepPopoverOpen();
  els.tpPopover.hidden = false;
  els.tpPopoverTitle.textContent = state.activeFamily.family;
  els.tpPopoverNote.textContent = `Supported TP sizes: ${state.activeFamily.supported_tp_sizes.join(", ")}`;
  els.tpCheckboxes.innerHTML = "";

  for (const tpSize of state.activeFamily.supported_tp_sizes) {
    const label = document.createElement("label");
    const checked = state.selectedTpSizes.includes(tpSize) ? "checked" : "";
    label.innerHTML = `
      <input type="checkbox" value="${tpSize}" ${checked} />
      <span>TP ${tpSize}</span>
    `;
    label.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) {
        if (!state.selectedTpSizes.includes(tpSize)) {
          state.selectedTpSizes.push(tpSize);
          state.selectedTpSizes.sort((a, b) => a - b);
        }
      } else {
        state.selectedTpSizes = state.selectedTpSizes.filter((value) => value !== tpSize);
      }
    });
    els.tpCheckboxes.appendChild(label);
  }

  const card = els.tpPopover.querySelector(".tp-popover-card");
  if (anchorRect) {
    const top = Math.min(anchorRect.top, window.innerHeight - 260);
    const left = Math.min(anchorRect.right + 16, window.innerWidth - 320);
    card.style.top = `${Math.max(20, top)}px`;
    card.style.left = `${Math.max(20, left)}px`;
  }
}

function selectionKey(selection) {
  return `${selection.family}::${selection.tp_sizes.join(",")}`;
}

function renderSelections() {
  els.selectionCount.textContent = String(state.selections.length);
  els.selectedSlots.innerHTML = "";
  if (state.selections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No workflow slots selected yet.";
    els.selectedSlots.appendChild(empty);
    return;
  }

  state.selections.forEach((selection, index) => {
    const chip = document.createElement("div");
    chip.className = "selection-chip";
    chip.innerHTML = `
      <div>
        <strong>Slot ${index + 1}: ${selection.family}</strong>
        <span>TP sizes: ${selection.tp_sizes.join(", ")}</span>
      </div>
      <button type="button">Remove</button>
    `;
    chip.querySelector("button").addEventListener("click", () => {
      state.selections = state.selections.filter(
        (item) => selectionKey(item) !== selectionKey(selection),
      );
      renderSelections();
    });
    els.selectedSlots.appendChild(chip);
  });
}

function addCurrentSelection() {
  if (!state.activeFamily) {
    updateOutput("Pick a model family first.", true);
    return;
  }
  if (state.selectedTpSizes.length === 0) {
    updateOutput("Select at least one TP size before adding the model selection.", true);
    return;
  }
  if (state.selections.length >= 8) {
    updateOutput("The current workflow supports at most 8 model selections.", true);
    return;
  }

  const selection = {
    family: state.activeFamily.family,
    tp_sizes: [...state.selectedTpSizes].sort((a, b) => a - b),
  };
  if (state.selections.some((item) => selectionKey(item) === selectionKey(selection))) {
    updateOutput("That model + TP selection is already in the workflow slot list.", true);
    return;
  }
  state.selections.push(selection);
  renderSelections();
  updateOutput(`Added ${selection.family} with TP sizes ${selection.tp_sizes.join(", ")}.`);
}

function filteredFamilies() {
  const query = els.familySearch.value.trim().toLowerCase();
  if (!query) {
    state.filteredFamilies = [...state.families];
    return;
  }
  state.filteredFamilies = state.families.filter((family) =>
    family.family.toLowerCase().includes(query),
  );
}

function buildPayload() {
  const parsePairs = (text) =>
    text
      .split(";")
      .map((pair) => pair.trim())
      .filter(Boolean)
      .map((pair) => pair.split(",").map((value) => Number(value.trim())));

  const parseCsv = (text) =>
    text
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);

  return {
    repository: els.repository.value.trim(),
    ref: els.ref.value.trim(),
    benchmark_client: els.benchmarkClient.value,
    model_selections: state.selections,
    isl_osl_pairs: parsePairs(els.islOslPairs.value),
    concurrency_values: parseCsv(els.concurrencyValues.value).map(Number),
    random_range_ratios: parseCsv(els.randomRangeRatios.value),
    oot_image: els.ootImage.value.trim(),
    publish_to_dashboard: els.publishToDashboard.checked,
    upload_to_custom_dashboard: els.uploadToCustomDashboard.checked,
  };
}

async function dispatchWorkflow() {
  try {
    if (state.selections.length === 0) {
      throw new Error("Add at least one model selection before triggering the workflow.");
    }
    const payload = buildPayload();
    updateOutput(`Dispatching workflow...\n\nPayload:\n${formatJson(payload)}`);
    const result = await fetchJson("/api/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    updateOutput(
      `Workflow dispatched successfully.\n\nCommand:\n${result.command.join(" ")}\n\nstdout:\n${result.stdout || "<empty>"}`,
    );
  } catch (error) {
    updateOutput(String(error.message || error), true);
  }
}

async function bootstrap() {
  Object.assign(els, {
    repository: q("repository"),
    ref: q("ref"),
    benchmarkClient: q("benchmark-client"),
    ootImage: q("oot-image"),
    islOslPairs: q("isl-osl-pairs"),
    concurrencyValues: q("concurrency-values"),
    randomRangeRatios: q("random-range-ratios"),
    publishToDashboard: q("publish-to-dashboard"),
    uploadToCustomDashboard: q("upload-to-custom-dashboard"),
    familySearch: q("family-search"),
    familyList: q("family-list"),
    selectedSlots: q("selected-slots"),
    selectionCount: q("selection-count"),
    dispatchOutput: q("dispatch-output"),
    tpPopover: q("tp-popover"),
    tpPopoverTitle: q("tp-popover-title"),
    tpPopoverNote: q("tp-popover-note"),
    tpCheckboxes: q("tp-checkboxes"),
    tpSelectAll: q("tp-select-all"),
    tpAddSelection: q("tp-add-selection"),
    dispatchRun: q("dispatch-run"),
    clearSelections: q("clear-selections"),
  });

  const [catalogResponse, configResponse] = await Promise.all([
    fetchJson("/api/catalog"),
    fetchJson("/api/config"),
  ]);
  state.config = configResponse;
  state.families = catalogResponse.families;
  state.filteredFamilies = [...state.families];

  els.repository.value = configResponse.repository;
  els.ref.value = configResponse.ref;
  els.islOslPairs.value = "1024,1024;1024,8192;8192,1024";
  els.concurrencyValues.value = "4,8,16,32,64,128,256,512";
  els.randomRangeRatios.value = "0.8";
  els.publishToDashboard.checked = false;
  els.uploadToCustomDashboard.checked = true;

  els.familySearch.addEventListener("input", () => {
    filteredFamilies();
    renderFamilyList();
  });
  els.tpPopover.addEventListener("mouseenter", keepPopoverOpen);
  els.tpPopover.addEventListener("mouseleave", hidePopoverSoon);
  els.tpSelectAll.addEventListener("click", () => {
    state.selectedTpSizes = [...state.activeFamily.supported_tp_sizes];
    renderTpPopover();
  });
  els.tpAddSelection.addEventListener("click", addCurrentSelection);
  els.dispatchRun.addEventListener("click", dispatchWorkflow);
  els.clearSelections.addEventListener("click", () => {
    state.selections = [];
    renderSelections();
    updateOutput("Selections cleared.");
  });

  renderFamilyList();
  renderSelections();
}

bootstrap().catch((error) => {
  updateOutput(`Failed to load local trigger UI: ${error.message || error}`, true);
});
