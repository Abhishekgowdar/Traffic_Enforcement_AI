// Traffic Enforcement AI — Dashboard (light theme, plate box, v5)

const $ = (id) => document.getElementById(id);
const els = {
    dropzone: $("dropzone"), fileInput: $("fileInput"),
    queue: $("queue"), queueItems: $("queueItems"), queueCount: $("queueCount"),
    addMoreBtn: $("addMoreBtn"), clearQueueBtn: $("clearQueueBtn"),
    analyzeBtn: $("analyzeBtn"), btnLabel: document.querySelector("#analyzeBtn .btn-label"),
    spinner: document.querySelector("#analyzeBtn .spinner"), newBtn: $("newBtn"),
    statusDot: $("statusDot"), statusText: $("statusText"),
    resultMode: $("resultMode"), resultEmpty: $("resultEmpty"), resultsList: $("resultsList"),
    clearResultsBtn: $("clearResultsBtn"),
    stats: $("stats"), statProcessed: $("statProcessed"), statFound: $("statFound"),
    statFoundLabel: $("statFoundLabel"), statAvg: $("statAvg"),
    progress: $("progress"), progressFill: $("progressFill"), progressText: $("progressText"),
    sampleChips: $("sampleChips"), engineInfo: $("engineInfo"), toast: $("toast"),
};

let queue = [], mode = "anpr", processing = false, uid = 0;
let stats = { processed: 0, found: 0, confSum: 0, confCount: 0 };

const SAMPLES = [
    { label: "Plate 1", url: "/static/samples/1.jpeg" },
    { label: "Plate 3", url: "/static/samples/3.jpeg" },
    { label: "Car", url: "/static/samples/car.jpg" },
    { label: "Traffic", url: "/static/samples/traffic.jpg" },
];

async function checkHealth() {
    try {
        const r = await fetch("/health");
        if (!r.ok) throw new Error();
        els.statusDot.className = "dot online";
        els.statusText.textContent = "Online";
    } catch {
        els.statusDot.className = "dot offline";
        els.statusText.textContent = "Offline";
    }
}

// Mode toggle
document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        if (processing) return;
        document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        mode = btn.dataset.mode;
        els.resultMode.textContent = mode === "anpr" ? "Plate Recognition" : "Vehicle Detection";
        els.statFoundLabel.textContent = mode === "anpr" ? "Plates found" : "Vehicles found";
    });
});

// Queue
function addFiles(fileList) {
    const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (!files.length) { showToast("Please choose valid image files.", "err"); return; }
    files.forEach((file) => queue.push({ id: ++uid, file, url: URL.createObjectURL(file), name: file.name }));
    renderQueue();
    showToast(`${files.length} image${files.length === 1 ? "" : "s"} added`, "ok");
}
function removeFromQueue(id) {
    const it = queue.find((q) => q.id === id);
    if (it) URL.revokeObjectURL(it.url);
    queue = queue.filter((q) => q.id !== id);
    renderQueue();
}
function clearQueue() { queue.forEach((q) => URL.revokeObjectURL(q.url)); queue = []; renderQueue(); }
function renderQueue() {
    const has = queue.length > 0;
    els.queue.hidden = !has;
    els.queueCount.textContent = `${queue.length} image${queue.length === 1 ? "" : "s"} ready`;
    els.analyzeBtn.disabled = !has || processing;
    els.queueItems.innerHTML = "";
    queue.forEach((q) => {
        const item = document.createElement("div");
        item.className = "queue-item";
        item.innerHTML = `<img src="${q.url}" alt="${escapeHtml(q.name)}" />
            <span class="qi-name" title="${escapeHtml(q.name)}">${escapeHtml(q.name)}</span>
            <button class="qi-remove" aria-label="Remove">&times;</button>`;
        item.querySelector(".qi-remove").addEventListener("click", () => removeFromQueue(q.id));
        els.queueItems.appendChild(item);
    });
}

// Dropzone
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); } });
els.fileInput.addEventListener("change", (e) => { addFiles(e.target.files); els.fileInput.value = ""; });
els.addMoreBtn.addEventListener("click", () => els.fileInput.click());
["dragover", "dragenter"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); }));
els.dropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
els.clearQueueBtn.addEventListener("click", clearQueue);

// Samples
function renderSamples() {
    SAMPLES.forEach((s) => {
        const chip = document.createElement("button");
        chip.className = "chip"; chip.textContent = s.label;
        chip.addEventListener("click", () => loadSample(s));
        els.sampleChips.appendChild(chip);
    });
}
async function loadSample(s) {
    try {
        const r = await fetch(s.url);
        if (!r.ok) throw new Error();
        const blob = await r.blob();
        addFiles([new File([blob], s.url.split("/").pop(), { type: blob.type || "image/jpeg" })]);
    } catch { showToast("Sample not available.", "err"); }
}

// Analyze
els.analyzeBtn.addEventListener("click", analyzeAll);
els.newBtn.addEventListener("click", resetAll);
els.clearResultsBtn.addEventListener("click", clearResults);

async function analyzeAll() {
    if (!queue.length || processing) return;
    processing = true; setBusy(true);
    clearResults(); resetStats();
    els.resultEmpty.hidden = true; els.resultsList.hidden = false; els.stats.hidden = false;

    const endpoint = mode === "anpr" ? "/api/v1/anpr" : "/api/v1/detect-visual";
    const total = queue.length;
    els.progress.hidden = false;

    for (let i = 0; i < total; i++) {
        const q = queue[i];
        updateProgress(i, total, q.name);
        const card = createPendingCard(q);
        els.resultsList.prepend(card);
        try {
            const form = new FormData();
            form.append("file", q.file);
            const r = await fetch(endpoint, { method: "POST", body: form });
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || "Request failed");
            mode === "anpr" ? fillAnprCard(card, data, q) : fillDetectCard(card, data, q);
        } catch (err) { fillErrorCard(card, q, err.message); }
        els.statProcessed.textContent = ++stats.processed;
        updateStats();
    }
    updateProgress(total, total, "");
    setTimeout(() => (els.progress.hidden = true), 700);
    processing = false; setBusy(false);
    els.clearResultsBtn.hidden = false;
    showToast(`Done — ${total} image${total === 1 ? "" : "s"} analyzed`, "ok");
}

function updateProgress(done, total, name) {
    els.progressFill.style.width = Math.round((done / total) * 100) + "%";
    els.progressText.textContent = done < total
        ? `Processing ${done + 1} of ${total}${name ? " — " + name : ""}…`
        : `Completed ${total} image${total === 1 ? "" : "s"}`;
}
function setBusy(on) {
    els.spinner.hidden = !on;
    els.btnLabel.textContent = on ? "Analyzing…" : "Analyze";
    els.analyzeBtn.disabled = on || !queue.length;
    els.newBtn.disabled = on; els.clearQueueBtn.disabled = on; els.addMoreBtn.disabled = on;
}
function resetStats() { stats = { processed: 0, found: 0, confSum: 0, confCount: 0 }; els.statProcessed.textContent = "0"; els.statFound.textContent = "0"; els.statAvg.textContent = "—"; }
function updateStats() { els.statFound.textContent = stats.found; els.statAvg.textContent = stats.confCount ? Math.round(stats.confSum / stats.confCount) + "%" : "—"; }

// Cards
function createPendingCard(q) {
    const card = document.createElement("div");
    card.className = "result-card pending";
    card.innerHTML = `<div class="rc-main"><div class="plate empty"><div class="plate-ind"><span>IND</span></div><div class="plate-num">…</div></div>
        <div class="rc-file">Analyzing ${escapeHtml(q.name)}…</div></div>`;
    return card;
}
function plateBox(text, empty) {
    return `<div class="plate ${empty ? "empty" : ""}"><div class="plate-ind"><span>IND</span></div><div class="plate-num">${escapeHtml(text)}</div></div>`;
}
function fillAnprCard(card, data, q) {
    card.classList.remove("pending");
    const ok = data.status === "success";
    card.classList.add(ok ? "ok" : "warnc");
    const found = data.plate_number && data.plate_number.trim();
    const conf = Math.round((data.confidence || 0) * 100);
    if (found) { stats.found++; stats.confSum += conf; stats.confCount++; }
    const display = found ? formatPlate(data.plate_number) : "NO PLATE";
    card.innerHTML = `
        <div class="rc-main">
            ${plateBox(display, !found)}
            <div class="rc-meta">
                <span class="badge ${statusClass(data.status)}">${escapeHtml((data.status || "unknown").replace(/_/g, " "))}</span>
                <span class="rc-file">${escapeHtml(q.name)}${data.corrected ? " · auto-corrected" : ""}</span>
            </div>
        </div>
        <div class="rc-side">
            <span class="rc-conf-num ${ok ? "" : "low"}">${conf}%</span>
            <span class="rc-conf-label">confidence</span>
            ${found ? '<button class="copy-btn">Copy</button>' : ""}
        </div>`;
    const cb = card.querySelector(".copy-btn");
    if (cb) cb.addEventListener("click", () => { navigator.clipboard.writeText(data.plate_number); showToast("Copied " + display, "ok"); });
}
function fillDetectCard(card, data, q) {
    card.classList.remove("pending");
    card.classList.add("ok");
    if (data.count) stats.found += data.count;
    const list = (data.detections || [])
        .map((d) => `<li><span class="cls">${escapeHtml(d.class_name)}</span><span class="cf">${Math.round((d.confidence || 0) * 100)}%</span></li>`).join("");
    card.innerHTML = `
        <div class="rc-main">
            <div class="rc-count">${data.count || 0} <span>vehicle${data.count === 1 ? "" : "s"}</span></div>
            <ul class="rc-vehlist">${list || "<li>No vehicles detected</li>"}</ul>
            <div class="rc-file">${escapeHtml(q.name)}</div>
        </div>`;
}
function fillErrorCard(card, q, msg) {
    card.classList.remove("pending");
    card.classList.add("err");
    card.innerHTML = `<div class="rc-main">${plateBox("ERROR", true)}<div class="rc-file">${escapeHtml(q.name)} — ${escapeHtml(msg)}</div></div>`;
}
function statusClass(s) { return s === "success" ? "success" : s === "low_confidence" ? "warn" : "error"; }

// Format a plate string like "MH14DT8831" -> "MH 14 DT 8831"
function formatPlate(s) {
    const t = String(s).replace(/\s+/g, "").toUpperCase();
    const m = t.match(/^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$/);
    return m ? [m[1], m[2], m[3], m[4]].filter(Boolean).join(" ") : s;
}

// Reset
function clearResults() { els.resultsList.innerHTML = ""; els.resultsList.hidden = true; els.stats.hidden = true; els.resultEmpty.hidden = false; els.clearResultsBtn.hidden = true; }
function resetAll() { if (processing) return; clearQueue(); clearResults(); els.progress.hidden = true; showToast("Cleared — ready for new images", "ok"); }

// Helpers
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
let toastTimer;
function showToast(msg, kind) {
    els.toast.textContent = msg;
    els.toast.className = "toast" + (kind ? " " + kind : "");
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (els.toast.hidden = true), 3000);
}
async function loadModelInfo() {
    try {
        const r = await fetch("/api/v1/models/info");
        if (!r.ok) return;
        const info = await r.json();
        const engines = info.ocr && info.ocr.engines ? info.ocr.engines.join(" + ") : "";
        const model = info.detector && info.detector.model_type ? info.detector.model_type : "";
        els.engineInfo.textContent = [model, engines].filter(Boolean).join(" · ");
    } catch { /* non-critical */ }
}

renderSamples();
checkHealth();
loadModelInfo();
