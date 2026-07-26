import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.7.76/pdf.min.mjs";
import * as PDFLib from "https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.esm.min.js";

pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.7.76/pdf.worker.min.mjs";

const API_BASE = "http://localhost:8000";

// 묶인 청크(그룹)를 서로 구분하기 위한 색 팔레트. 순서대로 돌려가며 배정하고,
// 같은 청크에 속한 박스(여러 줄에 걸친 경우 포함)는 전부 같은 색을 씀.
const CHUNK_COLORS = [
    { border: "#6c5ce7", bg: "rgba(108, 92, 231, 0.16)" },
    { border: "#e17055", bg: "rgba(225, 112, 85, 0.16)" },
    { border: "#00b894", bg: "rgba(0, 184, 148, 0.16)" },
    { border: "#0984e3", bg: "rgba(9, 132, 227, 0.16)" },
    { border: "#d63031", bg: "rgba(214, 48, 49, 0.16)" },
    { border: "#e0a800", bg: "rgba(224, 168, 0, 0.16)" },
    { border: "#e84393", bg: "rgba(232, 67, 147, 0.16)" },
    { border: "#00cec9", bg: "rgba(0, 206, 201, 0.16)" },
];

// ===================== PDF.js 렌더링 헬퍼 =====================
// canvas별로 진행 중인 렌더 작업을 추적해서, 같은 캔버스에 새 렌더 요청이
// 들어오면 이전 걸 취소함 (pdf.js는 같은 캔버스에 동시 렌더를 허용하지 않음).
const activeRenderTasks = new WeakMap();

async function renderPdfPageToCanvas(pdfDocument, pageNumber1Indexed, canvasEl, targetCssWidth) {
    const prevTask = activeRenderTasks.get(canvasEl);
    if (prevTask) {
        try { prevTask.cancel(); } catch (_) { /* 이미 끝난 작업이면 무시 */ }
    }

    const page = await pdfDocument.getPage(pageNumber1Indexed);
    const dpr = window.devicePixelRatio || 1;
    const baseViewport = page.getViewport({ scale: 1 });
    const scale = (targetCssWidth / baseViewport.width) * dpr;
    const viewport = page.getViewport({ scale });

    canvasEl.width = Math.round(viewport.width);
    canvasEl.height = Math.round(viewport.height);

    const ctx = canvasEl.getContext("2d");
    const renderTask = page.render({ canvasContext: ctx, viewport });
    activeRenderTasks.set(canvasEl, renderTask);

    try {
        await renderTask.promise;
    } catch (err) {
        if (err && err.name === "RenderingCancelledException") return;
        throw err;
    } finally {
        if (activeRenderTasks.get(canvasEl) === renderTask) activeRenderTasks.delete(canvasEl);
    }
}

// ===================== 서버 분석 API (위치값만 요청) =====================
// 서버는 이미지를 만들어 돌려주지 않음. 원본 PDF는 프론트가 로컬 파일을 그대로
// PDF.js로 렌더링하고, 서버는 청크 텍스트/위치(bbox, pt 단위)/점수만 JSON으로 줌.
// file_id는 서버가 이 업로드 원본을 아직 지우지 않고 들고 있다는 참조값 — 확정 시
// 페이지 이미지를 저장할 때 서버가 같은 파일을 다시 렌더링하는 데 씀(중복 렌더링/
// 대용량 이미지 업로드를 피하려고 클라이언트에서 다시 렌더링하지 않음).
async function analyzeOnServer(file, mode) {
    const form = new FormData();
    form.append("file", file);
    form.append("mode", mode);

    const res = await fetch(`${API_BASE}/api/v1/analyze/`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`분석 요청 실패 (${res.status})`);
    const { task_id } = await res.json();

    while (true) {
        await new Promise((r) => setTimeout(r, 900));
        const pollRes = await fetch(`${API_BASE}/api/v1/analyze/${task_id}`);
        const data = await pollRes.json();
        if (data.status === "done") return { pages: data.pages, fileId: data.file_id };
        if (data.status === "failed") throw new Error(data.error || "분석 실패");
    }
}

// 사용자가 직접 고친 최종 청킹(어절을 어떻게 묶었는지)을 서버에 저장 요청.
// 나중에 청킹 모델을 다시 학습시킬 때 "정답 청킹" 데이터로 쓰는 용도.
async function submitCorrection(payload) {
    const res = await fetch(`${API_BASE}/api/v1/corrections/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`저장 요청 실패 (${res.status})`);
    return res.json();
}

// ===================== 마스킹 PDF 생성(클라이언트) =====================
// 서버에 다시 업로드하지 않고, 로컬에 있는 원본 PDF 파일 위에 직접 검은 사각형을
// 그려서(pdf-lib) 마스킹된 PDF를 만들고 브라우저 다운로드로 내려줌.
//
// 주의: PyMuPDF(및 우리 bbox 데이터)는 페이지 크롭박스 좌상단을 (0,0)으로 정규화한
// 좌표계를 쓰지만, pdf-lib은 PDF 원본 절대좌표(좌하단 원점, y 위로 증가)를 그대로 씀.
// 대부분의 PDF는 크롭박스 원점이 절대좌표 (0,0)이라 문제가 안 되지만, 일부 PDF(예:
// Canva 내보내기)는 크롭박스 자체가 절대좌표상 (0,0)이 아닌 곳에 떠 있어서, 단순히
// (페이지 높이 - y2)만 하면 그 오프셋만큼 밀려서 그려짐 — 그래서 getCropBox()의
// 절대 원점(x, y)을 반드시 더해줘야 함.
async function buildMaskedPdfBlob(file, maskRegionsByPage) {
    const bytes = await file.arrayBuffer();
    const pdfDoc = await PDFLib.PDFDocument.load(bytes);
    const pdfPages = pdfDoc.getPages();

    for (const [pageNum, boxes] of maskRegionsByPage.entries()) {
        const pdfPage = pdfPages[pageNum];
        if (!pdfPage) continue;
        const cropBox = pdfPage.getCropBox();
        for (const [x1, y1, x2, y2] of boxes) {
            pdfPage.drawRectangle({
                x: cropBox.x + x1,
                y: cropBox.y + (cropBox.height - y2),
                width: Math.max(0, x2 - x1),
                height: Math.max(0, y2 - y1),
                color: PDFLib.rgb(0.1, 0.1, 0.1),
            });
        }
    }

    const outBytes = await pdfDoc.save();
    return new Blob([outBytes], { type: "application/pdf" });
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => {
    initReviewDemo();

    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("fileName");
    const submitBtn = document.getElementById("submitBtn");
    const uploadForm = document.getElementById("uploadForm");
    const statusMessage = document.getElementById("statusMessage");

    let selectedFile = null;

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        if (file.type !== "application/pdf") {
            showStatus("PDF 파일만 업로드 가능합니다.", "error");
            return;
        }
        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        submitBtn.disabled = false;
        showStatus("", "");
    }

    function showStatus(msg, type) {
        statusMessage.textContent = msg;
        statusMessage.className = "status-message";
        if (type === "error") statusMessage.classList.add("status-error");
        if (type === "success") statusMessage.classList.add("status-success");
    }

    // 실제 서버(FastAPI+Celery)에 업로드해서 청크 위치값을 받아오고,
    // 원본 PDF 자체는 프론트가 들고 있는 로컬 파일을 PDF.js로 직접 렌더링함.
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const mode = document.querySelector('input[name="mode"]:checked').value;
        const spinner = document.getElementById("spinner");

        submitBtn.disabled = true;
        spinner.style.display = "inline-block";
        showStatus("서버에서 문서를 분석하는 중...", "");

        try {
            const arrayBuffer = await selectedFile.arrayBuffer();
            const pdfDocument = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

            const { pages: serverPages, fileId } = await analyzeOnServer(selectedFile, mode);
            const pages = serverPages.map((p) => ({
                page_num: p.page_num,
                label: `페이지 ${p.page_num + 1}`,
                width: p.width,
                height: p.height,
                words: p.chunks.map((c) => ({
                    word_id: c.chunk_id,
                    text: c.text,
                    bbox: c.bbox,
                    random_score: c.score,
                })),
                // 서버가 제안하는 자동 청킹 묶음 — Shift+클릭으로 직접 묶은 것과 동일한
                // 형식(어절 id 리스트)이라 그대로 초기 그룹 상태로 반영함
                groups: p.groups || [],
            }));

            window.__setReviewPages(pages, pdfDocument, mode, selectedFile, fileId);
            window.__openChunkingReview();
            showStatus("", "");
        } catch (err) {
            showStatus(`분석 실패: ${err.message}`, "error");
        } finally {
            submitBtn.disabled = false;
            spinner.style.display = "none";
        }
    });
});

// ===================== 리뷰/검토 화면 (Canva 스타일 에디터, 어절 단위 + 직접 청킹) =====================
// 페이지 이미지는 서버가 렌더링해서 주는 게 아니라, 프론트가 로컬 PDF 파일을
// PDF.js로 직접 그리고, 서버가 준 bbox(pt 단위)를 페이지 크기 대비 %로 변환해 오버레이함.
function initReviewDemo() {
    const uploadScreen = document.getElementById("uploadScreen");
    const reviewScreen = document.getElementById("reviewScreen");
    const backBtn = document.getElementById("backToUploadBtn");
    const pageSidebar = document.getElementById("pageSidebar");
    const pageCanvas = document.getElementById("pageCanvas");
    const pageFrame = document.getElementById("pageFrame");
    const chunkListEl = document.getElementById("chunkList");
    const ratioSlider = document.getElementById("reviewMaskRatio");
    const ratioValueEl = document.getElementById("reviewRatioValue");
    const confirmBtn = document.getElementById("confirmBtn");
    const groupActionBar = document.getElementById("groupActionBar");
    const groupCountLabel = document.getElementById("groupCountLabel");
    const groupBtn = document.getElementById("groupBtn");
    const clearSelectionBtn = document.getElementById("clearSelectionBtn");

    let currentPageIdx = 0;
    let renderSeq = 0; // 페이지 전환 시 뒤늦게 도착한 이전 렌더 결과를 무시하기 위한 시퀀스
    let ratio = parseFloat(ratioSlider.value); // 청킹 확정 시 이 비율만큼 무작위로 가려짐
    const selection = new Set(); // 지금 페이지에서 클릭으로 고른 unitKey들 (그룹핑 대기)

    let PAGES = [];      // 서버 분석 결과 (setReviewPages로 채워짐)
    let pdfDoc = null;    // 로컬에서 연 PDF.js 문서 (원본 파일 자체, 서버 이미지 아님)
    let pageGroups = {};  // pageGroups[page_num] = [[word_id, ...], ...]
    let currentMode = "digital"; // 청킹 확정 시 서버에 같이 보낼 처리 모드
    let originalFile = null; // 마스킹 PDF를 만들 때 다시 읽어들일 로컬 원본 파일
    let currentFileId = null; // 서버가 들고 있는 업로드 원본 참조 (확정 시 페이지 이미지 렌더링용)

    function wordOrderIndex(wordId) {
        const m = String(wordId).match(/(\d+)$/);
        return m ? parseInt(m[1], 10) : 0;
    }
    function unitKey(ids) {
        return ids.join("+");
    }
    function getWord(page, wordId) {
        return page.words.find((w) => w.word_id === wordId);
    }

    // 두 bbox가 같은 줄(가로 행)에 있다고 볼 수 있는지 — 세로 위치만 봄
    // (서버 pipeline.py의 is_same_line과 같은 기준)
    function bboxSameLine(b1, b2) {
        const cy1 = (b1[1] + b1[3]) / 2, cy2 = (b2[1] + b2[3]) / 2;
        const h1 = b1[3] - b1[1], h2 = b2[3] - b2[1];
        const avgH = (h1 + h2) / 2;
        if (Math.abs(cy1 - cy2) < avgH * 0.3) return true;
        const overlap = Math.max(0, Math.min(b1[3], b2[3]) - Math.max(b1[1], b2[1]));
        return overlap / Math.max(h1, h2) > 0.3;
    }

    // 멤버들을 줄(세로 위치) 단위로 묶음 — 그룹이 여러 줄에 걸쳐 있으면 줄마다
    // 따로 묶어서, 하나의 거대한 min/max 사각형이 아니라 줄별로 딱 맞는 박스가 나오게 함
    function groupMembersByLine(members) {
        const sorted = [...members].sort((a, b) => (a.bbox[1] + a.bbox[3]) - (b.bbox[1] + b.bbox[3]));
        const lines = [[sorted[0]]];
        for (let i = 1; i < sorted.length; i++) {
            const w = sorted[i];
            const line = lines[lines.length - 1];
            if (bboxSameLine(line[line.length - 1].bbox, w.bbox)) {
                line.push(w);
            } else {
                lines.push([w]);
            }
        }
        return lines;
    }

    function computeUnit(page, ids) {
        const members = ids.map((id) => getWord(page, id));
        const text = members.map((m) => m.text).join(" ");
        const score = members.reduce((s, m) => s + m.random_score, 0) / members.length;

        // 같은 줄끼리만 min/max로 묶어서 박스를 만듦 (여러 줄에 걸친 그룹을 하나의
        // 큰 사각형으로 감싸면 실제 텍스트가 없는 영역까지 가리게 되므로 방지)
        const boxes = groupMembersByLine(members).map((line) => [
            Math.min(...line.map((m) => m.bbox[0])),
            Math.min(...line.map((m) => m.bbox[1])),
            Math.max(...line.map((m) => m.bbox[2])),
            Math.max(...line.map((m) => m.bbox[3])),
        ]);

        return { key: unitKey(ids), ids, text, boxes, random_score: score };
    }
    function currentPage() {
        return PAGES[currentPageIdx];
    }
    function unitsOf(page) {
        return pageGroups[page.page_num].map((ids) => computeUnit(page, ids));
    }

    function toggleSelect(key) {
        if (selection.has(key)) selection.delete(key);
        else selection.add(key);
        syncCurrentPageUI();
        updateGroupBar();
    }

    function clearSelection() {
        selection.clear();
        syncCurrentPageUI();
        updateGroupBar();
    }

    function updateGroupBar() {
        const n = selection.size;
        groupActionBar.style.display = n >= 2 ? "flex" : "none";
        groupCountLabel.textContent = `${n}개 선택됨`;
    }

    // 클릭으로 고른 여러 어절/그룹을 하나의 새 그룹(=직접 청킹)으로 합침
    function groupSelected() {
        const page = currentPage();
        const mergedIds = [];
        pageGroups[page.page_num] = pageGroups[page.page_num].filter((ids) => {
            if (selection.has(unitKey(ids))) {
                mergedIds.push(...ids);
                return false;
            }
            return true;
        });
        mergedIds.sort((a, b) => wordOrderIndex(a) - wordOrderIndex(b));
        pageGroups[page.page_num].push(mergedIds);
        // 합친 그룹이 리스트 맨 밑으로 밀리지 않도록, 원래 읽기 순서 기준으로 다시 정렬
        pageGroups[page.page_num].sort((a, b) => wordOrderIndex(a[0]) - wordOrderIndex(b[0]));

        selection.clear();
        updateGroupBar();
        renderCurrentPage();
    }

    // 합쳤던 그룹을 다시 개별 어절로 되돌림
    function ungroup(key) {
        const page = currentPage();
        const idx = pageGroups[page.page_num].findIndex((ids) => unitKey(ids) === key);
        if (idx === -1) return;
        const ids = pageGroups[page.page_num][idx];
        if (ids.length <= 1) return;
        pageGroups[page.page_num].splice(idx, 1, ...ids.map((id) => [id]));
        renderCurrentPage();
    }

    function syncCurrentPageUI() {
        const page = currentPage();
        if (!page) return;
        unitsOf(page).forEach((unit) => {
            const selected = selection.has(unit.key);
            const boxes = pageFrame.querySelectorAll(`[data-unit-key="${unit.key}"].chunk-box`);
            const item = chunkListEl.querySelector(`[data-unit-key="${unit.key}"].chunk-item`);
            boxes.forEach((box) => {
                box.classList.toggle("selecting", selected);
            });
            if (item) {
                item.classList.toggle("selecting", selected);
            }
        });
        pageSidebar.querySelectorAll(".page-thumb").forEach((el, idx) => {
            el.classList.toggle("active", idx === currentPageIdx);
        });
    }

    function renderSidebar() {
        pageSidebar.innerHTML = "";
        PAGES.forEach((page, idx) => {
            const thumb = document.createElement("div");
            thumb.className = "page-thumb" + (idx === currentPageIdx ? " active" : "");
            thumb.innerHTML = `
                <span class="thumb-num">${idx + 1}</span>
                <canvas></canvas>
                <span class="thumb-label">${page.label}</span>
            `;
            thumb.addEventListener("click", () => selectPage(idx));
            pageSidebar.appendChild(thumb);

            const thumbCanvas = thumb.querySelector("canvas");
            renderPdfPageToCanvas(pdfDoc, page.page_num + 1, thumbCanvas, 220).catch((err) => {
                if (!(err && err.name === "RenderingCancelledException")) console.error(err);
            });
        });
    }

    async function renderCurrentPage() {
        const seq = ++renderSeq;
        const page = currentPage();
        if (!page) return;

        await renderPdfPageToCanvas(pdfDoc, page.page_num + 1, pageCanvas, 1000);
        if (seq !== renderSeq) return; // 그 사이 다른 페이지로 넘어갔으면 폐기

        pageFrame.querySelectorAll(".chunk-box").forEach((el) => el.remove());
        chunkListEl.innerHTML = "";

        let groupColorIdx = 0; // 묶인 청크마다 순서대로 다른 색을 배정 (같은 청크=같은 색)

        unitsOf(page).forEach((unit) => {
            const isGrouped = unit.ids.length > 1;
            const color = isGrouped ? CHUNK_COLORS[groupColorIdx++ % CHUNK_COLORS.length] : null;

            // 여러 줄에 걸친 그룹은 줄마다 별도 박스로 그려서, 텍스트 없는 영역까지
            // 가리는 하나의 거대한 사각형이 되지 않게 함. 같은 청크의 박스는 전부 같은
            // 색으로 그려서 (여러 줄에 걸쳐 있어도) 한눈에 같은 묶음임을 알 수 있게 함
            unit.boxes.forEach(([x1, y1, x2, y2]) => {
                const leftPct = (x1 / page.width) * 100;
                const topPct = (y1 / page.height) * 100;
                const widthPct = ((x2 - x1) / page.width) * 100;
                const heightPct = ((y2 - y1) / page.height) * 100;

                const box = document.createElement("div");
                box.className = "chunk-box" + (isGrouped ? " grouped" : "");
                box.dataset.unitKey = unit.key;
                box.title = unit.text + (isGrouped ? ` (${unit.ids.length}개 청크 묶음)` : "");
                box.style.left = `${leftPct}%`;
                box.style.top = `${topPct}%`;
                box.style.width = `${widthPct}%`;
                box.style.height = `${heightPct}%`;
                if (color) {
                    box.style.setProperty("--chunk-color", color.border);
                    box.style.setProperty("--chunk-color-bg", color.bg);
                }
                box.addEventListener("click", () => toggleSelect(unit.key));
                pageFrame.appendChild(box);
            });

            const item = document.createElement("div");
            item.className = "chunk-item" + (isGrouped ? " grouped" : "");
            item.dataset.unitKey = unit.key;
            if (color) {
                item.style.setProperty("--chunk-color", color.border);
                item.style.setProperty("--chunk-color-bg", color.bg);
            }

            const label = document.createElement("label");
            label.className = "chunk-item-label";
            const span = document.createElement("span");
            span.className = "chunk-text";
            span.textContent = unit.text;
            label.appendChild(span);
            item.appendChild(label);

            if (isGrouped) {
                const ungroupBtn = document.createElement("button");
                ungroupBtn.type = "button";
                ungroupBtn.className = "ungroup-btn";
                ungroupBtn.textContent = "해제";
                ungroupBtn.title = "이 묶음을 다시 개별 청크로 되돌리기";
                ungroupBtn.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    ungroup(unit.key);
                });
                item.appendChild(ungroupBtn);
            }

            item.addEventListener("click", () => toggleSelect(unit.key));

            chunkListEl.appendChild(item);
        });

        syncCurrentPageUI();
    }

    function selectPage(idx) {
        currentPageIdx = idx;
        selection.clear();
        updateGroupBar();
        renderCurrentPage();
    }

    function openReview() {
        uploadScreen.style.display = "none";
        reviewScreen.style.display = "flex";
        currentPageIdx = 0;
        selection.clear();
        updateGroupBar();
        renderSidebar();
        renderCurrentPage();
    }

    function closeReview() {
        reviewScreen.style.display = "none";
        uploadScreen.style.display = "flex";
    }

    // 업로드 화면의 "문서 분석 시작" 제출에서 실제 서버 분석 결과를 넘겨줄 때 호출.
    // 서버가 제안한 묶음(page.groups)은 그대로 초기 그룹으로 반영하고, 어떤 묶음에도
    // 속하지 않은 어절은 각자 자기 혼자만의 그룹(=싱글턴)으로 시작함 — 둘 다 사용자가
    // 클릭으로 다시 묶거나(그룹으로 묶기) 풀 수 있음(해제).
    window.__setReviewPages = function (pages, pdfDocument, mode, file, fileId) {
        PAGES = pages;
        pdfDoc = pdfDocument;
        currentMode = mode || "digital";
        originalFile = file;
        currentFileId = fileId || null;
        for (const k of Object.keys(pageGroups)) delete pageGroups[k];
        PAGES.forEach((page) => {
            const grouped = new Set();
            const groups = (page.groups || []).map((ids) => {
                ids.forEach((id) => grouped.add(id));
                return ids.slice();
            });
            const singles = page.words.filter((w) => !grouped.has(w.word_id)).map((w) => [w.word_id]);
            const all = [...groups, ...singles];
            all.sort((a, b) => wordOrderIndex(a[0]) - wordOrderIndex(b[0]));
            pageGroups[page.page_num] = all;
        });
    };

    // 업로드 화면의 제출에서 호출할 수 있도록 전역에 노출
    window.__openChunkingReview = openReview;

    // TEMP DEBUG (제거 예정)
    window.__debug = { PAGES: () => PAGES, pageGroups: () => pageGroups, unitsOf, ratio: () => ratio };

    backBtn.addEventListener("click", closeReview);
    groupBtn.addEventListener("click", groupSelected);
    clearSelectionBtn.addEventListener("click", clearSelection);

    // 청킹 자체는 마스킹 여부와 무관하게 자유롭게 고칠 수 있게 하고(그래야 청킹 교정
    // 데이터가 깨끗하게 모임), 마스킹 비율은 "청킹이 끝난 뒤 이 비율만큼 가려진다"는
    // 별도 설정값으로만 두고 미리보기 화면에서 직접 토글하게 하지 않음.
    ratioSlider.addEventListener("input", (e) => {
        ratio = parseFloat(e.target.value);
        ratioValueEl.textContent = `${Math.round(ratio * 100)}%`;
    });

    // 확정 = ① 지금까지 사용자가 직접 고친 청킹(어절을 어떻게 묶었는지)을 서버에 저장(학습
    // 데이터용, 청크 정보만) + ② 그 청크에 마스킹 비율을 적용해 실제로 가린 PDF 파일을
    // 브라우저에서 바로 만들어 다운로드. 화면에 결과 모달은 띄우지 않음 — 다운로드 자체가
    // 결과 확인임.
    const originalConfirmBtnHtml = confirmBtn.innerHTML;
    confirmBtn.addEventListener("click", async () => {
        if (!originalFile) return;

        // 페이지 이미지는 여기서 다시 렌더링하지 않음 — 서버가 analyze 때 쓴 원본 파일을
        // file_id로 참조해서 직접 재렌더링/저장함(같은 렌더링을 클라이언트에서 중복으로
        // 하지 않기 위함).
        const payload = {
            mode: currentMode,
            mask_ratio: ratio,
            file_id: currentFileId,
            pages: PAGES.map((page) => ({
                page_num: page.page_num,
                width: page.width,
                height: page.height,
                words: page.words.map((w) => ({
                    word_id: w.word_id,
                    text: w.text,
                    bbox: w.bbox,
                    score: w.random_score,
                })),
                final_groups: pageGroups[page.page_num],
            })),
        };

        confirmBtn.disabled = true;
        confirmBtn.textContent = "처리 중...";

        // 서버 저장(청킹 데이터 + 페이지 이미지)은 백그라운드로 — 실패해도 다운로드는 그대로 진행함
        submitCorrection(payload)
            .then(({ correction_id }) => console.log("[청킹 확정] 서버 저장 완료:", correction_id))
            .catch((err) => console.error("[청킹 확정] 서버 저장 실패:", err));

        try {
            const maskRegionsByPage = new Map();
            PAGES.forEach((page) => {
                const boxes = unitsOf(page)
                    .filter((u) => u.random_score < ratio)
                    .flatMap((u) => u.boxes);
                if (boxes.length) maskRegionsByPage.set(page.page_num, boxes);
            });

            const blob = await buildMaskedPdfBlob(originalFile, maskRegionsByPage);
            downloadBlob(blob, `masked_${originalFile.name}`);
        } catch (err) {
            console.error("[청킹 확정] 마스킹 PDF 생성 실패:", err);
            alert(`마스킹 PDF를 만들지 못했어요: ${err.message}`);
        } finally {
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = originalConfirmBtnHtml;
        }
    });
}
