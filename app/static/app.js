const statusEl = document.getElementById("status");
const themeToggle = document.getElementById("theme-toggle");
const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const resultsEl = document.getElementById("results");
const resultCount = document.getElementById("result-count");
const categoriesEl = document.getElementById("categories");
const detailEl = document.getElementById("detail");
const clearFilters = document.getElementById("clear-filters");
const paginationEl = document.getElementById("pagination");
const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const question = document.getElementById("question");
const assistantShell = document.querySelector(".assistant-shell");
const assistantLauncher = document.getElementById("assistant-launcher");
const assistantPopover = document.getElementById("assistant-popover");
const assistantClose = document.getElementById("assistant-close");

const sessionKey = "library_rag_session_id";
const themeKey = "library_rag_theme";
const assistantPositionKey = "library_rag_assistant_position";
const categoryFallback = ["文学", "历史", "科幻", "计算机", "艺术", "哲学", "社科", "教育", "医学", "自然科学", "经济管理"];

let selectedCategory = "";
let currentPage = 1;
let pageSize = 10;
let sessionId = localStorage.getItem(sessionKey) || crypto.randomUUID();
let dragState = null;

localStorage.setItem(sessionKey, sessionId);

function fieldValue(id) {
  return document.getElementById(id).value.trim();
}

function buildSearchPayload() {
  return {
    query: queryInput.value.trim(),
    category: selectedCategory || null,
    title: fieldValue("field-title") || null,
    author: fieldValue("field-author") || null,
    isbn: fieldValue("field-isbn") || null,
    book_id: fieldValue("field-book-id") || null,
    call_number: fieldValue("field-call-number") || null,
    available_only: document.getElementById("available-only").checked,
    page: currentPage,
    limit: pageSize,
  };
}

function text(value, fallback = "未知") {
  return value === null || value === undefined || value === "" ? fallback : value;
}

function locationText(item) {
  if (item.shelf) return item.shelf;
  if (item.shelf_code) {
    return `${item.shelf_code}书架 第${text(item.shelf_row, "?")}行 第${text(item.shelf_col, "?")}列`;
  }
  return "位置信息未登记";
}

function availabilityText(item) {
  if (item.available_count !== null && item.available_count !== undefined) {
    return `${item.available_count}/${text(item.copy_count, item.available_count)} 可借`;
  }
  return text(item.availability, "状态未知");
}

function resetToFirstPage() {
  currentPage = 1;
}

function setTheme(theme) {
  const normalized = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", normalized);
  localStorage.setItem(themeKey, normalized);
  themeToggle.setAttribute("aria-pressed", String(normalized === "dark"));
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  setTheme(current === "dark" ? "light" : "dark");
}

function setAssistantOpen(open) {
  assistantPopover.classList.toggle("open", open);
  assistantPopover.setAttribute("aria-hidden", String(!open));
  assistantLauncher.setAttribute("aria-expanded", String(open));
  if (open) {
    question.focus();
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function saveAssistantPosition(left, top) {
  localStorage.setItem(assistantPositionKey, JSON.stringify({ left, top }));
}

function applyAssistantPosition(left, top) {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const launcherRect = assistantLauncher.getBoundingClientRect();
  const shellRect = assistantShell.getBoundingClientRect();
  const shellWidth = launcherRect.width || shellRect.width || 64;
  const shellHeight = launcherRect.height || shellRect.height || 64;
  const maxLeft = Math.max(12, viewportWidth - shellWidth - 12);
  const maxTop = Math.max(12, viewportHeight - shellHeight - 12);
  const nextLeft = clamp(left, 12, maxLeft);
  const nextTop = clamp(top, 12, maxTop);
  assistantShell.style.left = `${nextLeft}px`;
  assistantShell.style.top = `${nextTop}px`;
  assistantShell.style.right = "auto";
  assistantShell.style.bottom = "auto";
}

function initializeAssistantPosition() {
  const saved = localStorage.getItem(assistantPositionKey);
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      if (typeof parsed.left === "number" && typeof parsed.top === "number") {
        applyAssistantPosition(parsed.left, parsed.top);
        return;
      }
    } catch {}
  }

  const defaultLeft = window.innerWidth - 84;
  const defaultTop = window.innerHeight - 84;
  applyAssistantPosition(defaultLeft, defaultTop);
}

function renderCategories(categories = []) {
  const merged = Array.from(new Set(["全部", ...categoryFallback, ...categories.filter(Boolean)]));
  categoriesEl.innerHTML = "";
  merged.forEach((category) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = category === (selectedCategory || "全部") ? "category active" : "category";
    button.textContent = category;
    button.addEventListener("click", () => {
      selectedCategory = category === "全部" ? "" : category;
      resetToFirstPage();
      renderCategories(categories);
      runSearch();
    });
    categoriesEl.appendChild(button);
  });
}

function renderPagination(meta = {}) {
  const totalPages = meta.total_pages || 0;
  const page = meta.page || currentPage;
  paginationEl.innerHTML = "";
  if (totalPages <= 1 && (meta.total || 0) <= pageSize) return;

  const prev = document.createElement("button");
  prev.type = "button";
  prev.textContent = "上一页";
  prev.disabled = !meta.has_prev;
  prev.addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage -= 1;
      runSearch();
    }
  });

  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "下一页";
  next.disabled = !meta.has_next;
  next.addEventListener("click", () => {
    if (currentPage < totalPages) {
      currentPage += 1;
      runSearch();
    }
  });

  const info = document.createElement("span");
  info.textContent = totalPages ? `第 ${page} / ${totalPages} 页` : "第 0 / 0 页";

  const sizeSelect = document.createElement("select");
  [10, 20, 50].forEach((size) => {
    const option = document.createElement("option");
    option.value = String(size);
    option.textContent = `每页 ${size} 条`;
    option.selected = size === pageSize;
    sizeSelect.appendChild(option);
  });
  sizeSelect.addEventListener("change", () => {
    pageSize = Number(sizeSelect.value);
    resetToFirstPage();
    runSearch();
  });

  paginationEl.append(prev, info, next, sizeSelect);
}

function renderDetail(item) {
  detailEl.innerHTML = `
    <h2>${text(item.title, "未知书名")}</h2>
    <div class="detail-grid">
      <span>馆藏编号</span><strong>${text(item.book_id)}</strong>
      <span>作者</span><strong>${text(item.author)}</strong>
      <span>ISBN</span><strong>${text(item.isbn)}</strong>
      <span>索书号</span><strong>${text(item.call_number)}</strong>
      <span>分类</span><strong>${text(item.category)}</strong>
      <span>位置</span><strong>${locationText(item)}</strong>
      <span>楼层/区域</span><strong>${text(item.floor)} / ${text(item.area)}</strong>
      <span>可借状态</span><strong>${availabilityText(item)}</strong>
    </div>
    <section><h3>图书大意</h3><p>${text(item.plot_summary, "暂无内容概述。")}</p></section>
    <section><h3>主角与主题</h3><p>${text(item.main_characters, "暂无主角信息")}；${text(item.subjects, "暂无主题词")}</p></section>
    <section><h3>借阅信息</h3><p>${text(item.borrow_rule, "暂无借阅规则")}；开放时间：${text(item.open_time)}</p></section>
    <section><h3>检索片段</h3><p>${text(item.excerpt, "暂无片段")}</p></section>
  `;
}

function renderResults(items, meta = {}) {
  resultsEl.innerHTML = "";
  const total = meta.total || 0;
  const page = meta.page || currentPage;
  const totalPages = meta.total_pages || 0;

  if (items.length) {
    const start = (page - 1) * pageSize + 1;
    const end = start + items.length - 1;
    resultCount.textContent = `共 ${total} 条，显示第 ${start}-${end} 条`;
  } else {
    resultCount.textContent = "没有匹配结果";
  }

  if (!items.length) {
    resultsEl.innerHTML = `<div class="empty-card">没有找到匹配馆藏，可以换一个主角、主题词、ISBN 或分类再试。</div>`;
    renderPagination({ total, page, total_pages: totalPages, has_prev: false, has_next: false });
    return;
  }

  items.forEach((item, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "book-card";
    card.innerHTML = `
      <div class="book-main">
        <span class="badge">${text(item.category, "未分类")}</span>
        <strong>${text(item.title, "未知书名")}</strong>
        <span>${text(item.author, "未知作者")}</span>
      </div>
      <div class="book-meta">
        <span>编号 ${text(item.book_id)}</span>
        <span>索书号 ${text(item.call_number)}</span>
        <span>${locationText(item)}</span>
        <span>${availabilityText(item)}</span>
      </div>
    `;
    card.addEventListener("click", () => renderDetail(item));
    resultsEl.appendChild(card);
    if (index === 0) renderDetail(item);
  });

  renderPagination(meta);
}

function appendMessage(role, body, sources = []) {
  const block = document.createElement("div");
  block.className = `msg ${role}`;
  block.textContent = body;
  if (sources.length && role === "assistant") {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    sources.forEach((item) => {
      const line = document.createElement("div");
      line.textContent = `[${item.rank}] ${text(item.title, "未知书目")} | ${text(item.book_id)} | ${locationText(item)}`;
      sourceBox.appendChild(line);
    });
    block.appendChild(sourceBox);
  }
  chat.appendChild(block);
  chat.scrollTop = chat.scrollHeight;
}

async function runSearch() {
  statusEl.textContent = "检索中";
  try {
    const resp = await fetch("/api/catalog/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSearchPayload()),
    });
    const data = await resp.json();
    renderCategories(data.categories || []);
    renderResults(data.results || [], data);
    statusEl.textContent = data.fallback ? "未命中，使用兜底模式" : "馆藏已更新";
  } catch {
    statusEl.textContent = "服务不可用";
    renderResults([]);
  }
}

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    statusEl.textContent = data.vector_store_ready ? "Milvus 已连接" : "BM25 兜底模式";
  } catch {
    statusEl.textContent = "服务不可用";
  }
}

function beginDrag(event) {
  if (assistantPopover.contains(event.target)) return;
  dragState = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    shellLeft: assistantShell.getBoundingClientRect().left,
    shellTop: assistantShell.getBoundingClientRect().top,
    moved: false,
  };
  assistantShell.classList.add("dragging");
  assistantLauncher.setPointerCapture(event.pointerId);
}

function continueDrag(event) {
  if (!dragState || event.pointerId !== dragState.pointerId) return;
  const deltaX = event.clientX - dragState.startX;
  const deltaY = event.clientY - dragState.startY;
  if (!dragState.moved && (Math.abs(deltaX) > 4 || Math.abs(deltaY) > 4)) {
    dragState.moved = true;
  }
  const nextLeft = dragState.shellLeft + deltaX;
  const nextTop = dragState.shellTop + deltaY;
  applyAssistantPosition(nextLeft, nextTop);
}

function endDrag(event) {
  if (!dragState || event.pointerId !== dragState.pointerId) return;
  assistantShell.classList.remove("dragging");
  assistantLauncher.releasePointerCapture(event.pointerId);
  const rect = assistantShell.getBoundingClientRect();
  saveAssistantPosition(rect.left, rect.top);
  const moved = dragState.moved;
  dragState = null;
  if (!moved) {
    setAssistantOpen(!assistantPopover.classList.contains("open"));
  }
}

searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  resetToFirstPage();
  runSearch();
});

clearFilters.addEventListener("click", () => {
  selectedCategory = "";
  queryInput.value = "";
  ["field-title", "field-author", "field-isbn", "field-book-id", "field-call-number"].forEach((id) => {
    document.getElementById(id).value = "";
  });
  document.getElementById("available-only").checked = false;
  resetToFirstPage();
  runSearch();
});

themeToggle.addEventListener("click", toggleTheme);
assistantLauncher.addEventListener("pointerdown", beginDrag);
assistantLauncher.addEventListener("pointermove", continueDrag);
assistantLauncher.addEventListener("pointerup", endDrag);
assistantLauncher.addEventListener("pointercancel", endDrag);
assistantClose.addEventListener("click", () => setAssistantOpen(false));

window.addEventListener("resize", () => {
  const rect = assistantShell.getBoundingClientRect();
  applyAssistantPosition(rect.left, rect.top);
  saveAssistantPosition(assistantShell.getBoundingClientRect().left, assistantShell.getBoundingClientRect().top);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setAssistantOpen(false);
});

document.addEventListener("click", (event) => {
  if (!assistantPopover.classList.contains("open")) return;
  if (assistantPopover.contains(event.target) || assistantLauncher.contains(event.target)) return;
  setAssistantOpen(false);
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = question.value.trim();
  if (!body) return;
  appendMessage("user", body);
  question.value = "";
  statusEl.textContent = "回答生成中";
  setAssistantOpen(true);

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question: body }),
    });
    const data = await resp.json();
    sessionId = data.session_id || sessionId;
    localStorage.setItem(sessionKey, sessionId);
    appendMessage("assistant", data.answer, data.sources || []);
    statusEl.textContent = data.fallback ? "已返回兜底答案" : `置信度 ${Math.round((data.confidence || 0) * 100)}%`;
  } catch {
    appendMessage("assistant", "请求失败，请检查后端服务和 Milvus 连接。");
    statusEl.textContent = "请求失败";
  }
});

setTheme(localStorage.getItem(themeKey) || "light");
setAssistantOpen(false);
initializeAssistantPosition();
renderCategories();
loadHealth();
runSearch();
