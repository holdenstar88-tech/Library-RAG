const statusEl = document.getElementById("status");
const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const resultsEl = document.getElementById("results");
const resultCount = document.getElementById("result-count");
const categoriesEl = document.getElementById("categories");
const detailEl = document.getElementById("detail");
const clearFilters = document.getElementById("clear-filters");
const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const question = document.getElementById("question");
const sessionKey = "library_rag_session_id";
const categoryFallback = ["文学", "历史", "科幻", "计算机", "艺术", "哲学", "社科", "教育", "医学", "自然科学", "经济管理"];
let selectedCategory = "";
let lastResults = [];
let sessionId = localStorage.getItem(sessionKey) || crypto.randomUUID();
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
    limit: 24,
  };
}

function text(value, fallback = "未知") {
  return value === null || value === undefined || value === "" ? fallback : value;
}

function locationText(item) {
  if (item.shelf) return item.shelf;
  if (item.shelf_code) return `${item.shelf_code}书架 第${text(item.shelf_row, "?")}行 第${text(item.shelf_col, "?")}列`;
  return "位置未登记";
}

function availabilityText(item) {
  if (item.available_count !== null && item.available_count !== undefined) {
    return `${item.available_count}/${text(item.copy_count, item.available_count)} 可借`;
  }
  return text(item.availability, "状态未知");
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
      renderCategories(categories);
      runSearch();
    });
    categoriesEl.appendChild(button);
  });
}

function renderResults(items) {
  lastResults = items;
  resultsEl.innerHTML = "";
  resultCount.textContent = items.length ? `显示 ${items.length} 条结果` : "没有匹配结果";
  if (!items.length) {
    resultsEl.innerHTML = `<div class="empty-card">没有找到匹配馆藏，可以换一个主角、主题词、ISBN 或分类再试。</div>`;
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
    <section>
      <h3>书籍大意</h3>
      <p>${text(item.plot_summary, "暂无内容概述。")}</p>
    </section>
    <section>
      <h3>主角与主题</h3>
      <p>${text(item.main_characters, "暂无主角信息")}；${text(item.subjects, "暂无主题词")}</p>
    </section>
    <section>
      <h3>借阅信息</h3>
      <p>${text(item.borrow_rule, "暂无借阅规则")}；开放时间：${text(item.open_time)}</p>
    </section>
    <section>
      <h3>检索片段</h3>
      <p>${text(item.excerpt, "暂无片段")}</p>
    </section>
  `;
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
    renderResults(data.results || []);
    statusEl.textContent = data.fallback ? "未命中" : "馆藏已更新";
  } catch {
    statusEl.textContent = "服务不可用";
    renderResults([]);
  }
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

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    statusEl.textContent = data.vector_store_ready ? "Milvus 已连接" : "BM25 兜底模式";
  } catch {
    statusEl.textContent = "服务不可用";
  }
}

searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});

clearFilters.addEventListener("click", () => {
  selectedCategory = "";
  queryInput.value = "";
  ["field-title", "field-author", "field-isbn", "field-book-id", "field-call-number"].forEach((id) => {
    document.getElementById(id).value = "";
  });
  document.getElementById("available-only").checked = false;
  runSearch();
});

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = question.value.trim();
  if (!body) return;
  appendMessage("user", body);
  question.value = "";
  statusEl.textContent = "回答生成中";
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

renderCategories();
loadHealth();
runSearch();
