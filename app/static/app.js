const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const question = document.getElementById("question");
const status = document.getElementById("status");
const sessionKey = "library_rag_session_id";
let sessionId = localStorage.getItem(sessionKey) || crypto.randomUUID();
localStorage.setItem(sessionKey, sessionId);

function appendMessage(role, text, sources = []) {
  const block = document.createElement("div");
  block.className = `msg ${role}`;
  block.textContent = text;
  if (sources.length && role === "assistant") {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    sources.forEach((item) => {
      const line = document.createElement("div");
      line.textContent = `[${item.rank}] ${item.title || "未知书目"} | ${item.author || "未知作者"} | ${item.source || "未知来源"}`;
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
    status.textContent = data.vector_store_ready ? "Milvus 已连接" : "BM25 兜底模式";
  } catch {
    status.textContent = "服务不可用";
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  appendMessage("user", text);
  question.value = "";
  status.textContent = "回答生成中";

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question: text }),
    });
    const data = await resp.json();
    sessionId = data.session_id || sessionId;
    localStorage.setItem(sessionKey, sessionId);
    appendMessage("assistant", data.answer, data.sources || []);
    status.textContent = data.fallback ? "已返回兜底答案" : `置信度 ${Math.round((data.confidence || 0) * 100)}%`;
  } catch (error) {
    appendMessage("assistant", "请求失败，请检查后端服务和 Milvus 连接。");
    status.textContent = "请求失败";
  }
});

loadHealth();
