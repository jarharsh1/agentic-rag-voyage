// Minimal UI for the Agentic RAG Engine (same-origin: /health, /ask).

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addDetails(container, data) {
  const trace = data.trace || {};
  const sources = data.sources || [];
  if (!sources.length && !trace.subqueries) return;

  const d = document.createElement("details");
  d.className = "cites";
  const s = document.createElement("summary");
  const nsub = (trace.subqueries || []).length;
  const g = trace.grounding != null ? trace.grounding : "—";
  s.textContent = `reasoning trace — ${nsub} sub-quer${nsub === 1 ? "y" : "ies"}, grounding ${g}${trace.refined ? ", refined" : ""} · ${sources.length} source(s)`;
  d.appendChild(s);

  if (trace.subqueries && trace.subqueries.length) {
    const p = document.createElement("div");
    p.className = "trace";
    p.textContent = "sub-queries: " + trace.subqueries.map((q) => `“${q}”`).join("  ·  ");
    d.appendChild(p);
  }

  if (sources.length) {
    const ul = document.createElement("ul");
    for (const src of sources) {
      const li = document.createElement("li");
      li.textContent = `${src.source}: ${src.excerpt || ""}`;
      ul.appendChild(li);
    }
    d.appendChild(ul);
  }
  container.appendChild(d);
}

async function ask(question) {
  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  const bot = addMessage("…", "bot");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    bot.textContent = data.answer || "(no answer)";
    addDetails(bot, data);
  } catch (err) {
    bot.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) ask(q);
});

document.getElementById("examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

fetch("/health").then((r) => r.json()).then((h) => {
  document.getElementById("backend").textContent = `backend: ${h.backend}`;
  document.getElementById("chunks").textContent = `index: ${h.indexed_chunks} chunks`;
}).catch(() => {});
