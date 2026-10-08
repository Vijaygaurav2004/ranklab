"use strict";

const $ = (id) => document.getElementById(id);
let current = null;
let busy = false;

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = res.statusText;
    try {
      message = (await res.json()).detail || message;
    } catch (_) {
      /* not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

function toast(text, isError = false) {
  const box = $("toast");
  box.textContent = text;
  box.className = isError ? "toast error" : "toast";
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (box.hidden = true), 2800);
}

/* ---------- voting ---------- */

function setProgress({ voted, total }) {
  const pct = total ? Math.round((100 * voted) / total) : 0;
  $("progress-bar").style.width = `${pct}%`;
  $("progress-text").textContent = `${voted} of ${total} pairs rated (${pct}%)`;
}

async function loadNext() {
  const data = await api("/api/next");
  setProgress(data.progress);
  current = data.done ? null : data;
  $("vote-area").hidden = data.done;
  $("done").hidden = !data.done;
  if (data.done) return;
  $("prompt-category").textContent = data.prompt.category;
  $("prompt-text").textContent = data.prompt.text;
  $("answer-left").textContent = data.left;
  $("answer-right").textContent = data.right;
  $("answer-left").scrollTop = 0;
  $("answer-right").scrollTop = 0;
  $("reason").value = "";
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll(".choices button").forEach((b) => (b.disabled = value));
}

async function vote(choice) {
  if (!current || busy) return;
  setBusy(true);
  try {
    const result = await api("/api/vote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticket: current.ticket, choice, reason: $("reason").value.trim() }),
    });
    toast(`A was ${result.left_model} · B was ${result.right_model}`);
    await loadNext();
  } catch (err) {
    toast(`Vote not saved: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

async function undo() {
  try {
    await api("/api/undo", { method: "POST" });
    toast("Last vote removed");
    await loadNext();
  } catch (err) {
    toast(err.message, true);
  }
}

/* ---------- leaderboard ---------- */

const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const num = (x, digits = 0) => (x == null ? "–" : x.toFixed(digits));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

function table(headers, rows) {
  const head = el("tr", {}, ...headers.map(([label, cls]) => el("th", cls ? { class: cls } : {}, label)));
  const body = rows.map((cells) => el("tr", {}, ...cells.map(([value, cls]) => el("td", cls ? { class: cls } : {}, value))));
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
}

function ciBar(row, lo, hi) {
  const span = hi - lo || 1;
  const bar = el("div", { class: "ci", title: `95% CI ${num(row.ci_low)} to ${num(row.ci_high)}` });
  const range = el("span");
  range.style.left = `${((row.ci_low - lo) / span) * 100}%`;
  range.style.width = `${Math.max(((row.ci_high - row.ci_low) / span) * 100, 1)}%`;
  const tick = el("i");
  tick.style.left = `calc(${((row.rating - lo) / span) * 100}% - 1px)`;
  bar.append(range, tick);
  return bar;
}

const FEW_BATTLES = 10;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function ranking(rows, emptyText) {
  if (!rows.length) return el("p", { class: "empty" }, emptyText);
  const lo = Math.min(...rows.map((r) => r.ci_low));
  const hi = Math.max(...rows.map((r) => r.ci_high));
  const result = el(
    "div",
    {},
    table(
      [["#"], ["Model"], ["Rating", "num"], ["95% CI", "num"], ["Range", "range"], ["Win rate", "num"], ["W–L–T", "num"]],
      rows.map((r) => [
        [r.rank],
        [r.model],
        [num(r.rating), "num"],
        [`${num(r.ci_low)}–${num(r.ci_high)}`, "num"],
        [ciBar(r, lo, hi), "range"],
        [pct(r.win_rate), "num"],
        [`${r.wins}–${r.losses}–${r.ties}`, "num"],
      ])
    )
  );
  if (rows.some((r) => r.battles < FEW_BATTLES)) {
    result.append(el("p", { class: "note warn" }, `Some models have fewer than ${FEW_BATTLES} comparisons, so treat this ranking as rough until more votes are in.`));
  }
  return result;
}

function stat(label, value, detail) {
  return el("div", { class: "card stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value), el("div", { class: "detail" }, detail));
}

function categoryTable(s) {
  const models = Object.keys(s.by_category);
  if (!models.length) return el("span");
  return el(
    "div",
    {},
    el("h2", {}, "Rubric overall by category"),
    table(
      [["Model"], ...s.categories.map((c) => [c, "num"])],
      models.map((m) => [[m], ...s.categories.map((c) => [num(s.by_category[m][c], 2), "num"])])
    )
  );
}

async function loadBoard() {
  const s = await api("/api/leaderboard");
  const { human, judge, agreement } = s;
  $("board").replaceChildren(
    el(
      "div",
      { class: "stats" },
      stat("Human votes", `${human.votes}`, `of ${human.pairs_total} pairs`),
      stat("Judge verdicts", `${judge.pairs}`, judge.model ? `judge: ${judge.model}` : "no judge run yet"),
      stat(
        "Judge–human agreement",
        pct(agreement.agreement_rate),
        agreement.pairs ? `${plural(agreement.decisive_pairs, "decisive pair")} · κ ${num(agreement.kappa, 2)}` : "needs votes and judge verdicts"
      ),
      stat(
        "Judge flips on swap",
        pct(judge.inconsistency_rate),
        judge.pairs ? `${judge.inconsistent} of ${judge.pairs} pairs · ${judge.first_position_bias} favoured A, ${judge.second_position_bias} B` : "no judge run yet"
      )
    ),
    el("h2", {}, "Human preference ranking"),
    ranking(human.leaderboard, "No votes yet. Rate some pairs on the Vote tab."),
    el("p", { class: "note" }, "Bradley–Terry ratings on an Elo-like scale (1000 = average) with 95% bootstrap confidence intervals. Ties count as half a win."),
    el("h2", {}, "AI judge ranking"),
    ranking(judge.leaderboard, "No judge verdicts yet. Run: ranklab judge --run <name>"),
    el("p", { class: "note" }, "The judge sees every pair twice with the answers swapped. A verdict counts only when both orders agree; otherwise it is scored as a tie."),
    el("h2", {}, "Rubric scores (1–5, AI judge)"),
    s.rubric.length
      ? table(
          [["Model"], ["Correct", "num"], ["Follows instructions", "num"], ["Helpful", "num"], ["Clear", "num"], ["Overall", "num"]],
          s.rubric.map((r) => [[r.model], [num(r.correctness, 2), "num"], [num(r.instruction_following, 2), "num"], [num(r.helpfulness, 2), "num"], [num(r.clarity, 2), "num"], [num(r.overall, 2), "num"]])
        )
      : el("p", { class: "empty" }, "No rubric scores yet."),
    categoryTable(s),
    el("h2", {}, "Automatic instruction checks"),
    s.checks.length
      ? table([["Model"], ["Prompts passed", "num"], ["Pass rate", "num"]], s.checks.map((r) => [[r.model], [`${r.passed} / ${r.total}`, "num"], [pct(r.rate), "num"]]))
      : el("p", { class: "empty" }, "No checkable answers yet."),
    el("p", { class: "note" }, "Word limits, bullet and line counts, JSON-only output and required phrases are checked by code, with no model involved."),
    el(
      "div",
      { class: "toolbar" },
      el("a", { class: "button", href: "/api/export.jsonl?source=human", download: "" }, "Export human preferences (JSONL)"),
      el("a", { class: "button", href: "/api/export.jsonl?source=judge", download: "" }, "Export judge preferences (JSONL)")
    )
  );
}

/* ---------- wiring ---------- */

function show(view) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $("view-vote").hidden = view !== "vote";
  $("view-board").hidden = view !== "board";
  history.replaceState(null, "", view === "board" ? "#leaderboard" : "#vote");
  if (view === "board") loadBoard().catch((err) => toast(err.message, true));
}

document.addEventListener("keydown", (event) => {
  if ($("view-vote").hidden || event.metaKey || event.ctrlKey || event.altKey) return;
  if (document.activeElement === $("reason")) {
    if (event.key === "Escape") $("reason").blur();
    return;
  }
  const choice = { 1: "left", ArrowLeft: "left", 2: "right", ArrowRight: "right", 3: "tie", t: "tie", 4: "both_bad", x: "both_bad" }[event.key];
  if (choice) {
    event.preventDefault();
    vote(choice);
  }
});

document.querySelectorAll(".choices button").forEach((b) => b.addEventListener("click", () => vote(b.dataset.choice)));
document.querySelectorAll(".tab, [data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.view || b.dataset.goto)));
$("undo").addEventListener("click", undo);

api("/api/run")
  .then((run) => {
    $("run-name").textContent = `· ${run.name}`;
    document.title = `RankLab · ${run.name}`;
  })
  .catch(() => {});

show(location.hash === "#leaderboard" ? "board" : "vote");
loadNext().catch((err) => toast(err.message, true));
