const state = {
  attentionChart: null,
  gateChart: null,
};

function severityClass(score) {
  if (score >= 0.85) return "critical";
  if (score >= 0.6) return "high";
  if (score >= 0.4) return "watch";
  return "normal";
}

function severityLabel(score) {
  const cls = severityClass(score);
  return cls.charAt(0).toUpperCase() + cls.slice(1);
}

function fmtScore(score) { return score.toFixed(2); }
function fmtTime(ts) { return new Date(ts * 1000).toLocaleTimeString(); }
function shortHash(h) { return h.slice(0, 10) + "…" + h.slice(-4); }
function shortAddr(a) { return a.slice(0, 6) + "…" + a.slice(-4); }

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/feed`);
  const dot = document.getElementById("conn-dot");
  const label = document.getElementById("conn-label");

  ws.onopen = () => { dot.classList.add("live"); label.textContent = "live"; };
  ws.onclose = () => {
    dot.classList.remove("live");
    label.textContent = "offline";
    setTimeout(connectWebSocket, 2000);
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "event") prependFeedRow(msg.data, msg.alert);
  };
}

function prependFeedRow(ev, alert) {
  const onlyAnomalies = document.getElementById("only-anomalies").checked;
  if (onlyAnomalies && !ev.is_anomaly) return;
  const tbody = document.getElementById("feed-body");
  const tr = document.createElement("tr");
  tr.className = "flash";
  tr.dataset.eventId = ev.id;
  tr.innerHTML = `
    <td>${shortHash(ev.tx_hash)}</td>
    <td>${shortAddr(ev.entity)}</td>
    <td>${ev.role}</td>
    <td>${ev.action}</td>
    <td><span class="score-pill score-${severityClass(ev.score)}">${fmtScore(ev.score)}</span></td>
    <td>${ev.is_anomaly ? "⚑" : ""}</td>
  `;
  tr.addEventListener("click", () => loadEventDetail(ev.id));
  tbody.prepend(tr);
  while (tbody.children.length > 150) tbody.removeChild(tbody.lastChild);
}

async function loadEventDetail(eventId) {
  const res = await fetch(`/api/events/${eventId}`);
  if (!res.ok) return;
  const data = await res.json();
  document.getElementById("detail-empty").classList.add("hidden");
  document.getElementById("detail-content").classList.remove("hidden");
  document.getElementById("detail-hash").textContent = data.tx_hash;
  const badge = document.getElementById("detail-severity");
  const cls = severityClass(data.score);
  badge.textContent = severityLabel(data.score);
  badge.className = "severity-badge score-" + cls;
  document.getElementById("detail-meta").innerHTML = `
    <div>Entity <span>${shortAddr(data.entity)}</span></div>
    <div>Role <span>${data.role}</span></div>
    <div>Action <span>${data.action}</span></div>
    <div>Score <span>${fmtScore(data.score)}</span></div>
    <div>Block <span>#${data.block_number}</span></div>
    <div>Sensitive record <span>${data.sensitive_record_accessed ? "yes" : "no"}</span></div>
  `;
  renderAttentionChart(data.attention_weights);
  renderGateChart(data.feature_gate);

  const feedbackRow = document.getElementById("feedback-row");
  const feedbackDone = document.getElementById("feedback-done");
  feedbackDone.classList.add("hidden");
  if (data.alert_id && data.alert_status === "open") {
    feedbackRow.classList.remove("hidden");
    document.getElementById("btn-confirm").onclick = () => submitFeedback(data.alert_id, "true_positive");
    document.getElementById("btn-dismiss").onclick = () => submitFeedback(data.alert_id, "false_positive");
  } else if (data.alert_id) {
    feedbackRow.classList.add("hidden");
    feedbackDone.classList.remove("hidden");
    feedbackDone.textContent = `Reviewed: marked as ${data.alert_status}.`;
  } else {
    feedbackRow.classList.add("hidden");
  }
}

async function submitFeedback(alertId, verdict) {
  await fetch(`/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alert_id: alertId, verdict }),
  });
  document.getElementById("feedback-row").classList.add("hidden");
  const done = document.getElementById("feedback-done");
  done.classList.remove("hidden");
  done.textContent = verdict === "true_positive" ? "Marked as a confirmed anomaly." : "Dismissed as a false positive.";
}

function renderAttentionChart(weights) {
  const ctx = document.getElementById("attention-chart");
  if (state.attentionChart) state.attentionChart.destroy();
  state.attentionChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: weights.map((_, i) => `t-${weights.length - i}`),
      datasets: [{ data: weights, backgroundColor: "#3B6FE0" }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#6B7688", maxRotation: 0, autoSkip: true }, grid: { display: false } },
        y: { ticks: { color: "#6B7688" }, grid: { color: "#EEF1F6" } },
      },
    },
  });
}

function renderGateChart(gateData) {
  const ctx = document.getElementById("gate-chart");
  if (state.gateChart) state.gateChart.destroy();
  const sorted = [...gateData].sort((a, b) => b.value - a.value);
  state.gateChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map((g) => g.feature.replace(/_/g, " ")),
      datasets: [{ data: sorted.map((g) => g.value), backgroundColor: "#1E9E6B" }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, max: 1, ticks: { color: "#6B7688" }, grid: { color: "#EEF1F6" } },
        y: { ticks: { color: "#6B7688", font: { size: 10.5 } }, grid: { display: false } },
      },
    },
  });
}

document.getElementById("export-btn").addEventListener("click", () => {
  window.location.href = "/api/audit/export";
});

async function pollStats() {
  const res = await fetch("/api/stats");
  const stats = await res.json();
  document.getElementById("stat-total").textContent = stats.total_events;
  document.getElementById("stat-anomaly-rate").textContent = stats.anomaly_rate + "%";
  document.getElementById("stat-open-alerts").textContent = stats.open_alerts;
  document.getElementById("stat-latency").textContent = stats.avg_latency_ms;

  const chain = stats.chain;
  if (chain && chain.connected) {
    document.getElementById("chain-id").textContent = `Local Ethereum (chain id ${chain.chain_id})`;
    document.getElementById("chain-address").textContent = chain.contract_address;
    document.getElementById("chain-block").textContent = `#${chain.latest_block}`;
    document.getElementById("chain-accounts").textContent = chain.accounts;
  }
}

function init() {
  connectWebSocket();
  pollStats();
  setInterval(pollStats, 3000);
}

init();
