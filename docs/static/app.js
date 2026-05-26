const state = {
  dates: [],
  dateFiles: {},
  dateGroups: {},
  rows: [],
  filteredRows: [],
  activeDate: "",
  visibleCount: 50,
  listMode: "watchlist",
  sortField: "",
  sortDirection: "asc",
};

const DATA_BASE = "./data/predictions";
const STOCK_INDEX_PATH = "./data/predictions/stocks.json";
const ANALOGUES_PATH = "./data/analogues/latest.json";
const NETWORK_PATH = "./data/networks/latest.json";
const BACKTEST_PATH = "./data/backtesting/summary.json";
const EBACKTEST_PATH = "./data/e_backtesting/summary.json";
const GROUP_IMPORTANCE_BASE = "./data/group_importance";
const PAGE_SIZE = 50;
const NETWORK_PAGE_SIZE = 10;
const DRIVER_MONTH_PAGE_SIZE = 36;
const DRIVER_MONTH_EXPAND_SIZE = 24;
const DRIVER_CAP_GROUPS = ["ALL", "mega", "large", "small", "micro", "nano"];
const FEATURED_TICKERS = [
  "AAPL",
  "MSFT",
  "NVDA",
  "AMZN",
  "GOOGL",
  "GOOG",
  "META",
  "BRKA",
  "BRKB",
  "LLY",
  "AVGO",
  "TSLA",
  "JPM",
  "V",
  "UNH",
  "XOM",
  "MA",
  "COST",
  "HD",
  "PG",
  "JNJ",
  "WMT",
  "NFLX",
  "BAC",
  "ABBV",
  "CRM",
  "KO",
  "CVX",
  "AMD",
  "PEP",
  "ADBE",
  "TMO",
  "ORCL",
  "CSCO",
  "MCD",
  "ACN",
  "ABT",
  "GE",
  "IBM",
  "LIN",
  "INTC",
  "QCOM",
  "CAT",
  "DIS",
  "GS",
  "MS",
  "PFE",
  "MRK",
  "NKE",
  "AMGN",
];
const featuredRank = new Map(FEATURED_TICKERS.map((ticker, index) => [ticker, index]));
const NETWORK_FEATURED_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "WMT", "UNH", "CAT", "KO"];
const networkFeaturedRank = new Map(NETWORK_FEATURED_TICKERS.map((ticker, index) => [ticker, index]));

const elements = {
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  asOfDate: document.querySelector("#asOfDate"),
  stockCount: document.querySelector("#stockCount"),
  averageVar: document.querySelector("#averageVar"),
  averageEs: document.querySelector("#averageEs"),
  yearSelect: document.querySelector("#yearSelect"),
  monthSelect: document.querySelector("#monthSelect"),
  searchInput: document.querySelector("#searchInput"),
  refreshButton: document.querySelector("#refreshButton"),
  showMoreButton: document.querySelector("#showMoreButton"),
  downloadButton: document.querySelector("#downloadButton"),
  tableFooter: document.querySelector(".table-footer"),
  tableStatus: document.querySelector("#tableStatus"),
  predictionRows: document.querySelector("#predictionRows"),
  sortHeaders: Array.from(document.querySelectorAll("[data-sort-field]")),
  industryHeatmap: document.querySelector("#industryHeatmap"),
  backtestStatusRows: document.querySelector("#backtestStatusRows"),
  ebacktestMeter: document.querySelector("#ebacktestMeter"),
  ebacktestStats: document.querySelector("#ebacktestStats"),
  ebacktestNote: document.querySelector("#ebacktestNote"),
  analoguePreview: document.querySelector("#analoguePreview .home-analogue-preview"),
  analogueDetailLink: document.querySelector("#analogueDetailLink"),
  analogueSearchInput: document.querySelector("#analogueSearchInput"),
  analogueQuickPicks: document.querySelector("#analogueQuickPicks"),
  driverSizeTabs: document.querySelector("#driverSizeTabs"),
  driverOverallBars: document.querySelector("#driverOverallBars"),
  driverMonthlyHeatmap: document.querySelector("#driverMonthlyHeatmap"),
  driverPeriod: document.querySelector("#driverPeriod"),
  networkSummary: document.querySelector("#networkSummary"),
  networkSearchInput: document.querySelector("#networkSearchInput"),
  networkGroupList: document.querySelector("#networkGroupList"),
  networkShowMore: document.querySelector("#networkShowMore"),
};

const analogueState = {
  payload: null,
  stockRows: [],
  stockIndex: new Map(),
  selectedId: "",
};

const networkState = {
  allGroups: [],
  groups: [],
  query: "",
  stockIndex: new Map(),
  industryByCode: new Map(),
  visibleCount: NETWORK_PAGE_SIZE,
  targetMonth: "",
};

const driverState = {
  metadata: null,
  overall: [],
  monthly: [],
  activeCapGroup: "ALL",
  visibleMonths: DRIVER_MONTH_PAGE_SIZE,
};

function setStatus(kind, text) {
  if (!elements.statusDot || !elements.statusText) return;
  elements.statusDot.className = `status-dot ${kind}`;
  elements.statusText.textContent = text;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return (Number(value) * 100).toFixed(2);
}

function formatSignedPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const formatted = (Number(value) * 100).toFixed(2);
  return Number(value) > 0 ? `+${formatted}` : formatted;
}

function updateSummary(rows) {
  if (!rows.length) {
    elements.asOfDate.textContent = "-";
    elements.stockCount.textContent = "0";
    elements.averageVar.textContent = "-";
    elements.averageEs.textContent = "-";
    return;
  }
  elements.asOfDate.textContent = rows[0].target_month ?? nextMonthEnd(rows[0].eom) ?? "-";
  elements.stockCount.textContent = String(rows.length);
  elements.averageVar.textContent = formatPercent(mean(rows.map((row) => row.v)));
  elements.averageEs.textContent = formatPercent(mean(rows.map((row) => row.e)));
}

function applyFilters() {
  const query = elements.searchInput.value.trim().toLowerCase();
  const field = state.sortField;
  const multiplier = state.sortDirection === "desc" ? -1 : 1;

  const rows = state.rows
    .filter((row) => {
      const haystack =
        `${row.ticker ?? ""} ${row.name ?? ""} ${row.industry ?? ""} ${row.industry_short ?? ""} ${row.id ?? ""}`.toLowerCase();
      return haystack.includes(query);
    });

  if (field) {
    state.filteredRows = sortRows(rows, field, multiplier);
  } else {
    state.filteredRows = sortDefaultRows(rows);
  }
}

function sortRows(rows, field, multiplier) {
  return rows.slice().sort((a, b) => {
    if (field === "ticker" || field === "name" || field === "industry") {
      return String(a[field] ?? "").localeCompare(String(b[field] ?? "")) * multiplier;
    }
    return (Number(a[field]) - Number(b[field])) * multiplier;
  });
}

function sortDefaultRows(rows) {
  return rows.slice().sort((a, b) => {
    const aRank = featuredRank.get(normalizeTicker(a.ticker)) ?? Number.MAX_SAFE_INTEGER;
    const bRank = featuredRank.get(normalizeTicker(b.ticker)) ?? Number.MAX_SAFE_INTEGER;
    if (aRank !== bRank) return aRank - bRank;
    return String(a.ticker ?? "").localeCompare(String(b.ticker ?? ""));
  });
}

function defaultSortDirection(field) {
  return ["y", "v", "e"].includes(field) ? "desc" : "asc";
}

function renderTable() {
  if (!state.filteredRows.length) {
    elements.predictionRows.innerHTML =
      '<tr><td colspan="6" class="empty-state">No predictions found.</td></tr>';
    updateTableFooter(0, 0);
    updateSortHeaders();
    return;
  }

  const visibleRows = state.filteredRows.slice(0, state.visibleCount);
  elements.predictionRows.innerHTML = visibleRows
    .map(
      (row) => `
        <tr class="clickable-row" data-stock-id="${row.id}" title="Open stock details">
          <td>${row.ticker || "N/A"}</td>
          <td>${row.name || "-"}</td>
          <td>${row.industry || "-"}</td>
          <td class="return-value ${Number(row.y) >= 0 ? "is-positive" : "is-negative"}">${formatSignedPercent(row.y)}</td>
          <td class="risk-value">${formatPercent(row.v)}</td>
          <td class="risk-value">${formatPercent(row.e)}</td>
        </tr>
      `,
    )
    .join("");
  updateTableFooter(visibleRows.length, state.filteredRows.length);
  updateSortHeaders();
}

async function loadPredictions() {
  setStatus("", "Loading");
  try {
    state.rows = await fetchPredictionRows(state.activeDate);
    applyFilters();
    updateSummary(state.rows);
    renderTable();
    renderIndustryHeatmap(state.rows);
    setStatus("ok", "Live");
  } catch (error) {
    state.rows = [];
    state.filteredRows = [];
    updateSummary([]);
    renderIndustryHeatmap([]);
    elements.predictionRows.innerHTML =
      '<tr><td colspan="6" class="empty-state">Unable to load predictions.</td></tr>';
    updateTableFooter(0, 0);
    setStatus("error", "Unavailable");
    console.error(error);
  }
}

function updateSortHeaders() {
  elements.sortHeaders.forEach((button) => {
    const field = button.dataset.sortField;
    const icon = button.querySelector("span");
    const active = field === state.sortField;
    button.classList.toggle("active", active);
    button.setAttribute("aria-sort", active ? (state.sortDirection === "asc" ? "ascending" : "descending") : "none");
    if (icon) icon.textContent = active ? (state.sortDirection === "asc" ? "↑" : "↓") : "↕";
  });
}

function updateTableFooter(visible, total) {
  if (elements.tableFooter) {
    elements.tableFooter.hidden = false;
  }
  if (elements.tableStatus) {
    elements.tableStatus.textContent = `Showing ${visible.toLocaleString()} of ${total.toLocaleString()} stocks`;
  }
  if (!elements.showMoreButton) return;
  elements.showMoreButton.hidden = visible >= total;
}

function renderIndustryHeatmap(rows) {
  if (!elements.industryHeatmap) return;
  const grouped = new Map();
  rows.forEach((row) => {
    const industry = row.industry || "Unclassified";
    const item = grouped.get(industry) || { industry, count: 0, varValues: [], esValues: [] };
    item.count += 1;
    if (Number.isFinite(Number(row.v))) item.varValues.push(Number(row.v));
    if (Number.isFinite(Number(row.e))) item.esValues.push(Number(row.e));
    grouped.set(industry, item);
  });

  const industries = Array.from(grouped.values())
    .map((item) => ({
      ...item,
      avgVar: mean(item.varValues),
      avgEs: mean(item.esValues),
    }))
    .filter((item) => Number.isFinite(item.avgEs))
    .sort((a, b) => a.avgEs - b.avgEs)
    .slice(0, 12);

  if (!industries.length) {
    elements.industryHeatmap.innerHTML = '<p class="heatmap-empty">No industry data available.</p>';
    return;
  }

  const severities = industries.map((item) => Math.abs(item.avgEs));
  const minSeverity = Math.min(...severities);
  const maxSeverity = Math.max(...severities);
  const severitySpan = maxSeverity - minSeverity;

  elements.industryHeatmap.innerHTML = industries
    .map((item) => {
      const strength = severitySpan > 0 ? (Math.abs(item.avgEs) - minSeverity) / severitySpan : 0.75;
      const alpha = 0.07 + strength * 0.24;
      const accentAlpha = 0.04 + strength * 0.13;
      return `
        <button
          type="button"
          class="industry-heat-cell"
          style="--heat-alpha: ${alpha.toFixed(3)}; --heat-accent-alpha: ${accentAlpha.toFixed(3)}"
          data-industry="${escapeAttribute(item.industry)}"
          title="Filter table to ${escapeAttribute(item.industry)}"
        >
          <span>${item.industry}</span>
          <strong>${formatPercent(item.avgEs)}%</strong>
          <small>${item.count.toLocaleString()} stocks</small>
        </button>
      `;
    })
    .join("");
}

async function loadDates() {
  try {
    const response = await fetch(`${DATA_BASE}/dates.json`);
    if (!response.ok) return;
    state.dates = await response.json();
    state.dateFiles = Object.fromEntries(state.dates.map((row) => [row.date, row.file]));
    state.dateGroups = groupDatesByYear(state.dates);
    renderYearOptions();
    selectLatestDate();
  } catch (error) {
    console.error(error);
  }
}

function groupDatesByYear(rows) {
  return rows.reduce((groups, row) => {
    const targetMonth = row.target_month ?? nextMonthEnd(row.date) ?? row.date;
    const year = targetMonth.slice(0, 4);
    const item = { ...row, targetMonth, monthLabel: monthName(targetMonth) };
    groups[year] = groups[year] || [];
    groups[year].push(item);
    return groups;
  }, {});
}

function renderYearOptions() {
  const years = Object.keys(state.dateGroups).sort((a, b) => Number(b) - Number(a));
  elements.yearSelect.innerHTML = years.map((year) => `<option value="${year}">${year}</option>`).join("");
  elements.yearSelect.disabled = !years.length;
}

function renderMonthOptions(year) {
  const months = state.dateGroups[year] || [];
  elements.monthSelect.disabled = !months.length;
  elements.monthSelect.innerHTML = months
    .map((row) => `<option value="${row.date}">${row.monthLabel}</option>`)
    .join("");
  if (months.length) {
    elements.monthSelect.value = months[0].date;
  }
}

function selectLatestDate() {
  if (!state.dates.length) return;
  const latest = state.dates[0];
  const targetMonth = latest.target_month ?? nextMonthEnd(latest.date) ?? latest.date;
  const year = targetMonth.slice(0, 4);
  elements.yearSelect.value = year;
  renderMonthOptions(year);
  elements.monthSelect.value = latest.date;
  state.activeDate = latest.date;
}

async function fetchPredictionRows(date) {
  const staticPath = date && state.dateFiles[date] ? `${DATA_BASE}/${state.dateFiles[date]}` : `${DATA_BASE}/latest.json`;
  const response = await fetch(staticPath);
  if (!response.ok) {
    throw new Error(`Static prediction file returned ${response.status}`);
  }
  const rows = await response.json();
  return rows.filter(hasDisplayTicker);
}

async function loadBacktesting() {
  try {
    const response = await fetch(BACKTEST_PATH);
    if (!response.ok) return;
    const summary = await response.json();
    renderBacktestStatus(summary);
  } catch (error) {
    console.error(error);
  }
}

async function loadEBacktesting() {
  try {
    const response = await fetch(EBACKTEST_PATH);
    if (!response.ok) return;
    const summary = await response.json();
    renderEBacktesting(summary);
  } catch (error) {
    console.error(error);
  }
}

async function loadGroupImportance() {
  if (!elements.driverOverallBars || !elements.driverMonthlyHeatmap) return;
  try {
    const [metadata, overallText, monthlyText] = await Promise.all([
      fetchJson(`${GROUP_IMPORTANCE_BASE}/metadata.json`),
      fetchText(`${GROUP_IMPORTANCE_BASE}/overall_importance.csv`),
      fetchText(`${GROUP_IMPORTANCE_BASE}/monthly_importance.csv`),
    ]);
    driverState.metadata = metadata;
    driverState.overall = parseCsv(overallText).map(normalizeDriverRow);
    driverState.monthly = parseCsv(monthlyText).map(normalizeDriverRow);
    renderDriverModule();
  } catch (error) {
    elements.driverOverallBars.innerHTML = '<p class="chart-empty">Unable to load risk drivers.</p>';
    elements.driverMonthlyHeatmap.innerHTML = '<p class="chart-empty">Unable to load monthly driver map.</p>';
    console.error(error);
  }
}

async function loadAnaloguePreview() {
  if (!elements.analoguePreview) return;
  try {
    const [analogues, stockRows] = await Promise.all([
      fetchJson(ANALOGUES_PATH),
      fetchJson(STOCK_INDEX_PATH),
    ]);
    const displayRows = stockRows.filter(hasDisplayTicker);
    const stockIndex = new Map(displayRows.map((row) => [String(row.id), row]));
    analogueState.payload = analogues;
    analogueState.stockRows = displayRows;
    analogueState.stockIndex = stockIndex;
    const preferred = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"];
    const preferredIds = preferred
      .map((ticker) => stockRows.find((row) => normalizeTicker(row.ticker) === ticker)?.id)
      .filter((id) => id !== undefined);
    const selectedId = preferredIds.find((id) => analogues.items?.[String(id)]) ?? Object.keys(analogues.items || {})[0];
    renderAnalogueQuickPicks(preferredIds.filter((id) => analogues.items?.[String(id)]));
    renderSelectedAnaloguePreview(selectedId);
  } catch (error) {
    elements.analoguePreview.innerHTML = '<p class="chart-empty">Unable to load historical analogue preview.</p>';
    console.error(error);
  }
}

function renderAnalogueQuickPicks(ids) {
  if (!elements.analogueQuickPicks) return;
  elements.analogueQuickPicks.innerHTML = ids
    .slice(0, 5)
    .map((id) => {
      const meta = analogueState.stockIndex.get(String(id)) || {};
      return `<button type="button" data-analogue-id="${id}">${meta.ticker || id}</button>`;
    })
    .join("");
}

function renderSelectedAnaloguePreview(selectedId) {
  const id = String(selectedId || "");
  const entry = analogueState.payload?.items?.[id];
  const meta = analogueState.stockIndex.get(id) || {};
  if (!entry?.analogues?.length) {
    elements.analoguePreview.innerHTML = '<p class="chart-empty">Historical analogue data is not available for this stock.</p>';
    return;
  }
  analogueState.selectedId = id;
  if (elements.analogueSearchInput) {
    elements.analogueSearchInput.value = meta.ticker || meta.name || id;
  }
  if (elements.analogueDetailLink) {
    elements.analogueDetailLink.href = `./stock.html?id=${encodeURIComponent(id)}`;
    elements.analogueDetailLink.textContent = `Open ${meta.ticker || "Stock"} Detail`;
  }
  renderHomeAnaloguePreview(entry.analogues.slice(0, 10), analogueState.stockIndex, analogueState.payload.target_month, meta);
  updateAnalogueQuickPickState();
}

function updateAnalogueQuickPickState() {
  if (!elements.analogueQuickPicks) return;
  elements.analogueQuickPicks.querySelectorAll("[data-analogue-id]").forEach((button) => {
    button.classList.toggle("active", String(button.dataset.analogueId) === String(analogueState.selectedId));
  });
}

function selectAnalogueFromQuery() {
  const query = normalizeTicker(elements.analogueSearchInput?.value);
  if (!query) return;
  const match = findAnalogueSearchMatch(query);
  if (match) {
    renderSelectedAnaloguePreview(match.id);
  } else if (elements.analoguePreview) {
    elements.analoguePreview.innerHTML = '<p class="chart-empty">No historical analogue snapshot found for that search.</p>';
  }
}

function findAnalogueSearchMatch(query) {
  const queryLower = query.toLowerCase();
  const rows = analogueState.stockRows.filter((row) => analogueState.payload?.items?.[String(row.id)]);
  return (
    rows.find((row) => normalizeTicker(row.ticker) === query) ||
    rows.find((row) => normalizeTicker(row.ticker).startsWith(query)) ||
    rows.find((row) => String(row.name || "").trim().toLowerCase() === queryLower) ||
    rows.find((row) => normalizeTicker(row.ticker).includes(query)) ||
    rows.find((row) => String(row.name || "").toLowerCase().includes(queryLower)) ||
    null
  );
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.text();
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values;
}

function renderHomeAnaloguePreview(rows, stockIndex, targetMonth, meta) {
  const weights = rows.map((row) => Number(row.avg_weight)).filter(Number.isFinite);
  const minWeight = Math.min(...weights);
  const maxWeight = Math.max(...weights);
  const span = maxWeight - minWeight || 1;
  const bands = [1, 2, 3, 4].map((year) => ({
    year,
    rows: rows.filter((row) => Math.round(analogueYearsBack(row.retrieved_eom, targetMonth)) === year),
  }));

  elements.analoguePreview.innerHTML = `
    <div class="home-analogue-heading">
      <div>
        <span>Selected stock</span>
        <strong>${meta.ticker || "Stock"}${meta.name ? ` · ${meta.name}` : ""}</strong>
      </div>
      <div>
        <span>Forecast month</span>
        <strong>${targetMonth ? targetMonth.slice(0, 7) : "-"}</strong>
      </div>
    </div>
    <div class="home-analogue-bands">
      ${bands.map((band) => `
        <div class="home-analogue-band">
          <span>${formatYearsBack(band.year)} back</span>
          <div>
            ${band.rows.length ? band.rows.map((row) => {
              const retrievedMeta = stockIndex.get(String(row.retrieved_id)) || {};
              const strength = (Number(row.avg_weight) - minWeight) / span;
              return `
                <a
                  href="./stock.html?id=${encodeURIComponent(row.retrieved_id)}"
                  class="home-analogue-pill"
                  style="--pill-alpha:${(0.18 + strength * 0.52).toFixed(3)}"
                  title="${retrievedMeta.name || ""}"
                >
                  ${retrievedMeta.ticker || row.retrieved_id}
                </a>
              `;
            }).join("") : '<em>No top match</em>'}
          </div>
        </div>
      `).join("")}
    </div>
    <div class="home-analogue-summary">
      ${rows.slice(0, 5).map((row) => {
        const retrievedMeta = stockIndex.get(String(row.retrieved_id)) || {};
        return `
          <a href="./stock.html?id=${encodeURIComponent(row.retrieved_id)}">
            <span>${row.rank}</span>
            <strong>${retrievedMeta.ticker || row.retrieved_id}</strong>
            <small>${formatYearsBack(Math.round(analogueYearsBack(row.retrieved_eom, targetMonth)))} back</small>
          </a>
        `;
      }).join("")}
    </div>
  `;
}

async function loadNetworkPreview() {
  if (!elements.networkGroupList) return;
  try {
    const [network, stockRows] = await Promise.all([
      fetchJson(NETWORK_PATH),
      fetchJson(STOCK_INDEX_PATH),
    ]);
    const displayRows = stockRows.filter(hasDisplayTicker);
    networkState.stockIndex = new Map(displayRows.map((row) => [String(row.id), row]));
    networkState.industryByCode = buildIndustryLookup(displayRows);
    networkState.targetMonth = network.target_month || "";
    networkState.allGroups = (network.groups || []).map((group) => enrichNetworkGroup(group));
    networkState.visibleCount = NETWORK_PAGE_SIZE;
    updateNetworkGroups();
    renderNetworkPreview();
  } catch (error) {
    if (elements.networkSummary) {
      elements.networkSummary.innerHTML = '<p class="chart-empty">Unable to load latent neighborhoods.</p>';
    }
    if (elements.networkGroupList) {
      elements.networkGroupList.innerHTML = '<p class="chart-empty">Unable to load neighborhoods.</p>';
    }
    console.error(error);
  }
}

function buildIndustryLookup(rows) {
  const lookup = new Map();
  rows.forEach((row) => {
    const code = Number(row.industry_code);
    if (Number.isFinite(code) && row.industry && !lookup.has(code)) {
      lookup.set(code, row.industry);
    }
  });
  return lookup;
}

function enrichNetworkGroup(group) {
  const anchorMeta = networkState.stockIndex.get(String(group.anchor_id)) || {};
  const members = (group.members || []).slice(0, 6).map((member) => ({
    ...member,
    meta: networkState.stockIndex.get(String(member.id)) || {},
  }));
  const dominantIndustry =
    networkState.industryByCode.get(Number(group.dominant_industry_code)) ||
    anchorMeta.industry ||
    "Mixed industries";
  return {
    ...group,
    anchorMeta,
    members,
    dominantIndustry,
  };
}

function renderNetworkPreview() {
  if (!networkState.groups.length) {
    if (elements.networkSummary) {
      elements.networkSummary.innerHTML = '<p class="chart-empty">No latent neighborhoods are available yet.</p>';
    }
    if (elements.networkGroupList) {
      elements.networkGroupList.innerHTML = '<p class="chart-empty">No neighborhoods found.</p>';
    }
    return;
  }
  renderNetworkSummary();
  renderNetworkGroupList();
}

function renderNetworkSummary() {
  if (!elements.networkSummary) return;
  const groups = networkState.allGroups.length ? networkState.allGroups : networkState.groups;
  const avgVar = mean(groups.map((group) => group.avg_var));
  const avgEs = mean(groups.map((group) => group.avg_es));
  elements.networkSummary.innerHTML = `
    <div class="network-summary-grid">
      <div><span>Date</span><strong>${networkState.targetMonth ? networkState.targetMonth.slice(0, 7) : "-"}</strong></div>
      <div><span>Average VaR</span><strong>${formatPercent(avgVar)}%</strong></div>
      <div><span>Average ES</span><strong>${formatPercent(avgEs)}%</strong></div>
    </div>
  `;
}

function updateNetworkGroups() {
  const query = normalizeTicker(networkState.query);
  if (!query) {
    networkState.groups = networkState.allGroups
      .filter((group) => networkFeaturedRank.has(normalizeTicker(group.anchorMeta.ticker)))
      .sort((a, b) => {
        const aRank = networkFeaturedRank.get(normalizeTicker(a.anchorMeta.ticker)) ?? Number.MAX_SAFE_INTEGER;
        const bRank = networkFeaturedRank.get(normalizeTicker(b.anchorMeta.ticker)) ?? Number.MAX_SAFE_INTEGER;
        return aRank - bRank;
      });
    return;
  }

  const queryLower = query.toLowerCase();
  networkState.groups = networkState.allGroups
    .filter((group) => {
      const ticker = normalizeTicker(group.anchorMeta.ticker);
      const name = String(group.anchorMeta.name || "").toLowerCase();
      return ticker === query || ticker.startsWith(query) || ticker.includes(query) || name.includes(queryLower);
    })
    .sort((a, b) => networkSearchRank(a, query) - networkSearchRank(b, query));
}

function networkSearchRank(group, query) {
  const ticker = normalizeTicker(group.anchorMeta.ticker);
  const name = String(group.anchorMeta.name || "").toLowerCase();
  const queryLower = query.toLowerCase();
  if (ticker === query) return 0;
  if (ticker.startsWith(query)) return 1;
  if (name === queryLower) return 2;
  if (ticker.includes(query)) return 3;
  if (name.includes(queryLower)) return 4;
  return 5;
}

function renderNetworkGroupList() {
  if (!elements.networkGroupList) return;
  if (!networkState.groups.length) {
    elements.networkGroupList.innerHTML = '<p class="chart-empty">No matching network found.</p>';
    if (elements.networkShowMore) elements.networkShowMore.hidden = true;
    return;
  }
  const visible = networkState.groups.slice(0, networkState.visibleCount);
  elements.networkGroupList.innerHTML = visible.map((group, index) => renderNetworkGroupCard(group, index)).join("");
  if (elements.networkShowMore) {
    elements.networkShowMore.hidden = networkState.groups.length <= NETWORK_PAGE_SIZE;
    elements.networkShowMore.textContent =
      networkState.visibleCount >= networkState.groups.length ? "Show Less" : "Show More";
  }
}

function renderNetworkGroupCard(group, index) {
  const anchorTicker = group.anchorMeta.ticker || group.anchor_id;
  const anchorName = group.anchorMeta.name || "";
  const members = group.members.slice(0, 6)
    .map((member) => `<a href="./stock.html?id=${encodeURIComponent(member.id)}">${member.meta.ticker || member.id}</a>`)
    .join("");
  return `
    <article class="network-group-card" data-anchor-id="${group.anchor_id}">
      <a class="network-card-main" href="./network.html?anchor=${encodeURIComponent(group.anchor_id)}">
        <span class="network-rank">${index + 1}</span>
        <span>
          <strong>${anchorTicker}</strong>
          ${anchorName ? `<small>${anchorName}</small>` : ""}
        </span>
      </a>
      <p>${group.dominantIndustry}</p>
      <div class="network-card-metrics">
        <span>${Number(group.size || 0).toLocaleString()} linked stocks</span>
        <span>${formatPercent(group.avg_es)}% avg ES</span>
      </div>
      <div class="network-member-pills">${members}</div>
      <a class="network-detail-link" href="./network.html?anchor=${encodeURIComponent(group.anchor_id)}">View network</a>
    </article>
  `;
}

function renderBacktestStatus(summary) {
  if (!elements.backtestStatusRows) return;
  const rows = [
    {
      label: "Latest evaluated",
      value: summary.latest_evaluated,
      kind: "metric",
    },
    {
      label: "Stocks evaluated",
      value: Number(summary.stocks_evaluated ?? 0).toLocaleString(),
      kind: "metric",
    },
    {
      label: "VaR CC test",
      value: renderStatusBreakdown(summary.var_cc),
      kind: "test",
    },
    {
      label: "ES ASER test",
      value: renderStatusBreakdown(summary.es_aser),
      kind: "test",
    },
  ];
  elements.backtestStatusRows.innerHTML = rows
    .map(
      (row) => `
        <div class="status-tile ${row.kind === "test" ? "status-test" : "status-metric"}">
          <span>${row.label}</span>
          <strong>${row.value ?? "-"}</strong>
        </div>
      `,
    )
    .join("");
}

function renderStatusBreakdown(counts) {
  if (!counts) return "-";
  const pass = counts.pass ?? 0;
  const fail = counts.fail ?? 0;
  const notTestable = counts.not_testable ?? 0;
  const unavailable = counts.unavailable ?? 0;
  const total = Math.max(pass + fail + notTestable + unavailable, 1);
  const segments = [
    ["pass", pass, "Pass"],
    ["fail", fail, "Fail"],
    ["insufficient", notTestable, "Insufficient data"],
    ["na", unavailable, "N.A."],
  ].filter(([, value]) => value > 0);
  return `
    <span class="status-chip-row">
      ${segments
        .map(([type, value, label]) => `<em class="status-chip ${type}">${label} ${Number(value).toLocaleString()}</em>`)
        .join("")}
    </span>
    <span class="status-share-bar" aria-hidden="true">
      ${segments
        .map(([type, value]) => `<i class="${type}" style="width:${((Number(value) / total) * 100).toFixed(2)}%"></i>`)
        .join("")}
    </span>
  `;
}

function renderEBacktesting(summary) {
  if (!elements.ebacktestStats) return;
  const counts = summary.alert_counts || {};
  const green = counts.green ?? 0;
  const yellow = counts.yellow ?? 0;
  const red = counts.red ?? 0;
  const total = Math.max(green + yellow + red, 1);

  if (elements.ebacktestMeter) {
    elements.ebacktestMeter.innerHTML = [
      ["green", green],
      ["yellow", yellow],
      ["red", red],
    ]
      .filter(([, value]) => value > 0)
      .map(([type, value]) => `<span class="${type}" style="width:${((Number(value) / total) * 100).toFixed(2)}%"></span>`)
      .join("");
  }

  elements.ebacktestStats.innerHTML = [
    ["Green", green],
    ["Yellow", yellow],
    ["Red", red],
  ]
    .map(
      ([label, value]) => `
        <div>
          <span>${label}</span>
          <strong>${Number(value).toLocaleString()}</strong>
        </div>
      `,
    )
    .join("");

  if (elements.ebacktestNote) {
    const latest = summary.latest_evaluated || "-";
    elements.ebacktestNote.textContent = `Updated through ${latest}.`;
  }
}

function renderDriverModule() {
  renderDriverTabs();
  renderDriverOverall();
  renderDriverHeatmap();
}

function renderDriverTabs() {
  if (!elements.driverSizeTabs) return;
  const available = new Set(driverState.overall.map((row) => row.cap_group));
  elements.driverSizeTabs.innerHTML = DRIVER_CAP_GROUPS
    .filter((group) => available.has(group))
    .map(
      (group) => `
        <button
          type="button"
          class="${group === driverState.activeCapGroup ? "active" : ""}"
          data-driver-cap="${group}"
        >
          ${formatCapGroup(group)}
        </button>
      `,
    )
    .join("");
}

function renderDriverOverall() {
  if (!elements.driverOverallBars) return;
  const rows = driverState.overall
    .filter((row) => row.cap_group === driverState.activeCapGroup && row.mask !== "ALL")
    .sort((a, b) => Number(a.rank) - Number(b.rank))
    .slice(0, 10);

  if (!rows.length) {
    elements.driverOverallBars.innerHTML = '<p class="chart-empty">No overall importance data available.</p>';
    return;
  }

  const maxValue = Math.max(...rows.map((row) => Number(row.importance) || 0), 0.001);
  elements.driverOverallBars.innerHTML = rows
    .map((row) => {
      const width = Math.max(3, (Number(row.importance) / maxValue) * 100);
      return `
        <div class="driver-bar-row" title="${driverTooltip(row)}">
          <div class="driver-bar-label">
            <span>${row.display_name}</span>
          </div>
          <div class="driver-bar-track">
            <span style="width:${width.toFixed(1)}%"></span>
          </div>
          <strong>${formatShare(row.importance)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderDriverHeatmap() {
  if (!elements.driverMonthlyHeatmap) return;
  const rows = driverState.monthly.filter((row) => row.mask !== "ALL");
  if (!rows.length) {
    elements.driverMonthlyHeatmap.innerHTML = '<p class="chart-empty">No monthly importance data available.</p>';
    return;
  }

  const allMonths = Array.from(new Set(rows.map((row) => row.eom)))
    .sort((a, b) => String(b).localeCompare(String(a)));
  const months = allMonths.slice(0, driverState.visibleMonths);
  const monthSet = new Set(months);
  const recentRows = rows.filter((row) => monthSet.has(row.eom));
  const driverGroups = Array.from(groupBy(recentRows, "mask").entries())
    .map(([mask, items]) => ({
      mask,
      displayName: items[0]?.display_name || mask,
      avgImportance: mean(items.map((item) => item.importance)),
    }))
    .sort((a, b) => b.avgImportance - a.avgImportance);
  const maxImportance = Math.max(...recentRows.map((row) => Number(row.importance) || 0), 0.001);

  if (elements.driverPeriod) {
    const latest = months[0] ? nextMonthEnd(months[0]).slice(0, 7) : "-";
    const earliest = months.at(-1) ? nextMonthEnd(months.at(-1)).slice(0, 7) : "-";
    elements.driverPeriod.textContent = `${latest} back to ${earliest}`;
  }

  const canExpand = months.length < allMonths.length;
  const shownCount = Math.min(months.length, allMonths.length);
  const lookup = new Map(recentRows.map((row) => [`${row.mask}|${row.eom}`, row]));
  elements.driverMonthlyHeatmap.innerHTML = `
    <div class="driver-heatmap-scroll">
      <div class="driver-heatmap-grid" style="--month-count:${months.length}">
        <div class="driver-heatmap-corner"></div>
        ${months.map((month) => `<span class="driver-month-label">${nextMonthEnd(month).slice(2, 7)}</span>`).join("")}
        ${driverGroups.map((group) => `
          <div class="driver-group-label">${group.displayName}</div>
          ${months.map((month) => {
            const row = lookup.get(`${group.mask}|${month}`);
            const value = Number(row?.importance) || 0;
            const strength = value / maxImportance;
            return `
              <span
                class="driver-heat-cell"
                style="--driver-alpha:${(0.08 + strength * 0.72).toFixed(3)}"
                data-driver-tooltip="${escapeAttribute(`${group.displayName} · ${nextMonthEnd(month).slice(0, 7)} · ${formatShare(value)}`)}"
              ></span>
            `;
          }).join("")}
        `).join("")}
      </div>
      <div class="driver-tooltip" id="driverTooltip" hidden></div>
    </div>
    <div class="driver-heatmap-actions">
      <span>Showing ${shownCount} of ${allMonths.length} months</span>
      ${canExpand ? '<button type="button" data-driver-expand>Show Earlier Months</button>' : ""}
    </div>
  `;
}

function normalizeDriverRow(row) {
  const mask = row.mask || "";
  return {
    ...row,
    display_name: displayGroupName(mask),
    feature_count: Number(row.feature_count) || featureCount(mask),
    loss: Number(row.loss),
    baseline_loss: Number(row.baseline_loss),
    delta_abs: Number(row.delta_abs),
    delta_rel: Number(row.delta_rel),
    importance: Number(row.importance),
    rank: Number(row.rank),
  };
}

function featureCount(mask) {
  const group = driverState.metadata?.groups?.find((item) => item.mask === mask);
  return Number(group?.feature_count) || 0;
}

function displayGroupName(mask) {
  const group = driverState.metadata?.groups?.find((item) => item.mask === mask);
  if (group?.display_name) return group.display_name;
  const names = {
    ShortTermReversal: "Short-Term Reversal",
    LowRisk: "Low Risk",
    DebtIssuance: "Debt Issuance",
    LowLeverage: "Low Leverage",
    ProfitGrowth: "Profit Growth",
  };
  return names[mask] || String(mask).replace(/(?!^)([A-Z])/g, " $1");
}

function driverTooltip(row) {
  return `${row.display_name}: ${formatShare(row.importance)} of positive loss increase`;
}

function formatShare(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatCapGroup(value) {
  if (value === "ALL") return "All";
  return String(value || "").replace(/^\w/, (letter) => letter.toUpperCase());
}

function showDriverTooltip(event) {
  const cell = event.target.closest("[data-driver-tooltip]");
  const tooltip = document.querySelector("#driverTooltip");
  if (!cell || !tooltip) return;
  tooltip.textContent = cell.dataset.driverTooltip;
  tooltip.hidden = false;
  moveDriverTooltip(event);
}

function moveDriverTooltip(event) {
  const tooltip = document.querySelector("#driverTooltip");
  const container = elements.driverMonthlyHeatmap;
  if (!tooltip || tooltip.hidden || !container) return;
  const rect = container.getBoundingClientRect();
  const x = event.clientX - rect.left + 12;
  const y = event.clientY - rect.top - 10;
  tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}

function hideDriverTooltip(event) {
  if (event.relatedTarget?.closest?.("[data-driver-tooltip]")) return;
  const tooltip = document.querySelector("#driverTooltip");
  if (tooltip) tooltip.hidden = true;
}

function renderList(element, items) {
  if (!element || !Array.isArray(items)) return;
  element.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function downloadCsv() {
  const rows = state.filteredRows.length ? state.filteredRows : state.rows;
  if (!rows.length) return;
  const columns = ["id", "ticker", "name", "industry", "industry_code", "eom", "target_month", "y", "v", "e"];
  const csv = [
    columns.join(","),
    ...rows.map((row) =>
      columns
        .map((column) => {
          const value = row[column] ?? "";
          return `"${String(value).replaceAll('"', '""')}"`;
        })
        .join(","),
    ),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "resga_forecasts.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function mean(values) {
  const nums = values.map(Number).filter((value) => Number.isFinite(value));
  if (!nums.length) return NaN;
  return nums.reduce((total, value) => total + value, 0) / nums.length;
}

function groupBy(rows, key) {
  const grouped = new Map();
  rows.forEach((row) => {
    const value = row[key];
    const items = grouped.get(value) || [];
    items.push(row);
    grouped.set(value, items);
  });
  return grouped;
}

function hasDisplayTicker(row) {
  const ticker = normalizeTicker(row?.ticker);
  return ticker && ticker !== "NA" && ticker !== "N/A";
}

function normalizeTicker(value) {
  return String(value ?? "").trim().toUpperCase();
}

function nextMonthEnd(dateString) {
  if (!dateString || typeof dateString !== "string") return "";
  const [year, month] = dateString.split("-").map(Number);
  if (!Number.isInteger(year) || !Number.isInteger(month)) return "";
  return new Date(Date.UTC(year, month + 1, 0)).toISOString().slice(0, 10);
}

function analogueYearsBack(retrievedEom, targetMonth) {
  const retrievedTarget = nextMonthEnd(retrievedEom);
  if (!retrievedTarget || !targetMonth) return NaN;
  const [targetYear, targetMonthNumber] = targetMonth.split("-").map(Number);
  const [retrievedYear, retrievedMonthNumber] = retrievedTarget.split("-").map(Number);
  if (![targetYear, targetMonthNumber, retrievedYear, retrievedMonthNumber].every(Number.isFinite)) return NaN;
  return (targetYear - retrievedYear) + (targetMonthNumber - retrievedMonthNumber) / 12;
}

function formatYearsBack(value) {
  if (!Number.isFinite(value)) return "-";
  return Number.isInteger(value) ? `${value}Y` : `${value.toFixed(1)}Y`;
}

function monthName(dateString) {
  const date = new Date(`${dateString.slice(0, 7)}-01T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return dateString;
  return date.toLocaleString("en-US", { month: "long", timeZone: "UTC" });
}

function escapeAttribute(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

elements.searchInput.addEventListener("input", () => {
  state.listMode = elements.searchInput.value.trim() ? "market" : "watchlist";
  state.visibleCount = PAGE_SIZE;
  applyFilters();
  renderTable();
});
elements.sortHeaders.forEach((button) => {
  button.addEventListener("click", () => {
    const field = button.dataset.sortField;
    if (!field) return;
    if (state.sortField === field) {
      const firstDirection = defaultSortDirection(field);
      if (state.sortDirection === firstDirection) {
        state.sortDirection = firstDirection === "asc" ? "desc" : "asc";
      } else {
        state.sortField = "";
        state.sortDirection = defaultSortDirection(field);
      }
    } else {
      state.sortField = field;
      state.sortDirection = defaultSortDirection(field);
    }
    state.listMode = "market";
    state.visibleCount = PAGE_SIZE;
    applyFilters();
    renderTable();
  });
});
elements.yearSelect.addEventListener("change", () => {
  const year = elements.yearSelect.value;
  renderMonthOptions(year);
  state.activeDate = elements.monthSelect.value;
  state.visibleCount = PAGE_SIZE;
  loadPredictions();
});
elements.monthSelect.addEventListener("change", () => {
  state.activeDate = elements.monthSelect.value;
  state.visibleCount = PAGE_SIZE;
  loadPredictions();
});
elements.refreshButton.addEventListener("click", loadPredictions);
if (elements.showMoreButton) {
  elements.showMoreButton.addEventListener("click", () => {
    state.visibleCount += PAGE_SIZE;
    renderTable();
  });
}
elements.downloadButton.addEventListener("click", downloadCsv);
elements.predictionRows.addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-stock-id]");
  if (!row) return;
  window.location.href = `./stock.html?id=${encodeURIComponent(row.dataset.stockId)}`;
});
if (elements.industryHeatmap) {
  elements.industryHeatmap.addEventListener("click", (event) => {
    const cell = event.target.closest("[data-industry]");
    if (!cell) return;
    state.listMode = "market";
    elements.searchInput.value = cell.dataset.industry;
    applyFilters();
    renderTable();
  });
}
if (elements.analogueQuickPicks) {
  elements.analogueQuickPicks.addEventListener("click", (event) => {
    const button = event.target.closest("[data-analogue-id]");
    if (!button) return;
    renderSelectedAnaloguePreview(button.dataset.analogueId);
  });
}
if (elements.analogueSearchInput) {
  elements.analogueSearchInput.addEventListener("change", selectAnalogueFromQuery);
  elements.analogueSearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      selectAnalogueFromQuery();
    }
  });
}
if (elements.networkShowMore) {
  elements.networkShowMore.addEventListener("click", () => {
    if (networkState.visibleCount >= networkState.groups.length) {
      networkState.visibleCount = NETWORK_PAGE_SIZE;
    } else {
      networkState.visibleCount = Math.min(networkState.visibleCount + NETWORK_PAGE_SIZE, networkState.groups.length);
    }
    renderNetworkGroupList();
  });
}
if (elements.networkSearchInput) {
  elements.networkSearchInput.addEventListener("input", () => {
    networkState.query = elements.networkSearchInput.value.trim();
    networkState.visibleCount = NETWORK_PAGE_SIZE;
    updateNetworkGroups();
    renderNetworkGroupList();
  });
}
if (elements.driverSizeTabs) {
  elements.driverSizeTabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-driver-cap]");
    if (!button) return;
    driverState.activeCapGroup = button.dataset.driverCap;
    renderDriverTabs();
    renderDriverOverall();
  });
}
if (elements.driverMonthlyHeatmap) {
  elements.driverMonthlyHeatmap.addEventListener("click", (event) => {
    const button = event.target.closest("[data-driver-expand]");
    if (!button) return;
    driverState.visibleMonths += DRIVER_MONTH_EXPAND_SIZE;
    renderDriverHeatmap();
  });
  elements.driverMonthlyHeatmap.addEventListener("pointerover", showDriverTooltip);
  elements.driverMonthlyHeatmap.addEventListener("pointermove", moveDriverTooltip);
  elements.driverMonthlyHeatmap.addEventListener("pointerout", hideDriverTooltip);
}

loadDates().then(loadPredictions);
loadBacktesting();
loadEBacktesting();
loadAnaloguePreview();
loadNetworkPreview();
loadGroupImportance();
