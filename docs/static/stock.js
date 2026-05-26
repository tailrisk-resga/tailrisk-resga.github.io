const STOCK_SERIES_BASE = "./data/predictions/stocks";
const STOCK_INDEX_PATH = "./data/predictions/stocks.json";
const STOCK_BACKTEST_BASE = "./data/backtesting/stocks";
const STOCK_EBACKTEST_BASE = "./data/e_backtesting/stocks";
const RELATED_DATES_PATH = "./data/related/dates.json";
const RELATED_BASE = "./data/related";
const ANALOGUES_PATH = "./data/analogues/latest.json";
const NETWORK_CENTERS_PATH = "./data/networks/centers.json";
const ANALOGUE_DISPLAY_LIMIT = 10;

const elements = {
  title: document.querySelector("#stockTitle"),
  subtitle: document.querySelector("#stockSubtitle"),
  obs: document.querySelector("#detailObs"),
  date: document.querySelector("#detailDate"),
  industry: document.querySelector("#detailIndustry"),
  varValue: document.querySelector("#detailVar"),
  esValue: document.querySelector("#detailEs"),
  performanceChart: document.querySelector("#performanceChart"),
  chart: document.querySelector("#returnRiskChart"),
  backtestList: document.querySelector("#stockBacktestList"),
  relatedList: document.querySelector("#relatedStocksList"),
  relatedYearSelect: document.querySelector("#relatedYearSelect"),
  relatedMonthSelect: document.querySelector("#relatedMonthSelect"),
  relatedPeriod: document.querySelector("#relatedPeriod"),
  analogueList: document.querySelector("#analogueList"),
  analoguePeriod: document.querySelector("#analoguePeriod"),
  networkPanel: document.querySelector("#stockNetworkPanel"),
};

let stockIndex = new Map();
let relatedDates = [];
let relatedDateGroups = {};
const relatedPayloadCache = new Map();
let analoguePayload = null;
const stockSeriesCache = new Map();
const stockBacktestState = {
  formal: null,
  monitor: null,
  realized: null,
};

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return (Number(value) * 100).toFixed(2);
}

function formatSignedPercentValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(2)}%`;
}

function getStockId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("id");
}

async function loadStock() {
  const id = getStockId();
  if (!id) {
    showError("Missing stock id.");
    return;
  }

  try {
    const response = await fetch(`${STOCK_SERIES_BASE}/${encodeURIComponent(id)}.json`);
    if (!response.ok) {
      throw new Error(`Stock series returned ${response.status}`);
    }
    const payload = await response.json();
    if (!hasDisplayTicker(payload)) {
      showError("Stock is not available.");
      return;
    }
    renderStock(payload);
    loadStockBacktests(id);
    loadRelatedStocks(id);
    loadHistoricalAnalogues(id);
    loadStockNetworkCenter(id);
  } catch (error) {
    showError("Unable to load stock detail.");
    console.error(error);
  }
}

async function loadStockNetworkCenter(id) {
  if (!elements.networkPanel) return;
  try {
    const response = await fetch(NETWORK_CENTERS_PATH);
    if (!response.ok) {
      throw new Error(`Network center index returned ${response.status}`);
    }
    const payload = await response.json();
    const center = (payload.centers || []).find((row) => String(row.id) === String(id));
    renderStockNetworkPanel({
      id,
      center,
      targetMonth: payload.target_month,
    });
  } catch (error) {
    elements.networkPanel.innerHTML = `
      <p>Network availability is currently unavailable.</p>
      <button class="button-link secondary-link is-disabled" type="button" disabled>View Network</button>
    `;
    console.error(error);
  }
}

async function loadHistoricalAnalogues(id) {
  if (!elements.analogueList) return;
  renderAnalogueEmpty("Loading historical analogues...");
  try {
    const [payload, index] = await Promise.all([fetchAnaloguePayload(), loadStockIndex()]);
    stockIndex = stockIndex.size ? stockIndex : index;
    if (elements.analoguePeriod) {
      elements.analoguePeriod.textContent = payload.target_month ? payload.target_month.slice(0, 7) : "Latest";
    }
    const entry = payload.items?.[String(id)];
    if (!entry?.analogues?.length) {
      renderAnalogueEmpty("No historical analogues found for this stock.");
      return;
    }
    const rows = await enrichAnalogues(entry.analogues.slice(0, ANALOGUE_DISPLAY_LIMIT));
    renderHistoricalAnalogues(rows, payload.target_month);
  } catch (error) {
    renderAnalogueEmpty("Unable to load historical analogues.");
    console.error(error);
  }
}

async function fetchAnaloguePayload() {
  if (analoguePayload) return analoguePayload;
  const response = await fetch(ANALOGUES_PATH);
  if (!response.ok) {
    throw new Error(`Analogue file returned ${response.status}`);
  }
  analoguePayload = await response.json();
  return analoguePayload;
}

async function enrichAnalogues(rows) {
  const enriched = await Promise.all(
    rows.map(async (row) => {
      const id = String(row.retrieved_id);
      const meta = stockIndex.get(id) || {};
      const outcome = await fetchRetrievedOutcome(id, row.retrieved_eom);
      return { ...row, meta, outcome };
    }),
  );
  return enriched;
}

async function fetchRetrievedOutcome(id, eom) {
  try {
    const payload = await fetchStockSeries(id);
    const match = normalizeSeries(payload.series).find((row) => row.eom === eom);
    return match || null;
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function fetchStockSeries(id) {
  if (stockSeriesCache.has(id)) {
    return stockSeriesCache.get(id);
  }
  const response = await fetch(`${STOCK_SERIES_BASE}/${encodeURIComponent(id)}.json`);
  if (!response.ok) {
    throw new Error(`Stock series returned ${response.status}`);
  }
  const payload = await response.json();
  stockSeriesCache.set(id, payload);
  return payload;
}

async function loadStockBacktests(id) {
  if (!elements.backtestList) return;
  const [formal, monitor] = await Promise.all([fetchStockBacktest(id), fetchStockEBacktest(id)]);
  stockBacktestState.formal = formal;
  stockBacktestState.monitor = monitor;
  renderBacktestPanel();
}

async function fetchStockBacktest(id) {
  try {
    const response = await fetch(`${STOCK_BACKTEST_BASE}/${encodeURIComponent(id)}.json`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function fetchStockEBacktest(id) {
  try {
    const response = await fetch(`${STOCK_EBACKTEST_BASE}/${encodeURIComponent(id)}.json`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

async function loadRelatedStocks(id) {
  if (!elements.relatedList) return;
  try {
    const [dates, index] = await Promise.all([loadRelatedDates(), loadStockIndex()]);
    if (!dates.length) {
      setRelatedDateControl([]);
      renderRelatedEmpty("Related stock data is not available yet.");
      return;
    }
    stockIndex = index;
    relatedDates = dates;
    relatedDateGroups = groupRelatedDatesByYear(dates);
    setRelatedDateControl(dates, id);
    const latest = dates[dates.length - 1];
    await loadRelatedForDate(id, latest.date);
  } catch (error) {
    renderRelatedEmpty("Unable to load related stocks.");
    console.error(error);
  }
}

async function loadRelatedForDate(id, date) {
  const selected = relatedDates.find((row) => row.date === date);
  if (!selected) {
    renderRelatedEmpty("Related stock data is not available for this month.");
    return;
  }

  renderRelatedEmpty("Loading related stocks...");
  setRelatedPeriod(selected);

  try {
    const payload = await fetchRelatedPayload(selected);
    renderRelatedStocks(payload.items?.[String(id)] || [], payload.target_month);
  } catch (error) {
    renderRelatedEmpty("Unable to load related stocks for this month.");
    console.error(error);
  }
}

function setRelatedDateControl(dates, id) {
  const yearSelect = elements.relatedYearSelect;
  const monthSelect = elements.relatedMonthSelect;
  if (!yearSelect || !monthSelect) return;

  if (!dates.length) {
    yearSelect.innerHTML = "<option>No dates</option>";
    monthSelect.innerHTML = "<option>No dates</option>";
    yearSelect.disabled = true;
    monthSelect.disabled = true;
    return;
  }

  const latest = dates[dates.length - 1];
  const latestTargetMonth = latest.target_month ?? nextMonthEnd(latest.date) ?? latest.date;
  const latestYear = latestTargetMonth.slice(0, 4);
  const years = Object.keys(relatedDateGroups).sort((a, b) => Number(b) - Number(a));

  yearSelect.innerHTML = years.map((year) => `<option value="${year}">${year}</option>`).join("");
  yearSelect.value = latestYear;
  renderRelatedMonthOptions(latestYear);
  monthSelect.value = latest.date;
  yearSelect.disabled = false;
  monthSelect.disabled = false;

  yearSelect.onchange = () => {
    renderRelatedMonthOptions(yearSelect.value);
    loadRelatedForDate(id, monthSelect.value);
  };
  monthSelect.onchange = () => loadRelatedForDate(id, monthSelect.value);
}

function groupRelatedDatesByYear(rows) {
  return rows.reduce((groups, row) => {
    const targetMonth = row.target_month ?? nextMonthEnd(row.date) ?? row.date;
    const year = targetMonth.slice(0, 4);
    const item = { ...row, targetMonth, monthLabel: monthName(targetMonth) };
    groups[year] = groups[year] || [];
    groups[year].push(item);
    return groups;
  }, {});
}

function renderRelatedMonthOptions(year) {
  const monthSelect = elements.relatedMonthSelect;
  if (!monthSelect) return;
  const months = relatedDateGroups[year] || [];
  monthSelect.innerHTML = months
    .map((row) => `<option value="${row.date}">${row.monthLabel}</option>`)
    .join("");
  monthSelect.disabled = !months.length;
  if (months.length) {
    monthSelect.value = months[months.length - 1].date;
  }
}

function setRelatedPeriod(dateInfo) {
  if (!elements.relatedPeriod) return;
  elements.relatedPeriod.textContent = dateInfo?.target_month ? dateInfo.target_month.slice(0, 7) : "-";
}

async function fetchRelatedPayload(dateInfo) {
  if (relatedPayloadCache.has(dateInfo.date)) {
    return relatedPayloadCache.get(dateInfo.date);
  }

  const response = await fetch(`${RELATED_BASE}/${dateInfo.file}`);
  if (!response.ok) {
    throw new Error(`Related stock file returned ${response.status}`);
  }
  const payload = await response.json();
  relatedPayloadCache.set(dateInfo.date, payload);
  return payload;
}

async function loadRelatedDates() {
  const response = await fetch(RELATED_DATES_PATH);
  if (!response.ok) return [];
  return response.json();
}

async function loadStockIndex() {
  const response = await fetch(STOCK_INDEX_PATH);
  if (!response.ok) return new Map();
  const rows = await response.json();
  return new Map(rows.filter(hasDisplayTicker).map((row) => [String(row.id), row]));
}

function showError(message) {
  elements.title.textContent = message;
  if (elements.chart) {
    elements.chart.innerHTML = `<p class="chart-empty">${message}</p>`;
  }
}

function renderStock(payload) {
  const label = payload.ticker || payload.name || "Stock Detail";
  elements.title.textContent = label;
  elements.subtitle.textContent = payload.name || `Stock identifier ${payload.id}`;

  const points = normalizeSeries(payload.series);
  if (!points.length) {
    elements.chart.innerHTML = '<p class="chart-empty">No numeric observations available for this stock.</p>';
    return;
  }

  const latest = points[points.length - 1];
  elements.obs.textContent = String(points.length);
  elements.date.textContent = latest.targetDate;
  elements.industry.textContent = payload.industry || "-";
  elements.varValue.textContent = formatPercent(latest.v);
  elements.esValue.textContent = formatPercent(latest.e);
  renderPerformanceChart(buildCumulativeReturn(points));
  renderReturnRiskChart(points);
  renderRealizedSummary(points);
}

function renderStockNetworkPanel({ id, center, targetMonth }) {
  if (!elements.networkPanel) return;
  const dateLabel = targetMonth ? targetMonth.slice(0, 7) : "latest";
  if (center) {
    elements.networkPanel.innerHTML = `
      <p>This stock is available as a risk-network center for ${dateLabel}.</p>
      <div class="stock-network-stats">
        <span>Connected stocks</span>
        <strong>${Number(center.size || 0).toLocaleString()}</strong>
      </div>
      <a class="button-link" href="./network.html?anchor=${encodeURIComponent(id)}">View Network</a>
    `;
    return;
  }

  elements.networkPanel.innerHTML = `
    <p>This stock is not available as a standalone network center for ${dateLabel}.</p>
    <button class="button-link secondary-link is-disabled" type="button" disabled>View Network</button>
  `;
}

function normalizeSeries(series) {
  if (!Array.isArray(series)) return [];
  return series
    .map((row) => ({
      eom: row[0],
      targetDate: nextMonthEnd(row[0]),
      y: Number(row[1]),
      v: Number(row[2]),
      e: Number(row[3]),
    }))
    .filter((row) => row.targetDate && [row.y, row.v, row.e].every(Number.isFinite));
}

function nextMonthEnd(dateString) {
  if (!dateString || typeof dateString !== "string") return "";
  const [year, month] = dateString.split("-").map(Number);
  if (!Number.isInteger(year) || !Number.isInteger(month)) return "";
  return new Date(Date.UTC(year, month + 1, 0)).toISOString().slice(0, 10);
}

function monthName(dateString) {
  const date = new Date(`${dateString.slice(0, 7)}-01T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return dateString;
  return date.toLocaleString("en-US", { month: "long", timeZone: "UTC" });
}

function buildCumulativeReturn(points) {
  let wealth = 1;
  return [
    {
      date: points[0].eom,
      value: 0,
    },
    ...points.map((row) => {
      wealth *= 1 + row.y;
      return {
        date: row.targetDate,
        value: (wealth - 1) * 100,
      };
    }),
  ];
}

function renderPerformanceChart(points) {
  if (!elements.performanceChart) return;
  if (!points.length) {
    elements.performanceChart.innerHTML = '<p class="chart-empty">No performance history is available for this stock.</p>';
    return;
  }

  const width = 960;
  const height = 360;
  const pad = { top: 22, right: 28, bottom: 42, left: 64 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  let minY = Math.min(...points.map((row) => row.value));
  let maxY = Math.max(...points.map((row) => row.value));
  const span = maxY - minY || Math.max(maxY, 1);
  minY -= span * 0.08;
  maxY += span * 0.08;

  const x = (idx) => pad.left + (points.length === 1 ? innerW / 2 : (idx / (points.length - 1)) * innerW);
  const y = (value) => pad.top + ((maxY - value) / (maxY - minY)) * innerH;
  const line = points.map((row, idx) => `${x(idx).toFixed(2)},${y(row.value).toFixed(2)}`).join(" ");
  const firstLabel = points[0].date.slice(0, 7);
  const lastLabel = points[points.length - 1].date.slice(0, 7);

  elements.performanceChart.innerHTML = `
    <svg class="stock-chart detail-stock-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Cumulative return history">
      <line class="axis-line" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
      <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
      <polyline class="series-performance" points="${line}" />
      <text x="${pad.left}" y="${height - 12}" font-size="12" fill="#657181">${firstLabel}</text>
      <text x="${width - pad.right}" y="${height - 12}" font-size="12" fill="#657181" text-anchor="end">${lastLabel}</text>
      <text x="${pad.left + innerW / 2}" y="${height - 12}" font-size="12" fill="#657181" text-anchor="middle">Date</text>
      <text x="12" y="${pad.top + 4}" font-size="12" fill="#657181">${formatSignedPercentValue(maxY)}</text>
      <text x="12" y="${height - pad.bottom}" font-size="12" fill="#657181">${formatSignedPercentValue(minY)}</text>
      <g class="chart-hover" visibility="hidden" aria-hidden="true">
        <line class="hover-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
        <circle class="hover-dot hover-dot-performance" r="4" />
        <g class="hover-card">
          <rect class="hover-card-bg" width="154" height="58" rx="6" />
          <text class="hover-card-date" x="12" y="20"></text>
          <text class="hover-card-return" x="12" y="42"></text>
        </g>
      </g>
      <rect class="chart-hit-area" x="${pad.left}" y="${pad.top}" width="${innerW}" height="${innerH}" />
    </svg>
  `;
  attachPerformanceChartHover({ points, width, height, pad, innerW, x, y });
}

function attachPerformanceChartHover({ points, width, height, pad, innerW, x, y }) {
  const svg = elements.performanceChart.querySelector("svg");
  const hitArea = elements.performanceChart.querySelector(".chart-hit-area");
  const hover = elements.performanceChart.querySelector(".chart-hover");
  if (!svg || !hitArea || !hover) return;

  const hoverLine = hover.querySelector(".hover-line");
  const indexDot = hover.querySelector(".hover-dot-performance");
  const card = hover.querySelector(".hover-card");
  const dateText = hover.querySelector(".hover-card-date");
  const returnText = hover.querySelector(".hover-card-return");

  hitArea.addEventListener("mousemove", (event) => {
    const svgX = pointerXInViewBox(event, svg, width);
    const ratio = clamp((svgX - pad.left) / innerW, 0, 1);
    const idx = Math.round(ratio * (points.length - 1));
    const row = points[idx];
    const hoverX = x(idx);
    const hoverY = y(row.value);
    const cardX = hoverX > width - pad.right - 166 ? hoverX - 166 : hoverX + 12;
    const cardY = Math.max(pad.top + 8, Math.min(height - pad.bottom - 66, hoverY - 30));

    hover.setAttribute("visibility", "visible");
    hoverLine.setAttribute("x1", hoverX);
    hoverLine.setAttribute("x2", hoverX);
    setDot(indexDot, hoverX, hoverY);
    card.setAttribute("transform", `translate(${cardX.toFixed(2)} ${cardY.toFixed(2)})`);
    dateText.textContent = row.date;
    returnText.textContent = `Excess Return: ${formatSignedPercentValue(row.value)}`;
  });

  hitArea.addEventListener("mouseleave", () => {
    hover.setAttribute("visibility", "hidden");
  });
}

function renderReturnRiskChart(points) {
  const width = 960;
  const height = 420;
  const pad = { top: 24, right: 28, bottom: 42, left: 64 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const values = points.flatMap((row) => [row.y, row.v, row.e]);
  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  const span = maxY - minY || 1;
  minY -= span * 0.08;
  maxY += span * 0.08;

  const x = (idx) => pad.left + (points.length === 1 ? innerW / 2 : (idx / (points.length - 1)) * innerW);
  const y = (value) => pad.top + ((maxY - value) / (maxY - minY)) * innerH;
  const line = (field) => points.map((row, idx) => `${x(idx).toFixed(2)},${y(row[field]).toFixed(2)}`).join(" ");
  const zeroY = y(0);
  const breachPoints = points
    .map((row, idx) => ({ row, idx }))
    .filter(({ row }) => row.y < row.v)
    .map(({ row, idx }) => `<circle class="breach-point" cx="${x(idx).toFixed(2)}" cy="${y(row.y).toFixed(2)}" r="4.5" />`)
    .join("");
  const firstLabel = points[0].targetDate.slice(0, 7);
  const lastLabel = points[points.length - 1].targetDate.slice(0, 7);

  elements.chart.innerHTML = `
    <svg class="stock-chart detail-stock-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Stock excess return, VaR, and ES history">
      <line class="axis-line" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
      <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
      ${zeroY >= pad.top && zeroY <= height - pad.bottom ? `<line class="zero-line" x1="${pad.left}" y1="${zeroY.toFixed(2)}" x2="${width - pad.right}" y2="${zeroY.toFixed(2)}" />` : ""}
      <polyline class="series-return" points="${line("y")}" />
      <polyline class="series-var" points="${line("v")}" />
      <polyline class="series-es" points="${line("e")}" />
      ${breachPoints}
      <text x="${pad.left}" y="${height - 12}" font-size="12" fill="#657181">${firstLabel}</text>
      <text x="${width - pad.right}" y="${height - 12}" font-size="12" fill="#657181" text-anchor="end">${lastLabel}</text>
      <text x="${pad.left + innerW / 2}" y="${height - 12}" font-size="12" fill="#657181" text-anchor="middle">Date</text>
      <text x="12" y="${pad.top + 4}" font-size="12" fill="#657181">${formatPercent(maxY)}</text>
      <text x="12" y="${height - pad.bottom}" font-size="12" fill="#657181">${formatPercent(minY)}</text>
      <g class="chart-hover" visibility="hidden" aria-hidden="true">
        <line class="hover-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
        <circle class="hover-dot hover-dot-return" r="4" />
        <circle class="hover-dot hover-dot-var" r="4" />
        <circle class="hover-dot hover-dot-es" r="4" />
        <g class="hover-card">
          <rect class="hover-card-bg" width="178" height="92" rx="6" />
          <text class="hover-card-date" x="12" y="20"></text>
          <text class="hover-card-return" x="12" y="42"></text>
          <text class="hover-card-var" x="12" y="62"></text>
          <text class="hover-card-es" x="12" y="82"></text>
        </g>
      </g>
      <rect class="chart-hit-area" x="${pad.left}" y="${pad.top}" width="${innerW}" height="${innerH}" />
    </svg>
  `;
  attachStockChartHover({ points, width, height, pad, innerW, x, y });
}

function attachStockChartHover({ points, width, height, pad, innerW, x, y }) {
  const svg = elements.chart.querySelector("svg");
  const hitArea = elements.chart.querySelector(".chart-hit-area");
  const hover = elements.chart.querySelector(".chart-hover");
  if (!svg || !hitArea || !hover) return;

  const hoverLine = hover.querySelector(".hover-line");
  const returnDot = hover.querySelector(".hover-dot-return");
  const varDot = hover.querySelector(".hover-dot-var");
  const esDot = hover.querySelector(".hover-dot-es");
  const card = hover.querySelector(".hover-card");
  const dateText = hover.querySelector(".hover-card-date");
  const returnText = hover.querySelector(".hover-card-return");
  const varText = hover.querySelector(".hover-card-var");
  const esText = hover.querySelector(".hover-card-es");

  hitArea.addEventListener("mousemove", (event) => {
    const svgX = pointerXInViewBox(event, svg, width);
    const ratio = clamp((svgX - pad.left) / innerW, 0, 1);
    const idx = Math.round(ratio * (points.length - 1));
    const row = points[idx];
    const hoverX = x(idx);
    const cardX = hoverX > width - pad.right - 190 ? hoverX - 190 : hoverX + 12;
    const cardY = Math.max(pad.top + 8, Math.min(height - pad.bottom - 100, y(row.y) - 46));

    hover.setAttribute("visibility", "visible");
    hoverLine.setAttribute("x1", hoverX);
    hoverLine.setAttribute("x2", hoverX);
    setDot(returnDot, hoverX, y(row.y));
    setDot(varDot, hoverX, y(row.v));
    setDot(esDot, hoverX, y(row.e));
    card.setAttribute("transform", `translate(${cardX.toFixed(2)} ${cardY.toFixed(2)})`);
    dateText.textContent = row.targetDate;
    returnText.textContent = `Excess Return: ${formatPercent(row.y)}%`;
    varText.textContent = `VaR: ${formatPercent(row.v)}%`;
    esText.textContent = `ES: ${formatPercent(row.e)}%`;
  });

  hitArea.addEventListener("mouseleave", () => {
    hover.setAttribute("visibility", "hidden");
  });
}

function pointerXInViewBox(event, svg, viewBoxWidth) {
  const bounds = svg.getBoundingClientRect();
  if (!bounds.width) return 0;
  return ((event.clientX - bounds.left) / bounds.width) * viewBoxWidth;
}

function setDot(dot, cx, cy) {
  dot.setAttribute("cx", cx.toFixed(2));
  dot.setAttribute("cy", cy.toFixed(2));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function renderRealizedSummary(points) {
  const breaches = points.filter((row) => row.y < row.v).length;
  const avgReturn = mean(points.map((row) => row.y));
  const avgVar = mean(points.map((row) => row.v));
  const avgEs = mean(points.map((row) => row.e));
  stockBacktestState.realized = { breaches, n: points.length, avgReturn, avgVar, avgEs };
  renderBacktestPanel();
}

function renderBacktestPanel() {
  if (!elements.backtestList) return;
  const formal = stockBacktestState.formal;
  const monitor = stockBacktestState.monitor;
  const realized = stockBacktestState.realized;
  const items = [];

  if (formal) {
    const varTest = formal.var_cc || {};
    const esTest = formal.es_aser || {};
    const esLine =
      esTest.status === "not_testable"
        ? `${statusBadge(esTest.status)} ES ASER: not enough history`
        : `${statusBadge(esTest.status)} ES ASER p-value: ${formatPValue(esTest.p_value)}`;
    items.push(
      `${statusBadge(varTest.status)} VaR CC p-value: ${formatPValue(varTest.p_value)}`,
      esLine,
    );
  }

  if (monitor) {
    items.push(
      `${statusBadge(monitor.alert)} E-backtesting alert`,
    );
  }

  if (realized) {
    items.push(
      `VaR breach count: ${realized.breaches} / ${realized.n}`,
      `Average excess return: ${formatPercent(realized.avgReturn)}%`,
      `Average VaR / ES: ${formatPercent(realized.avgVar)}% / ${formatPercent(realized.avgEs)}%`,
    );
  }

  elements.backtestList.innerHTML = items.length
    ? items.map((item) => `<li>${item}</li>`).join("")
    : "<li>Backtest snapshot loading...</li>";
}

function statusBadge(status) {
  const label = statusLabel(status);
  return `<span class="status-badge status-${status || "unavailable"}">${label}</span>`;
}

function statusLabel(status) {
  if (status === "pass") return "Pass";
  if (status === "fail") return "Fail";
  if (status === "not_testable") return "History";
  if (status === "green") return "Green";
  if (status === "yellow") return "Yellow";
  if (status === "red") return "Red";
  return "Pending";
}

function formatPValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(3);
}

function renderRelatedStocks(rows, targetMonth) {
  const topRows = rows.filter((row) => stockIndex.has(String(row[0]))).slice(0, 10);
  if (!topRows.length) {
    renderRelatedEmpty("No related stocks found for this month.");
    return;
  }
  const scores = topRows.map((row) => Number(row[2])).filter(Number.isFinite);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const scoreSpan = maxScore - minScore;

  elements.relatedList.innerHTML = topRows
    .map((row, index) => {
      const [id, , similarity] = row;
      const meta = stockIndex.get(String(id)) || {};
      const label = meta.ticker || "N/A";
      const name = meta.name || "";
      const industry = meta.industry || "";
      const strength = normalizeRelatedStrength(similarity, minScore, scoreSpan);
      const heatStyle = [
        `--related-width: ${(34 + strength * 52).toFixed(1)}%`,
        `--related-color: ${relatedHeatColor(strength)}`,
        `--related-bg: ${relatedHeatColor(strength, 0.18 + strength * 0.34)}`,
      ].join("; ");
      return `
        <li style="${heatStyle}">
          <a href="./stock.html?id=${encodeURIComponent(id)}">
            <span class="related-rank">${index + 1}</span>
            <span class="related-main">
              <strong>${label}</strong>
              ${name ? `<small>${name}</small>` : ""}
              ${industry ? `<small>${industry}</small>` : ""}
            </span>
          </a>
        </li>
      `;
    })
    .join("");
  elements.relatedList.setAttribute("aria-label", `Related stocks for ${targetMonth}`);
}

function renderHistoricalAnalogues(rows, targetMonth) {
  if (!elements.analogueList) return;
  const weights = rows.map((row) => Number(row.avg_weight)).filter(Number.isFinite);
  const minWeight = Math.min(...weights);
  const maxWeight = Math.max(...weights);
  const weightSpan = maxWeight - minWeight;

  const cards = rows
    .map((row) => {
      const strength = normalizeRelatedStrength(row.avg_weight, minWeight, weightSpan);
      const meta = row.meta || {};
      const outcome = row.outcome;
      const ticker = meta.ticker || String(row.retrieved_id);
      const name = meta.name || "";
      const industry = meta.industry || "";
      const outcomeLine = outcome
        ? `
          <div class="analogue-metrics">
            <span><small>Excess Return</small><strong>${formatPercent(outcome.y)}%</strong></span>
            <span><small>VaR</small><strong>${formatPercent(outcome.v)}%</strong></span>
            <span><small>ES</small><strong>${formatPercent(outcome.e)}%</strong></span>
          </div>
        `
        : '<p class="analogue-muted">Outcome unavailable</p>';
      const breach = outcome ? outcome.y < outcome.v : false;
      const heatStyle = [
        `--analogue-strength: ${(strength * 100).toFixed(1)}%`,
        `--analogue-bg: ${relatedHeatColor(strength, 0.14 + strength * 0.32)}`,
      ].join("; ");

      return `
        <article class="analogue-card" style="${heatStyle}">
          <a class="analogue-card-link" href="./stock.html?id=${encodeURIComponent(row.retrieved_id)}">
            <span class="related-rank">${row.rank}</span>
            <span class="analogue-card-main">
              <strong>${ticker}</strong>
              ${name ? `<small>${name}</small>` : ""}
              ${industry ? `<small>${industry}</small>` : ""}
            </span>
          </a>
          <div class="analogue-card-meta">
            <span>${nextMonthEnd(row.retrieved_eom).slice(0, 7)}</span>
            ${row.is_self_stock ? "<span>Same stock</span>" : ""}
            ${breach ? '<span class="breach-chip">VaR breach</span>' : ""}
          </div>
          ${outcomeLine}
        </article>
      `;
    })
    .join("");

  elements.analogueList.innerHTML = `
    ${renderAnalogueMap(rows, targetMonth, minWeight, weightSpan)}
    <div class="analogue-card-grid">
      ${cards}
    </div>
  `;
  attachAnalogueMapHover();
}

function renderAnalogueMap(rows, targetMonth, minWeight, weightSpan) {
  const plottedRows = rows
    .map((row) => ({
      ...row,
      targetDate: nextMonthEnd(row.retrieved_eom),
      yearsBack: analogueYearsBack(row.retrieved_eom, targetMonth),
      strength: normalizeRelatedStrength(row.avg_weight, minWeight, weightSpan),
    }))
    .filter((row) => row.outcome && row.targetDate && Number.isFinite(row.yearsBack));

  if (!plottedRows.length) {
    return '<div class="analogue-map-panel"><p class="chart-empty">Historical outcome data is not available for these analogues.</p></div>';
  }

  const width = 940;
  const height = 330;
  const pad = { top: 58, right: 34, bottom: 46, left: 66 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const bands = [...new Set(plottedRows.map((row) => row.yearsBack))].sort((a, b) => a - b);
  const bandWidth = innerW / bands.length;
  const bandIndex = new Map(bands.map((band, index) => [band, index]));
  const groupedBandRows = bands.reduce((groups, band) => {
    groups[band] = plottedRows.filter((row) => row.yearsBack === band);
    return groups;
  }, {});
  const values = plottedRows.map((row) => row.outcome.y);
  let minY = Math.min(...values, 0);
  let maxY = Math.max(...values, 0);
  const spanY = maxY - minY || 1;
  minY -= spanY * 0.16;
  maxY += spanY * 0.16;

  const x = (row) => {
    const index = bandIndex.get(row.yearsBack) ?? 0;
    const rowsInBand = groupedBandRows[row.yearsBack] || [];
    const position = rowsInBand.findIndex((item) => item === row);
    const count = Math.max(rowsInBand.length, 1);
    const local = count === 1 ? 0.5 : (position + 0.5) / count;
    return pad.left + index * bandWidth + bandWidth * (0.18 + local * 0.64);
  };
  const y = (value) => pad.top + ((maxY - value) / (maxY - minY)) * innerH;
  const zeroY = y(0);
  const bandRects = bands
    .map((band, index) => {
      const x0 = pad.left + index * bandWidth;
      const fill = index % 2 ? "rgba(248, 250, 251, 0.72)" : "rgba(242, 250, 248, 0.9)";
      return `
        <rect class="analogue-band-bg" x="${x0.toFixed(2)}" y="${pad.top}" width="${bandWidth.toFixed(2)}" height="${innerH}" fill="${fill}" />
        <text x="${(x0 + bandWidth / 2).toFixed(2)}" y="28" font-size="13" fill="#17202a" text-anchor="middle" font-weight="800">${formatYearsBack(band)} back</text>
        <text x="${(x0 + bandWidth / 2).toFixed(2)}" y="45" font-size="11" fill="#657181" text-anchor="middle">${groupedBandRows[band].length} states</text>
      `;
    })
    .join("");
  const nodes = plottedRows
    .map((row, index) => {
      const meta = row.meta || {};
      const cx = x(row);
      const cy = y(row.outcome.y);
      const r = 6 + row.strength * 9;
      const breach = row.outcome.y < row.outcome.v;
      const ticker = meta.ticker || row.retrieved_id;
      return `
        <g
          class="analogue-node ${breach ? "is-breach" : ""}"
          data-index="${index}"
          data-cx="${cx.toFixed(2)}"
          data-cy="${cy.toFixed(2)}"
          data-r="${r.toFixed(2)}"
          data-ticker="${escapeAttribute(ticker)}"
          data-name="${escapeAttribute(meta.name || "")}"
          data-period="${row.targetDate.slice(0, 7)}"
          data-return="${formatPercent(row.outcome.y)}%"
          data-var="${formatPercent(row.outcome.v)}%"
          data-es="${formatPercent(row.outcome.e)}%"
          data-breach="${breach ? "VaR breach" : "No breach"}"
        >
          <circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${r.toFixed(2)}" style="--node-alpha:${(0.34 + row.strength * 0.56).toFixed(3)}" />
        </g>
      `;
    })
    .join("");
  const labels = plottedRows.slice(0, 5)
    .map((row) => {
      const meta = row.meta || {};
      return `<span>${meta.ticker || row.retrieved_id}</span>`;
    })
    .join("");
  const breachCount = plottedRows.filter((row) => row.outcome.y < row.outcome.v).length;
  const crossStockCount = rows.filter((row) => !row.is_self_stock).length;
  const avgReturn = mean(plottedRows.map((row) => row.outcome.y));

  return `
    <div class="analogue-map-panel">
      <div class="analogue-map-header">
        <div>
          <span>Historical similarity map</span>
          <strong>${plottedRows.length} closest annual echoes</strong>
        </div>
        <div class="analogue-map-stats">
          <span><small>Cross-stock</small><strong>${crossStockCount}</strong></span>
          <span><small>Breach cases</small><strong>${breachCount}</strong></span>
          <span><small>Avg excess return</small><strong>${formatPercent(avgReturn)}%</strong></span>
        </div>
      </div>
      <svg class="analogue-map" viewBox="0 0 ${width} ${height}" role="img" aria-label="Historical analogue map">
        ${bandRects}
        <line class="axis-line" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
        <line class="axis-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
        ${zeroY >= pad.top && zeroY <= height - pad.bottom ? `<line class="zero-line" x1="${pad.left}" y1="${zeroY.toFixed(2)}" x2="${width - pad.right}" y2="${zeroY.toFixed(2)}" />` : ""}
        <text x="12" y="${pad.top + 4}" font-size="12" fill="#657181">${formatPercent(maxY)}%</text>
        <text x="12" y="${height - pad.bottom}" font-size="12" fill="#657181">${formatPercent(minY)}%</text>
        ${nodes}
        <g class="analogue-hover-layer" visibility="hidden" aria-hidden="true">
          <line class="hover-line" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
          <g class="analogue-hover-card">
            <rect class="hover-card-bg" width="204" height="138" rx="8" />
            <text class="analogue-hover-ticker" x="12" y="22"></text>
            <text class="analogue-hover-name" x="12" y="42"></text>
            <text class="analogue-hover-period" x="12" y="62"></text>
            <text class="analogue-hover-return" x="12" y="86"></text>
            <text class="analogue-hover-var" x="12" y="106"></text>
            <text class="analogue-hover-es" x="12" y="126"></text>
          </g>
        </g>
        <text x="${pad.left + innerW / 2}" y="${height - 14}" font-size="12" fill="#657181" text-anchor="middle">Historical annual segment</text>
      </svg>
      <div class="analogue-map-footer">
        <span class="node-key"></span>
        <span>node size shows match strength; vertical position shows subsequent excess return</span>
        <span>bands use same-calendar-month annual states</span>
        ${labels}
      </div>
    </div>
  `;
}

function attachAnalogueMapHover() {
  const svg = elements.analogueList?.querySelector(".analogue-map");
  const hover = elements.analogueList?.querySelector(".analogue-hover-layer");
  if (!svg || !hover) return;

  const hoverLine = hover.querySelector(".hover-line");
  const card = hover.querySelector(".analogue-hover-card");
  const tickerText = hover.querySelector(".analogue-hover-ticker");
  const nameText = hover.querySelector(".analogue-hover-name");
  const periodText = hover.querySelector(".analogue-hover-period");
  const returnText = hover.querySelector(".analogue-hover-return");
  const varText = hover.querySelector(".analogue-hover-var");
  const esText = hover.querySelector(".analogue-hover-es");
  const nodes = Array.from(svg.querySelectorAll(".analogue-node"));
  const viewBox = svg.viewBox.baseVal;
  const width = viewBox?.width || 940;
  const height = viewBox?.height || 330;

  nodes.forEach((node) => {
    node.addEventListener("mouseenter", () => {
      const cx = Number(node.dataset.cx);
      const cy = Number(node.dataset.cy);
      const r = Number(node.dataset.r);
      const cardWidth = 204;
      const cardHeight = 138;
      const cardX = cx > width - cardWidth - 42 ? cx - cardWidth - 18 : cx + 18;
      const cardY = Math.max(12, Math.min(height - cardHeight - 12, cy - 62));

      hover.setAttribute("visibility", "visible");
      hoverLine.setAttribute("x1", cx.toFixed(2));
      hoverLine.setAttribute("x2", cx.toFixed(2));
      card.setAttribute("transform", `translate(${cardX.toFixed(2)} ${cardY.toFixed(2)})`);
      tickerText.textContent = node.dataset.ticker || "-";
      nameText.textContent = compactLabel(node.dataset.name || "", 24);
      periodText.textContent = `${node.dataset.period || "-"} · ${node.dataset.breach || ""}`;
      returnText.textContent = `Excess Return: ${node.dataset.return || "-"}`;
      varText.textContent = `VaR: ${node.dataset.var || "-"}`;
      esText.textContent = `ES: ${node.dataset.es || "-"}`;
    });
    node.addEventListener("mouseleave", () => {
      hover.setAttribute("visibility", "hidden");
    });
  });
}

function compactLabel(value, maxLength) {
  const label = String(value ?? "").trim();
  if (!label) return "";
  return label.length > maxLength ? `${label.slice(0, maxLength - 1)}...` : label;
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

function escapeAttribute(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderAnalogueEmpty(message) {
  if (!elements.analogueList) return;
  elements.analogueList.innerHTML = `<p class="chart-empty">${message}</p>`;
}

function renderRelatedEmpty(message) {
  elements.relatedList.innerHTML = `<li class="related-empty">${message}</li>`;
}

function normalizeRelatedStrength(value, minScore, scoreSpan) {
  const score = Number(value);
  if (!Number.isFinite(score)) return 0.5;
  if (!Number.isFinite(scoreSpan) || scoreSpan <= 0) return 1;
  return clamp((score - minScore) / scoreSpan, 0, 1);
}

function relatedHeatColor(strength, alpha) {
  const hue = 180 - strength * 145;
  const saturation = 58 + strength * 14;
  const lightness = 42 + strength * 3;
  if (Number.isFinite(alpha)) {
    return `hsla(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%, ${alpha.toFixed(3)})`;
  }
  return `hsl(${hue.toFixed(0)}, ${saturation.toFixed(0)}%, ${lightness.toFixed(0)}%)`;
}

function hasDisplayTicker(row) {
  const ticker = String(row?.ticker ?? "").trim().toUpperCase();
  return ticker && ticker !== "NA" && ticker !== "N/A";
}

function mean(values) {
  const nums = values.map(Number).filter((value) => Number.isFinite(value));
  if (!nums.length) return NaN;
  return nums.reduce((total, value) => total + value, 0) / nums.length;
}

loadStock();
