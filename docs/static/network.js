const NETWORK_PATH = "./data/networks/latest.json";
const STOCK_INDEX_PATH = "./data/predictions/stocks.json";
const MAX_VISIBLE_CLUSTER_EDGES = 28;

const elements = {
  title: document.querySelector("#networkTitle"),
  subtitle: document.querySelector("#networkSubtitle"),
  date: document.querySelector("#networkDate"),
  size: document.querySelector("#networkSize"),
  industry: document.querySelector("#networkIndustry"),
  varValue: document.querySelector("#networkVar"),
  esValue: document.querySelector("#networkEs"),
  graph: document.querySelector("#networkGraph"),
  members: document.querySelector("#networkMembers"),
  tooltip: document.querySelector("#networkTooltip"),
};

let stockIndex = new Map();
let industryByCode = new Map();
let selectedGroup = null;
let allGroups = [];
let centerIds = new Set();
let groupByAnchor = new Map();
let pinnedNetworkNode = null;

function getAnchorId() {
  return new URLSearchParams(window.location.search).get("anchor");
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

function hasDisplayTicker(row) {
  const ticker = String(row?.ticker || "").trim().toUpperCase();
  return ticker && ticker !== "NA" && ticker !== "NAN" && ticker !== "NULL" && ticker !== "-";
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return (Number(value) * 100).toFixed(2);
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(3);
}

function formatMaybePercent(value) {
  const formatted = formatPercent(value);
  return formatted === "-" ? "-" : `${formatted}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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

function enrichGroup(group) {
  const anchorMeta = stockIndex.get(String(group.anchor_id)) || {};
  const members = (group.members || [])
    .map((member) => ({
      ...member,
      meta: stockIndex.get(String(member.id)) || {},
    }))
    .filter((member) => hasDisplayTicker(member.meta))
    .sort((a, b) => Number(b.similarity || 0) - Number(a.similarity || 0));
  const dominantIndustry =
    industryByCode.get(Number(group.dominant_industry_code)) ||
    anchorMeta.industry ||
    "Mixed industries";

  return {
    ...group,
    anchorMeta,
    members,
    dominantIndustry,
  };
}

async function loadNetwork() {
  try {
    const [network, stocks] = await Promise.all([
      fetchJson(NETWORK_PATH),
      fetchJson(STOCK_INDEX_PATH),
    ]);
    const displayStocks = stocks.filter(hasDisplayTicker);
    stockIndex = new Map(displayStocks.map((row) => [String(row.id), row]));
    industryByCode = buildIndustryLookup(displayStocks);

    allGroups = (network.groups || [])
      .filter((group) => group.members?.length)
      .sort((a, b) => {
        if (Number(b.size) !== Number(a.size)) return Number(b.size) - Number(a.size);
        return Number(a.avg_es || 0) - Number(b.avg_es || 0);
      });
    centerIds = new Set(allGroups.map((group) => String(group.anchor_id)));
    groupByAnchor = new Map(allGroups.map((group) => [String(group.anchor_id), group]));

    const anchorId = getAnchorId();
    const rawGroup = allGroups.find((group) => String(group.anchor_id) === String(anchorId)) || allGroups[0];
    selectedGroup = rawGroup ? enrichGroup(rawGroup) : null;
    if (!selectedGroup) {
      showError("No network data is available.");
      return;
    }

    renderSummary(selectedGroup, network.target_month);
    renderGraph(selectedGroup);
    renderMembers(selectedGroup);
    renderStockCard(buildAnchorNode(selectedGroup));
  } catch (error) {
    showError("Unable to load risk network.");
    console.error(error);
  }
}

function renderSummary(group, targetMonth) {
  const anchorTicker = group.anchorMeta.ticker || group.anchor_id;
  const anchorName = group.anchorMeta.name || "";
  elements.title.textContent = `${anchorTicker} Risk Network`;
  elements.subtitle.textContent = anchorName
    ? `${anchorName} and its strongest connected stocks.`
    : "Strongest connected stocks and group-level risk profile.";
  elements.date.textContent = targetMonth ? targetMonth.slice(0, 7) : "-";
  elements.size.textContent = Number(group.size || group.members.length || 0).toLocaleString();
  elements.industry.textContent = group.dominantIndustry;
  elements.varValue.textContent = formatPercent(group.avg_var);
  elements.esValue.textContent = formatPercent(group.avg_es);
}

function normalize(value, min, max) {
  if (!Number.isFinite(value)) return 0.5;
  if (max <= min) return 0.65;
  return (value - min) / (max - min);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function hashUnit(value, salt = 0) {
  let hash = 2166136261 + salt * 16777619;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 10000) / 10000;
}

function similarityColor(value) {
  const strength = Math.max(0, Math.min(1, value));
  const lightness = 88 - strength * 42;
  const saturation = 34 + strength * 22;
  return `hsl(178 ${saturation.toFixed(0)}% ${lightness.toFixed(0)}%)`;
}

function buildAnchorNode(group) {
  return {
    id: group.anchor_id,
    meta: group.anchorMeta,
    similarity: 1,
    seed_count: 5,
    v: group.anchor_var,
    e: group.anchor_es,
    isAnchor: true,
  };
}

function enrichMember(member) {
  return {
    ...member,
    meta: stockIndex.get(String(member.id)) || {},
  };
}

function buildLocalSubgraph(group) {
  const mainId = String(group.anchor_id);
  const primaryMembers = group.members.map((member) => ({
    ...member,
    role: centerIds.has(String(member.id)) ? "subcenter" : "primary",
    layer: "primary",
    meta: member.meta,
    card: member,
    strength: Number(member.similarity || 0),
  }));
  const primaryIds = new Set(primaryMembers.map((member) => String(member.id)));
  primaryIds.add(mainId);

  const nodes = [
    {
      id: group.anchor_id,
      role: "main",
      layer: "main",
      strength: 1,
      meta: group.anchorMeta,
      card: buildAnchorNode(group),
    },
    ...primaryMembers,
  ];
  const edges = primaryMembers.map((member) => ({
    source: String(member.id),
    target: mainId,
    type: "primary",
    parent: mainId,
    similarity: Number(member.similarity || 0),
    visual: true,
  }));
  const secondaryIds = new Set();
  const clusterEdgeKeys = new Set();
  const clusterEdges = [];

  primaryMembers.forEach((member) => {
    const parentId = String(member.id);
    const subGroup = groupByAnchor.get(parentId);
    if (!subGroup) return;
    (subGroup.members || []).forEach((row) => {
      const rowId = String(row.id);
      if (!primaryIds.has(rowId) || rowId === parentId || rowId === mainId) return;
      const key = [parentId, rowId].sort().join("__");
      if (clusterEdgeKeys.has(key)) return;
      clusterEdgeKeys.add(key);
      clusterEdges.push({
        source: rowId,
        target: parentId,
        type: "cluster",
        parent: parentId,
        similarity: Number(row.similarity || 0),
        visual: false,
      });
    });

    const secondaryMembers = (subGroup.members || [])
      .map((row) => enrichMember(row))
      .filter((row) => hasDisplayTicker(row.meta))
      .filter((row) => !primaryIds.has(String(row.id)))
      .filter((row) => !secondaryIds.has(String(row.id)))
      .sort((a, b) => Number(b.similarity || 0) - Number(a.similarity || 0))
      .slice(0, 4);

    secondaryMembers.forEach((secondary) => {
      secondaryIds.add(String(secondary.id));
      nodes.push({
        ...secondary,
        role: "secondary",
        layer: "secondary",
        parent: parentId,
        strength: Number(secondary.similarity || 0),
        meta: secondary.meta,
        card: secondary,
      });
      edges.push({
        source: String(secondary.id),
        target: parentId,
        type: "secondary",
        parent: parentId,
        similarity: Number(secondary.similarity || 0),
        visual: true,
      });
    });
  });

  clusterEdges
    .sort((a, b) => Number(b.similarity || 0) - Number(a.similarity || 0))
    .forEach((edge, index) => {
      edges.push({
        ...edge,
        featured: index < MAX_VISIBLE_CLUSTER_EDGES,
        visual: true,
      });
    });

  return { nodes, edges };
}

function graphLayout(graphNodes, graphEdges) {
  const primaryNodes = graphNodes.filter((node) => node.layer === "primary");
  const secondaryNodes = graphNodes.filter((node) => node.layer === "secondary");
  const width = Math.max(4200, 3000 + primaryNodes.length * 22);
  const height = Math.max(3000, 2300 + primaryNodes.length * 16);
  const center = { x: width / 2, y: height / 2 };
  const strengthValues = primaryNodes.map((node) => Number(node.strength)).filter(Number.isFinite);
  const minStrength = strengthValues.length ? Math.min(...strengthValues) : 0;
  const maxStrength = strengthValues.length ? Math.max(...strengthValues) : 1;
  const edgeSimilarityValues = graphEdges.map((edge) => Number(edge.similarity)).filter(Number.isFinite);
  const minEdgeSimilarity = edgeSimilarityValues.length ? Math.min(...edgeSimilarityValues) : 0;
  const maxEdgeSimilarity = edgeSimilarityValues.length ? Math.max(...edgeSimilarityValues) : 1;

  const clusterParent = new Map();
  graphEdges
    .filter((edge) => edge.type === "cluster")
    .forEach((edge) => {
      const source = String(edge.source);
      const target = String(edge.target);
      const previous = clusterParent.get(source);
      if (!previous || Number(edge.similarity || 0) > Number(previous.similarity || 0)) {
        clusterParent.set(source, { id: target, similarity: Number(edge.similarity || 0) });
      }
    });

  const subcenters = primaryNodes
    .filter((node) => node.role === "subcenter")
    .sort((a, b) => Number(b.strength || 0) - Number(a.strength || 0));
  const hubCount = Math.min(subcenters.length, Math.max(7, Math.ceil(Math.sqrt(Math.max(subcenters.length, 1))) + 3));
  const hubSubcenters = subcenters.slice(0, hubCount);
  const hubIds = new Set(hubSubcenters.map((node) => String(node.id)));
  const subcenterIndex = new Map(subcenters.map((node, index) => [String(node.id), index]));
  const orphanNodes = primaryNodes.filter((node) => node.role !== "subcenter" && !clusterParent.has(String(node.id)));
  const orphanIndex = new Map(orphanNodes.map((node, index) => [String(node.id), index]));
  const assignedCounts = new Map();
  const primaryLayout = new Map();

  hubSubcenters.forEach((item, index) => {
    const strengthNorm = normalize(Number(item.strength), minStrength, maxStrength);
    const angle = -Math.PI / 2 + (index / Math.max(hubSubcenters.length, 1)) * Math.PI * 2;
    const sizeNorm = Math.max(0, (strengthNorm - 0.12) / 0.88) ** 1.55;
    const distance = 1120 + (1 - strengthNorm) * 520;
    const x = center.x + Math.cos(angle) * distance;
    const y = center.y + Math.sin(angle) * distance * 0.86;
    primaryLayout.set(String(item.id), {
      x,
      y,
      tx: x,
      ty: y,
      vx: 0,
      vy: 0,
      strengthNorm,
      radius: 58 + sizeNorm * 132,
      layer: item.layer,
      role: item.role,
    });
  });

  subcenters.slice(hubCount).forEach((item) => {
    const strengthNorm = normalize(Number(item.strength), minStrength, maxStrength);
    const sizeNorm = Math.max(0, (strengthNorm - 0.12) / 0.88) ** 1.55;
    let parentId = clusterParent.get(String(item.id))?.id;
    if (!hubIds.has(String(parentId))) {
      const nextParent = clusterParent.get(String(parentId))?.id;
      parentId = hubIds.has(String(nextParent)) ? nextParent : null;
    }
    if (!parentId && hubSubcenters.length) {
      parentId = String(hubSubcenters[Math.floor(hashUnit(item.id, 19) * hubSubcenters.length)]?.id);
    }
    const parentLayout = primaryLayout.get(String(parentId)) || { x: center.x, y: center.y };
    const count = assignedCounts.get(parentId) || 0;
    assignedCounts.set(parentId, count + 1);
    const parentIndex = subcenterIndex.get(String(parentId)) || 0;
    const angle = parentIndex * 0.48 + count * 1.47 + hashUnit(item.id, 13) * 0.42;
    const distance = 620 + (1 - strengthNorm) * 620 + (count % 4) * 96;
    const x = parentLayout.x + Math.cos(angle) * distance;
    const y = parentLayout.y + Math.sin(angle) * distance * 0.88;
    primaryLayout.set(String(item.id), {
      x,
      y,
      tx: x,
      ty: y,
      vx: 0,
      vy: 0,
      strengthNorm,
      radius: 48 + sizeNorm * 112,
      layer: item.layer,
      role: item.role,
    });
  });

  primaryNodes
    .filter((node) => node.role !== "subcenter")
    .forEach((item) => {
      const strengthNorm = normalize(Number(item.strength), minStrength, maxStrength);
      const sizeNorm = Math.max(0, (strengthNorm - 0.12) / 0.88) ** 1.55;
      const assigned = clusterParent.get(String(item.id));
      const parentLayout = assigned ? primaryLayout.get(String(assigned.id)) : null;
      let x;
      let y;
      if (parentLayout) {
        const count = assignedCounts.get(assigned.id) || 0;
        assignedCounts.set(assigned.id, count + 1);
        const parentIndex = subcenterIndex.get(String(assigned.id)) || 0;
        const angle = parentIndex * 0.43 + count * 1.71 + hashUnit(item.id, 3) * 0.38;
        const distance = 720 + (1 - strengthNorm) * 680 + (count % 3) * 130;
        x = parentLayout.x + Math.cos(angle) * distance;
        y = parentLayout.y + Math.sin(angle) * distance * 0.9;
      } else {
        const index = orphanIndex.get(String(item.id)) || 0;
        const capacity = Math.max(orphanNodes.length, 1);
        const angle = -Math.PI / 2 + (index / capacity) * Math.PI * 2 + hashUnit(item.id, 8) * 0.18;
        const distance = 1900 + (1 - strengthNorm) * 1280;
        x = center.x + Math.cos(angle) * distance;
        y = center.y + Math.sin(angle) * distance * 0.9;
      }
      primaryLayout.set(String(item.id), {
        x,
        y,
        tx: x,
        ty: y,
        vx: 0,
        vy: 0,
        strengthNorm,
        radius: 42 + sizeNorm * 118,
        layer: item.layer,
        role: item.role,
      });
    });

  const primaryEdges = graphEdges
    .map((edge) => ({
      ...edge,
      sourceNode: primaryLayout.get(String(edge.source)),
      targetNode: primaryLayout.get(String(edge.target)),
      similarityNorm: normalize(Number(edge.similarity), minEdgeSimilarity, maxEdgeSimilarity),
    }))
    .filter((edge) => edge.sourceNode && edge.targetNode);

  for (let tick = 0; tick < 220; tick += 1) {
    primaryLayout.forEach((node) => {
      node.vx += (node.tx - node.x) * 0.038;
      node.vy += (node.ty - node.y) * 0.038;
    });

    primaryEdges.forEach((edge) => {
      const source = edge.sourceNode;
      const target = edge.targetNode;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const strength = Number(edge.similarityNorm || 0);
      const preferred =
        edge.type === "cluster" ? 760 + (1 - strength) * 520 :
        edge.type === "primary" ? 920 + (1 - strength) * 1280 :
        240;
      const pull = (distance - preferred) * (edge.type === "cluster" ? 0.0038 : 0.0018);
      const px = (dx / distance) * pull;
      const py = (dy / distance) * pull;
      source.vx += px;
      source.vy += py;
      target.vx -= px;
      target.vy -= py;
    });

    for (let left = 0; left < primaryNodes.length; left += 1) {
      const a = primaryLayout.get(String(primaryNodes[left].id));
      if (!a) continue;
      for (let right = left + 1; right < primaryNodes.length; right += 1) {
        const b = primaryLayout.get(String(primaryNodes[right].id));
        if (!b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distance = Math.max(Math.hypot(dx, dy), 1);
        const minDistance = a.radius + b.radius + 320;
        if (distance < minDistance) {
          const push = (minDistance - distance) * 0.09;
          const px = (dx / distance) * push;
          const py = (dy / distance) * push;
          a.vx -= px;
          a.vy -= py;
          b.vx += px;
          b.vy += py;
        }
      }
    }

    primaryLayout.forEach((node) => {
      node.vx *= 0.68;
      node.vy *= 0.68;
      node.x = Math.min(width - 360, Math.max(360, node.x + node.vx));
      node.y = Math.min(height - 360, Math.max(360, node.y + node.vy));
    });
  }

  const mainNode = graphNodes.find((node) => node.layer === "main");
  const nodes = [];
  if (mainNode) {
    nodes.push({
      ...mainNode,
      x: center.x,
      y: center.y,
      radius: 138,
      similarityNorm: 1,
      fill: similarityColor(1),
    });
  }

  primaryNodes.forEach((item) => {
    const layout = primaryLayout.get(String(item.id)) || center;
    const strengthNorm = layout.strengthNorm ?? normalize(Number(item.strength), minStrength, maxStrength);
    nodes.push({
      ...item,
      x: layout.x,
      y: layout.y,
      radius: layout.radius || (item.role === "subcenter" ? 46 + strengthNorm * 54 : 34 + strengthNorm * 50),
      similarityNorm: strengthNorm,
      fill: similarityColor(strengthNorm),
    });
  });

  secondaryNodes.forEach((item) => {
    const parent = primaryLayout.get(String(item.parent)) || center;
    const parentId = String(item.parent || item.id);
    const siblingHash = hashUnit(`${item.id}-${parentId}`, 5);
    const angle = siblingHash * Math.PI * 2;
    const distance = 560 + hashUnit(`${item.id}-${parentId}`, 9) * 260;
    const strengthNorm = 0.2 + normalize(Number(item.strength), minStrength, maxStrength) * 0.35;
    nodes.push({
      ...item,
      x: parent.x + Math.cos(angle) * distance,
      y: parent.y + Math.sin(angle) * distance * 0.96,
      radius: 16 + strengthNorm * 16,
      similarityNorm: strengthNorm,
      fill: similarityColor(strengthNorm),
    });
  });

  const nodeById = new Map(nodes.map((node) => [String(node.id), node]));
  const edges = graphEdges
    .map((edge) => ({
      ...edge,
      sourceNode: nodeById.get(String(edge.source)),
      targetNode: nodeById.get(String(edge.target)),
    }))
    .filter((edge) => edge.visual !== false && edge.sourceNode && edge.targetNode);

  return { width, height, center, nodes, edges };
}

function renderGraph(group) {
  if (!elements.graph) return;
  const localGraph = buildLocalSubgraph(group);
  if (!localGraph.nodes.length) {
    elements.graph.innerHTML = '<p class="chart-empty">No connected stocks found.</p>';
    return;
  }
  const { width, height, nodes, edges: layoutEdges } = graphLayout(localGraph.nodes, localGraph.edges);

  const edges = layoutEdges.map((edge) => `
    <line
      class="network-detail-edge ${edge.type === "primary" ? "is-primary" : ""} ${edge.type === "secondary" ? "is-secondary" : ""} ${edge.type === "cluster" ? "is-cluster" : ""} ${edge.type === "cluster" && edge.featured === false ? "is-cluster-hidden" : ""}"
      x1="${edge.sourceNode.x.toFixed(2)}"
      y1="${edge.sourceNode.y.toFixed(2)}"
      x2="${edge.targetNode.x.toFixed(2)}"
      y2="${edge.targetNode.y.toFixed(2)}"
      stroke-width="${edge.type === "cluster" ? "2.05" : edge.type === "secondary" ? "1.8" : "2.65"}"
      stroke-opacity="${edge.type === "cluster" ? "1" : edge.type === "secondary" ? "0.5" : "0.62"}"
      data-source="${edge.source}"
      data-target="${edge.target}"
      data-parent-id="${edge.parent || ""}"
      data-edge-type="${edge.type}"
    />
  `).join("");

  const nodeMarkup = nodes.map((node, index) => {
    const meta = node.meta || {};
    const ticker = escapeHtml(meta.ticker || node.id);
    const labelInside = node.radius >= 52 || node.role === "main";
    const showLabel =
      node.role === "main" ||
      node.radius >= 58 ||
      (node.role === "subcenter" && node.radius >= 48) ||
      (node.layer === "primary" && index < 12 && node.radius >= 44);
    const labelClass = labelInside ? "network-detail-label is-inside" : "network-detail-label";
    const labelY = labelInside ? node.y + 4 : node.y - node.radius - 8;
    const nodeClass = node.role === "main" ? "network-detail-anchor" : `network-detail-node ${node.role === "subcenter" ? "is-subcenter" : ""}`;
    return `
      <g class="network-node-wrap ${showLabel ? "is-major" : ""} ${node.role === "main" ? "is-anchor is-selected" : ""} ${node.role === "subcenter" ? "is-subcenter" : ""} ${node.layer === "secondary" ? "is-secondary" : ""}" data-id="${node.id}" data-parent-id="${node.parent || ""}">
        <circle
          class="${nodeClass}"
          cx="${node.x.toFixed(2)}"
          cy="${node.y.toFixed(2)}"
          r="${node.radius.toFixed(2)}"
          fill="${node.fill}"
          data-base-radius="${node.radius.toFixed(2)}"
          data-id="${node.id}"
          data-parent-id="${node.parent || ""}"
        ></circle>
        <text class="${labelClass}" x="${node.x.toFixed(2)}" y="${labelY.toFixed(2)}" text-anchor="middle">${ticker}</text>
      </g>
    `;
  }).join("");

  elements.graph.innerHTML = `
    <svg class="network-detail-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Latent risk network graph">
      <g class="network-viewport">
        <g class="network-edge-layer">${edges}</g>
        <g class="network-node-layer">${nodeMarkup}</g>
      </g>
    </svg>
  `;

  const nodesById = new Map(nodes.map((node) => [String(node.id), node.card]));
  nodesById.set(String(group.anchor_id), buildAnchorNode(group));
  const anchorId = String(group.anchor_id);
  pinnedNetworkNode = buildAnchorNode(group);
  let activeViewId = anchorId;
  const viewport = elements.graph.querySelector(".network-viewport");
  const svg = elements.graph.querySelector(".network-detail-svg");
  let currentView = initialNetworkView(width, height);
  let dragState = null;

  const setSelectedNode = (id) => {
    elements.graph.querySelectorAll(".network-detail-node, .network-detail-anchor").forEach((item) => {
      item.classList.toggle("is-selected", String(item.dataset.id) === String(id));
    });
  };

  const setView = (nextView) => {
    if (!viewport) return;
    currentView = {
      scale: clamp(nextView.scale, 0.55, 2.6),
      tx: nextView.tx,
      ty: nextView.ty,
    };
    viewport.setAttribute(
      "transform",
      `matrix(${currentView.scale.toFixed(3)} 0 0 ${currentView.scale.toFixed(3)} ${currentView.tx.toFixed(2)} ${currentView.ty.toFixed(2)})`
    );
  };

  const pinView = (id) => {
    activeViewId = String(id);
    setSelectedNode(activeViewId);
    setGraphFocus(activeViewId, true);
    const member = nodesById.get(activeViewId);
    if (member) {
      pinnedNetworkNode = member;
      renderStockCard(member);
    }
  };

  const svgPointFromEvent = (event, target = svg) => {
    if (!target) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = target.getScreenCTM();
    return matrix ? point.matrixTransform(matrix.inverse()) : null;
  };

  setView(currentView);

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const svgPoint = svgPointFromEvent(event, svg);
    const graphPoint = svgPointFromEvent(event, viewport);
    if (!svgPoint || !graphPoint) return;
    const zoomFactor = Math.exp(-event.deltaY * 0.0015);
    const nextScale = clamp(currentView.scale * zoomFactor, 0.55, 2.6);
    setView({
      scale: nextScale,
      tx: svgPoint.x - graphPoint.x * nextScale,
      ty: svgPoint.y - graphPoint.y * nextScale,
    });
  }, { passive: false });

  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest(".network-node-wrap")) return;
    const point = svgPointFromEvent(event, svg);
    if (!point) return;
    dragState = { point, view: { ...currentView } };
    svg.classList.add("is-panning");
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!dragState) return;
    const point = svgPointFromEvent(event, svg);
    if (!point) return;
    setView({
      scale: dragState.view.scale,
      tx: dragState.view.tx + point.x - dragState.point.x,
      ty: dragState.view.ty + point.y - dragState.point.y,
    });
  });

  const stopPan = (event) => {
    dragState = null;
    svg.classList.remove("is-panning");
    if (event.pointerId !== undefined && svg.hasPointerCapture(event.pointerId)) {
      svg.releasePointerCapture(event.pointerId);
    }
  };
  svg.addEventListener("pointerup", stopPan);
  svg.addEventListener("pointercancel", stopPan);

  elements.graph.querySelectorAll(".network-detail-node, .network-detail-anchor").forEach((node) => {
    node.addEventListener("mouseenter", () => {
      const baseRadius = Number(node.dataset.baseRadius || 10);
      node.setAttribute("r", (baseRadius * 1.2).toFixed(2));
      node.classList.add("is-hovered");
      setGraphFocus(String(node.dataset.id), false);
      const member = nodesById.get(String(node.dataset.id));
      if (member) renderStockCard(member);
    });
    node.addEventListener("mouseleave", () => {
      node.setAttribute("r", node.dataset.baseRadius || "10");
      node.classList.remove("is-hovered");
      setGraphFocus(activeViewId, true);
      if (pinnedNetworkNode) renderStockCard(pinnedNetworkNode);
    });
    node.addEventListener("click", () => {
      const nextId = String(node.dataset.id);
      pinView(activeViewId === nextId && nextId !== anchorId ? anchorId : nextId);
    });
  });
  const requestedFocus = new URLSearchParams(window.location.search).get("focus");
  pinView(requestedFocus && nodesById.has(String(requestedFocus)) ? requestedFocus : anchorId);
}

function initialNetworkView(width, height) {
  return {
    scale: 0.9,
    tx: width * 0.05,
    ty: height * 0.03,
  };
}

function setGraphFocus(id, pinned) {
  if (!elements.graph) return;
  const isAnchorFocus = selectedGroup && String(id) === String(selectedGroup.anchor_id);
  let focusedWrap = null;
  elements.graph.querySelectorAll(".network-node-wrap").forEach((wrap) => {
    if (String(wrap.dataset.id) === String(id)) {
      focusedWrap = wrap;
    }
  });
  const focusedParent = focusedWrap?.dataset?.parentId || "";
  const revealParent = isAnchorFocus ? "" : String(id);
  const connectedNodeIds = new Set();

  if (!isAnchorFocus) {
    elements.graph.querySelectorAll(".network-detail-edge").forEach((edge) => {
      const source = String(edge.dataset.source || "");
      const target = String(edge.dataset.target || "");
      if (source === String(id) && target) connectedNodeIds.add(target);
      if (target === String(id) && source) connectedNodeIds.add(source);
    });
  }

  elements.graph.querySelectorAll(".network-node-wrap").forEach((wrap) => {
    const wrapId = String(wrap.dataset.id);
    const active = wrapId === String(id);
    const isSecondary = wrap.classList.contains("is-secondary");
    const belongsToFocus = String(wrap.dataset.parentId || "") === revealParent;
    const belongsToPinnedParent = focusedParent && wrapId === focusedParent;
    const connectedToFocus = connectedNodeIds.has(wrapId);
    wrap.classList.toggle("is-focused", active);
    wrap.classList.toggle("is-revealed", !isAnchorFocus && isSecondary && belongsToFocus);
    wrap.classList.toggle("is-context", !isAnchorFocus && (belongsToPinnedParent || connectedToFocus));
    wrap.classList.toggle(
      "is-muted",
      Boolean(id) && !isAnchorFocus && !active && !belongsToFocus && !belongsToPinnedParent && !connectedToFocus,
    );
  });

  elements.graph.querySelectorAll(".network-detail-edge").forEach((edge) => {
    const primaryHit = String(edge.dataset.source) === String(id) || String(edge.dataset.target) === String(id);
    const secondaryHit = !isAnchorFocus && String(edge.dataset.parentId || "") === revealParent;
    edge.classList.toggle("is-focused", !isAnchorFocus && (primaryHit || secondaryHit));
    edge.classList.toggle("is-revealed", secondaryHit);
    edge.classList.toggle("is-muted", Boolean(id) && !isAnchorFocus && !primaryHit && !secondaryHit);
  });
  elements.graph.querySelector(".network-detail-svg")?.classList.toggle("has-pinned-focus", pinned);
}

function renderStockCard(member) {
  if (!elements.tooltip) return;
  const meta = member.meta || {};
  const stockHref = `./stock.html?id=${encodeURIComponent(member.id)}`;
  const graphHref = `./network.html?anchor=${encodeURIComponent(member.id)}`;
  const canBuildGraph = centerIds.has(String(member.id));
  const role = member.isAnchor ? "Current center" : "Connected stock";
  elements.tooltip.innerHTML = `
    <span>${role}</span>
    <strong>${escapeHtml(meta.ticker || member.id)}</strong>
    <p class="network-company-name">${escapeHtml(meta.name || "Company name unavailable")}</p>
    <div class="network-info-grid">
      <div><span>Industry</span><b>${escapeHtml(meta.industry || "Unavailable")}</b></div>
      <div><span>Similarity</span><b>${member.isAnchor ? "Center" : formatScore(member.similarity)}</b></div>
      <div><span>VaR</span><b>${formatMaybePercent(member.v)}</b></div>
      <div><span>ES</span><b>${formatMaybePercent(member.e)}</b></div>
    </div>
    <div class="network-action-row">
      <a class="button-link" href="${stockHref}">Open stock detail</a>
      ${
        canBuildGraph
          ? `<a class="button-link secondary-link" href="${graphHref}">View Network</a>`
          : '<button class="button-link secondary-link is-disabled" type="button" disabled>View Network</button>'
      }
    </div>
  `;
}

function renderMembers(group) {
  if (!elements.members) return;
  const members = group.members.slice(0, 30);
  elements.members.innerHTML = members.map((member, index) => {
    const meta = member.meta || {};
    return `
      <a class="network-member-row" href="./stock.html?id=${encodeURIComponent(member.id)}">
        <b>${index + 1}</b>
        <div>
          <strong>${escapeHtml(meta.ticker || member.id)}</strong>
          <small>${escapeHtml(meta.name || "Company name unavailable")}</small>
        </div>
        <div><span>Similarity</span><b>${formatScore(member.similarity)}</b></div>
        <div><span>VaR</span><b>${formatPercent(member.v)}%</b></div>
        <div><span>ES</span><b>${formatPercent(member.e)}%</b></div>
      </a>
    `;
  }).join("");
}

function showError(message) {
  if (elements.title) elements.title.textContent = message;
  if (elements.graph) elements.graph.innerHTML = `<p class="chart-empty">${message}</p>`;
  if (elements.members) elements.members.innerHTML = `<p class="chart-empty">${message}</p>`;
}

loadNetwork();
