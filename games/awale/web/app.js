// Awale in the browser: the server (core/server.py) holds the game and searches AI moves.
// API paths are relative: the server serves page and API under a secret token prefix.
// AI moves are asked for one at a time, so the page can show the thinking time and
// pace AI vs AI games with the minimum delay.

const SVG_NS = "http://www.w3.org/2000/svg";
const BOARD_SIZE = 12;
const HALF = 6;
const ROW_Y = [225, 105];  // Player 0's holes along the bottom, player 1's along the top
const HOLE_RADIUS = 42;
const TOTAL_SEEDS = 48;
const MAX_PLIES = 100;  // The game stops after this many moves and is scored on captured seeds

const $ = (selector) => document.querySelector(selector);

let configs = null;       // The two player settings of the current match
let game = null;          // Latest state from the server
let minDelay = 0.5;
let paused = false;
let searching = false;    // A move being sent or searched and not shown yet
let generation = 0;       // Bumped by each new game, so late answers for an old one are dropped
let lastShown = 0;        // When the last move was shown, for the minimum delay
let report = "";          // What the last move was, shown before the status
let resultClosed = false; // The result card was closed to look at the final board

// ---------- Server ----------

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    let detail = data.detail ?? response.statusText;
    if (Array.isArray(detail)) detail = detail.map((error) => `${error.loc.slice(-1)[0]}: ${error.msg}`).join("; ");
    throw new Error(detail);
  }
  return data;
}

// ---------- Setup ----------

function setupCards(options) {
  const template = $("#player-fields");
  for (const card of document.querySelectorAll(".player-card")) {
    card.append(template.content.cloneNode(true));
    const kind = card.querySelector('[name="kind"]');
    kind.value = card.dataset.side === "0" ? "human" : "ai";
    kind.addEventListener("change", () => showFields(card));
    const difficulty = card.querySelector('[name="difficulty"]');
    difficulty.min = options.min_difficulty;
    difficulty.max = options.max_difficulty;
    const hint = card.querySelector('[data-role="difficulty-hint"]');
    const showDifficulty = () => {
      const label = options.labels[difficulty.value - options.min_difficulty];
      hint.textContent = `${difficulty.value} / ${options.max_difficulty} · ${label}`;
    };
    difficulty.addEventListener("input", showDifficulty);
    showDifficulty();
    showFields(card);
  }
  if (!options.weights) {
    for (const option of document.querySelectorAll('option[value="ai"]')) {
      option.disabled = true;
      option.textContent += " (no weights in models/)";
    }
    for (const kind of document.querySelectorAll('[name="kind"]')) kind.value = "human";
    for (const card of document.querySelectorAll(".player-card")) showFields(card);
  }
}

function showFields(card) {
  const kind = card.querySelector('[name="kind"]').value;
  for (const label of card.querySelectorAll("[data-for]")) {
    label.hidden = !label.dataset.for.split(" ").includes(kind);
  }
}

function readConfig(card) {
  const field = (name) => card.querySelector(`[name="${name}"]`);
  return { kind: field("kind").value, difficulty: Number(field("difficulty").value) };
}

async function startMatch() {
  configs = [...document.querySelectorAll(".player-card")].map(readConfig);
  minDelay = Math.max(0, Number($("#min-delay").value) || 0);
  $("#setup-error").textContent = "";
  try {
    await newGame();
  } catch (error) {
    $("#setup-error").textContent = error.message;
    return;
  }
  $("#setup").hidden = true;
  $("#game").hidden = false;
}

async function newGame() {
  const data = await api("api/games", { players: configs });
  generation += 1;
  game = data;
  searching = false;
  report = "";
  resultClosed = false;
  hideResult();
  lastShown = performance.now();
  render();
  advance();
}

// ---------- Turns ----------

const sideName = (side) => `P${side} (${game.players[side].name})`;
const isHumanTurn = () => !game.over && game.players[game.current_player].human;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setStatus(text) {
  $("#status").textContent = report + text;
}

function advance() {
  render();
  if (game.over) {
    const [s0, s1] = game.seeds;
    const result = game.winner === null
      ? `Draw, ${s0}-${s1}.`
      : `${sideName(game.winner)} wins ${game.seeds[game.winner]}-${game.seeds[1 - game.winner]}.`;
    setStatus(`Game over: ${result}`);
    showResult();
  } else if (isHumanTurn()) {
    setStatus(`${sideName(game.current_player)} to play: click a highlighted hole.`);
  } else if (paused) {
    setStatus("Paused: press Space or Resume.");
  } else {
    aiMove();
  }
}

async function aiMove() {
  const myGeneration = generation;
  const side = game.current_player;
  const started = performance.now();
  searching = true;
  const tick = () => setStatus(`${sideName(side)} thinking... ${((performance.now() - started) / 1000).toFixed(1)} s`);
  tick();
  const timer = setInterval(tick, 100);
  let data;
  try {
    data = await api(`api/games/${game.id}/ai-move`, { ply: game.ply });
  } catch (error) {
    clearInterval(timer);
    if (myGeneration === generation) {
      searching = false;
      setStatus(`${sideName(side)} failed: ${error.message}`);
    }
    return;
  }
  clearInterval(timer);
  // Fast moves wait so that each one stays on screen at least minDelay
  await sleep(minDelay * 1000 - (performance.now() - lastShown));
  if (myGeneration !== generation) return;  // A new game started meanwhile
  searching = false;
  showMove(data, `${sideName(side)} played ${data.last_move} in ${data.move_seconds.toFixed(2)} s. `);
}

async function humanMove(move) {
  if (!isHumanTurn() || searching) return;
  const myGeneration = generation;
  const side = game.current_player;
  searching = true;  // Ignores further clicks until the server answers
  drawBoard();
  try {
    const data = await api(`api/games/${game.id}/move`, { ply: game.ply, move });
    if (myGeneration !== generation) return;
    searching = false;
    showMove(data, `${sideName(side)} played ${move}. `);
  } catch (error) {
    if (myGeneration !== generation) return;
    searching = false;
    drawBoard();
    setStatus(`Move refused: ${error.message}`);
  }
}

function showMove(data, moveReport) {
  game = data;
  report = moveReport;
  lastShown = performance.now();
  advance();
}

function togglePause() {
  paused = !paused;
  $("#pause").textContent = paused ? "Resume" : "Pause";
  // A running search goes on: its move is shown, then the pause applies
  if (!searching) advance();
}

// ---------- Result ----------

// Why the game stopped, as a sentence
function endReason() {
  const majority = game.seeds.findIndex((seeds) => seeds > TOTAL_SEEDS / 2);
  if (majority >= 0) return `P${majority} captured more than half the seeds.`;
  if (game.ply >= MAX_PLIES) return `The ${MAX_PLIES}-move limit was reached: the most seeds captured wins.`;
  return "A side ran out of seeds.";
}

function resultTexts() {
  const winner = game.winner;
  const humans = [0, 1].filter((side) => game.players[side].human);
  const [s0, s1] = game.seeds;
  if (winner === null) return { title: "Draw", score: `${s0} – ${s1}`, detail: endReason() };
  const score = `${game.seeds[winner]} – ${game.seeds[1 - winner]}`;
  if (humans.length === 1) {
    const won = winner === humans[0];
    return {
      title: won ? "You win!" : "You lose",
      score,
      detail: (won ? "" : `${game.players[winner].name} wins. `) + endReason(),
    };
  }
  return { title: `Player ${winner} wins`, score, detail: `${game.players[winner].name}. ${endReason()}` };
}

function showResult() {
  const panel = $("#result");
  const wasHidden = panel.hidden;
  const { title, score, detail } = resultTexts();
  const humans = [0, 1].filter((side) => game.players[side].human);
  panel.classList.toggle("draw", game.winner === null);
  panel.classList.toggle("lost", humans.length === 1 && game.winner !== null && game.winner !== humans[0]);
  $("#result-title").textContent = title;
  $("#result-score").textContent = score;
  $("#result-detail").textContent = detail;
  panel.hidden = resultClosed;
  if (wasHidden && !panel.hidden) $("#result-again").focus({ preventScroll: true });
}

function hideResult() {
  $("#result").hidden = true;
}

// ---------- Drawing ----------

function svg(tag, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (parent) parent.append(element);
  return element;
}

function holeCenter(index) {
  // Sowing goes counter-clockwise: player 0's holes (0-5) left to right along the bottom,
  // player 1's (6-11) right to left along the top
  const column = index < HALF ? index : BOARD_SIZE - 1 - index;
  return [180 + 100 * column, ROW_Y[index < HALF ? 0 : 1]];
}

function drawBoard() {
  const board = $("#board");
  board.replaceChildren();
  svg("rect", { x: 4, y: 4, width: 852, height: 322, rx: 60, class: "board-wood" }, board);

  // Each player's store is on their right: player 0's at the right end, player 1's at the left
  for (const [side, x] of [[1, 60], [0, 800]]) {
    const storeClass = game.over && game.winner === side ? "store winner" : "store";
    svg("rect", { x: x - 38, y: 45, width: 76, height: 240, rx: 38, class: storeClass }, board);
    svg("text", { x, y: 150, class: "store-label" }, board).textContent = `P${side}`;
    svg("text", { x, y: 190, class: "store-count" }, board).textContent = game.seeds[side];
  }

  const playable = new Set(isHumanTurn() && !searching ? game.legal_moves : []);
  for (let index = 0; index < BOARD_SIZE; index++) {
    const [x, y] = holeCenter(index);
    const classes = ["hole"];
    if (playable.has(index)) classes.push("playable");
    if (index === game.last_move) classes.push("last");
    const group = svg("g", { class: classes.join(" ") }, board);
    svg("circle", { cx: x, cy: y, r: HOLE_RADIUS }, group);
    svg("text", { x, y, class: "seeds" }, group).textContent = game.board[index];
    const labelY = index < HALF ? y + HOLE_RADIUS + 18 : y - HOLE_RADIUS - 18;
    svg("text", { x, y: labelY, class: "index" }, board).textContent = index;
    if (playable.has(index)) {
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
      group.setAttribute("aria-label", `Play hole ${index}, ${game.board[index]} seeds`);
      group.addEventListener("click", () => humanMove(index));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.stopPropagation();
          humanMove(index);
        }
      });
    }
  }
}

function drawBars() {
  for (const side of [0, 1]) {
    const player = game.players[side];
    const bar = $(`#bar-${side}`);
    bar.classList.toggle("to-move", !game.over && game.current_player === side);
    bar.classList.toggle("winner", game.over && game.winner === side);
    bar.classList.toggle("loser", game.over && game.winner === 1 - side);
    bar.classList.toggle("draw", game.over && game.winner === null);
    const stats = [`${game.seeds[side]} captured`];
    bar.replaceChildren();
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `P${side}: ${player.name}`;
    const details = document.createElement("span");
    details.className = "stats";
    details.textContent = stats.join(" · ");
    bar.append(name, details);
    if (game.over && (game.winner === side || game.winner === null)) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = game.winner === null ? "Draw" : "Winner";
      bar.prepend(badge);
    }
  }
}

function render() {
  drawBoard();
  drawBars();
}

// ---------- Wiring ----------

$("#start").addEventListener("click", startMatch);
$("#new-game").addEventListener("click", () => newGame().catch((error) => setStatus(`New game failed: ${error.message}`)));
$("#pause").addEventListener("click", togglePause);
$("#change-players").addEventListener("click", () => {
  generation += 1;  // Drops any move still being searched
  searching = false;
  hideResult();
  $("#game").hidden = true;
  $("#setup").hidden = false;
});
$("#result-again").addEventListener("click", () => $("#new-game").click());
$("#result-players").addEventListener("click", () => $("#change-players").click());
$("#result-close").addEventListener("click", () => {
  resultClosed = true;
  hideResult();
  $("#new-game").focus({ preventScroll: true });
});

document.addEventListener("keydown", (event) => {
  if ($("#game").hidden) return;
  if (event.key === "Escape" && !$("#result").hidden) {
    $("#result-close").click();
    return;
  }
  if (event.target.closest("input, select, button")) return;
  if (event.key === "n" || event.key === "N") $("#new-game").click();
  if (event.key === " ") {
    event.preventDefault();
    togglePause();
  }
});

api("api/options")
  .then(setupCards)
  .catch((error) => { $("#setup-error").textContent = `Cannot reach the server: ${error.message}`; });
