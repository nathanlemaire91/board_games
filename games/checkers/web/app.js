// Checkers in the browser: the server (core/server.py --game checkers) holds the game and searches AI moves.
// API paths are relative: the server serves page and API under a secret token prefix.
// AI moves are asked for one at a time, so the page can show the thinking time and
// pace AI vs AI games with the minimum delay. Each jump of a multiple jump is a move
// of its own, the same player moving again.
//
// Squares are numbered 0-31 row by row from player 0's side (games/checkers/rules.py),
// and shown with chess-style coordinates, a1 being player 0's left corner.

const SVG_NS = "http://www.w3.org/2000/svg";
const ROWS = 8;
const CELL = 100;   // Board units per square
const MARGIN = 28;  // Around the squares, for the coordinates
const EMPTY = 0;
const UP_RIGHT = 1, DOWN_RIGHT = 3;  // Directions: up-left 0, up-right 1, down-left 2, down-right 3

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
let selected = null;      // The square of the piece a human picked, to move it
let flipped = false;      // Player 1's side at the bottom

// ---------- Board geometry, as in rules.py ----------

const pieceOwner = (piece) => (piece - 1) >> 1;
const isKing = (piece) => piece !== EMPTY && piece % 2 === 0;
const rowOf = (square) => square >> 2;
const columnOf = (square) => 2 * (square & 3) + (rowOf(square) & 1);  // 0-7 from player 0's left

function step(square, direction) {
  const row = rowOf(square) + (direction <= UP_RIGHT ? 1 : -1);
  const column = columnOf(square) + (direction === UP_RIGHT || direction === DOWN_RIGHT ? 1 : -1);
  return row >= 0 && row < ROWS && column >= 0 && column < ROWS ? row * 4 + (column >> 1) : null;
}

const squareName = (square) => "abcdefgh"[columnOf(square)] + (rowOf(square) + 1);

// The squares of the move that led to the current board: a step lands next to its square,
// a jump leaves that square empty and lands one further
function lastMoveSquares() {
  if (game.last_move === null) return null;
  const from = game.last_move >> 2, direction = game.last_move & 3;
  const next = step(from, direction);
  if (game.board[next] !== EMPTY) return { from, to: next, captured: null };
  return { from, to: step(next, direction), captured: next };
}

function describeMove({ from, to, captured }) {
  return `${squareName(from)}${captured === null ? "-" : "x"}${squareName(to)}`;
}

function pieceCounts(side) {
  const own = game.board.filter((piece) => piece !== EMPTY && pieceOwner(piece) === side);
  return { pieces: own.length, kings: own.filter(isKing).length };
}

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
  // A lone human playing second sees the board from their side
  flipped = configs[1].kind === "human" && configs[0].kind !== "human";
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
  advance();
}

// ---------- Turns ----------

const sideName = (side) => `P${side} (${game.players[side].name})`;
const isHumanTurn = () => !game.over && game.players[game.current_player].human;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setStatus(text) {
  $("#status").textContent = report + text;
}

// The squares of the pieces the human can move, and the selection: the jumping piece in the
// middle of a multiple jump, or the only piece that can move
function movableSquares() {
  if (!isHumanTurn() || searching) return new Set();
  return new Set(game.moves.map((move) => move.from));
}

function autoSelect() {
  const movable = movableSquares();
  if (game.jumping !== null) selected = game.jumping;
  else if (movable.size === 1) selected = [...movable][0];
  else if (!movable.has(selected)) selected = null;
}

function advance() {
  autoSelect();
  render();
  if (game.over) {
    setStatus(`Game over: ${resultTexts().summary}`);
    showResult();
  } else if (isHumanTurn()) {
    const side = sideName(game.current_player);
    if (game.jumping !== null) {
      setStatus(`${side}: keep jumping with the same piece.`);
    } else {
      const capture = game.moves[0].captured !== null ? " Capturing is mandatory." : "";
      setStatus(`${side} to play: pick a highlighted piece, then where it goes.${capture}`);
    }
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
  showMove(data, (move) => `${sideName(side)} played ${move} in ${data.move_seconds.toFixed(2)} s. `);
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
    showMove(data, (description) => `${sideName(side)} played ${description}. `);
  } catch (error) {
    if (myGeneration !== generation) return;
    searching = false;
    drawBoard();
    setStatus(`Move refused: ${error.message}`);
  }
}

// `moveReport` makes the report from the move's description, read from the new board
function showMove(data, moveReport) {
  game = data;
  report = moveReport(describeMove(lastMoveSquares()));
  lastShown = performance.now();
  advance();
}

function togglePause() {
  paused = !paused;
  $("#pause").textContent = paused ? "Resume" : "Pause";
  // A running search goes on: its move is shown, then the pause applies
  if (!searching) advance();
}

function clickSquare(square) {
  if (!isHumanTurn() || searching) return;
  const move = game.moves.find((m) => m.from === selected && m.to === square);
  if (move) {
    humanMove(move.move);
  } else if (movableSquares().has(square) && game.jumping === null) {
    selected = selected === square ? null : square;
    drawBoard();
  }
}

// ---------- Result ----------

function resultTexts() {
  const winner = game.winner;
  const humans = [0, 1].filter((side) => game.players[side].human);
  const counts = [pieceCounts(0).pieces, pieceCounts(1).pieces];
  const score = `${counts[0]} – ${counts[1]}`;
  if (winner === null) {
    const reason = game.quiet_moves >= game.quiet_limit
      ? `${game.quiet_limit / 2} moves each without a capture or a man move.`
      : `The ${game.max_plies}-move limit was reached.`;
    return { title: "Draw", score, detail: reason, summary: `draw. ${reason}` };
  }
  const loser = 1 - winner;
  const reason = counts[loser] === 0 ? `P${loser} has no pieces left.` : `P${loser} cannot move.`;
  const summary = `${sideName(winner)} wins. ${reason}`;
  if (humans.length === 1) {
    const won = winner === humans[0];
    return { title: won ? "You win!" : "You lose", score, detail: (won ? "" : `${game.players[winner].name} wins. `) + reason, summary };
  }
  return { title: `Player ${winner} wins`, score, detail: `${game.players[winner].name}. ${reason}`, summary };
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

// Top left corner of a cell on screen, player 0's side at the bottom unless flipped
function cellCorner(row, column) {
  const x = flipped ? ROWS - 1 - column : column;
  const y = flipped ? row : ROWS - 1 - row;
  return [MARGIN + x * CELL, MARGIN + y * CELL];
}

function squareCenter(square) {
  const [x, y] = cellCorner(rowOf(square), columnOf(square));
  return [x + CELL / 2, y + CELL / 2];
}

// Makes a board element play like a button, by click or keyboard
function makeButton(element, label, action) {
  element.setAttribute("tabindex", "0");
  element.setAttribute("role", "button");
  element.setAttribute("aria-label", label);
  element.addEventListener("click", action);
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      action();
    }
  });
}

function drawBoard() {
  const board = $("#board");
  board.replaceChildren();
  const size = ROWS * CELL + 2 * MARGIN;
  svg("rect", { x: 0, y: 0, width: size, height: size, rx: 14, class: "board-edge" }, board);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < ROWS; column++) {
      const [x, y] = cellCorner(row, column);
      svg("rect", { x, y, width: CELL, height: CELL, class: (row + column) % 2 ? "cell light" : "cell dark" }, board);
    }
  }
  for (let index = 0; index < ROWS; index++) {
    const [x] = cellCorner(0, index), [, y] = cellCorner(index, 0);
    svg("text", { x: x + CELL / 2, y: size - MARGIN / 2, class: "coordinate" }, board).textContent = "abcdefgh"[index];
    svg("text", { x: MARGIN / 2, y: y + CELL / 2, class: "coordinate" }, board).textContent = index + 1;
  }

  const last = lastMoveSquares();
  if (last) {
    for (const square of [last.from, last.to]) {
      const [x, y] = squareCenter(square);
      svg("rect", { x: x - CELL / 2, y: y - CELL / 2, width: CELL, height: CELL, class: "last" }, board);
    }
  }

  const movable = movableSquares();
  const targets = game.moves.filter((move) => move.from === selected && movable.has(selected));
  for (let square = 0; square < game.board.length; square++) {
    const piece = game.board[square];
    if (piece === EMPTY) continue;
    const [x, y] = squareCenter(square);
    const classes = ["piece", `p${pieceOwner(piece)}`];
    if (movable.has(square)) classes.push("movable");
    if (square === selected && movable.has(square)) classes.push("selected");
    if (square === game.jumping) classes.push("jumping");
    const group = svg("g", { class: classes.join(" ") }, board);
    svg("circle", { cx: x, cy: y, r: 38 }, group);
    svg("circle", { cx: x, cy: y, r: 27, class: "rim" }, group);
    if (isKing(piece)) {
      const crown = `M ${x - 17} ${y + 9} L ${x - 17} ${y - 7} L ${x - 8} ${y + 1} L ${x} ${y - 13} L ${x + 8} ${y + 1} L ${x + 17} ${y - 7} L ${x + 17} ${y + 9} Z`;
      svg("path", { d: crown, class: "crown" }, group);
    }
    if (movable.has(square) && game.jumping === null) {
      const kind = isKing(piece) ? "king" : "man";
      makeButton(group, `Pick the ${kind} on ${squareName(square)}`, () => clickSquare(square));
    }
  }
  if (last && last.captured !== null) {
    const [x, y] = squareCenter(last.captured);
    svg("circle", { cx: x, cy: y, r: 30, class: "captured" }, board);
  }
  for (const move of targets) {
    const [x, y] = squareCenter(move.to);
    const target = svg("g", { class: "target" }, board);
    svg("rect", { x: x - CELL / 2, y: y - CELL / 2, width: CELL, height: CELL }, target);
    svg("circle", { cx: x, cy: y, r: 16 }, target);
    makeButton(target, `Move to ${squareName(move.to)}`, () => clickSquare(move.to));
  }
}

function drawBars() {
  const bottomSide = flipped ? 1 : 0;
  for (const [id, side] of [["#bar-bottom", bottomSide], ["#bar-top", 1 - bottomSide]]) {
    const player = game.players[side];
    const bar = $(id);
    bar.classList.toggle("to-move", !game.over && game.current_player === side);
    bar.classList.toggle("winner", game.over && game.winner === side);
    bar.classList.toggle("loser", game.over && game.winner === 1 - side);
    bar.classList.toggle("draw", game.over && game.winner === null);
    const { pieces, kings } = pieceCounts(side);
    const stats = [`${pieces} piece${pieces === 1 ? "" : "s"}`, `${kings} king${kings === 1 ? "" : "s"}`];
    bar.replaceChildren();
    const chip = document.createElement("span");
    chip.className = `chip p${side}`;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `P${side}: ${player.name}`;
    const details = document.createElement("span");
    details.className = "stats";
    details.textContent = stats.join(" · ");
    bar.append(chip, name, details);
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
$("#flip").addEventListener("click", () => {
  flipped = !flipped;
  render();
});
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
  if (event.key === "f" || event.key === "F") $("#flip").click();
  if (event.key === " ") {
    event.preventDefault();
    togglePause();
  }
});

api("api/options")
  .then(setupCards)
  .catch((error) => { $("#setup-error").textContent = `Cannot reach the server: ${error.message}`; });
