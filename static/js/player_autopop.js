// ============================================================================
// ULTRA-DEBUG VERSION - Find the exact issue
// ============================================================================

const TEAM_STATE = {
  team1: { name: null, players: [], loaded: false, loading: false },
  team2: { name: null, players: [], loaded: false, loading: false },
};

const DEFAULT_TEAM_1 = "MI";
const DEFAULT_TEAM_2 = "CSK";
const PLAYERS_PER_TEAM = 11;

console.log("🚀 ULTRA-DEBUG VERSION LOADED");

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showNotification(message, type = "info") {
  const colors = {
    success: "bg-green-500",
    error: "bg-red-500",
    info: "bg-blue-500",
  };
  const notification = document.createElement("div");
  notification.className = `fixed top-4 right-4 ${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg z-50 transition-opacity duration-300`;
  notification.textContent = message;
  document.body.appendChild(notification);
  setTimeout(() => {
    notification.style.opacity = "0";
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

function getCategoryBadge(category) {
  const badges = {
    BATTER:
      '<span class="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-[10px] font-medium">BAT</span>',
    BOWLER:
      '<span class="px-2 py-0.5 bg-red-500/20 text-red-400 rounded text-[10px] font-medium">BOWL</span>',
    "ALL-ROUNDER":
      '<span class="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-[10px] font-medium">AR</span>',
    "WK-BATTER":
      '<span class="px-2 py-0.5 bg-green-500/20 text-green-400 rounded text-[10px] font-medium">WK</span>',
  };
  return badges[category] || badges["ALL-ROUNDER"];
}

function logDOMState(context) {
  console.log(`\n🔍 DOM STATE CHECK (${context}):`);
  const team1List = document.getElementById("team1-list");
  const team2List = document.getElementById("team2-list");
  console.log(
    `  Team 1 list children: ${team1List ? team1List.children.length : "NULL"}`
  );
  console.log(
    `  Team 2 list children: ${team2List ? team2List.children.length : "NULL"}`
  );
  if (team1List && team1List.children.length > 0) {
    console.log(
      `  Team 1 first player: ${team1List.children[0].dataset.playerName}`
    );
  }
  if (team2List && team2List.children.length > 0) {
    console.log(
      `  Team 2 first player: ${team2List.children[0].dataset.playerName}`
    );
  }
}

// ============================================================================
// TEAM 1 HANDLER
// ============================================================================

function handleTeam1Change() {
  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("🔵 TEAM 1 CHANGE EVENT FIRED");
  logDOMState("BEFORE Team 1 change");

  const team1Select = document.getElementById("team1-select");
  if (!team1Select) {
    console.error("❌ team1-select not found!");
    return;
  }

  const selectedTeam = team1Select.value;
  console.log(`  Selected: ${selectedTeam}`);
  console.log(
    `  Current state: ${TEAM_STATE.team1.name}, loaded: ${TEAM_STATE.team1.loaded}`
  );

  if (TEAM_STATE.team1.name === selectedTeam && TEAM_STATE.team1.loaded) {
    console.log("✅ Team 1 already loaded, skipping");
    return;
  }

  if (TEAM_STATE.team1.loading) {
    console.log("⏳ Team 1 already loading, skipping");
    return;
  }

  console.log("➡️ Loading Team 1...");
  loadPlayersForTeam1(selectedTeam);
}

// ============================================================================
// TEAM 2 HANDLER
// ============================================================================

function handleTeam2Change() {
  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("🔴 TEAM 2 CHANGE EVENT FIRED");
  logDOMState("BEFORE Team 2 change");

  const team2Select = document.getElementById("team2-select");
  if (!team2Select) {
    console.error("❌ team2-select not found!");
    return;
  }

  const selectedTeam = team2Select.value;
  console.log(`  Selected: ${selectedTeam}`);
  console.log(
    `  Current state: ${TEAM_STATE.team2.name}, loaded: ${TEAM_STATE.team2.loaded}`
  );

  if (TEAM_STATE.team2.name === selectedTeam && TEAM_STATE.team2.loaded) {
    console.log("✅ Team 2 already loaded, skipping");
    return;
  }

  if (TEAM_STATE.team2.loading) {
    console.log("⏳ Team 2 already loading, skipping");
    return;
  }

  console.log("➡️ Loading Team 2...");
  loadPlayersForTeam2(selectedTeam);
}

// ============================================================================
// LOAD TEAM 1
// ============================================================================

async function loadPlayersForTeam1(team) {
  console.log(`\n📥 [TEAM 1] Starting load for: ${team}`);

  const listElement = document.getElementById("team1-list");
  if (!listElement) {
    console.error("❌ [TEAM 1] team1-list not found!");
    return;
  }

  console.log(
    `  [TEAM 1] List element found, current children: ${listElement.children.length}`
  );
  TEAM_STATE.team1.loading = true;

  try {
    const response = await fetch(`/meta/players/${team}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (!data.success || !data.players) throw new Error("Invalid response");

    console.log(`✅ [TEAM 1] Fetched ${data.players.length} players`);

    // Update state
    TEAM_STATE.team1.name = team;
    TEAM_STATE.team1.players = data.players.slice(0, PLAYERS_PER_TEAM);
    TEAM_STATE.team1.loaded = true;
    TEAM_STATE.team1.loading = false;

    console.log(
      `  [TEAM 1] Clearing list (current: ${listElement.children.length})`
    );
    listElement.innerHTML = "";
    console.log(
      `  [TEAM 1] List cleared (now: ${listElement.children.length})`
    );

    const topPlayers = data.players.slice(0, PLAYERS_PER_TEAM);
    console.log(`  [TEAM 1] Adding ${topPlayers.length} players to DOM...`);

    topPlayers.forEach((player, index) => {
      addPlayerToList(player, listElement, 1);
      if (index === 0) console.log(`    First player added: ${player.name}`);
    });

    console.log(
      `✅ [TEAM 1] DOM populated: ${listElement.children.length} players`
    );
    logDOMState("AFTER Team 1 load");
    updatePlayerCounter(1);
  } catch (error) {
    console.error(`❌ [TEAM 1] Load failed:`, error);
    TEAM_STATE.team1.loading = false;
    showNotification(`Failed to load ${team}: ${error.message}`, "error");
  }
}

// ============================================================================
// LOAD TEAM 2
// ============================================================================

async function loadPlayersForTeam2(team) {
  console.log(`\n📥 [TEAM 2] Starting load for: ${team}`);

  const listElement = document.getElementById("team2-list");
  if (!listElement) {
    console.error("❌ [TEAM 2] team2-list not found!");
    return;
  }

  console.log(
    `  [TEAM 2] List element found, current children: ${listElement.children.length}`
  );
  TEAM_STATE.team2.loading = true;

  try {
    const response = await fetch(`/meta/players/${team}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (!data.success || !data.players) throw new Error("Invalid response");

    console.log(`✅ [TEAM 2] Fetched ${data.players.length} players`);

    // Update state
    TEAM_STATE.team2.name = team;
    TEAM_STATE.team2.players = data.players.slice(0, PLAYERS_PER_TEAM);
    TEAM_STATE.team2.loaded = true;
    TEAM_STATE.team2.loading = false;

    console.log(
      `  [TEAM 2] Clearing list (current: ${listElement.children.length})`
    );
    listElement.innerHTML = "";
    console.log(
      `  [TEAM 2] List cleared (now: ${listElement.children.length})`
    );

    const topPlayers = data.players.slice(0, PLAYERS_PER_TEAM);
    console.log(`  [TEAM 2] Adding ${topPlayers.length} players to DOM...`);

    topPlayers.forEach((player, index) => {
      addPlayerToList(player, listElement, 2);
      if (index === 0) console.log(`    First player added: ${player.name}`);
    });

    console.log(
      `✅ [TEAM 2] DOM populated: ${listElement.children.length} players`
    );
    logDOMState("AFTER Team 2 load");
    updatePlayerCounter(2);
  } catch (error) {
    console.error(`❌ [TEAM 2] Load failed:`, error);
    TEAM_STATE.team2.loading = false;
    showNotification(`Failed to load ${team}: ${error.message}`, "error");
  }
}

// ============================================================================
// ADD PLAYER TO LIST
// ============================================================================

function addPlayerToList(player, listElement, teamNumber) {
  const playerCard = document.createElement("div");
  playerCard.className =
    "group relative bg-slate-900/50 border border-slate-800 rounded-lg p-3 hover:border-blue-500/50 transition-all";
  playerCard.dataset.playerName = player.name;
  playerCard.dataset.teamNumber = teamNumber;

  const batOvr = player.bat_ovr
    ? `<span class="text-blue-400">${player.bat_ovr}</span>`
    : '<span class="text-slate-600">-</span>';
  const bowlOvr = player.bowl_ovr
    ? `<span class="text-red-400">${player.bowl_ovr}</span>`
    : '<span class="text-slate-600">-</span>';

  playerCard.innerHTML = `
        <div class="flex items-start justify-between gap-3">
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 mb-1">
                    ${getCategoryBadge(player.category)}
                    <span class="text-[11px] font-medium text-slate-200 truncate">${
                      player.name
                    }</span>
                </div>
                <div class="flex items-center gap-3 text-[10px]">
                    <div class="flex items-center gap-1">
                        <span class="text-slate-500">BAT</span>
                        ${batOvr}
                    </div>
                    <div class="flex items-center gap-1">
                        <span class="text-slate-500">BOWL</span>
                        ${bowlOvr}
                    </div>
                    <div class="flex items-center gap-1">
                        <span class="text-slate-500">OVR</span>
                        <span class="text-emerald-400 font-semibold">${
                          player.overall_ovr
                        }</span>
                    </div>
                </div>
            </div>
            <button onclick="removePlayer(this)" 
                    class="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-400 p-1">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                </svg>
            </button>
        </div>
    `;

  listElement.appendChild(playerCard);
}

// ============================================================================
// REMOVE PLAYER
// ============================================================================

function removePlayer(button) {
  const playerCard = button.closest("[data-player-name]");
  const playerName = playerCard.dataset.playerName;
  const teamNumber = parseInt(playerCard.dataset.teamNumber);

  console.log(`🗑️ Removing ${playerName} from Team ${teamNumber}`);
  logDOMState("BEFORE remove");

  playerCard.remove();

  const stateKey = `team${teamNumber}`;
  TEAM_STATE[stateKey].players = TEAM_STATE[stateKey].players.filter(
    (p) => p.name !== playerName
  );

  logDOMState("AFTER remove");
  updatePlayerCounter(teamNumber);
}

// ============================================================================
// UPDATE COUNTER
// ============================================================================

function updatePlayerCounter(teamNumber) {
  const listId = `team${teamNumber}-list`;
  const counterId = `team${teamNumber}-count`;

  const listElement = document.getElementById(listId);
  const counterElement = document.getElementById(counterId);

  if (listElement && counterElement) {
    const count = listElement.children.length;
    counterElement.textContent = count;

    if (count === 11) {
      counterElement.className = "text-green-400 font-bold";
    } else if (count > 0) {
      counterElement.className = "text-yellow-400 font-bold";
    } else {
      counterElement.className = "text-red-400 font-bold";
    }
  }
}

// ============================================================================
// RUN PREDICTION
// ============================================================================

async function runPredictionEnhanced() {
  console.log("\n🎯 Running prediction...");
  logDOMState("BEFORE prediction");

  const team1Select = document.getElementById("team1-select");
  const team2Select = document.getElementById("team2-select");

  if (!team1Select || !team2Select) {
    showNotification("Team selection not found", "error");
    return;
  }

  const team1 = team1Select.value;
  const team2 = team2Select.value;

  const team1List = document.getElementById("team1-list");
  const team2List = document.getElementById("team2-list");

  const team1Players = Array.from(team1List.children).map(
    (card) => card.dataset.playerName
  );
  const team2Players = Array.from(team2List.children).map(
    (card) => card.dataset.playerName
  );

  console.log(`📊 Team 1 (${team1}): ${team1Players.length} players`);
  console.log(`📊 Team 2 (${team2}): ${team2Players.length} players`);

  if (team1Players.length === 0 || team2Players.length === 0) {
    showNotification("Both teams need players!", "error");
    return;
  }

  const predictBtn = document.querySelector('[onclick*="runPrediction"]');
  if (predictBtn) {
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team1: team1,
        team2: team2,
        team1_players: team1Players,
        team2_players: team2Players,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}`);
    }

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.error || "Prediction failed");
    }

    console.log("✅ Prediction successful:", result);
    updateResultScreenEnhanced(result);
    showStep(3);
    showNotification("Prediction complete!", "success");
  } catch (error) {
    console.error("❌ Prediction error:", error);
    showNotification(`Prediction failed: ${error.message}`, "error");
  } finally {
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.textContent = "Run Prediction";
    }
  }
}

// ============================================================================
// UPDATE RESULT SCREEN
// ============================================================================

function updateResultScreenEnhanced(result) {
  const team1Name = document.getElementById("result-team1-name");
  const team2Name = document.getElementById("result-team2-name");
  if (team1Name) team1Name.textContent = result.team1;
  if (team2Name) team2Name.textContent = result.team2;

  const team1Prob = document.getElementById("result-team1-prob");
  const team2Prob = document.getElementById("result-team2-prob");
  if (team1Prob) team1Prob.textContent = `${result.team1_prob}%`;
  if (team2Prob) team2Prob.textContent = `${result.team2_prob}%`;

  const winnerBadge = document.getElementById("winner-badge");
  if (winnerBadge) {
    winnerBadge.textContent = `${result.winner} wins`;
    winnerBadge.className =
      "px-4 py-2 bg-green-500/20 text-green-400 rounded-lg font-semibold";
  }

  const confidenceLevel = document.getElementById("confidence-level");
  if (confidenceLevel) {
    confidenceLevel.textContent = result.confidence.level;
    confidenceLevel.style.color = result.confidence.color;
  }

  if (result.source_weights) {
    const ovrWeight = document.getElementById("weight-ovr");
    const h2hWeight = document.getElementById("weight-h2h");
    const formWeight = document.getElementById("weight-form");
    const pvpWeight = document.getElementById("weight-pvp");

    if (ovrWeight) ovrWeight.textContent = `${result.source_weights.ovr}%`;
    if (h2hWeight) h2hWeight.textContent = `${result.source_weights.h2h}%`;
    if (formWeight) formWeight.textContent = `${result.source_weights.form}%`;
    if (pvpWeight) pvpWeight.textContent = `${result.source_weights.pvp}%`;
  }

  console.log("✅ Results UI updated");
}

// ============================================================================
// INITIALIZE
// ============================================================================

document.addEventListener("DOMContentLoaded", function () {
  console.log("\n🎬 ULTRA-DEBUG: Initializing...");
  logDOMState("Initial state");

  const team1Select = document.getElementById("team1-select");
  const team2Select = document.getElementById("team2-select");

  if (!team1Select || !team2Select) {
    console.error("❌ Team select elements not found");
    return;
  }

  // Set defaults
  if (!team1Select.value) team1Select.value = DEFAULT_TEAM_1;
  if (!team2Select.value) team2Select.value = DEFAULT_TEAM_2;

  console.log(`  Default Team 1: ${team1Select.value}`);
  console.log(`  Default Team 2: ${team2Select.value}`);

  // Load with delay
  setTimeout(() => {
    console.log(`🚀 Loading Team 1: ${team1Select.value}`);
    loadPlayersForTeam1(team1Select.value);
  }, 100);

  setTimeout(() => {
    console.log(`🚀 Loading Team 2: ${team2Select.value}`);
    loadPlayersForTeam2(team2Select.value);
  }, 300);

  // Attach listeners
  team1Select.addEventListener("change", handleTeam1Change);
  team2Select.addEventListener("change", handleTeam2Change);

  console.log("✅ Event listeners attached");
  console.log("✅ ULTRA-DEBUG initialization complete");
});

// ============================================================================
// EXPOSE GLOBALLY
// ============================================================================

window.removePlayer = removePlayer;
window.runPredictionEnhanced = runPredictionEnhanced;
window.handleTeam1Change = handleTeam1Change;
window.handleTeam2Change = handleTeam2Change;
window.logDOMState = logDOMState;
window.TEAM_STATE = TEAM_STATE;

console.log(
  "✅ ULTRA-DEBUG VERSION LOADED - Type logDOMState('manual check') to inspect"
);
