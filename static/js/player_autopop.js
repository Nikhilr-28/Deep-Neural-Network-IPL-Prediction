/**
 * AUTO-POPULATE PLAYERS - INJECTION SCRIPT
 * This script adds auto-population when teams are selected in Step 1
 * Add this BEFORE the closing </script> tag in your existing HTML
 */

// =============================================================================
// AUTO-POPULATE PLAYERS ON TEAM SELECTION
// =============================================================================

let team1PlayersCache = [];
let team2PlayersCache = [];

// Override the existing team selection handler
document.addEventListener("DOMContentLoaded", function () {
  console.log("🚀 Auto-populate players enabled");

  // Hook into team1 selection
  const team1Select = document.getElementById("team1-select");
  if (team1Select) {
    team1Select.addEventListener("change", async function () {
      const team = this.value;
      if (!team) return;

      console.log(`Loading players for Team 1: ${team}`);
      await loadAndPopulatePlayers(team, "team1-list", 1);
    });
  }

  // Hook into team2 selection
  const team2Select = document.getElementById("team2-select");
  if (team2Select) {
    team2Select.addEventListener("change", async function () {
      const team = this.value;
      if (!team) return;

      console.log(`Loading players for Team 2: ${team}`);
      await loadAndPopulatePlayers(team, "team2-list", 2);
    });
  }
});

/**
 * Load players from backend and populate the team list
 */
async function loadAndPopulatePlayers(team, listId, teamNumber) {
  try {
    // Fetch players from backend (sorted by OVR)
    const response = await fetch(`/api/players/${team}`);
    const data = await response.json();

    if (!data.success) {
      console.error(`Failed to load players for ${team}:`, data.error);
      return;
    }

    const players = data.players;
    console.log(`✅ Loaded ${players.length} players for ${team}`);

    // Cache players
    if (teamNumber === 1) {
      team1PlayersCache = players;
    } else {
      team2PlayersCache = players;
    }

    // Clear existing list
    const listElement = document.getElementById(listId);
    if (!listElement) {
      console.error(`List element ${listId} not found`);
      return;
    }

    listElement.innerHTML = "";

    // Populate with ALL players (sorted by OVR descending)
    players.forEach((player, index) => {
      const div = document.createElement("div");
      div.className =
        "group flex items-center justify-between bg-slate-900 px-2 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700";

      // Build OVR display
      let ovrDisplay = "";
      if (player.bat_ovr && player.bowl_ovr) {
        ovrDisplay = `BAT: ${player.bat_ovr} | BOWL: ${player.bowl_ovr}`;
      } else if (player.bat_ovr) {
        ovrDisplay = `BAT: ${player.bat_ovr}`;
      } else if (player.bowl_ovr) {
        ovrDisplay = `BOWL: ${player.bowl_ovr}`;
      } else {
        ovrDisplay = `OVR: ${player.overall_ovr}`;
      }

      // Get category badge color
      let badgeClass = "bg-slate-700 text-slate-300";
      if (player.category === "BATTER") {
        badgeClass = "bg-blue-500/20 text-blue-300";
      } else if (player.category === "BOWLER") {
        badgeClass = "bg-red-500/20 text-red-300";
      } else if (player.category === "ALL-ROUNDER") {
        badgeClass = "bg-purple-500/20 text-purple-300";
      } else if (player.category === "WK-BATTER") {
        badgeClass = "bg-amber-500/20 text-amber-300";
      }

      div.innerHTML = `
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <span class="text-slate-200 text-[11px] font-medium">${player.name}</span>
                        <span class="text-[9px] px-1.5 py-0.5 rounded ${badgeClass}">${player.category}</span>
                    </div>
                    <div class="text-[10px] text-slate-400 mt-0.5">
                        ${ovrDisplay} · Overall: ${player.overall_ovr}
                    </div>
                </div>
                <div class="hidden group-hover:flex items-center gap-1">
                    <button
                        class="text-[9px] text-red-400 hover:text-red-300"
                        onclick="this.parentElement.parentElement.remove()"
                        title="Remove player"
                    >
                        ×
                    </button>
                </div>
            `;

      listElement.appendChild(div);
    });

    console.log(`✅ Populated ${players.length} players in ${listId}`);

    // Show notification
    showNotification(
      `Loaded ${players.length} players for ${team} (sorted by OVR)`,
      "success"
    );
  } catch (error) {
    console.error("Error loading players:", error);
    showNotification(`Failed to load players for ${team}`, "error");
  }
}

/**
 * Show notification toast
 */
function showNotification(message, type = "info") {
  // Create toast element
  const toast = document.createElement("div");
  toast.className = `fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg text-sm font-medium z-50 transition-all duration-300 transform translate-y-0 opacity-100`;

  // Set colors based on type
  if (type === "success") {
    toast.className += " bg-emerald-500 text-white";
  } else if (type === "error") {
    toast.className += " bg-red-500 text-white";
  } else {
    toast.className += " bg-slate-700 text-slate-100";
  }

  toast.textContent = message;
  document.body.appendChild(toast);

  // Auto-remove after 3 seconds
  setTimeout(() => {
    toast.classList.add("translate-y-2", "opacity-0");
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

/**
 * Update runPrediction to use new API endpoint
 */
async function runPredictionEnhanced() {
  const team1 = document.getElementById("team1-select").value;
  const team2 = document.getElementById("team2-select").value;

  // Get player names from lists
  const team1Players = [...document.querySelectorAll("#team1-list > div")]
    .map((div) => {
      const nameSpan = div.querySelector(".text-slate-200");
      return nameSpan ? nameSpan.textContent.trim() : "";
    })
    .filter((name) => name);

  const team2Players = [...document.querySelectorAll("#team2-list > div")]
    .map((div) => {
      const nameSpan = div.querySelector(".text-slate-200");
      return nameSpan ? nameSpan.textContent.trim() : "";
    })
    .filter((name) => name);

  console.log("Sending prediction request:", {
    team1,
    team2,
    team1_players_count: team1Players.length,
    team2_players_count: team2Players.length,
  });

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

    const result = await response.json();

    if (!result.success) {
      showNotification(result.error || "Prediction failed", "error");
      return;
    }

    console.log("✅ Prediction result:", result);

    // Update UI with results
    updateResultScreenEnhanced(result);

    // Show success notification
    showNotification(
      `Prediction complete: ${result.winner} (${result.winner_prob}%)`,
      "success"
    );

    // Go to results step
    goToStep(3);
  } catch (error) {
    console.error("Prediction error:", error);
    showNotification("Failed to run prediction", "error");
  }
}

/**
 * Update result screen with enhanced data
 */
function updateResultScreenEnhanced(result) {
  // Update team names
  const t1 = document.getElementById("result-team1");
  const t2 = document.getElementById("result-team2");
  if (t1) t1.textContent = result.team1;
  if (t2) t2.textContent = result.team2;

  // Update probabilities
  const pA = document.getElementById("result-prob-a");
  const pB = document.getElementById("result-prob-b");
  if (pA)
    pA.innerHTML = `${result.team1_prob.toFixed(
      1
    )}<span class="text-2xl text-emerald-400">%</span>`;
  if (pB)
    pB.innerHTML = `${result.team2_prob.toFixed(
      1
    )}<span class="text-2xl text-sky-400">%</span>`;

  // Highlight winner
  const leftCard = pA?.parentElement;
  const rightCard = pB?.parentElement;

  if (result.team1_prob > result.team2_prob) {
    leftCard?.classList.add("from-emerald-500/25");
    rightCard?.classList.remove("from-sky-500/15");
  } else {
    rightCard?.classList.add("from-sky-500/15");
    leftCard?.classList.remove("from-emerald-500/25");
  }

  console.log("✅ Result screen updated");
}

// Override the original runPrediction function
window.runPrediction = runPredictionEnhanced;

console.log("✅ Enhanced player auto-population loaded");
