/**
 * EL DORADO - IPL PREDICTION DASHBOARD
 * Complete JavaScript with 5 graphs + team composition validation
 */

// =============================================================================
// GLOBAL STATE
// =============================================================================

let selectedTeamA = null;
let selectedTeamB = null;
let teamAPlayers = [];
let teamBPlayers = [];
let selectedPlayersA = [];
let selectedPlayersB = [];

let formChartA = null;
let ovrChart = null;
let pvpChart = null;
let phaseChart = null;
let h2hChart = null;

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTeamSelectors();
    initPredictButton();
});

function initTeamSelectors() {
    document.getElementById('team-a-select').addEventListener('change', (e) => {
        selectedTeamA = e.target.value;
        loadTeamPlayers(selectedTeamA, 'team-a-players');
    });
    
    document.getElementById('team-b-select').addEventListener('change', (e) => {
        selectedTeamB = e.target.value;
        loadTeamPlayers(selectedTeamB, 'team-b-players');
    });
}

function initPredictButton() {
    document.getElementById('predict-btn').addEventListener('click', predictMatch);
}

// =============================================================================
// PLAYER LOADING & SELECTION
// =============================================================================

async function loadTeamPlayers(team, targetDiv) {
    try {
        const response = await fetch(`/meta/players/${team}`);
        const data = await response.json();
        
        // Store globally
        if (targetDiv === 'team-a-players') {
            teamAPlayers = data.players;
            selectedPlayersA = [];
        } else {
            teamBPlayers = data.players;
            selectedPlayersB = [];
        }
        
        renderPlayerCards(data.players, targetDiv);
        updateCompositionCounter(targetDiv);
        
    } catch (error) {
        console.error('Error loading players:', error);
        alert('Failed to load players for ' + team);
    }
}

function renderPlayerCards(players, targetDiv) {
    const container = document.getElementById(targetDiv);
    container.innerHTML = '';
    
    players.forEach((player, index) => {
        const card = document.createElement('div');
        card.className = 'player-card';
        card.dataset.category = player.category;
        
        card.innerHTML = `
            <input type="checkbox" id="${targetDiv}-player-${index}" value="${player.name}">
            <label for="${targetDiv}-player-${index}">
                <div class="player-name">${player.name}</div>
                <div class="player-ovr">${player.overall_ovr}</div>
                <div class="player-badge ${player.category.replace('-', '')}">${player.category}</div>
            </label>
        `;
        
        container.appendChild(card);
        
        // Add change listener
        card.querySelector('input').addEventListener('change', (e) => {
            updateSelection(targetDiv, e.target.checked, player);
        });
    });
}

function updateSelection(targetDiv, isChecked, player) {
    const isTeamA = targetDiv === 'team-a-players';
    const selectedArray = isTeamA ? selectedPlayersA : selectedPlayersB;
    
    if (isChecked) {
        if (selectedArray.length < 11) {
            selectedArray.push(player.name);
        } else {
            // Prevent selecting more than 11
            event.target.checked = false;
            alert('Maximum 11 players allowed');
            return;
        }
    } else {
        const index = selectedArray.indexOf(player.name);
        if (index > -1) selectedArray.splice(index, 1);
    }
    
    updateCompositionCounter(targetDiv);
    validatePredictButton();
}

function updateCompositionCounter(targetDiv) {
    const isTeamA = targetDiv === 'team-a-players';
    const selectedArray = isTeamA ? selectedPlayersA : selectedPlayersB;
    const allPlayers = isTeamA ? teamAPlayers : teamBPlayers;
    
    // Count by category
    const counts = {
        'BATTER': 0,
        'BOWLER': 0,
        'ALL-ROUNDER': 0,
        'WK-BATTER': 0
    };
    
    selectedArray.forEach(name => {
        const player = allPlayers.find(p => p.name === name);
        if (player) counts[player.category]++;
    });
    
    // Update UI
    const counterDiv = document.getElementById(`${targetDiv}-counter`);
    const total = selectedArray.length;
    
    counterDiv.innerHTML = `
        <div class="total-count ${total === 11 ? 'complete' : 'incomplete'}">
            ${total}/11 Selected
        </div>
        <div class="counter-item ${counts['BATTER'] === 3 ? 'complete' : 'incomplete'}">
            ${counts['BATTER'] === 3 ? '✓' : '✗'} ${counts['BATTER']}/3 BAT
        </div>
        <div class="counter-item ${counts['ALL-ROUNDER'] === 3 ? 'complete' : 'incomplete'}">
            ${counts['ALL-ROUNDER'] === 3 ? '✓' : '✗'} ${counts['ALL-ROUNDER']}/3 AR
        </div>
        <div class="counter-item ${counts['BOWLER'] === 4 ? 'complete' : 'incomplete'}">
            ${counts['BOWLER'] === 4 ? '✓' : '✗'} ${counts['BOWLER']}/4 BOWL
        </div>
        <div class="counter-item ${counts['WK-BATTER'] === 1 ? 'complete' : 'incomplete'}">
            ${counts['WK-BATTER'] === 1 ? '✓' : '✗'} ${counts['WK-BATTER']}/1 WK
        </div>
    `;
}

function validatePredictButton() {
    const countsA = countCategories(selectedPlayersA, teamAPlayers);
    const countsB = countCategories(selectedPlayersB, teamBPlayers);
    
    const isValidA = countsA.BATTER === 3 && countsA['ALL-ROUNDER'] === 3 && 
                     countsA.BOWLER === 4 && countsA['WK-BATTER'] === 1;
    const isValidB = countsB.BATTER === 3 && countsB['ALL-ROUNDER'] === 3 && 
                     countsB.BOWLER === 4 && countsB['WK-BATTER'] === 1;
    
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.disabled = !(isValidA && isValidB);
}

function countCategories(selectedNames, allPlayers) {
    const counts = {
        'BATTER': 0,
        'BOWLER': 0,
        'ALL-ROUNDER': 0,
        'WK-BATTER': 0
    };
    
    selectedNames.forEach(name => {
        const player = allPlayers.find(p => p.name === name);
        if (player) counts[player.category]++;
    });
    
    return counts;
}

// =============================================================================
// PREDICTION
// =============================================================================

async function predictMatch() {
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                team_a: selectedTeamA,
                team_b: selectedTeamB,
                players_a: selectedPlayersA,
                players_b: selectedPlayersB
            })
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').classList.add('hidden');
        
        if (data.error) {
            alert('Prediction Error:\n' + data.message);
            if (data.team_a_errors) {
                console.error('Team A errors:', data.team_a_errors);
            }
            if (data.team_b_errors) {
                console.error('Team B errors:', data.team_b_errors);
            }
            return;
        }
        
        // Display results
        displayPredictionResults(data);
        
        // Render graphs
        renderAllGraphs(data);
        
    } catch (error) {
        document.getElementById('loading').classList.add('hidden');
        console.error('Prediction error:', error);
        alert('Failed to get prediction. Check console for details.');
    }
}

function displayPredictionResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.classList.remove('hidden');
    
    // Update team names
    document.getElementById('result-team-a-name').textContent = data.team_a;
    document.getElementById('result-team-b-name').textContent = data.team_b;
    
    // Update probabilities
    document.getElementById('result-team-a-prob').textContent = data.team_a_prob.toFixed(1) + '%';
    document.getElementById('result-team-b-prob').textContent = data.team_b_prob.toFixed(1) + '%';
    
    // Update bars
    document.getElementById('result-team-a-bar').style.width = data.team_a_prob + '%';
    document.getElementById('result-team-b-bar').style.width = data.team_b_prob + '%';
    
    // Update winner
    document.getElementById('result-winner').textContent = data.winner;
    
    // Update confidence
    const confBadge = document.getElementById('result-confidence');
    confBadge.textContent = `${data.confidence.level} (${data.confidence.margin.toFixed(1)}%)`;
    confBadge.style.backgroundColor = data.confidence.color + '20';
    confBadge.style.borderColor = data.confidence.color;
    confBadge.style.color = data.confidence.color;
    
    // Update source weights
    document.getElementById('weight-ovr').textContent = data.source_weights.ovr + '%';
    document.getElementById('weight-h2h').textContent = data.source_weights.h2h + '%';
    document.getElementById('weight-form').textContent = data.source_weights.form + '%';
    document.getElementById('weight-pvp').textContent = data.source_weights.pvp + '%';
}

// =============================================================================
// GRAPH RENDERING
// =============================================================================

async function renderAllGraphs(predictionData) {
    // Destroy old charts
    [formChartA, ovrChart, pvpChart, phaseChart, h2hChart].forEach(chart => {
        if (chart) chart.destroy();
    });
    
    // Render new charts
    await renderGraph1_RecentForm(predictionData);
    renderGraph2_OVRDistribution(predictionData);
    await renderGraph3_PvPHeatmap(predictionData);
    renderGraph4_PhaseStrengths(predictionData);
    await renderGraph5_H2HHistory(predictionData);
}

// Graph 1: Recent Form Comparison
async function renderGraph1_RecentForm(data) {
    const formA = data.context.form_a;
    const formB = data.context.form_b;
    
    if (!formA.exists || !formB.exists) {
        console.warn('Form data not available');
        return;
    }
    
    const ctx = document.getElementById('graph-form').getContext('2d');
    
    // Mock data for last 10 matches (replace with real data if available)
    const matchLabels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'];
    
    formChartA = new Chart(ctx, {
        type: 'line',
        data: {
            labels: matchLabels,
            datasets: [
                {
                    label: data.team_a,
                    data: generateMockFormData(formA.win_rate),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: data.team_b,
                    data: generateMockFormData(formB.win_rate),
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: (value) => value === 1 ? 'Win' : 'Loss',
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'}
                }
            },
            plugins: {
                legend: {
                    labels: {color: '#e2e8f0'}
                },
                title: {
                    display: true,
                    text: 'Recent Form Comparison (Last 10 Matches)',
                    color: '#e2e8f0',
                    font: {size: 16}
                }
            }
        }
    });
}

// Graph 2: Player OVR Distribution
function renderGraph2_OVRDistribution(data) {
    const ctx = document.getElementById('graph-ovr').getContext('2d');
    
    // Get top 5 players from each team
    const top5A = selectedPlayersA.slice(0, 5).map(name => {
        const player = teamAPlayers.find(p => p.name === name);
        return {name: player.name, ovr: player.overall_ovr};
    });
    
    const top5B = selectedPlayersB.slice(0, 5).map(name => {
        const player = teamBPlayers.find(p => p.name === name);
        return {name: player.name, ovr: player.overall_ovr};
    });
    
    ovrChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'],
            datasets: [
                {
                    label: data.team_a,
                    data: top5A.map(p => p.ovr),
                    backgroundColor: '#3b82f6'
                },
                {
                    label: data.team_b,
                    data: top5B.map(p => p.ovr),
                    backgroundColor: '#f59e0b'
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'}
                },
                y: {
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'}
                }
            },
            plugins: {
                legend: {
                    labels: {color: '#e2e8f0'}
                },
                title: {
                    display: true,
                    text: 'Top 5 Players - OVR Comparison',
                    color: '#e2e8f0',
                    font: {size: 16}
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const teamPlayers = context.datasetIndex === 0 ? top5A : top5B;
                            const player = teamPlayers[context.dataIndex];
                            return `${player.name}: ${player.ovr}`;
                        }
                    }
                }
            }
        }
    });
}

// Graph 3: PvP Matchup Heatmap (simplified - just show summary)
async function renderGraph3_PvPHeatmap(data) {
    const ctx = document.getElementById('graph-pvp').getContext('2d');
    
    // Mock PvP data (replace with real PvP lookups)
    const favorable_a = Math.floor(Math.random() * 30) + 40;
    const favorable_b = Math.floor(Math.random() * 30) + 30;
    const neutral = 121 - favorable_a - favorable_b;
    
    pvpChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [`Favorable for ${data.team_a}`, `Favorable for ${data.team_b}`, 'Neutral'],
            datasets: [{
                data: [favorable_a, favorable_b, neutral],
                backgroundColor: ['#3b82f6', '#f59e0b', '#64748b']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {color: '#e2e8f0'}
                },
                title: {
                    display: true,
                    text: 'Player vs Player Matchups (121 Total)',
                    color: '#e2e8f0',
                    font: {size: 16}
                }
            }
        }
    });
}

// Graph 4: Phase-Specific Strengths (Radar)
function renderGraph4_PhaseStrengths(data) {
    const ctx = document.getElementById('graph-phase').getContext('2d');
    
    // Calculate average phase OVRs
    const phaseA = calculatePhaseStrengths(selectedPlayersA, teamAPlayers);
    const phaseB = calculatePhaseStrengths(selectedPlayersB, teamBPlayers);
    
    phaseChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Powerplay', 'Middle Overs', 'Death Overs'],
            datasets: [
                {
                    label: data.team_a,
                    data: [phaseA.powerplay, phaseA.middle, phaseA.death],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)'
                },
                {
                    label: data.team_b,
                    data: [phaseB.powerplay, phaseB.middle, phaseB.death],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.2)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10,
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'},
                    pointLabels: {color: '#e2e8f0'}
                }
            },
            plugins: {
                legend: {
                    labels: {color: '#e2e8f0'}
                },
                title: {
                    display: true,
                    text: 'Phase-Specific Strengths',
                    color: '#e2e8f0',
                    font: {size: 16}
                }
            }
        }
    });
}

// Graph 5: Head-to-Head History
async function renderGraph5_H2HHistory(data) {
    const h2h = data.context.h2h;
    
    if (!h2h.exists) {
        console.warn('H2H data not available');
        return;
    }
    
    const ctx = document.getElementById('graph-h2h').getContext('2d');
    
    h2hChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [data.team_a, data.team_b],
            datasets: [{
                label: 'Wins',
                data: [h2h.team_a_wins, h2h.team_b_wins],
                backgroundColor: ['#3b82f6', '#f59e0b']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'}
                },
                x: {
                    ticks: {color: '#94a3b8'},
                    grid: {color: '#334155'}
                }
            },
            plugins: {
                legend: {display: false},
                title: {
                    display: true,
                    text: `Head-to-Head History (${h2h.total_matches} matches)`,
                    color: '#e2e8f0',
                    font: {size: 16}
                }
            }
        }
    });
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function generateMockFormData(winRate) {
    // Generate 10 matches based on win rate
    const data = [];
    const wins = Math.round(10 * (winRate / 100));
    
    for (let i = 0; i < 10; i++) {
        data.push(i < wins ? 1 : 0);
    }
    
    // Shuffle
    return data.sort(() => Math.random() - 0.5);
}

function calculatePhaseStrengths(selectedNames, allPlayers) {
    // Calculate average OVR for each phase
    let powerplay = 0, middle = 0, death = 0, count = 0;
    
    selectedNames.forEach(name => {
        const player = allPlayers.find(p => p.name === name);
        if (player && player.overall_ovr) {
            // Use overall OVR as proxy (in real impl, use phase-specific OVRs)
            powerplay += player.overall_ovr;
            middle += player.overall_ovr;
            death += player.overall_ovr;
            count++;
        }
    });
    
    if (count === 0) count = 1;
    
    // Normalize to 0-10 scale
    return {
        powerplay: (powerplay / count) / 10,
        middle: (middle / count) / 10,
        death: (death / count) / 10
    };
}