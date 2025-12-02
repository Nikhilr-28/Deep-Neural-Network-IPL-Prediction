from flask import Flask, render_template, request, jsonify
import sys
import os

app = Flask(__name__)

# =============================================================================
# FIX IMPORT PATH FOR YOUR CURRENT STRUCTURE
# =============================================================================

BASE = os.path.dirname(os.path.abspath(__file__))

DL_MODELS_PATH = os.path.join(
    BASE,
    "Nikhil",
    "Dataset(s) and code",
    "dataset",
    "DL_MODELS"
)

sys.path.append(DL_MODELS_PATH)

# Import your ML classes
from ipl_match_predictor import Predictor, PlayerLookup, norm_team

# =============================================================================
# DEFINE CORRECT MAC PATHS TO YOUR DATA FILES
# =============================================================================

BATTING_OVR = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "OVR",
    "PRE_OVR_BATTING_20251129_154628.csv"
)

BOWLING_OVR = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "OVR",
    "PRE_OVR_BOWLING_20251129_154628.csv"
)

H2H_CSV = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "DL_MODELS", "outputs",
    "team_h2h_matrix_2025.csv"
)

FORM_CSV = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "DL_MODELS", "outputs",
    "team_recent_form_2025.csv"
)

PVP_JSON = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "DL_MODELS", "outputs",
    "pvp_nested_dict.json"  # ← NEW FILENAME
)

MODEL_PATH = os.path.join(
    BASE, "Nikhil", "Dataset(s) and code", "dataset", "DL_MODELS",
    "best_model.pt"
)

# =============================================================================
# INSTANTIATE LOOKUP + MODEL
# =============================================================================

print("Loading Player Lookup...")
lookup = PlayerLookup(BATTING_OVR, BOWLING_OVR)

print("Loading Predictor...")
predictor = Predictor(
    model_path=MODEL_PATH,
    lookup=lookup,
    h2h_csv=H2H_CSV,
    form_csv=FORM_CSV,
    pvp_json=PVP_JSON
)

print("🔥 Backend Ready")


# =============================================================================
# TEAMS + HOME GROUNDS (BASE VERSION)
# Later you can move to CSV if needed
# =============================================================================

TEAM_HOME_GROUNDS = {
    "CSK": ["M. A. Chidambaram Stadium · Chennai"],
    "MI": ["Wankhede Stadium · Mumbai"],
    "KKR": ["Eden Gardens · Kolkata"],
    "RCB": ["Chinnaswamy Stadium · Bengaluru"],
    "SRH": ["Rajiv Gandhi Intl Stadium · Hyderabad"],
    "DC": ["Arun Jaitley Stadium · Delhi"],
    "RR": ["Sawai Mansingh Stadium · Jaipur"],
    "LSG": ["Ekana Stadium · Lucknow"],
    "GT": ["Narendra Modi Stadium · Ahmedabad"],
    "PK": ["IS Bindra Stadium · Mohali", "HPCA Stadium · Dharamshala"]
}


# =============================================================================
# ROUTES - HTML + API
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# PROVIDE TEAMS + HOME GROUNDS
# -----------------------------
@app.route("/meta/teams")
def meta_teams():
    import pandas as pd

    batting_df = pd.read_csv(BATTING_OVR)
    bowling_df = pd.read_csv(BOWLING_OVR)

    # Extract team names from correct column
    teams_from_bat = set(batting_df["IPL_Team_2025"].dropna().unique())
    teams_from_bowl = set(bowling_df["IPL_Team_2025"].dropna().unique())

    TEAM_LIST = sorted(teams_from_bat | teams_from_bowl)

    # Dummy home grounds for now; will upgrade later
    TEAM_META = {
        team: {
            "home_grounds": [
                f"{team} Stadium 1",
                f"{team} Stadium 2"
            ]
        }
        for team in TEAM_LIST
    }

    return jsonify({
        "teams": TEAM_LIST,
        "grounds": TEAM_META
    })



# -----------------------------
# PROVIDE PLAYERS LOADED FROM CSV
# (Lookup already preprocessed them)
# -----------------------------
@app.route("/meta/players")
def meta_players():
    players = list(lookup.p2i.keys())  # All recognized players
    return jsonify({"players": players})


@app.route("/meta/players/<team>")
def meta_players1(team):
    import pandas as pd

    batting_df = pd.read_csv(BATTING_OVR)
    bowling_df = pd.read_csv(BOWLING_OVR)

    # Filter by IPL team
    bat = batting_df[batting_df["IPL_Team_2025"] == team][
        ["Player_Name", "Player_Type", "BASE_OVR"]
    ].rename(columns={"BASE_OVR": "bat_ovr"})

    bowl = bowling_df[bowling_df["IPL_Team_2025"] == team][
        ["Player_Name", "Player_Type", "BASE_OVR"]
    ].rename(columns={"BASE_OVR": "bowl_ovr"})

    # Outer merge so players that appear only in batting OR bowling are still included
    merged = pd.merge(
        bat,
        bowl,
        on=["Player_Name", "Player_Type"],
        how="outer",
        suffixes=("", "_bowl"),
    )

    players = []
    for _, row in merged.iterrows():
        players.append(
            {
                "name": row["Player_Name"],
                "type": row["Player_Type"],
                "bat_ovr": float(row["bat_ovr"]) if not pd.isna(row.get("bat_ovr")) else None,
                "bowl_ovr": float(row["bowl_ovr"]) if not pd.isna(row.get("bowl_ovr")) else None,
            }
        )

    return jsonify({
        "team": team,
        "players": players
    })




# -----------------------------
# MAKE PREDICTION
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    match = {
        "team_a": norm_team(data["team_a"]),
        "team_b": norm_team(data["team_b"]),
        "team_a_xi": data["players_a"],
        "team_b_xi": data["players_b"]
    }

    result = predictor.predict(match)
    return jsonify(result)


# =============================================================================
# START SERVER
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True)
