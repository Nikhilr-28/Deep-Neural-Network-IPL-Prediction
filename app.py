import sys
import os
import json
import logging
import traceback
from functools import wraps

import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ── Environment ──────────────────────────────────────────────────────────────
PRODUCTION = os.getenv("PRODUCTION", "NO").upper() == "YES"
PORT       = int(os.getenv("PORT", 5000))
BASE       = os.path.dirname(os.path.abspath(__file__))

# ── Logging ──────────────────────────────────────────────────────────────────
log_level = logging.WARNING if PRODUCTION else logging.DEBUG
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    DEBUG=not PRODUCTION,
    TESTING=False,
    PROPAGATE_EXCEPTIONS=False,
)

# ── Model / data paths ────────────────────────────────────────────────────────
sys.path.append(os.path.join(BASE, "data_forge/dataset/DL_MODELS"))

BATTING_OVR = f"{BASE}/data_forge/dataset/OVR/PRE_OVR_BATTING_20251129_154628.csv"
BOWLING_OVR = f"{BASE}/data_forge/dataset/OVR/PRE_OVR_BOWLING_20251129_154628.csv"
H2H_CSV     = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/team_h2h_matrix_2025.csv"
FORM_CSV    = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/team_recent_form_2025.csv"
PVP_JSON    = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/pvp_nested_dict.json"
MODEL_PATH  = f"{BASE}/data_forge/dataset/DL_MODELS/best_model.pt"

# ── Startup: load data once ───────────────────────────────────────────────────
try:
    from ipl_match_predictor import Predictor, PlayerLookup, norm_team

    lookup      = PlayerLookup(BATTING_OVR, BOWLING_OVR)
    predictor   = Predictor(MODEL_PATH, lookup, H2H_CSV, FORM_CSV, PVP_JSON)
    batting_df  = pd.read_csv(BATTING_OVR)
    bowling_df  = pd.read_csv(BOWLING_OVR)
    h2h_df      = pd.read_csv(H2H_CSV)
    form_df     = pd.read_csv(FORM_CSV)
    logger.info("Backend ready — all models and data loaded.")
except Exception as exc:
    logger.critical("Failed to initialise backend: %s", exc)
    raise SystemExit(1) from exc

# ── Home grounds lookup ───────────────────────────────────────────────────────
HOME_GROUNDS = {
    "CSK": ["MA Chidambaram Stadium, Chennai"],
    "DC":  ["Arun Jaitley Stadium, Delhi"],
    "GT":  ["Narendra Modi Stadium, Ahmedabad"],
    "KKR": ["Eden Gardens, Kolkata"],
    "LSG": ["BRSABV Ekana Cricket Stadium, Lucknow"],
    "MI":  ["Wankhede Stadium, Mumbai"],
    "PK":  ["Punjab Cricket Association Stadium, Mohali"],
    "RCB": ["M Chinnaswamy Stadium, Bengaluru"],
    "RR":  ["Sawai Mansingh Stadium, Jaipur"],
    "SRH": ["Rajiv Gandhi International Stadium, Hyderabad"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def categorize(pt: str) -> str:
    pt = str(pt).upper()
    if "WK" in pt or "WICKET" in pt:
        return "WK-BATTER"
    if "ALL" in pt or "ROUNDER" in pt:
        return "ALL-ROUNDER"
    if "BOWL" in pt:
        return "BOWLER"
    if "BAT" in pt:
        return "BATTER"
    return "ALL-ROUNDER"


def _safe_float(value, default=0.0):
    try:
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def get_players_for_team(team: str) -> list:
    bat = (
        batting_df[batting_df["IPL_Team_2025"] == team][
            ["Player_Name", "Player_Type", "BASE_OVR",
             "TOP_ORDER_OVR", "MIDDLE_ORDER_OVR", "FINISHER_OVR"]
        ].rename(columns={"BASE_OVR": "bat_ovr"})
    )
    bowl = (
        bowling_df[bowling_df["IPL_Team_2025"] == team][
            ["Player_Name", "Player_Type", "BASE_OVR",
             "POWERPLAY_OVR", "MIDDLE_OVERS_OVR", "DEATH_OVERS_OVR"]
        ].rename(columns={"BASE_OVR": "bowl_ovr"})
    )

    merged = pd.merge(bat, bowl, on=["Player_Name", "Player_Type"], how="outer")

    players = []
    for _, r in merged.iterrows():
        bat_ovr  = _safe_float(r.get("bat_ovr"))
        bowl_ovr = _safe_float(r.get("bowl_ovr"))
        overall  = max(bat_ovr, bowl_ovr)

        players.append({
            "name":        r["Player_Name"],
            "Player_Name": r["Player_Name"],
            "type":        r["Player_Type"],
            "Player_Type": r["Player_Type"],
            "category":    categorize(r["Player_Type"]),
            "bat_ovr":     round(bat_ovr, 1)  if bat_ovr  > 0 else None,
            "bowl_ovr":    round(bowl_ovr, 1) if bowl_ovr > 0 else None,
            "overall_ovr": round(overall, 1),
            "BASE_OVR":    round(overall, 1),
            "top_ovr":     round(_safe_float(r.get("TOP_ORDER_OVR")), 1)    or None,
            "middle_ovr":  round(_safe_float(r.get("MIDDLE_ORDER_OVR")), 1) or None,
            "finisher_ovr":round(_safe_float(r.get("FINISHER_OVR")), 1)     or None,
            "pp_ovr":      round(_safe_float(r.get("POWERPLAY_OVR")), 1)    or None,
            "death_ovr":   round(_safe_float(r.get("DEATH_OVERS_OVR")), 1)  or None,
        })

    players.sort(key=lambda x: x["overall_ovr"], reverse=True)
    return players


def get_h2h_data(team1: str, team2: str) -> dict:
    try:
        ta, tb = norm_team(team1), norm_team(team2)
        mask = (
            ((h2h_df["team_a"] == ta) & (h2h_df["team_b"] == tb)) |
            ((h2h_df["team_a"] == tb) & (h2h_df["team_b"] == ta))
        )
        rows = h2h_df[mask]
        if rows.empty:
            return {"exists": False}

        row = rows.iloc[0]
        if row["team_a"] == ta:
            wins_a, wins_b = int(row["team_a_wins"]), int(row["team_b_wins"])
        else:
            wins_a, wins_b = int(row["team_b_wins"]), int(row["team_a_wins"])

        return {"exists": True, "team1_wins": wins_a, "team2_wins": wins_b,
                "total": wins_a + wins_b}
    except Exception:
        logger.debug("H2H lookup failed", exc_info=True)
        return {"exists": False}


def get_form_data(team: str) -> dict:
    try:
        rows = form_df[form_df["team"] == norm_team(team)]
        if rows.empty:
            return {"exists": False}

        row = rows.iloc[0]
        return {
            "exists":      True,
            "last_5_wins": int(row["last_5_wins"]),
            "win_rate":    round(float(row["last_5_win_rate"]) * 100, 1),
        }
    except Exception:
        logger.debug("Form lookup failed", exc_info=True)
        return {"exists": False}


def _resolve_probabilities(result: dict):
    """Extract (prob_a, prob_b) from whatever the predictor returns."""
    prob_a = result.get("team_a_prob") or result.get("prob_a")
    prob_b = result.get("team_b_prob") or result.get("prob_b")

    if prob_a is None or prob_b is None:
        pred = result.get("prediction", [0.5, 0.5])
        if hasattr(pred, "item"):          # torch.Tensor scalar
            prob_a = pred.item(); prob_b = 1 - prob_a
        elif isinstance(pred, (list, tuple)) and len(pred) >= 2:
            prob_a, prob_b = pred[0], pred[1]
        elif isinstance(pred, (int, float)):
            prob_a = float(pred); prob_b = 1 - prob_a
        else:
            prob_a = prob_b = 0.5

    try:
        prob_a, prob_b = float(prob_a), float(prob_b)
    except (TypeError, ValueError):
        prob_a = prob_b = 0.5

    total = prob_a + prob_b
    if total > 0:
        prob_a /= total
        prob_b /= total

    return prob_a, prob_b


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.error("Unhandled 500: %s", e, exc_info=True)
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams", methods=["GET"])
@app.route("/meta/teams", methods=["GET"])
def get_teams():
    teams = sorted(
        set(batting_df["IPL_Team_2025"].dropna()) |
        set(bowling_df["IPL_Team_2025"].dropna())
    )
    grounds = {k: {"home_grounds": v} for k, v in HOME_GROUNDS.items()}
    return jsonify({"teams": teams, "grounds": grounds})


@app.route("/api/players/<team>", methods=["GET"])
@app.route("/meta/players/<team>", methods=["GET"])
def get_players(team):
    try:
        players = get_players_for_team(team)
        return jsonify({"success": True, "team": team,
                        "count": len(players), "players": players})
    except Exception as exc:
        logger.error("Error fetching players for %s: %s", team, exc, exc_info=not PRODUCTION)
        return jsonify({"success": False, "error": "Failed to fetch players"}), 500


@app.route("/api/search_players", methods=["GET"])
def search_players():
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])

    all_teams = sorted(
        set(batting_df["IPL_Team_2025"].dropna()) |
        set(bowling_df["IPL_Team_2025"].dropna())
    )
    results = []
    for team in all_teams:
        for p in get_players_for_team(team):
            if query in p["name"].lower():
                p["team"] = team
                results.append(p)
                if len(results) >= 20:
                    break
        if len(results) >= 20:
            break

    return jsonify(results)


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON body"}), 400

        team1          = data.get("team1") or data.get("team_a")
        team2          = data.get("team2") or data.get("team_b")
        team1_players  = data.get("team1_players") or data.get("players_a", [])
        team2_players  = data.get("team2_players") or data.get("players_b", [])

        if not team1 or not team2:
            return jsonify({"success": False, "error": "Both teams must be specified"}), 400
        if not team1_players or not team2_players:
            return jsonify({
                "success": False,
                "error": f"Both teams need players. Got {len(team1_players)} and {len(team2_players)}."
            }), 400

        logger.info("Prediction request: %s (%d) vs %s (%d)",
                    team1, len(team1_players), team2, len(team2_players))

        match = {
            "team_a":    norm_team(team1),
            "team_b":    norm_team(team2),
            "team_a_xi": team1_players,
            "team_b_xi": team2_players,
        }

        result = predictor.predict(match)
        logger.debug("Predictor result keys: %s", list(result.keys()))

        prob_a, prob_b = _resolve_probabilities(result)

        margin = abs(prob_a - prob_b)
        if margin >= 0.25:
            confidence, conf_color = "HIGH",   "#22c55e"
        elif margin >= 0.10:
            confidence, conf_color = "MEDIUM", "#f59e0b"
        else:
            confidence, conf_color = "LOW",    "#ef4444"

        winner = team1 if prob_a >= prob_b else team2
        logger.info("Result: %s wins (%.1f%%)", winner, max(prob_a, prob_b) * 100)

        return jsonify({
            "success":    True,
            "team1":      team1,
            "team2":      team2,
            "team_a":     team1,
            "team_b":     team2,
            "team1_prob": round(prob_a * 100, 2),
            "team2_prob": round(prob_b * 100, 2),
            "prob_a":     prob_a,
            "prob_b":     prob_b,
            "winner":     winner,
            "winner_prob":round(max(prob_a, prob_b) * 100, 2),
            "confidence": {
                "level":  confidence,
                "margin": round(margin * 100, 2),
                "color":  conf_color,
            },
            "source_weights": {
                "ovr":  round(result.get("w_ovr",  0.23) * 100, 1),
                "h2h":  round(result.get("w_h2h",  0.27) * 100, 1),
                "form": round(result.get("w_form", 0.25) * 100, 1),
                "pvp":  round(result.get("w_pvp",  0.26) * 100, 1),
            },
            "context": {
                "h2h":  get_h2h_data(team1, team2),
                "form": {
                    "team1": get_form_data(team1),
                    "team2": get_form_data(team2),
                },
            },
        })

    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=not PRODUCTION)
        msg = str(exc) if not PRODUCTION else "Prediction failed. Please try again."
        return jsonify({"success": False, "error": msg}), 500


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "production": PRODUCTION})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if PRODUCTION:
        # In production use a proper WSGI server (gunicorn / waitress).
        # This branch exists only as a fallback / local smoke-test.
        try:
            from waitress import serve
            logger.warning("Starting waitress on port %d", PORT)
            serve(app, host="0.0.0.0", port=PORT)
        except ImportError:
            logger.warning(
                "waitress not installed; falling back to Flask dev server. "
                "Install waitress or use gunicorn for real production deployments."
            )
            app.run(host="0.0.0.0", port=PORT, debug=False)
    else:
        app.run(debug=True, port=PORT)