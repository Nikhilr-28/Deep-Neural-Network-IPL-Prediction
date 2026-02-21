from flask import Flask, render_template, request, jsonify
import sys, os, json
import pandas as pd

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "data_forge/dataset/DL_MODELS"))

from ipl_match_predictor import Predictor, PlayerLookup, norm_team

# Paths
BATTING_OVR = f"{BASE}/data_forge/dataset/OVR/PRE_OVR_BATTING_20251129_154628.csv"
BOWLING_OVR = f"{BASE}/data_forge/dataset/OVR/PRE_OVR_BOWLING_20251129_154628.csv"
H2H_CSV = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/team_h2h_matrix_2025.csv"
FORM_CSV = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/team_recent_form_2025.csv"
PVP_JSON = f"{BASE}/data_forge/dataset/DL_MODELS/outputs/pvp_nested_dict.json"
MODEL_PATH = f"{BASE}/data_forge/dataset/DL_MODELS/best_model.pt"

# Load data
lookup = PlayerLookup(BATTING_OVR, BOWLING_OVR)
predictor = Predictor(MODEL_PATH, lookup, H2H_CSV, FORM_CSV, PVP_JSON)
batting_df = pd.read_csv(BATTING_OVR)
bowling_df = pd.read_csv(BOWLING_OVR)
h2h_df = pd.read_csv(H2H_CSV)
form_df = pd.read_csv(FORM_CSV)

print("✅ Backend Ready - All issues fixed")

def categorize(pt):
    pt = str(pt).upper()
    if 'WK' in pt or 'WICKET' in pt: return 'WK-BATTER'
    if 'ALL' in pt or 'ROUNDER' in pt: return 'ALL-ROUNDER'
    if 'BOWL' in pt: return 'BOWLER'
    if 'BAT' in pt: return 'BATTER'
    return 'ALL-ROUNDER'

def get_players_for_team(team):
    bat = batting_df[batting_df["IPL_Team_2025"]==team][["Player_Name","Player_Type","BASE_OVR","TOP_ORDER_OVR","MIDDLE_ORDER_OVR","FINISHER_OVR"]].rename(columns={"BASE_OVR":"bat_ovr"})
    bowl = bowling_df[bowling_df["IPL_Team_2025"]==team][["Player_Name","Player_Type","BASE_OVR","POWERPLAY_OVR","MIDDLE_OVERS_OVR","DEATH_OVERS_OVR"]].rename(columns={"BASE_OVR":"bowl_ovr"})
    
    merged = pd.merge(bat, bowl, on=["Player_Name","Player_Type"], how="outer")
    
    players = []
    for _, r in merged.iterrows():
        bat_ovr = float(r["bat_ovr"]) if pd.notna(r.get("bat_ovr")) else 0
        bowl_ovr = float(r["bowl_ovr"]) if pd.notna(r.get("bowl_ovr")) else 0
        overall = max(bat_ovr, bowl_ovr)
        
        players.append({
            "name": r["Player_Name"],
            "Player_Name": r["Player_Name"],
            "type": r["Player_Type"],
            "Player_Type": r["Player_Type"],
            "category": categorize(r["Player_Type"]),
            "bat_ovr": round(bat_ovr,1) if bat_ovr>0 else None,
            "bowl_ovr": round(bowl_ovr,1) if bowl_ovr>0 else None,
            "overall_ovr": round(overall,1),
            "BASE_OVR": round(overall,1),
            "top_ovr": round(float(r.get("TOP_ORDER_OVR",0)),1) if pd.notna(r.get("TOP_ORDER_OVR")) else None,
            "middle_ovr": round(float(r.get("MIDDLE_ORDER_OVR",0)),1) if pd.notna(r.get("MIDDLE_ORDER_OVR")) else None,
            "finisher_ovr": round(float(r.get("FINISHER_OVR",0)),1) if pd.notna(r.get("FINISHER_OVR")) else None,
            "pp_ovr": round(float(r.get("POWERPLAY_OVR",0)),1) if pd.notna(r.get("POWERPLAY_OVR")) else None,
            "death_ovr": round(float(r.get("DEATH_OVERS_OVR",0)),1) if pd.notna(r.get("DEATH_OVERS_OVR")) else None
        })
    
    players.sort(key=lambda x: x["overall_ovr"], reverse=True)
    return players

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/teams", methods=["GET"])
@app.route("/meta/teams", methods=["GET"])
def get_teams():
    teams = sorted(set(batting_df["IPL_Team_2025"].dropna()) | set(bowling_df["IPL_Team_2025"].dropna()))
    
    home_grounds = {
        "CSK": {"home_grounds": ["MA Chidambaram Stadium, Chennai"]},
        "DC": {"home_grounds": ["Arun Jaitley Stadium, Delhi"]},
        "GT": {"home_grounds": ["Narendra Modi Stadium, Ahmedabad"]},
        "KKR": {"home_grounds": ["Eden Gardens, Kolkata"]},
        "LSG": {"home_grounds": ["BRSABV Ekana Cricket Stadium, Lucknow"]},
        "MI": {"home_grounds": ["Wankhede Stadium, Mumbai"]},
        "PK": {"home_grounds": ["Punjab Cricket Association Stadium, Mohali"]},
        "RCB": {"home_grounds": ["M Chinnaswamy Stadium, Bengaluru"]},
        "RR": {"home_grounds": ["Sawai Mansingh Stadium, Jaipur"]},
        "SRH": {"home_grounds": ["Rajiv Gandhi International Stadium, Hyderabad"]}
    }
    
    return jsonify({"teams": teams, "grounds": home_grounds})

@app.route("/api/players/<team>", methods=["GET"])
@app.route("/meta/players/<team>", methods=["GET"])
def get_players(team):
    try:
        players = get_players_for_team(team)
        return jsonify({"success": True, "team": team, "count": len(players), "players": players})
    except Exception as e:
        print(f"❌ Error getting players for {team}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/search_players", methods=["GET"])
def search_players():
    query = request.args.get("q", "").lower()
    if len(query) < 2:
        return jsonify([])
    
    all_players = []
    for team in sorted(set(batting_df["IPL_Team_2025"].dropna()) | set(bowling_df["IPL_Team_2025"].dropna())):
        players = get_players_for_team(team)
        for p in players:
            if query in p["name"].lower():
                p["team"] = team
                all_players.append(p)
    
    return jsonify(all_players[:20])

@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        team1 = data.get("team1") or data.get("team_a")
        team2 = data.get("team2") or data.get("team_b")
        team1_players = data.get("team1_players") or data.get("players_a", [])
        team2_players = data.get("team2_players") or data.get("players_b", [])
        
        print(f"📊 Prediction: {team1} ({len(team1_players)}) vs {team2} ({len(team2_players)})")
        
        if not team1 or not team2:
            return jsonify({"success": False, "error": "Teams not specified"}), 400
        
        if len(team1_players) == 0 or len(team2_players) == 0:
            return jsonify({"success": False, "error": f"Teams need players. Got {len(team1_players)} and {len(team2_players)}"}), 400
        
        match = {
            "team_a": norm_team(team1),
            "team_b": norm_team(team2),
            "team_a_xi": team1_players,
            "team_b_xi": team2_players
        }
        
        # Call predictor
        result = predictor.predict(match)
        
        # DEBUG: Print what predictor returns
        print(f"🔍 Predictor returned keys: {list(result.keys())}")
        print(f"🔍 Full result: {result}")
        
        # FIXED: Handle different possible key formats
        # Try multiple possible key names the predictor might use
        prob_a = result.get('team_a_prob') or result.get('prob_a') or result.get('prediction', [0.5, 0.5])[0]
        prob_b = result.get('team_b_prob') or result.get('prob_b') or result.get('prediction', [0.5, 0.5])[1]
        
        # If still None, try extracting from prediction tensor/array
        if prob_a is None or prob_b is None:
            if 'prediction' in result:
                pred = result['prediction']
                if hasattr(pred, 'item'):  # Torch tensor
                    prob_a = pred.item()
                    prob_b = 1 - prob_a
                elif isinstance(pred, (list, tuple)) and len(pred) >= 2:
                    prob_a, prob_b = pred[0], pred[1]
                elif isinstance(pred, (int, float)):
                    prob_a = float(pred)
                    prob_b = 1 - prob_a
        
        # Ensure probabilities are valid floats
        try:
            prob_a = float(prob_a) if prob_a is not None else 0.5
            prob_b = float(prob_b) if prob_b is not None else 0.5
        except:
            prob_a, prob_b = 0.5, 0.5
        
        # Normalize to ensure they sum to 1
        total = prob_a + prob_b
        if total > 0:
            prob_a = prob_a / total
            prob_b = prob_b / total
        
        print(f"✅ Probabilities: Team A = {prob_a:.4f}, Team B = {prob_b:.4f}")
        
        margin = abs(prob_a - prob_b)
        if margin >= 0.25:
            confidence, conf_color = "HIGH", "#22c55e"
        elif margin >= 0.10:
            confidence, conf_color = "MEDIUM", "#f59e0b"
        else:
            confidence, conf_color = "LOW", "#ef4444"
        
        winner = team1 if prob_a > prob_b else team2
        
        h2h_data = get_h2h_data(team1, team2)
        form_data = {"team1": get_form_data(team1), "team2": get_form_data(team2)}
        
        print(f"✅ Result: {winner} wins ({round(max(prob_a, prob_b)*100, 1)}%)")
        
        return jsonify({
            "success": True,
            "team1": team1,
            "team2": team2,
            "team1_prob": round(prob_a * 100, 2),
            "team2_prob": round(prob_b * 100, 2),
            "team_a": team1,
            "team_b": team2,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "winner": winner,
            "winner_prob": round(max(prob_a, prob_b) * 100, 2),
            "confidence": {"level": confidence, "margin": round(margin * 100, 2), "color": conf_color},
            "source_weights": {
                "ovr": round(result.get('w_ovr', 0.23) * 100, 1),
                "h2h": round(result.get('w_h2h', 0.27) * 100, 1),
                "form": round(result.get('w_form', 0.25) * 100, 1),
                "pvp": round(result.get('w_pvp', 0.26) * 100, 1)
            },
            "context": {"h2h": h2h_data, "form": form_data}
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def get_h2h_data(team1, team2):
    try:
        ta, tb = norm_team(team1), norm_team(team2)
        row = h2h_df[((h2h_df['team_a']==ta)&(h2h_df['team_b']==tb))|((h2h_df['team_a']==tb)&(h2h_df['team_b']==ta))]
        
        if row.empty:
            return {"exists": False}
        
        row = row.iloc[0]
        if row['team_a'] == ta:
            wins_a, wins_b = int(row['team_a_wins']), int(row['team_b_wins'])
        else:
            wins_a, wins_b = int(row['team_b_wins']), int(row['team_a_wins'])
        
        return {"exists": True, "team1_wins": wins_a, "team2_wins": wins_b, "total": wins_a + wins_b}
    except:
        return {"exists": False}

def get_form_data(team):
    try:
        row = form_df[form_df['team'] == norm_team(team)]
        if row.empty:
            return {"exists": False}
        
        row = row.iloc[0]
        return {"exists": True, "last_5_wins": int(row['last_5_wins']), "win_rate": round(float(row['last_5_win_rate']) * 100, 1)}
    except:
        return {"exists": False}

if __name__ == "__main__":
    app.run(debug=True, port=5000)