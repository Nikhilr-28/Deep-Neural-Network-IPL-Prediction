"""
Flask API for IPL Match Prediction
Models loaded in memory for instant predictions (<100ms)

Endpoints:
  POST /api/simulate/pre-match - Full match prediction
  GET /api/teams - Team metadata
  GET /api/venues - Venue metadata
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("INITIALIZING IPL PREDICTION API")
print("=" * 80)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset"

CONFIG = {
    'tab_model': rf"{BASE_DIR}\DL_MODELS\outputs\tabtransformer_model.pt",
    'tab_encoders': rf"{BASE_DIR}\DL_MODELS\outputs\tabtransformer_encoders.json",
    'gat_pvp': rf"{BASE_DIR}\DL_MODELS\outputs\gat_enhanced_pvp.json",
    'batting_ovr': rf"{BASE_DIR}\OVR\PRE_OVR_BATTING_20251129_154628.csv",
    'bowling_ovr': rf"{BASE_DIR}\OVR\PRE_OVR_BOWLING_20251129_154628.csv",
    'city_dev_batting': rf"{BASE_DIR}\stadium_matching\CITY_DEVIATION_BATTING_2022_2024.csv",
    'city_dev_bowling': rf"{BASE_DIR}\stadium_matching\CITY_DEVIATION_BOWLING_2022_2024.csv",
    'venue_mapping': rf"{BASE_DIR}\stadium_matching\VENUE_TO_CITY_MAPPING.csv"
}

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# TABTRANSFORMER MODEL DEFINITION
# ============================================================================

class TabTransformer(nn.Module):
    def __init__(self, categorical_dims, continuous_dim, embedding_dim=128, 
                 num_layers=4, num_heads=8, ff_dim=256, dropout=0.3):
        super().__init__()
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in categorical_dims.items()
        })
        
        self.continuous_proj = nn.Linear(continuous_dim, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, categorical, continuous):
        embeddings = []
        for name, emb_layer in self.embeddings.items():
            embeddings.append(emb_layer(categorical[name].squeeze(1)))
        
        cont_emb = self.continuous_proj(continuous)
        all_features = torch.stack(embeddings + [cont_emb], dim=1)
        encoded = self.transformer(all_features)
        pooled = encoded.mean(dim=1)
        output = self.mlp(pooled)
        
        return output.squeeze()

# ============================================================================
# LOAD MODELS AND DATA (ON STARTUP - ONCE!)
# ============================================================================

print(f"\n📦 LOADING MODELS AND DATA INTO MEMORY...")

# Global variables to hold models/data
tab_model = None
encoders = None
pvp_data = None
ovr_lookup = None
city_dev_lookup = None
venue_to_city = None

def load_all_resources():
    """Load all models and data ONCE on startup"""
    global tab_model, encoders, pvp_data, ovr_lookup, city_dev_lookup, venue_to_city
    
    print(f"\n1️⃣  Loading TabTransformer model...")
    checkpoint = torch.load(CONFIG['tab_model'], map_location=device)
    
    # Create model
    tab_model = TabTransformer(
        categorical_dims=checkpoint['categorical_dims'],
        continuous_dim=checkpoint['continuous_dim'],
        embedding_dim=checkpoint['config']['embedding_dim'],
        num_layers=checkpoint['config']['num_layers'],
        num_heads=checkpoint['config']['num_heads'],
        ff_dim=checkpoint['config']['ff_dim'],
        dropout=checkpoint['config']['dropout']
    ).to(device)
    
    tab_model.load_state_dict(checkpoint['model_state_dict'])
    tab_model.eval()
    print(f"   ✅ Model loaded (accuracy: {checkpoint['val_acc']*100:.1f}%)")
    
    print(f"\n2️⃣  Loading encoders...")
    with open(CONFIG['tab_encoders'], 'r') as f:
        encoders = json.load(f)
    print(f"   ✅ Encoders loaded")
    
    print(f"\n3️⃣  Loading GAT PvP data...")
    with open(CONFIG['gat_pvp'], 'r') as f:
        pvp_data = json.load(f)
    
    # Create PvP lookup
    pvp_lookup = {}
    for edge in pvp_data['pvp_edges']:
        batter = edge['batter']['kaggle_name'].lower().strip()
        bowler = edge['bowler']['kaggle_name'].lower().strip()
        key = f"{batter}_{bowler}"
        pvp_lookup[key] = edge['gat_enhanced']['gat_advantage']
    pvp_data['lookup'] = pvp_lookup
    print(f"   ✅ PvP data loaded ({len(pvp_lookup)} matchups)")
    
    print(f"\n4️⃣  Loading player OVRs...")
    batting_ovr = pd.read_csv(CONFIG['batting_ovr'])
    bowling_ovr = pd.read_csv(CONFIG['bowling_ovr'])
    
    ovr_lookup = {}
    for _, row in batting_ovr.iterrows():
        if pd.notna(row['Kaggle_Match_Name']):
            name = row['Kaggle_Match_Name'].lower().strip()
            ovr_lookup[name] = row['BASE_OVR']
    
    for _, row in bowling_ovr.iterrows():
        if pd.notna(row['Kaggle_Match_Name']):
            name = row['Kaggle_Match_Name'].lower().strip()
            if name not in ovr_lookup:
                ovr_lookup[name] = row['BASE_OVR']
    
    print(f"   ✅ OVR data loaded ({len(ovr_lookup)} players)")
    
    print(f"\n5️⃣  Loading city deviations...")
    city_bat = pd.read_csv(CONFIG['city_dev_batting'])
    city_bowl = pd.read_csv(CONFIG['city_dev_bowling'])
    
    city_dev_lookup = {
        'batting': {},
        'bowling': {}
    }
    
    for _, row in city_bat.iterrows():
        key = f"{row['Player_Name'].lower().strip()}_{row['City'].lower().strip()}"
        city_dev_lookup['batting'][key] = row['Strike_Rate_Deviation']
    
    for _, row in city_bowl.iterrows():
        key = f"{row['Player_Name'].lower().strip()}_{row['City'].lower().strip()}"
        city_dev_lookup['bowling'][key] = row['Economy_Deviation']
    
    print(f"   ✅ City deviations loaded")
    
    print(f"\n6️⃣  Loading venue mapping...")
    venue_map = pd.read_csv(CONFIG['venue_mapping'])
    venue_to_city = {}
    for _, row in venue_map.iterrows():
        venue_to_city[row['Venue'].lower().strip()] = row['City'].lower().strip()
    print(f"   ✅ Venue mapping loaded ({len(venue_to_city)} venues)")
    
    print(f"\n✅ ALL RESOURCES LOADED IN MEMORY!")

# Load on startup
load_all_resources()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_team_ovr(players):
    """Calculate team average OVR"""
    ovrs = [ovr_lookup.get(p.lower().strip(), 70.0) for p in players]
    return np.mean(ovrs)

def calculate_pvp_boost(team_batters, opponent_bowlers):
    """Calculate PvP boost from GAT"""
    advantages = []
    for batter in team_batters:
        for bowler in opponent_bowlers:
            key = f"{batter.lower().strip()}_{bowler.lower().strip()}"
            if key in pvp_data['lookup']:
                advantages.append(pvp_data['lookup'][key])
    
    return np.mean(advantages) * 10.0 if advantages else 0.0

def calculate_city_deviation(players, city, dev_type='batting'):
    """Calculate city deviation"""
    city_lower = city.lower().strip()
    deviations = []
    
    for player in players:
        key = f"{player.lower().strip()}_{city_lower}"
        if key in city_dev_lookup[dev_type]:
            deviations.append(city_dev_lookup[dev_type][key])
    
    return np.mean(deviations) if deviations else 0.0

def apply_dew_factor(batting_first_ovr, chasing_ovr, dew_factor):
    """
    Apply dew factor (0.0 to 1.0)
    Chasing team gets advantage
    """
    dew_advantage = dew_factor * 0.15  # Up to 15% advantage
    
    batting_first_final = batting_first_ovr * (1 - dew_advantage)
    chasing_final = chasing_ovr * (1 + dew_advantage)
    
    return batting_first_final, chasing_final

def encode_categorical(value, feature_name):
    """Encode categorical feature"""
    classes = encoders[feature_name]['classes']
    try:
        idx = classes.index(str(value))
        return idx
    except ValueError:
        return 0  # Default to first class if not found

def normalize_continuous(values):
    """Normalize continuous features"""
    mean = np.array(encoders['scaler']['mean'])
    scale = np.array(encoders['scaler']['scale'])
    return (values - mean) / scale

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': tab_model is not None,
        'device': str(device)
    })

@app.route('/api/simulate/pre-match', methods=['POST'])
def simulate_pre_match():
    """
    Full match prediction
    
    Input:
    {
        "home_team": "CSK",
        "away_team": "MI",
        "venue": "Chepauk",
        "pitch_type": "SPIN",
        "dew_factor": 0.7,
        "toss_winner": "CSK",
        "toss_decision": "bat",
        "home_xi": ["MS Dhoni", "Ruturaj Gaikwad", ...],
        "away_xi": ["Rohit Sharma", "Ishan Kishan", ...]
    }
    """
    try:
        data = request.json
        
        # Extract inputs
        home_team = data['home_team']
        away_team = data['away_team']
        venue = data['venue']
        dew_factor = float(data.get('dew_factor', 0.0))
        toss_winner = data['toss_winner']
        toss_decision = data['toss_decision']
        home_xi = data['home_xi']
        away_xi = data['away_xi']
        
        # Get city
        city = venue_to_city.get(venue.lower().strip(), 'unknown')
        if city == 'unknown':
            return jsonify({'error': f'Unknown venue: {venue}'}), 400
        
        # Determine batting first
        if toss_winner == home_team:
            batting_first = 'home' if toss_decision == 'bat' else 'away'
        else:
            batting_first = 'away' if toss_decision == 'bat' else 'home'
        
        # Calculate features
        home_base_ovr = calculate_team_ovr(home_xi)
        away_base_ovr = calculate_team_ovr(away_xi)
        
        home_pvp = calculate_pvp_boost(home_xi, away_xi)
        away_pvp = calculate_pvp_boost(away_xi, home_xi)
        
        home_city_bat = calculate_city_deviation(home_xi, city, 'batting')
        away_city_bat = calculate_city_deviation(away_xi, city, 'batting')
        
        home_city_bowl = calculate_city_deviation(home_xi, city, 'bowling')
        away_city_bowl = calculate_city_deviation(away_xi, city, 'bowling')
        
        home_advantage = 1.03
        
        # Apply dew factor
        if batting_first == 'home':
            home_final_ovr, away_final_ovr = apply_dew_factor(
                home_base_ovr + home_pvp + home_city_bat,
                away_base_ovr + away_pvp + away_city_bat,
                dew_factor
            )
        else:
            away_final_ovr, home_final_ovr = apply_dew_factor(
                away_base_ovr + away_pvp + away_city_bat,
                home_base_ovr + home_pvp + home_city_bat,
                dew_factor
            )
        
        # Prepare model input
        categorical = {
            'home_team': torch.LongTensor([[encode_categorical(home_team, 'home_team')]]).to(device),
            'away_team': torch.LongTensor([[encode_categorical(away_team, 'away_team')]]).to(device),
            'venue': torch.LongTensor([[encode_categorical(venue, 'venue')]]).to(device),
            'toss_winner': torch.LongTensor([[encode_categorical(toss_winner, 'toss_winner')]]).to(device),
            'toss_decision': torch.LongTensor([[encode_categorical(toss_decision, 'toss_decision')]]).to(device),
            'batting_first': torch.LongTensor([[encode_categorical(batting_first, 'batting_first')]]).to(device)
        }
        
        continuous = np.array([[
            home_base_ovr, away_base_ovr,
            home_pvp, away_pvp,
            home_city_bat, away_city_bat,
            home_city_bowl, away_city_bowl,
            home_advantage
        ]])
        continuous = normalize_continuous(continuous)
        continuous = torch.FloatTensor(continuous).to(device)
        
        # Predict
        with torch.no_grad():
            prob_home_win = tab_model(categorical, continuous).item()
        
        # Return response
        return jsonify({
            'win_probability': {
                home_team: round(prob_home_win, 4),
                away_team: round(1 - prob_home_win, 4)
            },
            'team_metrics': {
                home_team: {
                    'base_ovr': round(home_base_ovr, 2),
                    'pvp_boost': round(home_pvp, 2),
                    'city_dev_batting': round(home_city_bat, 2),
                    'city_dev_bowling': round(home_city_bowl, 2),
                    'dew_adjustment': round((home_final_ovr - (home_base_ovr + home_pvp + home_city_bat)), 2) if batting_first == 'home' else 0,
                    'final_ovr': round(home_final_ovr, 2)
                },
                away_team: {
                    'base_ovr': round(away_base_ovr, 2),
                    'pvp_boost': round(away_pvp, 2),
                    'city_dev_batting': round(away_city_bat, 2),
                    'city_dev_bowling': round(away_city_bowl, 2),
                    'dew_adjustment': round((away_final_ovr - (away_base_ovr + away_pvp + away_city_bat)), 2) if batting_first == 'away' else 0,
                    'final_ovr': round(away_final_ovr, 2)
                }
            },
            'match_context': {
                'venue': venue,
                'city': city,
                'dew_factor': dew_factor,
                'batting_first': batting_first,
                'toss_winner': toss_winner,
                'toss_decision': toss_decision
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get team metadata"""
    teams = encoders['home_team']['classes']
    return jsonify({'teams': teams})

@app.route('/api/venues', methods=['GET'])
def get_venues():
    """Get venue metadata"""
    venues = list(venue_to_city.keys())
    return jsonify({'venues': venues, 'mapping': venue_to_city})

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print(f"\n" + "=" * 80)
    print("✅ API READY!")
    print("=" * 80)
    print(f"\nEndpoints:")
    print(f"  GET  /api/health")
    print(f"  POST /api/simulate/pre-match")
    print(f"  GET  /api/teams")
    print(f"  GET  /api/venues")
    print(f"\n🚀 Starting server on http://localhost:5000")
    print("=" * 80)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
