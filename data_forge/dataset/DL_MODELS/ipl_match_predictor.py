"""
IPL Match Prediction Model - Production Ready
Usage: python ipl_predictor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

BATTING_OVR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BATTING_20251129_154628.csv"
BOWLING_OVR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BOWLING_20251129_154628.csv"
TEST_JSON = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\2025\test_matches_10.json"  # Update this
MODEL_PATH = r"best_model.pt"  # After training

# Context files - NOW ENABLED
H2H_CSV = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\team_h2h_matrix_2025.csv"
FORM_CSV = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\team_recent_form_2025.csv"
PVP_JSON = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\gat_enhanced_pvp.json"
PVP_JSON = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\gat_enhanced_pvp.json"

# ============================================================================
# TEAM MAPPING
# ============================================================================

TEAM_MAP = {
    'Chennai Super Kings': 'CSK', 'Delhi Capitals': 'DC', 'Gujarat Titans': 'GT',
    'Kolkata Knight Riders': 'KKR', 'Lucknow Super Giants': 'LSG', 'Mumbai Indians': 'MI',
    'Punjab Kings': 'PK', 'Rajasthan Royals': 'RR', 'Royal Challengers Bengaluru': 'RCB',
    'Royal Challengers Bangalore': 'RCB', 'Sunrisers Hyderabad': 'SRH',
    'CSK': 'CSK', 'DC': 'DC', 'GT': 'GT', 'KKR': 'KKR', 'LSG': 'LSG',
    'MI': 'MI', 'PK': 'PK', 'RR': 'RR', 'RCB': 'RCB', 'SRH': 'SRH'
}

def norm_team(name: str) -> str:
    return TEAM_MAP.get(name, name)


# ============================================================================
# PLAYER LOOKUP
# ============================================================================

class PlayerLookup:
    def __init__(self, bat_csv: str, bowl_csv: str):
        bat = pd.read_csv(bat_csv)
        bowl = pd.read_csv(bowl_csv)
        
        # Fix empty names (debut players)
        bat['Kaggle_Match_Name'] = bat.apply(
            lambda r: r['Player_Name'] if pd.isna(r['Kaggle_Match_Name']) or r['Kaggle_Match_Name'] == '' 
            else r['Kaggle_Match_Name'], axis=1
        )
        bowl['Kaggle_Match_Name'] = bowl.apply(
            lambda r: r['Player_Name'] if pd.isna(r['Kaggle_Match_Name']) or r['Kaggle_Match_Name'] == '' 
            else r['Kaggle_Match_Name'], axis=1
        )
        
        self.p2i = {}
        self.i2p = {}
        self.feat = {}
        
        idx = 1
        for _, r in bat.iterrows():
            n = str(r['Kaggle_Match_Name']).strip()
            if n and n != 'nan' and pd.notna(n) and n not in self.p2i:
                self.p2i[n] = idx
                self.i2p[idx] = n
                self.feat[idx] = [r.get('BASE_OVR', 0), r.get('TOP_ORDER_OVR', 0),
                                  r.get('MIDDLE_ORDER_OVR', 0), r.get('FINISHER_OVR', 0)]
                idx += 1
        
        for _, r in bowl.iterrows():
            n = str(r['Kaggle_Match_Name']).strip()
            if n and n != 'nan' and pd.notna(n) and n not in self.p2i:
                self.p2i[n] = idx
                self.i2p[idx] = n
                self.feat[idx] = [r.get('BASE_OVR', 0), r.get('POWERPLAY_OVR', 0),
                                  r.get('MIDDLE_OVERS_OVR', 0), r.get('DEATH_OVERS_OVR', 0)]
                idx += 1
        
        print(f"✅ Loaded {len(self.p2i)} players")
    
    def get(self, name: str) -> int:
        return self.p2i.get(str(name).strip(), 0)
    
    def get_matrix(self, dim: int = 32) -> torch.Tensor:
        n = len(self.p2i) + 1
        m = torch.zeros(n, dim)
        for i in range(1, n):
            ovr = [x / 100.0 for x in self.feat.get(i, [0, 0, 0, 0])]
            m[i, :4] = torch.tensor(ovr)
            if dim > 4:
                m[i, 4:] = torch.randn(dim - 4) * 0.001
        return m


# ============================================================================
# MODEL
# ============================================================================

class Model(nn.Module):
    def __init__(self, n_players: int, dim: int = 32, heads: int = 2, 
                 hidden: int = 64, drop: float = 0.2, pre: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.emb = nn.Embedding(n_players, dim, padding_idx=0)
        if pre is not None:
            self.emb.weight.data = pre
        
        self.att = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.team_enc = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(drop)
        )
        self.h2h_enc = nn.Sequential(nn.Linear(4, hidden // 2), nn.ReLU(), nn.Dropout(drop))
        self.form_enc = nn.Sequential(nn.Linear(7, hidden // 2), nn.ReLU(), nn.Dropout(drop))
        self.pvp_enc = nn.Sequential(nn.Linear(2, hidden // 4), nn.ReLU(), nn.Dropout(drop))
        
        self.weights = nn.Parameter(torch.ones(4) / 4)  # 4 sources
        
        self.pred = nn.Sequential(
            nn.Linear(hidden + hidden // 2 + hidden // 2 + hidden // 4, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(drop), 
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, a, b, h2h, form, pvp):
        ea, eb = self.emb(a), self.emb(b)
        aa, _ = self.att(ea, ea, ea)
        ab, _ = self.att(eb, eb, eb)
        
        ma = (a == 0).unsqueeze(-1)
        mb = (b == 0).unsqueeze(-1)
        
        ra = (aa * ~ma).sum(1) / (~ma).sum(1).clamp(min=1)
        rb = (ab * ~mb).sum(1) / (~mb).sum(1).clamp(min=1)
        
        ta = self.team_enc(ra)
        tb = self.team_enc(rb)
        diff = ta - tb
        
        h = self.h2h_enc(h2h)
        f = self.form_enc(form)
        p = self.pvp_enc(pvp)
        
        w = F.softmax(self.weights, dim=0)
        comb = torch.cat([diff * w[0], h * w[1], f * w[2], p * w[3]], dim=1)
        
        return self.pred(comb).squeeze(-1), w


# ============================================================================
# PREDICTOR
# ============================================================================

class Predictor:
    def __init__(self, model_path: str, lookup: PlayerLookup, 
                 h2h_csv: Optional[str] = None, form_csv: Optional[str] = None,
                 pvp_json: Optional[str] = None):
        
        self.lookup = lookup
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.h2h = {}
        if h2h_csv and Path(h2h_csv).exists():
            df = pd.read_csv(h2h_csv)
            for _, r in df.iterrows():
                ta, tb = norm_team(r['team_a']), norm_team(r['team_b'])
                self.h2h[(ta, tb)] = {'wr': r['h2h_win_rate'], 'dom': r['h2h_dominance'],
                                      'mar': r['h2h_avg_margin'], 'w': r['h2h_wins']}
                self.h2h[(tb, ta)] = {'wr': 1 - r['h2h_win_rate'], 'dom': -r['h2h_dominance'],
                                      'mar': -r['h2h_avg_margin'], 'w': r['h2h_losses']}
        
        self.form = {}
        if form_csv and Path(form_csv).exists():
            df = pd.read_csv(form_csv)
            for _, r in df.iterrows():
                t = norm_team(r['IPL_Team_2025'])
                self.form[t] = {'l5': r['last_5_win_rate'], 'tr': r.get('form_trend', 0),
                                'l10': r.get('last_10_win_rate', 0.5)}
        
        self.pvp = {}
        if pvp_json and Path(pvp_json).exists():
            try:
                with open(pvp_json) as f:
                    pvp_data = json.load(f)
                for batter, bowlers in pvp_data.items():
                    if not isinstance(bowlers, dict):
                        continue
                    if batter not in self.pvp:
                        self.pvp[batter] = {}
                    for bowler, stats in bowlers.items():
                        if isinstance(stats, dict):
                            self.pvp[batter][bowler] = {
                                'adv': stats.get('advantage_score', 0),
                                'sr': stats.get('strike_rate', 100),
                                'avg': stats.get('batting_avg', 25)
                            }
            except Exception as e:
                print(f"⚠️ PvP load failed: {e}")
                self.pvp = {}
        
        n = len(lookup.p2i) + 1
        pre = lookup.get_matrix(32)
        self.model = Model(n, 32, 2, 64, 0.2, pre).to(self.dev)
        
        ckpt = torch.load(model_path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded: {ckpt.get('val_acc', 0):.2f}% acc")
    
    def predict(self, match: Dict) -> Dict:
        ta = norm_team(match['team_a'])
        tb = norm_team(match['team_b'])
        
        xa = [self.lookup.get(p) for p in match['team_a_xi']] + [0] * (11 - len(match['team_a_xi']))
        xb = [self.lookup.get(p) for p in match['team_b_xi']] + [0] * (11 - len(match['team_b_xi']))
        
        h2h = self.h2h.get((ta, tb), {'wr': 0.5, 'dom': 0, 'mar': 0, 'w': 0})
        fa = self.form.get(ta, {'l5': 0.5, 'tr': 0, 'l10': 0.5})
        fb = self.form.get(tb, {'l5': 0.5, 'tr': 0, 'l10': 0.5})
        
        hf = [h2h['wr'], h2h['dom'], h2h['mar'] / 50, float(h2h['w'])]
        ff = [fa['l5'], fa['tr'], fa['l10'], fb['l5'], fb['tr'], fb['l10'], fa['l5'] - fb['l5']]
        
        # Compute PvP advantage
        pvp_score = 0.0
        pvp_count = 0
        if self.pvp:
            for batter_name in match['team_a_xi']:
                for bowler_name in match['team_b_xi']:
                    if batter_name in self.pvp and bowler_name in self.pvp[batter_name]:
                        pvp_score += self.pvp[batter_name][bowler_name]['adv']
                        pvp_count += 1
        
        pvp_avg = (pvp_score / pvp_count) if pvp_count > 0 else 0.0
        pf = [pvp_avg, float(pvp_count) / 121.0]
        
        at = torch.tensor([xa], dtype=torch.long).to(self.dev)
        bt = torch.tensor([xb], dtype=torch.long).to(self.dev)
        ht = torch.tensor([hf], dtype=torch.float32).to(self.dev)
        ft = torch.tensor([ff], dtype=torch.float32).to(self.dev)
        pt = torch.tensor([pf], dtype=torch.float32).to(self.dev)
        
        with torch.no_grad():
            logits, w = self.model(at, bt, ht, ft, pt)
            pa = torch.sigmoid(logits).item()
        
        return {
            'team_a': ta, 'team_b': tb, 'prob_a': pa, 'prob_b': 1 - pa,
            'winner': ta if pa > 0.5 else tb, 'conf': max(pa, 1 - pa),
            'w_ovr': w[0].item(), 'w_h2h': w[1].item(), 
            'w_form': w[2].item(), 'w_pvp': w[3].item()
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("IPL MATCH PREDICTION")
    print("="*80)
    
    print("\n[1/3] Loading players...")
    lookup = PlayerLookup(BATTING_OVR, BOWLING_OVR)
    
    print("\n[2/3] Loading model...")
    pred = Predictor(MODEL_PATH, lookup, H2H_CSV, FORM_CSV, PVP_JSON)
    
    print("\n[3/3] Predicting...")
    with open(TEST_JSON) as f:
        matches = json.load(f)
    
    correct = 0
    total = 0
    
    for k, m in matches.items():
        print(f"\n{'='*80}")
        print(f"{k.upper()}: {m['team_a']} vs {m['team_b']}")
        
        r = pred.predict(m)
        
        print(f"\n📊 {r['team_a']}: {r['prob_a']*100:.1f}% | {r['team_b']}: {r['prob_b']*100:.1f}%")
        print(f"🏆 Winner: {r['winner']} ({r['conf']*100:.1f}% conf)")
        print(f"🔍 OVR={r['w_ovr']*100:.0f}% H2H={r['w_h2h']*100:.0f}% Form={r['w_form']*100:.0f}% PvP={r['w_pvp']*100:.0f}%")
        
        actual = norm_team(m.get('winner', ''))
        is_correct = False
        if actual:
            ok = (r['winner'] == actual)
            is_correct = ok
            correct += int(ok)
            total += 1
            print(f"{'✅' if ok else '❌'} Actual: {actual}")
    
    if total:
        print(f"\n{'='*80}")
        print(f"ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()