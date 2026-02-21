"""
Training Script - IPL Match Prediction
Usage: python train_ipl.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path

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
    
    def get_matrix(self, dim: int = 32):
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
                 hidden: int = 64, drop: float = 0.2, pre = None):
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
        
        self.weights = nn.Parameter(torch.ones(4) / 4)  # 4 sources now
        
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
# CONFIGURATION - UPDATE THESE
# ============================================================================

BATTING_OVR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BATTING_20251129_154628.csv"
BOWLING_OVR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BOWLING_20251129_154628.csv"
TRAIN_JSON = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\training_matches_with_xi_20251201_053037.json"

# Context files - NOW ENABLED
H2H_CSV = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\team_h2h_matrix_2025.csv"
FORM_CSV = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\team_recent_form_2025.csv"
PVP_JSON = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\gat_enhanced_pvp.json"

# Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LR = 0.0001  # Lower LR (was 0.0003)
TRAIN_SPLIT = 0.8
DROPOUT = 0.3  # Higher dropout (was 0.2)
WEIGHT_DECAY = 5e-5  # Stronger regularization (was 1e-5)

# ============================================================================
# DATASET
# ============================================================================

class MatchDataset(Dataset):
    def __init__(self, matches, lookup, h2h_dict, form_dict, pvp_dict=None):
        self.matches = matches
        self.lookup = lookup
        self.h2h = h2h_dict
        self.form = form_dict
        self.pvp = pvp_dict if pvp_dict else {}
    
    def __len__(self):
        return len(self.matches)
    
    def __getitem__(self, idx):
        m = self.matches[idx]
        
        xa = [self.lookup.get(p) for p in m['team_a_xi']] + [0] * (11 - len(m['team_a_xi']))
        xb = [self.lookup.get(p) for p in m['team_b_xi']] + [0] * (11 - len(m['team_b_xi']))
        
        ta, tb = norm_team(m['team_a']), norm_team(m['team_b'])
        
        h2h = self.h2h.get((ta, tb), {'wr': 0.5, 'dom': 0, 'mar': 0, 'w': 0})
        fa = self.form.get(ta, {'l5': 0.5, 'tr': 0, 'l10': 0.5})
        fb = self.form.get(tb, {'l5': 0.5, 'tr': 0, 'l10': 0.5})
        
        hf = [h2h['wr'], h2h['dom'], h2h['mar'] / 50, float(h2h['w'])]
        ff = [fa['l5'], fa['tr'], fa['l10'], fb['l5'], fb['tr'], fb['l10'], fa['l5'] - fb['l5']]
        
        # Compute PvP advantage (team_a batters vs team_b bowlers)
        pvp_score = 0.0
        pvp_count = 0
        if self.pvp:
            for batter_name in m['team_a_xi']:
                for bowler_name in m['team_b_xi']:
                    if batter_name in self.pvp and bowler_name in self.pvp[batter_name]:
                        pvp_score += self.pvp[batter_name][bowler_name]['adv']
                        pvp_count += 1
        
        pvp_avg = (pvp_score / pvp_count) if pvp_count > 0 else 0.0
        pf = [pvp_avg, float(pvp_count) / 121.0]  # Normalized count (max 11x11=121)
        
        label = 1.0 if m['winner'] == m['team_a'] else 0.0
        
        return {
            'a': torch.tensor(xa, dtype=torch.long),
            'b': torch.tensor(xb, dtype=torch.long),
            'h2h': torch.tensor(hf, dtype=torch.float32),
            'form': torch.tensor(ff, dtype=torch.float32),
            'pvp': torch.tensor(pf, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)  # Scalar, not [label]
        }


# ============================================================================
# TRAINING
# ============================================================================

def train():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    
    # Load
    print("\n[1/5] Loading players...")
    lookup = PlayerLookup(BATTING_OVR, BOWLING_OVR)
    
    print("\n[2/5] Loading matches...")
    with open(TRAIN_JSON) as f:
        matches = json.load(f)
    print(f"✅ {len(matches)} matches")
    
    # Context
    print("\n[3/5] Loading context...")
    h2h_dict = {}
    if H2H_CSV:
        df = pd.read_csv(H2H_CSV)
        for _, r in df.iterrows():
            ta, tb = norm_team(r['team_a']), norm_team(r['team_b'])
            h2h_dict[(ta, tb)] = {'wr': r['h2h_win_rate'], 'dom': r['h2h_dominance'],
                                   'mar': r['h2h_avg_margin'], 'w': r['h2h_wins']}
            h2h_dict[(tb, ta)] = {'wr': 1 - r['h2h_win_rate'], 'dom': -r['h2h_dominance'],
                                   'mar': -r['h2h_avg_margin'], 'w': r['h2h_losses']}
        print(f"✅ Loaded H2H: {len(df)} pairs")
    
    form_dict = {}
    if FORM_CSV:
        df = pd.read_csv(FORM_CSV)
        for _, r in df.iterrows():
            t = norm_team(r['IPL_Team_2025'])
            form_dict[t] = {'l5': r['last_5_win_rate'], 'tr': r.get('form_trend', 0),
                            'l10': r.get('last_10_win_rate', 0.5)}
        print(f"✅ Loaded Form: {len(df)} teams")
    
    # Load PvP data
    pvp_dict = {}
    if PVP_JSON and Path(PVP_JSON).exists():
        try:
            with open(PVP_JSON) as f:
                pvp_data = json.load(f)
            
            # Build batter vs bowler lookup
            for batter, bowlers in pvp_data.items():
                if not isinstance(bowlers, dict):
                    continue  # Skip if not nested dict
                if batter not in pvp_dict:
                    pvp_dict[batter] = {}
                for bowler, stats in bowlers.items():
                    if isinstance(stats, dict):
                        pvp_dict[batter][bowler] = {
                            'adv': stats.get('advantage_score', 0),  # -1 to 1
                            'sr': stats.get('strike_rate', 100),
                            'avg': stats.get('batting_avg', 25)
                        }
            print(f"✅ Loaded PvP: {len(pvp_dict)} batters")
        except Exception as e:
            print(f"⚠️ PvP load failed: {e}")
            pvp_dict = {}
    
    # Dataset
    print("\n[4/5] Creating dataset...")
    ds = MatchDataset(matches, lookup, h2h_dict, form_dict, pvp_dict)
    
    train_sz = int(TRAIN_SPLIT * len(ds))
    val_sz = len(ds) - train_sz
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_sz, val_sz])
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Train: {train_sz} | Val: {val_sz}")
    
    # Model
    print("\n[5/5] Model...")
    n = len(lookup.p2i) + 1
    pre = lookup.get_matrix(32)
    model = Model(n, 32, 2, 64, DROPOUT, pre).to(dev)  # Use DROPOUT variable
    print(f"✅ Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'weights' not in n], 'lr': LR},
        {'params': model.weights, 'lr': LR * 10}
    ], weight_decay=WEIGHT_DECAY)  # Use WEIGHT_DECAY variable
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=15)
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best = 0.0
    patience = 0
    
    for ep in range(EPOCHS):
        # Train
        model.train()
        t_loss = 0
        t_corr = 0
        t_tot = 0
        
        for batch in train_dl:
            a = batch['a'].to(dev)
            b = batch['b'].to(dev)
            h2h = batch['h2h'].to(dev)
            form = batch['form'].to(dev)
            pvp = batch['pvp'].to(dev)
            lbl = batch['label'].to(dev)
            
            opt.zero_grad()
            logits, _ = model(a, b, h2h, form, pvp)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            t_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            t_corr += (pred == lbl).sum().item()
            t_tot += lbl.size(0)
        
        t_loss /= len(train_dl)
        t_acc = 100 * t_corr / t_tot
        
        # Val
        model.eval()
        v_corr = 0
        v_tot = 0
        v_probs = []
        v_lbls = []
        
        with torch.no_grad():
            for batch in val_dl:
                a = batch['a'].to(dev)
                b = batch['b'].to(dev)
                h2h = batch['h2h'].to(dev)
                form = batch['form'].to(dev)
                pvp = batch['pvp'].to(dev)
                lbl = batch['label'].to(dev)
                
                logits, _ = model(a, b, h2h, form, pvp)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()
                
                v_corr += (pred == lbl).sum().item()
                v_tot += lbl.size(0)
                v_probs.extend(prob.cpu().numpy())
                v_lbls.extend(lbl.cpu().numpy())
        
        v_acc = 100 * v_corr / v_tot
        brier = np.mean([(p - l)**2 for p, l in zip(v_probs, v_lbls)])
        
        sched.step(v_acc)
        
        if (ep + 1) % 10 == 0:
            w = torch.softmax(model.weights, dim=0).detach().cpu().numpy()
            print(f"\nEpoch {ep+1}/{EPOCHS}")
            print(f"  Train: {t_loss:.4f} loss, {t_acc:.2f}% acc")
            print(f"  Val: {v_acc:.2f}% acc, {brier:.4f} brier")
            print(f"  Weights: OVR={w[0]*100:.0f}% H2H={w[1]*100:.0f}% Form={w[2]*100:.0f}% PvP={w[3]*100:.0f}%")
        
        if v_acc > best:
            best = v_acc
            patience = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': v_acc,
                'brier_score': brier
            }, 'best_model.pt')
            if (ep + 1) % 10 == 0:
                print(f"  ✅ Saved")
        else:
            patience += 1
        
        if patience >= 30:
            print(f"\nEarly stop at epoch {ep+1}")
            break
    
    print("\n" + "="*80)
    print(f"DONE - Best: {best:.2f}%")
    print("Model: best_model.pt")
    print("="*80)


if __name__ == "__main__":
    train()