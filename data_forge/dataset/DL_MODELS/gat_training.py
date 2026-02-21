"""
GAT Training for IPL PvP Prediction
Version: 1.0
Date: Nov 30, 2025

Graph Structure:
- Nodes: 108 players (56 batters + 65 bowlers)
- Edges: 482 PvP matchups
- Task: Learn enhanced PvP advantages for team-level predictions
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from datetime import datetime
import os
from typing import Tuple, Dict, Any

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ImportError:
    print("Warning: sklearn not found. Install with: pip install scikit-learn")
    # Create a simple fallback scaler
    class StandardScaler:  # type: ignore
        def fit_transform(self, X):
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            return (X - self.mean_) / (self.std_ + 1e-8)
        
        def transform(self, X):
            return (X - self.mean_) / (self.std_ + 1e-8)

print("=" * 80)
print("GAT TRAINING - IPL PVP PREDICTION")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# CRITICAL: Update these paths for your Windows system
BASE_DIR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset"

CONFIG = {
    # Input files - CORRECTED PATHS
    'pvp_file': rf"{BASE_DIR}\pvp\PVP_RAW_STATS_20251130_222935.json",
    'batting_ovr': rf"{BASE_DIR}\OVR\PRE_OVR_BATTING_20251129_154628.csv",
    'bowling_ovr': rf"{BASE_DIR}\OVR\PRE_OVR_BOWLING_20251129_154628.csv",
    'batting_master': rf"{BASE_DIR}\Master_Datasets\BATTING_MASTER_2025_20251129_052900.csv",
    'bowling_master': rf"{BASE_DIR}\Master_Datasets\BOWLING_MASTER_2025_20251129_052900.csv",
    
    # GAT Architecture
    'node_features': 11,
    'edge_features': 4,
    'hidden_dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.3,
    
    # Training
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'train_split': 0.8,
    
    # Output
    'output_dir': rf"{BASE_DIR}\DL_MODELS\outputs",
    'model_name': 'gat_pvp_model.pt',
    'predictions_name': 'gat_enhanced_pvp.json'
}

# Ensure output directory exists
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# GPU Setup
print("\n🖥️  HARDWARE DETECTION:")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"  ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ PyTorch Version: {torch.__version__}")
    print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  🚀 Using GPU for training!")
else:
    device = torch.device('cpu')
    print(f"  ⚠️  CUDA Not Available")
    print(f"  💻 Using CPU (will be slower)")
    print(f"  💡 To use GPU, install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print(f"\n📁 FILE PATHS:")
print(f"  PvP Dataset: {CONFIG['pvp_file']}")
print(f"  Batting OVR: {CONFIG['batting_ovr']}")
print(f"  Bowling OVR: {CONFIG['bowling_ovr']}")
print(f"  Output Dir: {CONFIG['output_dir']}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n" + "=" * 80)
print("STEP 1: LOADING DATA FILES")
print("=" * 80)

# Load PvP edges
print(f"\n📂 Loading PvP Dataset...")
print(f"   Path: {CONFIG['pvp_file']}")
try:
    with open(CONFIG['pvp_file'], 'r') as f:
        pvp_data = json.load(f)
    print(f"   ✅ SUCCESS: Loaded {len(pvp_data['pvp_edges'])} PvP edges")
    print(f"   ✅ Version: {pvp_data['metadata'].get('version', 'N/A')}")
    print(f"   ✅ Batters: {pvp_data['metadata']['players_covered']['batters']}")
    print(f"   ✅ Bowlers: {pvp_data['metadata']['players_covered']['bowlers']}")
except FileNotFoundError:
    print(f"   ❌ ERROR: File not found!")
    print(f"   💡 Check that file exists at: {CONFIG['pvp_file']}")
    exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# Load OVRs
print(f"\n📊 Loading Batting OVRs...")
print(f"   Path: {CONFIG['batting_ovr']}")
try:
    batting_ovr = pd.read_csv(CONFIG['batting_ovr'])
    print(f"   ✅ SUCCESS: {len(batting_ovr)} batters loaded")
    print(f"   ✅ Columns: {list(batting_ovr.columns[:5])}...")
except FileNotFoundError:
    print(f"   ❌ ERROR: File not found!")
    print(f"   💡 Check that file exists at: {CONFIG['batting_ovr']}")
    exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

print(f"\n📊 Loading Bowling OVRs...")
print(f"   Path: {CONFIG['bowling_ovr']}")
try:
    bowling_ovr = pd.read_csv(CONFIG['bowling_ovr'])
    print(f"   ✅ SUCCESS: {len(bowling_ovr)} bowlers loaded")
    print(f"   ✅ Columns: {list(bowling_ovr.columns[:5])}...")
except FileNotFoundError:
    print(f"   ❌ ERROR: File not found!")
    print(f"   💡 Check that file exists at: {CONFIG['bowling_ovr']}")
    exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# Load MASTER files for innings data
print(f"\n📊 Loading Batting MASTER (for innings data)...")
print(f"   Path: {CONFIG['batting_master']}")
try:
    batting_master = pd.read_csv(CONFIG['batting_master'])
    print(f"   ✅ SUCCESS: {len(batting_master)} batters loaded")
except FileNotFoundError:
    print(f"   ❌ ERROR: File not found!")
    print(f"   💡 Check that file exists at: {CONFIG['batting_master']}")
    exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

print(f"\n📊 Loading Bowling MASTER (for innings data)...")
print(f"   Path: {CONFIG['bowling_master']}")
try:
    bowling_master = pd.read_csv(CONFIG['bowling_master'])
    print(f"   ✅ SUCCESS: {len(bowling_master)} bowlers loaded")
except FileNotFoundError:
    print(f"   ❌ ERROR: File not found!")
    print(f"   💡 Check that file exists at: {CONFIG['bowling_master']}")
    exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

print(f"\n✅ ALL DATA FILES LOADED SUCCESSFULLY!")

# ============================================================================
# BUILD NODE INDEX & FEATURES
# ============================================================================

print(f"\n" + "=" * 80)
print("STEP 2: BUILDING GRAPH STRUCTURE")
print("=" * 80)

print(f"\n🔨 Extracting unique players from PvP edges...")

# Extract unique players from PvP edges
all_batters = set()
all_bowlers = set()

for edge in pvp_data['pvp_edges']:
    all_batters.add(edge['batter']['kaggle_name'])
    all_bowlers.add(edge['bowler']['kaggle_name'])

print(f"   ✅ Found {len(all_batters)} unique batters")
print(f"   ✅ Found {len(all_bowlers)} unique bowlers")

# Create node index mapping
print(f"\n🔢 Creating node index mapping...")
node_to_idx = {}
idx_to_node = {}
current_idx = 0

# Add batters first
for batter in sorted(all_batters):
    node_to_idx[batter] = current_idx
    idx_to_node[current_idx] = {'name': batter, 'type': 'batter'}
    current_idx += 1

print(f"   ✅ Indexed {current_idx} batters (nodes 0-{current_idx-1})")

# Add bowlers
bowlers_start = current_idx
for bowler in sorted(all_bowlers):
    if bowler not in node_to_idx:  # Some bowlers might also bat
        node_to_idx[bowler] = current_idx
        idx_to_node[current_idx] = {'name': bowler, 'type': 'bowler'}
        current_idx += 1

print(f"   ✅ Indexed {current_idx - bowlers_start} bowlers (nodes {bowlers_start}-{current_idx-1})")

num_nodes = len(node_to_idx)
print(f"\n✅ Total Graph Nodes: {num_nodes}")

# ============================================================================
# EXTRACT NODE FEATURES
# ============================================================================

print(f"\n🎯 Preparing OVR and innings lookups...")

# OVR lookups from PRE_OVR files
batting_ovr['kaggle_lower'] = batting_ovr['Kaggle_Match_Name'].str.lower()
bowling_ovr['kaggle_lower'] = bowling_ovr['Kaggle_Match_Name'].str.lower()

batting_ovr_map = batting_ovr.set_index('kaggle_lower')['BASE_OVR'].to_dict()
bowling_ovr_map = bowling_ovr.set_index('kaggle_lower')['BASE_OVR'].to_dict()

batting_type_map = batting_ovr.set_index('kaggle_lower')['Player_Type'].to_dict()
bowling_type_map = bowling_ovr.set_index('kaggle_lower')['Player_Type'].to_dict()

# Innings lookups from MASTER files
batting_master['kaggle_lower'] = batting_master['Kaggle_Match_Name'].str.lower()
bowling_master['kaggle_lower'] = bowling_master['Kaggle_Match_Name'].str.lower()

batting_innings_map = batting_master.set_index('kaggle_lower')['Total_Innings'].to_dict()
bowling_innings_map = bowling_master.set_index('kaggle_lower')['Total_Innings_Bowled'].to_dict()

print(f"   ✅ Batting OVR lookup: {len(batting_ovr_map)} players")
print(f"   ✅ Bowling OVR lookup: {len(bowling_ovr_map)} players")
print(f"   ✅ Batting innings lookup: {len(batting_innings_map)} players")
print(f"   ✅ Bowling innings lookup: {len(bowling_innings_map)} players")

# Player type encodings
batting_types = ['Batter', 'All-Rounder', 'WK-Batter']
bowling_types = ['Bowler', 'PACE', 'SPIN', 'PACE_AR', 'SPIN_AR']

def encode_player_type(player_type, is_batter):
    """One-hot encode player type"""
    if is_batter:
        encoding = [1 if player_type == t else 0 for t in batting_types]
        encoding.extend([0] * len(bowling_types))  # Pad with zeros
    else:
        encoding = [0] * len(batting_types)
        encoding.extend([1 if player_type == t else 0 for t in bowling_types])
    return encoding

# Build node feature matrix
node_features = []

for node_idx in range(num_nodes):
    node_info = idx_to_node[node_idx]
    node_name = node_info['name']
    node_type = node_info['type']
    
    # Get BASE_OVR
    if node_type == 'batter':
        base_ovr = batting_ovr_map.get(node_name, 70.0)
        player_type = batting_type_map.get(node_name, 'Batter')
        innings = batting_innings_map.get(node_name, 0.0)
        node_type_flag = 0.0
    else:
        base_ovr = bowling_ovr_map.get(node_name, 70.0)
        player_type = bowling_type_map.get(node_name, 'Bowler')
        innings = bowling_innings_map.get(node_name, 0.0)
        node_type_flag = 1.0
    
    # Encode player type
    type_encoding = encode_player_type(player_type, node_type == 'batter')
    
    # Normalize innings (log scale)
    innings_norm = np.log1p(innings) / 5.0  # Max ~5 for 150 innings
    
    # Concatenate features
    features = [base_ovr] + type_encoding + [innings_norm, node_type_flag]
    node_features.append(features)

node_features = np.array(node_features, dtype=np.float32)

# Normalize continuous features (BASE_OVR and innings)
scaler = StandardScaler()
node_features[:, [0, -2]] = scaler.fit_transform(node_features[:, [0, -2]])

print(f"  Node Features Shape: {node_features.shape}")
print(f"  Features per node: {node_features.shape[1]}")

# ============================================================================
# BUILD EDGE INDEX & FEATURES
# ============================================================================

print(f"\n🔗 BUILDING EDGES...")

edge_index = []
edge_features = []
edge_labels = []  # Ground truth advantages

for edge in pvp_data['pvp_edges']:
    batter_name = edge['batter']['kaggle_name']
    bowler_name = edge['bowler']['kaggle_name']
    
    # Get node indices
    batter_idx = node_to_idx[batter_name]
    bowler_idx = node_to_idx[bowler_name]
    
    # Edge: batter → bowler
    edge_index.append([batter_idx, bowler_idx])
    
    # Edge features
    advantage = edge['advantage_metrics']['batter_advantage']
    confidence = edge['advantage_metrics']['confidence_weight']
    balls = edge['raw_statistics']['balls_faced']
    dismissals = edge['raw_statistics']['dismissals']
    
    # Normalize balls and dismissals
    balls_norm = min(balls / 150.0, 1.0)  # Max 150 balls
    dismissals_norm = min(dismissals / 10.0, 1.0)  # Max 10 dismissals
    
    edge_features.append([advantage, confidence, balls_norm, dismissals_norm])
    edge_labels.append(advantage)  # Target: predict enhanced advantage

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_features = torch.tensor(edge_features, dtype=torch.float)
edge_labels = torch.tensor(edge_labels, dtype=torch.float)

print(f"  Total Edges: {edge_index.shape[1]}")
print(f"  Edge Features Shape: {edge_features.shape}")

# ============================================================================
# CREATE PYTORCH GEOMETRIC GRAPH
# ============================================================================

print(f"\n🎨 CREATING GRAPH...")

x = torch.tensor(node_features, dtype=torch.float)

graph = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_features,
    y=edge_labels
)

print(f"  Graph: {graph}")

# ============================================================================
# GAT MODEL DEFINITION
# ============================================================================

print(f"\n🧠 DEFINING GAT MODEL...")

class IPL_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, num_heads, dropout=0.3):
        super(IPL_GAT, self).__init__()
        
        # Layer 1: Node features → Hidden
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=num_heads, 
            edge_dim=edge_dim,
            dropout=dropout
        )
        
        # Layer 2: Hidden → Hidden (with multi-head)
        self.conv2 = GATConv(
            hidden_channels * num_heads, 
            hidden_channels, 
            heads=num_heads, 
            edge_dim=edge_dim,
            dropout=dropout,
            concat=False  # Average heads in final layer
        )
        
        # Edge predictor: Concatenated node embeddings → Advantage
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Edge prediction
        row, col = edge_index
        # Concatenate: source node + target node + edge features
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)
        enhanced_advantage = self.edge_predictor(edge_emb).squeeze()
        
        return enhanced_advantage, x

model = IPL_GAT(
    in_channels=CONFIG['node_features'],
    hidden_channels=CONFIG['hidden_dim'],
    edge_dim=CONFIG['edge_features'],
    num_heads=CONFIG['num_heads'],
    dropout=CONFIG['dropout']
).to(device)

print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print(f"\n🏋️ TRAINING SETUP...")

# Split edges for train/val
num_edges = edge_index.shape[1]
num_train = int(num_edges * CONFIG['train_split'])

indices = torch.randperm(num_edges)
train_idx = indices[:num_train]
val_idx = indices[num_train:]

print(f"  Train Edges: {len(train_idx)}")
print(f"  Val Edges: {len(val_idx)}")

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

criterion = torch.nn.MSELoss()

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"\n🚀 STARTING TRAINING...")
print("=" * 80)

graph = graph.to(device)

best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(CONFIG['epochs']):
    # Train
    model.train()
    optimizer.zero_grad()
    
    pred_advantage, node_embeddings = model(graph.x, graph.edge_index, graph.edge_attr)
    
    # Type-safe tensor indexing
    train_loss = criterion(
        pred_advantage[train_idx.to(device)],  # type: ignore
        graph.y[train_idx.to(device)]  # type: ignore
    )
    train_loss.backward()
    optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        pred_advantage, _ = model(graph.x, graph.edge_index, graph.edge_attr)
        val_loss = criterion(
            pred_advantage[val_idx.to(device)],  # type: ignore
            graph.y[val_idx.to(device)]  # type: ignore
        )
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state': model.state_dict(),
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node,
            'scaler': scaler,
            'config': CONFIG
        }, os.path.join(CONFIG['output_dir'], CONFIG['model_name']))
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    if patience_counter >= patience:
        print(f"\n⏹️  Early stopping at epoch {epoch+1}")
        break

print("=" * 80)
print(f"✅ TRAINING COMPLETE!")
print(f"   Best Val Loss: {best_val_loss:.4f}")

# ============================================================================
# GENERATE ENHANCED PVP PREDICTIONS
# ============================================================================

print(f"\n📊 GENERATING ENHANCED PVP PREDICTIONS...")

# Load best model
checkpoint = torch.load(os.path.join(CONFIG['output_dir'], CONFIG['model_name']))
model.load_state_dict(checkpoint['model_state'])

model.eval()
with torch.no_grad():
    enhanced_advantages, final_node_embeddings = model(graph.x, graph.edge_index, graph.edge_attr)

enhanced_advantages = enhanced_advantages.cpu().numpy()

# ============================================================================
# SAVE ENHANCED PVP DATASET
# ============================================================================

print(f"\n💾 SAVING ENHANCED PVP DATASET...")

enhanced_pvp_edges = []

for i, edge in enumerate(pvp_data['pvp_edges']):
    enhanced_edge = edge.copy()
    
    # Add GAT predictions
    enhanced_edge['gat_enhanced'] = {
        'original_advantage': float(edge['advantage_metrics']['batter_advantage']),
        'gat_advantage': float(enhanced_advantages[i]),
        'delta': float(enhanced_advantages[i] - edge['advantage_metrics']['batter_advantage'])
    }
    
    enhanced_pvp_edges.append(enhanced_edge)

# Create output JSON
output_data = {
    'metadata': pvp_data['metadata'].copy(),
    'gat_training': {
        'timestamp': datetime.now().isoformat(),
        'model_config': CONFIG,
        'training_metrics': {
            'best_val_loss': float(best_val_loss),
            'train_edges': int(len(train_idx)),
            'val_edges': int(len(val_idx))
        }
    },
    'pvp_edges': enhanced_pvp_edges
}

output_data['metadata']['version'] = '3.0_GAT_Enhanced'

output_path = os.path.join(CONFIG['output_dir'], CONFIG['predictions_name'])
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"  ✅ Saved: {output_path}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n📈 ENHANCEMENT STATISTICS:")
original_adv = np.array([e['advantage_metrics']['batter_advantage'] for e in pvp_data['pvp_edges']])
gat_adv = enhanced_advantages

print(f"  Original Advantage:")
print(f"    Mean: {original_adv.mean():.3f}, Std: {original_adv.std():.3f}")
print(f"    Range: [{original_adv.min():.3f}, {original_adv.max():.3f}]")

print(f"\n  GAT Enhanced Advantage:")
print(f"    Mean: {gat_adv.mean():.3f}, Std: {gat_adv.std():.3f}")
print(f"    Range: [{gat_adv.min():.3f}, {gat_adv.max():.3f}]")

delta = gat_adv - original_adv
print(f"\n  Delta (GAT - Original):")
print(f"    Mean: {delta.mean():.3f}, Std: {delta.std():.3f}")
print(f"    Range: [{delta.min():.3f}, {delta.max():.3f}]")

# Top adjustments
top_positive = np.argsort(delta)[-5:][::-1]
top_negative = np.argsort(delta)[:5]

print(f"\n  Top 5 Positive Adjustments (GAT increased advantage):")
for idx in top_positive:
    edge = pvp_data['pvp_edges'][idx]
    print(f"    {edge['batter']['name']} vs {edge['bowler']['name']}: "
          f"{original_adv[idx]:+.3f} → {gat_adv[idx]:+.3f} (Δ {delta[idx]:+.3f})")

print(f"\n  Top 5 Negative Adjustments (GAT decreased advantage):")
for idx in top_negative:
    edge = pvp_data['pvp_edges'][idx]
    print(f"    {edge['batter']['name']} vs {edge['bowler']['name']}: "
          f"{original_adv[idx]:+.3f} → {gat_adv[idx]:+.3f} (Δ {delta[idx]:+.3f})")

print("\n" + "=" * 80)
print("✅ GAT TRAINING COMPLETE!")
print("=" * 80)
print(f"\nOutputs:")
print(f"  1. Model: {os.path.join(CONFIG['output_dir'], CONFIG['model_name'])}")
print(f"  2. Enhanced PvP: {output_path}")
print(f"\nNext Step: Use enhanced PvP in TabTransformer for match prediction!")