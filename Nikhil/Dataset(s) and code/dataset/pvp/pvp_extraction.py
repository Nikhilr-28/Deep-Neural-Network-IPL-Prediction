"""
PVP (Player vs Player) Extraction Script
==========================================
El Dorado Project - CSCI 566

Purpose: Extract batter vs bowler head-to-head statistics from ball-by-ball data
Data Range: 2021-2024 IPL seasons
Output: PVP_RAW_STATS.json (foundation for GAT training)

Rules:
  - Create PvP edge IF:
    * Dismissals ≥ 3, OR
    * Strike Rate > 145 (vs PACE/MEDIUM bowlers), OR
    * Strike Rate > 140 (vs SPIN bowlers)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect base path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Try to find data files
possible_bases = [
    Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil"),
    project_root / "Kaggle_download",
    Path("./data"),
]

DATA_BASE = None
for base in possible_bases:
    if (base / "Kaggle_download" / "rajsengo" / "all_season_details.csv").exists():
        DATA_BASE = base
        break
    elif (base / "rajsengo" / "all_season_details.csv").exists():
        DATA_BASE = base / "Kaggle_download"
        break

if DATA_BASE is None:
    print("❌ ERROR: Cannot find data files!")
    print("\nPlease set DATA_BASE manually to the path containing Kaggle_download folder")
    print("Or place season_details.csv files in ./data/ folder")
    sys.exit(1)

# Data paths
DATA_2021_2023 = DATA_BASE / "Kaggle_download" / "rajsengo" / "all_season_details.csv"
DATA_2024 = DATA_BASE / "Kaggle_download" / "rajsengo" / "2024" / "season_details.csv"

# Master lists
MASTER_BASE = DATA_BASE / "Dataset(s) and code" / "dataset" / "Master_Datasets"
BATTING_MASTER = MASTER_BASE / "BATTING_MASTER_2025_20251129_052900.csv"
BOWLING_MASTER = MASTER_BASE / "BOWLING_MASTER_2025_20251129_052900.csv"
BOWLER_TYPE_MAP = DATA_BASE / "Dataset(s) and code" / "dataset" / "stadium_matching" / "BOWLER_TYPE_MAPPING_COMPLETE_170.csv"

# Output
OUTPUT_DIR = script_dir
OUTPUT_JSON = OUTPUT_DIR / f"PVP_RAW_STATS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Thresholds
MIN_BALLS = 20  # Minimum balls for ANY PvP edge (strict quality filter)
MIN_DISMISSALS = 3  # Bowler domination via wickets
SR_THRESHOLD_PACE = 145.0  # Batter domination threshold
SR_THRESHOLD_SPIN = 140.0  # Batter domination threshold
MIN_BALLS_FOR_DEVIATION = 30  # Deviation calculation threshold (higher than MIN_BALLS)

print("=" * 80)
print("PVP EXTRACTION - Configuration")
print("=" * 80)
print(f"Data Base: {DATA_BASE}")
print(f"2021-2023: {DATA_2021_2023}")
print(f"2024:      {DATA_2024}")
print(f"Masters:   {MASTER_BASE}")
print(f"Output:    {OUTPUT_JSON}")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_confidence_weight(balls, threshold=30, full_confidence_balls=120):
    """
    Tiered confidence weight system:
    - Below 20: 0.0 (below qualification)
    - 20-29: Linear scale 0.15 to 0.28 (base advantage zone)
    - 30+: Sigmoid curve (standard formula)
    """
    if balls < 20:
        return 0.0
    
    elif balls < 30:
        # Linear interpolation for base advantage zone
        # 20 balls = 0.15, 29 balls = 0.28
        confidence = 0.15 + (balls - 20) * 0.014
        return round(confidence, 3)
    
    elif balls >= full_confidence_balls:
        return 0.90 + min(0.05, (balls - full_confidence_balls) / 2000)
    
    else:
        # Sigmoid for 30+ balls
        x = (balls - threshold) / (full_confidence_balls - threshold)
        sigmoid = 1 / (1 + np.exp(-10 * (x - 0.5)))
        confidence = 0.3 + (sigmoid * 0.55)
        
        return round(confidence, 3)


def calculate_deviation(actual_sr, overall_sr, baseline_sr):
    """
    Calculate normalized deviation
    Same formula as pace/spin matchup system
    
    Args:
        actual_sr: SR against this specific bowler
        overall_sr: Batter's overall SR
        baseline_sr: Expected SR (145 for PACE, 138 for SPIN)
    """
    if actual_sr == 0 or overall_sr == 0:
        return 0.0
    
    personal_delta = actual_sr - overall_sr
    normalization_factor = baseline_sr / 145.0
    denominator = baseline_sr * normalization_factor
    
    if denominator == 0:
        return 0.0
    
    normalized_delta = personal_delta / denominator
    deviation = np.clip(normalized_delta, -1.0, 1.0)
    
    return round(deviation, 3)


def determine_matchup_winner(advantage, threshold=0.20):
    """Classify matchup winner based on advantage score"""
    if advantage > threshold:
        return "BATTER"
    elif advantage < -threshold:
        return "BOWLER"
    else:
        return "NEUTRAL"


# ============================================================================
# LOAD MASTER LISTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Loading Master Lists")
print("=" * 80)

try:
    batting_master = pd.read_csv(BATTING_MASTER)
    bowling_master = pd.read_csv(BOWLING_MASTER)
    bowler_types = pd.read_csv(BOWLER_TYPE_MAP)
    
    print(f"✅ Batters:      {len(batting_master)} players")
    print(f"✅ Bowlers:      {len(bowling_master)} players")
    print(f"✅ Bowler Types: {len(bowler_types)} classified")
    
except Exception as e:
    print(f"❌ ERROR loading master lists: {e}")
    sys.exit(1)

# Create lookup dictionaries
batter_names = set(batting_master['Kaggle_Match_Name'].str.strip().str.lower())
bowler_names = set(bowling_master['Kaggle_Match_Name'].str.strip().str.lower())

# Get batter overall SRs
batter_sr_map = dict(zip(
    batting_master['Kaggle_Match_Name'].str.strip().str.lower(),
    batting_master['Strike_Rate']
))

# Create bowler type mapping
bowler_type_map = dict(zip(
    bowler_types['Kaggle_Match_Name'].str.strip().str.lower(),
    bowler_types['Bowler_Type']
))

# Classify as PACE or SPIN
def get_bowler_category(bowler_type):
    """Simplify to PACE or SPIN"""
    if pd.isna(bowler_type):
        return "UNKNOWN"
    bowler_type = str(bowler_type).upper()
    if 'SPIN' in bowler_type:
        return "SPIN"
    elif 'PACE' in bowler_type or 'MEDIUM' in bowler_type:
        return "PACE"
    else:
        return "UNKNOWN"

print("\nBowler type distribution:")
bowler_categories = {name: get_bowler_category(btype) 
                     for name, btype in bowler_type_map.items()}
from collections import Counter
category_counts = Counter(bowler_categories.values())
for cat, count in category_counts.items():
    print(f"  {cat}: {count}")

# ============================================================================
# LOAD BALL-BY-BALL DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Loading Ball-by-Ball Data")
print("=" * 80)

all_balls = []

# Load 2021-2023
try:
    print(f"Loading 2021-2023: {DATA_2021_2023}")
    df_2021_2023 = pd.read_csv(DATA_2021_2023, low_memory=False)
    print(f"  ✅ Loaded {len(df_2021_2023):,} balls")
    all_balls.append(df_2021_2023)
except Exception as e:
    print(f"  ❌ Error: {e}")

# Load 2024
try:
    print(f"Loading 2024: {DATA_2024}")
    df_2024 = pd.read_csv(DATA_2024, low_memory=False)
    print(f"  ✅ Loaded {len(df_2024):,} balls")
    all_balls.append(df_2024)
except Exception as e:
    print(f"  ❌ Error: {e}")

if not all_balls:
    print("❌ ERROR: No data loaded!")
    sys.exit(1)

# Combine all data
df_all = pd.concat(all_balls, ignore_index=True)
print(f"\n✅ Total balls: {len(df_all):,}")

# ============================================================================
# EXTRACT PVP STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Extracting PvP Statistics")
print("=" * 80)

# Map actual columns to our needs
# This dataset uses: batsman1_name, bowler1_name, runs, wicket info

required_cols = ['batsman1_name', 'bowler1_name', 'runs']
missing = [col for col in required_cols if col not in df_all.columns]

if missing:
    print(f"❌ ERROR: Missing columns in data: {missing}")
    print(f"Available columns: {df_all.columns.tolist()}")
    sys.exit(1)

print("Columns found:")
print(f"  Batter (batsman1_name): {df_all['batsman1_name'].dtype}")
print(f"  Bowler (bowler1_name): {df_all['bowler1_name'].dtype}")
print(f"  Runs: {df_all['runs'].dtype}")

# Normalize names
df_all['batter_clean'] = df_all['batsman1_name'].str.strip().str.lower()
df_all['bowler_clean'] = df_all['bowler1_name'].str.strip().str.lower()

# Filter to IPL 2025 players only
print("\nFiltering to IPL 2025 players...")
df_pvp = df_all[
    (df_all['batter_clean'].isin(batter_names)) &
    (df_all['bowler_clean'].isin(bowler_names))
].copy()

print(f"✅ Balls involving IPL 2025 players: {len(df_pvp):,}")

# Determine dismissals
# This dataset has: wicket_id, wkt_batsman_name, wkt_bowler_name
# A dismissal occurs when wicket_id is not null AND bowler gets the wicket
df_pvp['is_dismissal'] = (
    df_pvp['wicket_id'].notna() & 
    (df_pvp['wkt_bowler_name'] == df_pvp['bowler1_name'])
)

# Group by batter-bowler pairs
print("\nAggregating batter-bowler pairs...")

pvp_stats = df_pvp.groupby(['batter_clean', 'bowler_clean']).agg({
    'runs': ['sum', 'count'],
    'is_dismissal': 'sum'
}).reset_index()

pvp_stats.columns = ['batter', 'bowler', 'runs', 'balls', 'dismissals']

print(f"✅ Unique batter-bowler pairs: {len(pvp_stats):,}")

# Calculate SR and average
pvp_stats['strike_rate'] = (pvp_stats['runs'] / pvp_stats['balls'] * 100).round(2)
pvp_stats['batting_average'] = np.where(
    pvp_stats['dismissals'] > 0,
    (pvp_stats['runs'] / pvp_stats['dismissals']).round(2),
    np.nan
)

# ============================================================================
# APPLY QUALIFICATION FILTERS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Applying Qualification Filters")
print("=" * 80)

# Get bowler type for each pair
pvp_stats['bowler_type'] = pvp_stats['bowler'].map(bowler_type_map)
pvp_stats['bowler_category'] = pvp_stats['bowler_type'].apply(get_bowler_category)

# Determine SR thresholds based on bowler type
pvp_stats['sr_threshold'] = pvp_stats['bowler_category'].map({
    'PACE': SR_THRESHOLD_PACE,
    'SPIN': SR_THRESHOLD_SPIN,
    'UNKNOWN': SR_THRESHOLD_PACE  # Default to pace
})

# Two qualification conditions (BOTH require minimum 20 balls)

# Condition 1: Batter Domination (high SR)
pvp_stats['batter_dominates'] = (
    (pvp_stats['balls'] >= MIN_BALLS) &
    (pvp_stats['strike_rate'] > pvp_stats['sr_threshold'])
)

# Condition 2: Bowler Dismissals (wickets)
pvp_stats['bowler_dismisses'] = (
    (pvp_stats['balls'] >= MIN_BALLS) &
    (pvp_stats['dismissals'] >= MIN_DISMISSALS)
)

# Qualify if EITHER condition is met
pvp_stats['qualifies'] = (
    pvp_stats['batter_dominates'] |
    pvp_stats['bowler_dismisses']
)

# Legacy columns for backward compatibility
pvp_stats['meets_wickets'] = pvp_stats['bowler_dismisses']
pvp_stats['meets_sr'] = pvp_stats['batter_dominates']

# Filter to qualified edges only
pvp_qualified = pvp_stats[pvp_stats['qualifies']].copy()

print(f"Before filtering: {len(pvp_stats):,} pairs")
print(f"After filtering:  {len(pvp_qualified):,} pairs (min {MIN_BALLS} balls)")
print(f"\nQualification breakdown:")
print(f"  Batter Domination (SR >{SR_THRESHOLD_PACE}/{SR_THRESHOLD_SPIN}): {pvp_stats['batter_dominates'].sum():,}")
print(f"  Bowler Dismissals (≥{MIN_DISMISSALS} wickets): {pvp_stats['bowler_dismisses'].sum():,}")
print(f"  Both conditions: {(pvp_stats['batter_dominates'] & pvp_stats['bowler_dismisses']).sum():,}")
print(f"\nMinimum balls enforced: {MIN_BALLS}")

# ============================================================================
# CALCULATE DEVIATIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Calculating Deviations")
print("=" * 80)

# Get batter overall SR
pvp_qualified['overall_sr'] = pvp_qualified['batter'].map(batter_sr_map)

# Calculate deviation
deviations = []
for idx, row in pvp_qualified.iterrows():
    actual_sr = row['strike_rate']
    overall_sr = row['overall_sr']
    baseline_sr = row['sr_threshold']
    balls = row['balls']
    batter_dominates = row['batter_dominates']
    bowler_dismisses = row['bowler_dismisses']
    
    # BASE ADVANTAGE SYSTEM (20-29 balls)
    if 20 <= balls < 30:
        if batter_dominates and bowler_dismisses:
            # Both conditions → conflicting signals → neutral
            deviation = 0.0
        elif batter_dominates:
            # Batter dominated with limited sample → base advantage
            deviation = +0.25
        elif bowler_dismisses:
            # Bowler dominated with limited sample → base disadvantage
            deviation = -0.25
        else:
            # Should never happen (qualification logic prevents this)
            deviation = 0.0
    
    # CALCULATED ADVANTAGE (30+ balls)
    else:
        deviation = calculate_deviation(actual_sr, overall_sr, baseline_sr)
    
    deviations.append(deviation)

pvp_qualified['batter_advantage'] = deviations

# Calculate confidence
pvp_qualified['confidence_weight'] = pvp_qualified['balls'].apply(calculate_confidence_weight)

# Determine matchup winner
pvp_qualified['matchup_winner'] = pvp_qualified['batter_advantage'].apply(determine_matchup_winner)

# Qualification reason (simplified - only 2 conditions now)
def get_qualification_reason(row):
    reasons = []
    
    # Check each condition
    if row['batter_dominates']:
        if row['bowler_category'] == 'PACE':
            reasons.append('BATTER_DOM_PACE_SR_145+')
        elif row['bowler_category'] == 'SPIN':
            reasons.append('BATTER_DOM_SPIN_SR_140+')
        else:
            reasons.append('BATTER_DOM_SR_HIGH')
    
    if row['bowler_dismisses']:
        reasons.append('BOWLER_DOM_WICKETS_3+')
    
    # Combine multiple reasons
    if len(reasons) > 1:
        return ' + '.join(reasons)
    elif len(reasons) == 1:
        return reasons[0]
    else:
        return 'UNKNOWN'

pvp_qualified['qualification_reason'] = pvp_qualified.apply(get_qualification_reason, axis=1)

print(f"✅ Deviations calculated for {len(pvp_qualified)} edges")
print(f"\nDeviation stats:")
print(pvp_qualified['batter_advantage'].describe())

print(f"\nMatchup winner distribution:")
print(pvp_qualified['matchup_winner'].value_counts())

# ============================================================================
# MAP BACK TO ORIGINAL NAMES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Mapping to Original Names")
print("=" * 80)

# Create reverse mappings
batter_name_map = dict(zip(
    batting_master['Kaggle_Match_Name'].str.strip().str.lower(),
    batting_master['Player_Name']
))

bowler_name_map = dict(zip(
    bowling_master['Kaggle_Match_Name'].str.strip().str.lower(),
    bowling_master['Player_Name']
))

pvp_qualified['batter_display_name'] = pvp_qualified['batter'].map(batter_name_map)
pvp_qualified['bowler_display_name'] = pvp_qualified['bowler'].map(bowler_name_map)

# ============================================================================
# GENERATE OUTPUT JSON
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Generating Output JSON")
print("=" * 80)

# Build JSON structure
pvp_edges = []

for idx, row in pvp_qualified.iterrows():
    edge = {
        "batter": {
            "name": row['batter_display_name'],
            "kaggle_name": row['batter'],
            "overall_strike_rate": float(row['overall_sr']) if not pd.isna(row['overall_sr']) else None
        },
        "bowler": {
            "name": row['bowler_display_name'],
            "kaggle_name": row['bowler'],
            "type": row['bowler_type'] if not pd.isna(row['bowler_type']) else "UNKNOWN",
            "category": row['bowler_category']
        },
        "raw_statistics": {
            "balls_faced": int(row['balls']),
            "runs_scored": int(row['runs']),
            "dismissals": int(row['dismissals']),
            "strike_rate": float(row['strike_rate']),
            "batting_average": float(row['batting_average']) if not pd.isna(row['batting_average']) else None
        },
        "qualification": {
            "reason": row['qualification_reason'],
            "sr_threshold": float(row['sr_threshold']),
            "meets_wickets": bool(row['meets_wickets']),
            "meets_sr": bool(row['meets_sr'])
        },
        "advantage_metrics": {
            "batter_advantage": float(row['batter_advantage']),
            "confidence_weight": float(row['confidence_weight']),
            "matchup_winner": row['matchup_winner']
        }
    }
    pvp_edges.append(edge)

# Metadata
metadata = {
    "generation_date": datetime.now().isoformat(),
    "data_range": "2021-2024 IPL",
    "version": "2.2",
    "total_edges": len(pvp_edges),
    "qualification_rules": {
        "min_balls": MIN_BALLS,
        "min_dismissals": MIN_DISMISSALS,
        "sr_threshold_pace_high": SR_THRESHOLD_PACE,
        "sr_threshold_spin_high": SR_THRESHOLD_SPIN
    },
    "players_covered": {
        "batters": len(pvp_qualified['batter'].unique()),
        "bowlers": len(pvp_qualified['bowler'].unique())
    },
    "matchup_distribution": {
        "batter_wins": int((pvp_qualified['matchup_winner'] == 'BATTER').sum()),
        "bowler_wins": int((pvp_qualified['matchup_winner'] == 'BOWLER').sum()),
        "neutral": int((pvp_qualified['matchup_winner'] == 'NEUTRAL').sum())
    },
    "qualification_breakdown": {
        "batter_domination": int(pvp_qualified['batter_dominates'].sum()),
        "bowler_dismissals": int(pvp_qualified['bowler_dismisses'].sum()),
        "both_conditions": int((pvp_qualified['batter_dominates'] & pvp_qualified['bowler_dismisses']).sum())
    },
    "confidence_distribution": {
        "high_confidence_0.8+": int((pvp_qualified['confidence_weight'] >= 0.8).sum()),
        "medium_confidence_0.5_to_0.8": int(
            ((pvp_qualified['confidence_weight'] >= 0.5) & 
             (pvp_qualified['confidence_weight'] < 0.8)).sum()
        ),
        "low_confidence_below_0.5": int((pvp_qualified['confidence_weight'] < 0.5).sum())
    }
}

# Final JSON output
output_data = {
    "pvp_edges": pvp_edges,
    "metadata": metadata
}

# Save to file
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✅ JSON saved: {OUTPUT_JSON}")
print(f"   Size: {OUTPUT_JSON.stat().st_size / 1024:.2f} KB")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("PVP EXTRACTION COMPLETE - SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset Statistics:")
print(f"  Total PvP edges: {len(pvp_edges):,}")
print(f"  Unique batters: {len(pvp_qualified['batter'].unique())}")
print(f"  Unique bowlers: {len(pvp_qualified['bowler'].unique())}")

print(f"\n🎯 Qualification Breakdown:")
for reason, count in pvp_qualified['qualification_reason'].value_counts().items():
    print(f"  {reason}: {count:,}")

print(f"\n⚖️  Matchup Winners:")
for winner, count in pvp_qualified['matchup_winner'].value_counts().items():
    print(f"  {winner}: {count:,}")

print(f"\n📈 Top 5 Batter Dominators (Highest Advantage):")
top_batters = pvp_qualified.nlargest(5, 'batter_advantage')[
    ['batter_display_name', 'bowler_display_name', 'balls', 'strike_rate', 'batter_advantage', 'confidence_weight']
]
print(top_batters.to_string(index=False))

print(f"\n📉 Top 5 Bowler Dominators (Lowest Advantage):")
top_bowlers = pvp_qualified.nsmallest(5, 'batter_advantage')[
    ['batter_display_name', 'bowler_display_name', 'balls', 'dismissals', 'batter_advantage', 'confidence_weight']
]
print(top_bowlers.to_string(index=False))

print("\n" + "=" * 80)
print("✅ PVP EXTRACTION SUCCESSFUL")
print("=" * 80)
print(f"\nOutput file: {OUTPUT_JSON}")
print("\nThis dataset is ready for GAT training (Phase 2)")
print("=" * 80)