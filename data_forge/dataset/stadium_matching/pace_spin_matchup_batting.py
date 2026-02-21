"""
================================================================================
PACE vs SPIN MATCHUP DEVIATION CALCULATION - BATTING
================================================================================
Project: El Dorado - CSCI 566
Purpose: Calculate batter performance vs pace/spin (2022-2024)

Uses: BOWLER_TYPE_MAPPING_COMPLETE_170.csv (bowler type classification)

Command: python pace_spin_matchup_batting.py
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# ============================================================================
# AUTO-DETECT PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent.parent

print("=" * 80)
print("PACE vs SPIN MATCHUP DEVIATION - BATTING")
print("=" * 80)
print(f"Script Directory: {SCRIPT_DIR}")
print(f"Auto-detected Root: {ROOT_DIR}")

# Input paths
KAGGLE_DIR = ROOT_DIR / "Kaggle_download" / "rajsengo"
DATASET_DIR = ROOT_DIR / "Dataset(s) and code" / "dataset"
MASTER_DIR = DATASET_DIR / "Master_Datasets"
OUTPUT_DIR = SCRIPT_DIR

# Input files
ALL_DETAILS_FILE = KAGGLE_DIR / "all_season_details.csv"
DETAILS_2024_FILE = KAGGLE_DIR / "2024" / "season_details.csv"
BATTING_MASTER_FILE = MASTER_DIR / "BATTING_MASTER_2025_20251129_052900.csv"
BOWLER_TYPE_FILE = OUTPUT_DIR / "BOWLER_TYPE_MAPPING_COMPLETE_170.csv"

# Output files
OUTPUT_FILE = OUTPUT_DIR / "PACE_SPIN_MATCHUP_BATTING_2022_2024.csv"

print(f"Output Directory: {OUTPUT_DIR}")

# Check bowler type mapping exists
if not BOWLER_TYPE_FILE.exists():
    print(f"\n❌ ERROR: Bowler type mapping not found!")
    print(f"Expected: {BOWLER_TYPE_FILE}")
    print(f"\nPlease run the bowler type classification script first.")
    sys.exit(1)

print("=" * 80)

# ============================================================================
# PART 1: LOAD BOWLER TYPE MAPPING
# ============================================================================

print("\n[1/5] Loading Bowler Type Classification...")

bowler_types = pd.read_csv(BOWLER_TYPE_FILE)
print(f"  ✓ Loaded {len(bowler_types)} bowler classifications")

# Distribution
type_dist = bowler_types['Bowler_Type'].value_counts()
print(f"\n  Bowler type distribution:")
for btype, count in type_dist.items():
    print(f"    {btype:15s}: {count:3d} bowlers")

# Create simplified classification: PACE vs SPIN
def classify_pace_spin(bowler_type):
    """Simplify to PACE or SPIN."""
    if pd.isna(bowler_type):
        return 'UNKNOWN'
    if 'PACE' in bowler_type or 'MEDIUM' in bowler_type:
        return 'PACE'
    elif 'SPIN' in bowler_type:
        return 'SPIN'
    else:
        return 'UNKNOWN'

bowler_types['Pace_Spin'] = bowler_types['Bowler_Type'].apply(classify_pace_spin)

pace_spin_dist = bowler_types['Pace_Spin'].value_counts()
print(f"\n  Simplified classification:")
for category, count in pace_spin_dist.items():
    print(f"    {category:10s}: {count:3d} bowlers")

# Create bowler name → pace/spin lookup
bowler_lookup = bowler_types.set_index('Kaggle_Match_Name')['Pace_Spin'].to_dict()
print(f"\n  ✓ Created lookup for {len(bowler_lookup)} bowlers")

# ============================================================================
# PART 2: LOAD BALL-BY-BALL DATA
# ============================================================================

print("\n[2/5] Loading Ball-by-Ball Data...")

details_chunks = []

print(f"  Reading: {ALL_DETAILS_FILE.name}...")
df_old = pd.read_csv(ALL_DETAILS_FILE, low_memory=False)
print(f"    ✓ Loaded {len(df_old):,} rows")
details_chunks.append(df_old)

print(f"  Reading: {DETAILS_2024_FILE.name}...")
df_2024 = pd.read_csv(DETAILS_2024_FILE, low_memory=False)
print(f"    ✓ Loaded {len(df_2024):,} rows")
details_chunks.append(df_2024)

details_df = pd.concat(details_chunks, ignore_index=True)
print(f"\n  ✓ Total ball-by-ball records: {len(details_df):,}")

# Filter: 2022-2024 only
before = len(details_df)
details_df = details_df[details_df['season'].isin([2022, 2023, 2024])]
print(f"  ✓ Filtered to 2022-2024: {len(details_df):,} rows (removed {before-len(details_df):,})")

# ============================================================================
# PART 3: CLASSIFY EACH BALL AS PACE/SPIN
# ============================================================================

print("\n[3/5] Classifying Each Ball by Bowler Type...")

# Map bowler to pace/spin
details_df['bowler'] = details_df['bowler1_name']
details_df['bowler_type'] = details_df['bowler'].map(bowler_lookup)

# Count classifications
type_counts = details_df['bowler_type'].value_counts()
print(f"\n  Ball classification:")
for btype, count in type_counts.items():
    pct = count / len(details_df) * 100
    print(f"    {btype:10s}: {count:7,} balls ({pct:5.1f}%)")

# Filter: Keep only PACE and SPIN (drop UNKNOWN)
before = len(details_df)
details_df = details_df[details_df['bowler_type'].isin(['PACE', 'SPIN'])]
print(f"\n  ✓ Filtered to classified balls: {len(details_df):,} (removed {before-len(details_df):,} unknown)")

# ============================================================================
# PART 4: CALCULATE BATTER vs PACE/SPIN STATS
# ============================================================================

print("\n[4/5] Calculating Batter Performance vs PACE and SPIN...")

# Identify batter (batsman1 is on strike)
details_df['batter'] = details_df['batsman1_name']

# Calculate stats per batter + bowler_type
print(f"  Aggregating by batter + bowler_type...")

matchup_stats = details_df.groupby(['batter', 'bowler_type']).agg({
    'runs': 'sum',
    'match_id': 'count',  # Total balls faced
    'wicket_id': lambda x: x.notna().sum()  # Dismissals
}).reset_index()

matchup_stats.columns = ['batter', 'bowler_type', 'runs', 'balls', 'dismissals']

# Calculate innings (unique matches)
innings_count = details_df.groupby(['batter', 'bowler_type'])['match_id'].nunique().reset_index()
innings_count.columns = ['batter', 'bowler_type', 'innings']

matchup_stats = matchup_stats.merge(innings_count, on=['batter', 'bowler_type'], how='left')

# Calculate strike rate
matchup_stats['strike_rate'] = np.where(
    matchup_stats['balls'] > 0,
    (matchup_stats['runs'] / matchup_stats['balls'] * 100).round(2),
    np.nan
)

# Calculate average
matchup_stats['average'] = np.where(
    matchup_stats['dismissals'] > 0,
    (matchup_stats['runs'] / matchup_stats['dismissals']).round(2),
    np.nan
)

# Calculate boundary % (4s and 6s)
boundary_data = details_df[details_df['isBoundary'] == True].groupby(['batter', 'bowler_type']).size().reset_index(name='boundaries')
matchup_stats = matchup_stats.merge(boundary_data, on=['batter', 'bowler_type'], how='left')
matchup_stats['boundaries'] = matchup_stats['boundaries'].fillna(0)
matchup_stats['boundary_pct'] = np.where(
    matchup_stats['balls'] > 0,
    (matchup_stats['boundaries'] / matchup_stats['balls'] * 100).round(2),
    0.0
)

print(f"  ✓ Calculated stats for {len(matchup_stats):,} batter-bowlertype combinations")

# Distribution
print(f"\n  Sample size distribution:")
for btype in ['PACE', 'SPIN']:
    subset = matchup_stats[matchup_stats['bowler_type'] == btype]
    print(f"    vs {btype:4s}: {len(subset):4d} batters")
    
    balls_dist = subset['balls'].describe()
    print(f"      Balls faced - Mean: {balls_dist['mean']:6.1f}, Median: {balls_dist['50%']:6.1f}, Max: {balls_dist['max']:6.0f}")

# ============================================================================
# PART 5: CALCULATE DEVIATIONS
# ============================================================================

print("\n[5/5] Calculating Matchup Deviations and Merging with Master...")

# Load batting master
batting_master = pd.read_csv(BATTING_MASTER_FILE)
print(f"  ✓ Loaded {len(batting_master)} players from BATTING_MASTER")

# IPL BASELINE STRIKE RATES (T20 specific)
IPL_BASELINE_VS_PACE = 145.0  # IPL avg SR vs pace
IPL_BASELINE_VS_SPIN = 138.0  # IPL avg SR vs spin (slightly lower - spinners in middle overs)

print(f"\n  Baseline Strike Rates (IPL T20):")
print(f"    vs PACE: {IPL_BASELINE_VS_PACE}")
print(f"    vs SPIN: {IPL_BASELINE_VS_SPIN}")

def calculate_confidence_weight(balls, min_threshold=30, full_confidence=120):
    """Sigmoid curve for confidence weight."""
    if balls < min_threshold:
        return 0.0
    if balls >= full_confidence:
        return min(0.95, 0.85 + (balls - full_confidence) / 1000)
    
    x = (balls - min_threshold) / (full_confidence - min_threshold)
    sigmoid = 1 / (1 + np.exp(-10 * (x - 0.5)))
    return 0.3 + (sigmoid * 0.55)

def calculate_matchup_deviation(batter_sr_vs_type, batter_overall_sr, baseline_sr):
    """
    Calculate matchup deviation.
    
    Formula:
      personal_delta = batter_sr_vs_type - batter_overall_sr
      normalization_factor = baseline_sr / 145.0  # 145 = overall IPL avg
      normalized_delta = personal_delta / (baseline_sr * normalization_factor)
      return clip(normalized_delta, -1.0, 1.0)
    
    Positive = performs BETTER vs this type than overall
    Negative = performs WORSE vs this type than overall
    """
    if pd.isna(batter_overall_sr) or batter_overall_sr == 0:
        return 0.0
    if pd.isna(baseline_sr) or baseline_sr == 0:
        return 0.0
    
    personal_delta = batter_sr_vs_type - batter_overall_sr
    normalization_factor = baseline_sr / 145.0
    
    if normalization_factor == 0:
        return 0.0
    
    normalized_delta = personal_delta / (baseline_sr * normalization_factor)
    return np.clip(normalized_delta, -1.0, 1.0)

# Pivot matchup_stats to have PACE and SPIN as separate columns
pivot_data = matchup_stats.pivot_table(
    index='batter',
    columns='bowler_type',
    values=['strike_rate', 'average', 'balls', 'innings', 'runs', 'dismissals', 'boundary_pct'],
    aggfunc='first'
).reset_index()

# Flatten column names
pivot_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in pivot_data.columns.values]

print(f"\n  ✓ Pivoted data: {len(pivot_data)} unique batters")

# Validation tracking
unmatched_batters = {}
matched_count = 0

# Calculate deviations
results = []

for _, row in pivot_data.iterrows():
    batter = row['batter']
    
    # Match with master
    master_row = batting_master[batting_master['Kaggle_Match_Name'] == batter]
    
    if len(master_row) == 0:
        # Track unmatched
        total_balls = row.get('balls_PACE', 0) + row.get('balls_SPIN', 0)
        if pd.isna(total_balls):
            total_balls = 0
        
        if batter not in unmatched_batters:
            unmatched_batters[batter] = int(total_balls)
        continue
    
    matched_count += 1
    master_row = master_row.iloc[0]
    
    overall_sr = master_row['Strike_Rate']
    overall_avg = master_row['Batting_Average']
    overall_boundary_pct = master_row['Boundary_Percentage']
    
    # Get vs PACE stats
    sr_vs_pace = row.get('strike_rate_PACE', np.nan)
    avg_vs_pace = row.get('average_PACE', np.nan)
    balls_vs_pace = row.get('balls_PACE', 0)
    innings_vs_pace = row.get('innings_PACE', 0)
    runs_vs_pace = row.get('runs_PACE', 0)
    dismissals_vs_pace = row.get('dismissals_PACE', 0)
    boundary_pct_vs_pace = row.get('boundary_pct_PACE', 0)
    
    # Get vs SPIN stats
    sr_vs_spin = row.get('strike_rate_SPIN', np.nan)
    avg_vs_spin = row.get('average_SPIN', np.nan)
    balls_vs_spin = row.get('balls_SPIN', 0)
    innings_vs_spin = row.get('innings_SPIN', 0)
    runs_vs_spin = row.get('runs_SPIN', 0)
    dismissals_vs_spin = row.get('dismissals_SPIN', 0)
    boundary_pct_vs_spin = row.get('boundary_pct_SPIN', 0)
    
    # Handle NaNs
    if pd.isna(balls_vs_pace):
        balls_vs_pace = 0
    if pd.isna(balls_vs_spin):
        balls_vs_spin = 0
    
    # Calculate deviations
    deviation_vs_pace = calculate_matchup_deviation(sr_vs_pace, overall_sr, IPL_BASELINE_VS_PACE)
    deviation_vs_spin = calculate_matchup_deviation(sr_vs_spin, overall_sr, IPL_BASELINE_VS_SPIN)
    
    # Calculate confidence
    confidence_vs_pace = calculate_confidence_weight(balls_vs_pace)
    confidence_vs_spin = calculate_confidence_weight(balls_vs_spin)
    
    # CRITICAL FIX: Force deviation to 0 if insufficient sample size
    # This prevents unreliable extreme values from affecting OVR
    if balls_vs_pace < 30:
        deviation_vs_pace = 0.0
    if balls_vs_spin < 30:
        deviation_vs_spin = 0.0
    
    results.append({
        'Player_Name': master_row['Player_Name'],
        'Kaggle_Match_Name': batter,
        
        # Overall stats
        'Overall_SR': round(overall_sr, 2) if not pd.isna(overall_sr) else 0.0,
        'Overall_Avg': round(overall_avg, 2) if not pd.isna(overall_avg) else 0.0,
        'Overall_Boundary_Pct': round(overall_boundary_pct, 2) if not pd.isna(overall_boundary_pct) else 0.0,
        
        # vs PACE
        'Balls_vs_Pace': int(balls_vs_pace),
        'Innings_vs_Pace': int(innings_vs_pace) if not pd.isna(innings_vs_pace) else 0,
        'SR_vs_Pace': round(sr_vs_pace, 2) if not pd.isna(sr_vs_pace) else 0.0,
        'Avg_vs_Pace': round(avg_vs_pace, 2) if not pd.isna(avg_vs_pace) else 0.0,
        'Boundary_Pct_vs_Pace': round(boundary_pct_vs_pace, 2),
        'Deviation_vs_Pace': round(deviation_vs_pace, 3),
        'Confidence_vs_Pace': round(confidence_vs_pace, 3),
        
        # vs SPIN
        'Balls_vs_Spin': int(balls_vs_spin),
        'Innings_vs_Spin': int(innings_vs_spin) if not pd.isna(innings_vs_spin) else 0,
        'SR_vs_Spin': round(sr_vs_spin, 2) if not pd.isna(sr_vs_spin) else 0.0,
        'Avg_vs_Spin': round(avg_vs_spin, 2) if not pd.isna(avg_vs_spin) else 0.0,
        'Boundary_Pct_vs_Spin': round(boundary_pct_vs_spin, 2),
        'Deviation_vs_Spin': round(deviation_vs_spin, 3),
        'Confidence_vs_Spin': round(confidence_vs_spin, 3),
        
        # Matchup strength indicator
        'Pace_Advantage': round(deviation_vs_pace - deviation_vs_spin, 3),  # Positive = better vs pace
    })

output_df = pd.DataFrame(results)
print(f"\n  ✓ Calculated matchup stats for {len(output_df)} batters")
print(f"  ✓ Successfully matched: {matched_count} batters")

# Report unmatched
if unmatched_batters:
    print(f"\n  ⚠️  WARNING: {len(unmatched_batters)} batters in rajsengo NOT FOUND in master:")
    
    sorted_unmatched = sorted(unmatched_batters.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top unmatched batters (by balls faced):")
    for batter, balls in sorted_unmatched[:20]:
        print(f"    {batter:30s} - {balls:4d} balls")
    
    if len(sorted_unmatched) > 20:
        print(f"    ... and {len(sorted_unmatched) - 20} more")
    
    unmatched_df = pd.DataFrame([
        {'Batter_Name': name, 'Total_Balls': balls}
        for name, balls in sorted_unmatched
    ])
    unmatched_file = OUTPUT_DIR / "UNMATCHED_BATTERS_PACE_SPIN.csv"
    unmatched_df.to_csv(unmatched_file, index=False)
    print(f"\n  ✓ Saved unmatched list: {unmatched_file.name}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print("\n[6/6] Saving Output...")

output_df = output_df.sort_values('Player_Name')
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  ✓ Saved: {OUTPUT_FILE.name}")
print(f"  ✓ Total players: {len(output_df)}")

# ============================================================================
# VALIDATION & STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION - FAMOUS MATCHUPS")
print("=" * 80)

famous_batters = [
    'Virat Kohli',
    'Rohit Sharma', 
    'Suryakumar Yadav',
    'Hardik Pandya',
]

for player in famous_batters:
    subset = output_df[output_df['Player_Name'].str.contains(player, case=False, na=False)]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n{player}:")
        print(f"  Overall SR: {row['Overall_SR']:.2f}")
        print(f"  vs PACE: SR {row['SR_vs_Pace']:.2f} (Deviation: {row['Deviation_vs_Pace']:+.3f}, Confidence: {row['Confidence_vs_Pace']:.2f}) [{row['Balls_vs_Pace']} balls]")
        print(f"  vs SPIN: SR {row['SR_vs_Spin']:.2f} (Deviation: {row['Deviation_vs_Spin']:+.3f}, Confidence: {row['Confidence_vs_Spin']:.2f}) [{row['Balls_vs_Spin']} balls]")
        print(f"  Pace Advantage: {row['Pace_Advantage']:+.3f} {'(PACE destroyer)' if row['Pace_Advantage'] > 0.2 else '(SPIN destroyer)' if row['Pace_Advantage'] < -0.2 else '(balanced)'}")

print("\n" + "-" * 80)
print("TOP 10 PACE DESTROYERS (High Deviation vs PACE, Min 60 balls)")
print("-" * 80)
top_pace = output_df[output_df['Balls_vs_Pace'] >= 60].nlargest(10, 'Deviation_vs_Pace')
print(top_pace[['Player_Name', 'SR_vs_Pace', 'Overall_SR', 'Deviation_vs_Pace', 'Balls_vs_Pace']].to_string(index=False))

print("\n" + "-" * 80)
print("TOP 10 SPIN DESTROYERS (High Deviation vs SPIN, Min 60 balls)")
print("-" * 80)
top_spin = output_df[output_df['Balls_vs_Spin'] >= 60].nlargest(10, 'Deviation_vs_Spin')
print(top_spin[['Player_Name', 'SR_vs_Spin', 'Overall_SR', 'Deviation_vs_Spin', 'Balls_vs_Spin']].to_string(index=False))

print("\n" + "-" * 80)
print("BIGGEST PACE ADVANTAGE (Best vs Pace, Struggles vs Spin)")
print("-" * 80)
pace_adv = output_df[(output_df['Balls_vs_Pace'] >= 60) & (output_df['Balls_vs_Spin'] >= 60)].nlargest(10, 'Pace_Advantage')
print(pace_adv[['Player_Name', 'SR_vs_Pace', 'SR_vs_Spin', 'Pace_Advantage', 'Deviation_vs_Pace', 'Deviation_vs_Spin']].to_string(index=False))

print("\n" + "-" * 80)
print("BIGGEST SPIN ADVANTAGE (Best vs Spin, Struggles vs Pace)")
print("-" * 80)
spin_adv = output_df[(output_df['Balls_vs_Pace'] >= 60) & (output_df['Balls_vs_Spin'] >= 60)].nsmallest(10, 'Pace_Advantage')
print(spin_adv[['Player_Name', 'SR_vs_Pace', 'SR_vs_Spin', 'Pace_Advantage', 'Deviation_vs_Pace', 'Deviation_vs_Spin']].to_string(index=False))

print("\n" + "=" * 80)
print("✅ COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput: {OUTPUT_FILE.name}")
print("\nHow to use in OVR calculation:")
print("  1. Venue has pitch type (PACE-friendly or SPIN-friendly)")
print("  2. If PACE pitch → Use Deviation_vs_Pace to adjust OVR")
print("  3. If SPIN pitch → Use Deviation_vs_Spin to adjust OVR")
print("  4. Weight by Confidence (higher confidence = more adjustment)")
print("\nFormula:")
print("  Effective_OVR = BASE_OVR + (Deviation_vs_Type × Confidence × BASE_OVR)")
print("=" * 80)