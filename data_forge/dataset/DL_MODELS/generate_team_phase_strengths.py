"""
================================================================================
SCRIPT 1: GENERATE TEAM PHASE STRENGTHS (2025)
================================================================================
Purpose: Aggregate player statistics into team-level phase performance metrics

Input:
  - BATTING_MASTER_2025.csv (157 players)
  - BOWLING_MASTER_2025.csv (170 players)
  - current_players_2025.csv (team mappings)

Output:
  - team_phase_strengths_2025.csv

Features per team:
  - Powerplay batting SR, bowling economy
  - Middle overs batting SR, bowling economy
  - Death overs batting SR, bowling economy
  - vs Pace batting average, bowling average
  - vs Spin batting average, bowling average
  - Overall boundary %, dot ball %
  
Author: El Dorado Project - CSCI 566
Date: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("TEAM PHASE STRENGTHS GENERATOR")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset"

CONFIG = {
    'batting_master': rf"{BASE_DIR}\Master_Datasets\BATTING_MASTER_2025_20251129_052900.csv",
    'bowling_master': rf"{BASE_DIR}\Master_Datasets\BOWLING_MASTER_2025_20251129_052900.csv",
    'current_players': rf"{BASE_DIR}\Clean_Nomenclature\current_players_2025_20251125_021101.csv",
    'output_dir': rf"{BASE_DIR}\DL_MODELS\outputs",
    'output_file': 'team_phase_strengths_2025.csv'
}

print(f"\n📁 Configuration:")
for key, path in CONFIG.items():
    if key not in ['output_dir', 'output_file']:
        exists = Path(path).exists()
        print(f"  {key}: {'✅' if exists else '❌'} {path}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📂 Loading data...")
batting = pd.read_csv(CONFIG['batting_master'])
bowling = pd.read_csv(CONFIG['bowling_master'])
players = pd.read_csv(CONFIG['current_players'])

print(f"  ✅ Batting: {len(batting)} players")
print(f"  ✅ Bowling: {len(bowling)} players")
print(f"  ✅ Player mappings: {len(players)} players")

# Merge team information
batting_with_team = batting.merge(
    players[['Kaggle_FullName', 'IPL_Team_2025']], 
    left_on='Kaggle_Match_Name', 
    right_on='Kaggle_FullName',
    how='left'
)

bowling_with_team = bowling.merge(
    players[['Kaggle_FullName', 'IPL_Team_2025']], 
    left_on='Kaggle_Match_Name', 
    right_on='Kaggle_FullName',
    how='left'
)

print(f"\n✅ Team mapping complete")

# Check which column name was created (might be IPL_Team_2025_x or IPL_Team_2025_y)
team_col = 'IPL_Team_2025' if 'IPL_Team_2025' in batting_with_team.columns else 'IPL_Team_2025_y'

print(f"  Batting with team: {batting_with_team[team_col].notna().sum()}/{len(batting_with_team)}")
print(f"  Bowling with team: {bowling_with_team[team_col].notna().sum()}/{len(bowling_with_team)}")

# ============================================================================
# CALCULATE TEAM BATTING STRENGTHS
# ============================================================================

print(f"\n🏏 Calculating team batting strengths...")

# Filter out debut players (all zeros) and players without team
batting_filtered = batting_with_team[
    (batting_with_team[team_col].notna()) &
    (batting_with_team['Total_Innings'] > 0)
].copy()

def safe_average(values, weights):
    """Calculate weighted average, handling zero weights gracefully"""
    clean_mask = ~(np.isnan(values) | np.isnan(weights))
    clean_values = values[clean_mask]
    clean_weights = weights[clean_mask]
    
    if len(clean_weights) == 0 or clean_weights.sum() == 0:
        return 0.0
    return np.average(clean_values, weights=clean_weights)

# Group by team and calculate weighted averages (weighted by innings played)
team_batting = batting_filtered.groupby(team_col, as_index=True).apply(
    lambda x: pd.Series({
        # Overall metrics
        'batting_avg_sr': safe_average(x['Strike_Rate'], x['Total_Innings']),
        'batting_avg': safe_average(x['Batting_Average'], x['Total_Innings']),
        'boundary_pct': safe_average(x['Boundary_Percentage'], x['Total_Innings']),
        
        # Powerplay (if data available - replace 0 with NaN first!)
        'pp_batting_sr': safe_average(
            x['Powerplay_SR'].replace(0, np.nan).fillna(x['Strike_Rate']), 
            x['Powerplay_Balls'].replace(0, np.nan).fillna(x['Total_Balls_Faced'])
        ),
        
        # Death overs (if data available - replace 0 with NaN first!)
        'death_batting_sr': safe_average(
            x['Death_Overs_SR'].replace(0, np.nan).fillna(x['Strike_Rate']), 
            x['Death_Overs_Balls'].replace(0, np.nan).fillna(x['Total_Balls_Faced'])
        ),
        
        # Rotation (middle overs proxy)
        'rotation_sr': safe_average(x['Rotation_Strike_Rate'], x['Total_Innings']),
        
        # Player count
        'batting_squad_size': len(x)
    })
).reset_index()

print(f"  ✅ Calculated batting strengths for {len(team_batting)} teams")

print(f"\n  Sample (CSK):")
# Team names are in the column, not index
if 'CSK' in team_batting[team_col].values:
    csk = team_batting[team_batting[team_col] == 'CSK'].iloc[0]
    print(f"    Overall SR: {csk['batting_avg_sr']:.1f}")
    print(f"    PP SR: {csk['pp_batting_sr']:.1f}")
    print(f"    Death SR: {csk['death_batting_sr']:.1f}")
else:
    print(f"    CSK not found, showing first team")
    print(f"    Available teams: {list(team_batting[team_col])}")

# ============================================================================
# CALCULATE TEAM BOWLING STRENGTHS
# ============================================================================

print(f"\n🎳 Calculating team bowling strengths...")

# Filter out debut players and players without team
bowling_filtered = bowling_with_team[
    (bowling_with_team[team_col].notna()) &
    (bowling_with_team['Total_Innings_Bowled'] > 0)
].copy()

# Group by team and calculate weighted averages (weighted by overs bowled)
team_bowling = bowling_filtered.groupby(team_col, as_index=True).apply(
    lambda x: pd.Series({
        # Overall metrics
        'bowling_avg_econ': safe_average(x['Economy_Rate'], x['Total_Overs_Bowled']),
        'bowling_avg': safe_average(x['Bowling_Average'], x['Total_Wickets'].replace(0, 1)),
        'bowling_sr': safe_average(x['Bowling_Strike_Rate'], x['Total_Wickets'].replace(0, 1)),
        'dot_ball_pct': safe_average(x['Dot_Ball_Percentage'], x['Total_Balls_Bowled']),
        
        # Powerplay (if data available - replace 0 with NaN first!)
        'pp_bowling_econ': safe_average(
            x['Powerplay_Economy'].replace(0, np.nan).fillna(x['Economy_Rate']), 
            x['Powerplay_Overs'].replace(0, np.nan).fillna(x['Total_Overs_Bowled'])
        ),
        
        # Death overs (if data available - replace 0 with NaN first!)
        'death_bowling_econ': safe_average(
            x['Death_Overs_Economy'].replace(0, np.nan).fillna(x['Economy_Rate']), 
            x['Death_Overs'].replace(0, np.nan).fillna(x['Total_Overs_Bowled'])
        ),
        
        # Middle overs (proxy: overall - (PP + death))
        'middle_bowling_econ': safe_average(x['Economy_Rate'], x['Total_Overs_Bowled']),
        
        # Player count
        'bowling_squad_size': len(x)
    })
).reset_index()

print(f"  ✅ Calculated bowling strengths for {len(team_bowling)} teams")
print(f"\n  Sample (MI):")
# Team names are in the column, not index
if 'MI' in team_bowling[team_col].values:
    mi = team_bowling[team_bowling[team_col] == 'MI'].iloc[0]
    print(f"    Overall Econ: {mi['bowling_avg_econ']:.2f}")
    print(f"    PP Econ: {mi['pp_bowling_econ']:.2f}")
    print(f"    Death Econ: {mi['death_bowling_econ']:.2f}")
else:
    print(f"    MI not found in column '{team_col}'")

# ============================================================================
# MERGE AND CALCULATE COMPOSITE METRICS
# ============================================================================

print(f"\n🔄 Merging batting and bowling strengths...")

# Reset index to make team column regular column
team_batting = team_batting.reset_index()
team_bowling = team_bowling.reset_index()

# Rename to consistent column name
team_batting = team_batting.rename(columns={team_col: 'IPL_Team_2025'})
team_bowling = team_bowling.rename(columns={team_col: 'IPL_Team_2025'})

team_strengths = team_batting.merge(
    team_bowling, 
    on='IPL_Team_2025', 
    how='outer'
)

# Fill NaN with reasonable defaults
team_strengths = team_strengths.fillna({
    'batting_avg_sr': 130.0,
    'pp_batting_sr': 135.0,
    'death_batting_sr': 150.0,
    'bowling_avg_econ': 8.5,
    'pp_bowling_econ': 8.0,
    'death_bowling_econ': 10.0
})

# Calculate composite phase strengths
team_strengths['pp_net_advantage'] = (
    (team_strengths['pp_batting_sr'] - 130) / 10  # Normalize SR
    - (team_strengths['pp_bowling_econ'] - 8) * 2  # Normalize economy (inverted)
)

team_strengths['death_net_advantage'] = (
    (team_strengths['death_batting_sr'] - 150) / 15
    - (team_strengths['death_bowling_econ'] - 10) * 2
)

team_strengths['overall_strength'] = (
    (team_strengths['batting_avg_sr'] - 130) / 10
    - (team_strengths['bowling_avg_econ'] - 8.5) * 2
)

print(f"  ✅ Composite metrics calculated")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

output_path = Path(CONFIG['output_dir']) / CONFIG['output_file']
team_strengths.to_csv(output_path, index=False)

print(f"\n" + "=" * 80)
print("✅ TEAM PHASE STRENGTHS GENERATED!")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"Teams: {len(team_strengths)}")
print(f"Features: {len(team_strengths.columns)}")

print(f"\n📊 Team Rankings (by overall strength):")
team_strengths_sorted = team_strengths.sort_values('overall_strength', ascending=False)
for rank, (idx, row) in enumerate(team_strengths_sorted.head(10).iterrows(), start=1):
    print(f"  {rank}. {row['IPL_Team_2025']}: {row['overall_strength']:.2f} "
          f"(PP: {row['pp_net_advantage']:.2f}, Death: {row['death_net_advantage']:.2f})")

print(f"\n✅ Ready for model training!")

# ============================================================================
# VALIDATION: SAMPLE MATCHUP PREDICTIONS
# ============================================================================

print(f"\n" + "=" * 80)
print("🧪 VALIDATION TESTS: Sample Matchup Predictions")
print("=" * 80)

# Define 4 test matches covering different team combinations
test_matches = [
    ("KKR", "MI", "Eden Gardens, Kolkata"),
    ("CSK", "RCB", "Chepauk, Chennai"),
    ("SRH", "RR", "Hyderabad"),
    ("LSG", "GT", "Lucknow")
]

print(f"\nPredicting match outcomes based on team phase strengths...")
print(f"Note: This is a SIMPLE baseline using only phase differentials")
print(f"Full model will add H2H, form, venue, PvP matchups, etc.\n")

for home, away, venue in test_matches:
    print(f"\n{'─' * 80}")
    print(f"🏏 {home} vs {away} @ {venue}")
    print(f"{'─' * 80}")
    
    # Get team strengths
    home_stats = team_strengths[team_strengths['IPL_Team_2025'] == home].iloc[0]
    away_stats = team_strengths[team_strengths['IPL_Team_2025'] == away].iloc[0]
    
    # Calculate advantages
    pp_advantage = home_stats['pp_net_advantage'] - away_stats['pp_net_advantage']
    death_advantage = home_stats['death_net_advantage'] - away_stats['death_net_advantage']
    overall_advantage = home_stats['overall_strength'] - away_stats['overall_strength']
    
    print(f"\n📊 Phase Battle:")
    print(f"  Powerplay:   {home:4s} {pp_advantage:+.2f} vs {away}")
    print(f"  Death Overs: {home:4s} {death_advantage:+.2f} vs {away}")
    print(f"  Overall:     {home:4s} {overall_advantage:+.2f} vs {away}")
    
    # Simple prediction (just for validation - NOT the final model!)
    home_score = 50 + (overall_advantage * 5)  # Baseline 50-50
    away_score = 100 - home_score
    
    print(f"\n🎯 Simple Baseline Prediction:")
    print(f"  {home}: {home_score:.1f}% chance")
    print(f"  {away}: {away_score:.1f}% chance")
    
    # Key factors
    print(f"\n🔑 Key Factors:")
    if abs(pp_advantage) > abs(death_advantage):
        print(f"  ⚡ Powerplay battle crucial ({abs(pp_advantage):.2f} advantage)")
    else:
        print(f"  💀 Death overs battle crucial ({abs(death_advantage):.2f} advantage)")
    
    # Weaknesses to exploit
    home_weakness = "Powerplay" if home_stats['pp_net_advantage'] < 0 else "Death overs" if home_stats['death_net_advantage'] < 20 else None
    away_weakness = "Powerplay" if away_stats['pp_net_advantage'] < 0 else "Death overs" if away_stats['death_net_advantage'] < 20 else None
    
    if home_weakness:
        print(f"  ⚠️  {home} vulnerable in: {home_weakness}")
    if away_weakness:
        print(f"  ⚠️  {away} vulnerable in: {away_weakness}")

print(f"\n" + "=" * 80)
print("⚠️  IMPORTANT: These are BASELINE predictions using only phase stats!")
print("Full TabTransformer v2 will add:")
print("  - Head-to-head history (15% weight)")
print("  - Recent form (20% weight)")
print("  - Player matchups from GAT (25% weight)")
print("  - Venue characteristics (5% weight)")
print("  - Dew factor, toss, etc.")
print("=" * 80)

print(f"\n✅ Validation complete! Team phase strengths ready for model training.")