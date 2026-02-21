"""
================================================================================
TEAM HEAD-TO-HEAD STATISTICS GENERATOR
================================================================================
Generate historical H2H records for all IPL 2025 team pairs from 2022-2024 data

Features Generated (per team pair):
  - h2h_matches: Total matches played
  - h2h_wins: Wins for team A
  - h2h_losses: Losses for team A  
  - h2h_win_rate: Win percentage
  - h2h_avg_margin: Average winning margin (runs/wickets converted to runs)
  - h2h_recent_form: Last 3 H2H results (W=1, L=0, averaged)
  - h2h_home_wins: Wins at home venue
  - h2h_away_wins: Wins at away venue
  - h2h_dominance: Psychological edge (-1 to +1 scale)

Data Source: rajsengo season_summary files (2022-2024)
Output: team_h2h_matrix_2025.csv (10x10 teams, asymmetric)

Author: Nikhil Ravichandran
Date: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
BASE_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Kaggle_download\rajsengo")

# Use all_season_summary.csv which has ALL data including 2021
SEASON_FILES = {
    'all': BASE_DIR / "all_season_summary.csv"  # Contains 2021-2023 combined
}

# If all_season file doesn't exist, fall back to individual files
if not (BASE_DIR / "all_season_summary.csv").exists():
    print("⚠️  all_season_summary.csv not found, using individual season files")
    SEASON_FILES = {
        2022: BASE_DIR / "2022" / "season_summary.csv",
        2023: BASE_DIR / "2023" / "summary.csv",
        2024: BASE_DIR / "2024" / "season_summary.csv"
    }

OUTPUT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPL 2025 teams (using new names where applicable)
IPL_2025_TEAMS = [
    'CSK', 'DC', 'GT', 'KKR', 'LSG', 
    'MI', 'PK', 'RCB', 'RR', 'SRH'
]

# Team name mappings (data uses abbreviations, handle franchise changes)
TEAM_NAME_MAPPING = {
    # Current teams (already abbreviated)
    'CSK': 'CSK',
    'DC': 'DC',
    'GT': 'GT',
    'KKR': 'KKR',
    'LSG': 'LSG',
    'MI': 'MI',
    'RCB': 'RCB',
    'RR': 'RR',
    'SRH': 'SRH',
    
    # Franchise changes
    'PBKS': 'PK',  # Punjab Kings (was KXIP)
    'KXIP': 'PK',  # Old name for Punjab Kings
    
    # Defunct teams (exclude from 2025)
    'GL': None,    # Gujarat Lions (defunct)
    'Kochi': None, # Kochi Tuskers Kerala (defunct)
    'PWI': None,   # Pune Warriors India (defunct)
    'RPS': None    # Rising Pune Supergiant (defunct)
}

# Venue to team mapping (for home/away classification)
TEAM_HOME_VENUES = {
    'CSK': ['Chepauk', 'Chennai', 'MA Chidambaram'],
    'DC': ['Arun Jaitley', 'Delhi', 'Feroz Shah Kotla'],
    'GT': ['Narendra Modi', 'Ahmedabad'],
    'KKR': ['Eden Gardens', 'Kolkata'],
    'LSG': ['Lucknow', 'Ekana'],
    'MI': ['Wankhede', 'Mumbai'],
    'PK': ['Mohali', 'Punjab', 'Dharamsala', 'Mullanpur'],
    'RCB': ['Chinnaswamy', 'Bangalore', 'Bengaluru'],
    'RR': ['Sawai Mansingh', 'Jaipur'],
    'SRH': ['Hyderabad', 'Rajiv Gandhi', 'Uppal']
}

print("=" * 80)
print("TEAM HEAD-TO-HEAD STATISTICS GENERATOR")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📁 Configuration:")
for year, path in SEASON_FILES.items():
    status = "✅" if path.exists() else "❌"
    print(f"  {year}: {status} {path}")

print(f"\n📂 Loading match data...")

all_matches = []
for year, filepath in SEASON_FILES.items():
    if not filepath.exists():
        print(f"  ⚠️  {year}: File not found, skipping")
        continue
    
    df = pd.read_csv(filepath)
    df['season'] = pd.to_numeric(df['season'], errors='coerce')  # Convert to numeric
    all_matches.append(df)
    print(f"  ✅ {year}: {len(df)} matches")

matches_df = pd.concat(all_matches, ignore_index=True)

# Filter to 2021-2024 seasons only
matches_df = matches_df[matches_df['season'].between(2021, 2024)]

print(f"\n✅ Total matches loaded: {len(matches_df)} (2021-2024 only)")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print(f"\n🔧 Preprocessing match data...")

# Use correct column names
matches_df = matches_df.rename(columns={
    'home_team': 'team1',
    'away_team': 'team2',
    'venue_name': 'venue',
    'result': 'match_result'
})

print(f"  ✅ Columns standardized (home_team→team1, away_team→team2)")

# Standardize team names
matches_df['team1_clean'] = matches_df['team1'].map(TEAM_NAME_MAPPING)
matches_df['team2_clean'] = matches_df['team2'].map(TEAM_NAME_MAPPING)

print(f"  ✅ Team names mapped")
print(f"     Before mapping: {matches_df['team1'].nunique()} unique teams")
print(f"     After mapping: {matches_df['team1_clean'].nunique()} unique teams")

# Filter to IPL 2025 teams only (and remove defunct teams = None)
matches_df = matches_df[
    matches_df['team1_clean'].isin(IPL_2025_TEAMS) & 
    matches_df['team2_clean'].isin(IPL_2025_TEAMS) &
    matches_df['team1_clean'].notna() &
    matches_df['team2_clean'].notna()
].copy()

print(f"  ✅ Filtered to IPL 2025 teams: {len(matches_df)} matches")

# Determine winner
matches_df['winner_clean'] = matches_df['winner'].map(TEAM_NAME_MAPPING)

# Parse match result to extract margin
def parse_margin(result_text):
    """Extract winning margin in runs (convert wickets to runs equivalent)"""
    if pd.isna(result_text):
        return 0
    
    result_lower = str(result_text).lower()
    
    # Extract runs margin
    if 'run' in result_lower:
        try:
            runs = int(''.join(filter(str.isdigit, result_lower.split('run')[0])))
            return runs
        except:
            return 20  # Default moderate margin
    
    # Convert wickets to runs (1 wicket ≈ 15 runs in T20)
    elif 'wicket' in result_lower:
        try:
            wickets = int(''.join(filter(str.isdigit, result_lower.split('wicket')[0])))
            return wickets * 15
        except:
            return 30  # Default moderate margin
    
    return 20  # Default

matches_df['margin_runs'] = matches_df['match_result'].apply(parse_margin)

# Determine home/away
def is_home_match(row, team):
    """Check if match is at team's home venue"""
    venue = str(row.get('venue', '')).lower()
    if team not in TEAM_HOME_VENUES:
        return False
    
    home_keywords = [v.lower() for v in TEAM_HOME_VENUES[team]]
    return any(keyword in venue for keyword in home_keywords)

print(f"  ✅ Match margins parsed")
print(f"  ✅ Home/away classification ready")

# ============================================================================
# CALCULATE H2H STATISTICS
# ============================================================================

print(f"\n📊 Calculating head-to-head statistics...")

h2h_records = []

for team_a in IPL_2025_TEAMS:
    for team_b in IPL_2025_TEAMS:
        if team_a == team_b:
            continue  # Skip self-matchups
        
        # Get all matches between team_a and team_b
        h2h_matches = matches_df[
            ((matches_df['team1_clean'] == team_a) & (matches_df['team2_clean'] == team_b)) |
            ((matches_df['team1_clean'] == team_b) & (matches_df['team2_clean'] == team_a))
        ].copy()
        
        if len(h2h_matches) == 0:
            # No history - neutral record
            h2h_records.append({
                'team_a': team_a,
                'team_b': team_b,
                'h2h_matches': 0,
                'h2h_wins': 0,
                'h2h_losses': 0,
                'h2h_win_rate': 0.5,  # Neutral
                'h2h_avg_margin': 0.0,
                'h2h_recent_form': 0.5,  # Neutral
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_dominance': 0.0  # Neutral
            })
            continue
        
        # Calculate wins/losses for team_a
        wins = (h2h_matches['winner_clean'] == team_a).sum()
        losses = (h2h_matches['winner_clean'] == team_b).sum()
        total = len(h2h_matches)
        
        # Win rate
        win_rate = wins / total if total > 0 else 0.5
        
        # Average margin (for team_a wins)
        team_a_wins = h2h_matches[h2h_matches['winner_clean'] == team_a]
        avg_margin = team_a_wins['margin_runs'].mean() if len(team_a_wins) > 0 else 0.0
        
        # Recent form (last 3 matches)
        # Sort by season (numeric) or by index if season is missing
        if matches_df['season'].dtype in ['int64', 'float64']:
            recent_matches = h2h_matches.nlargest(3, 'season')
        else:
            # Fallback: use most recent rows (assume chronological order)
            recent_matches = h2h_matches.tail(3)
        
        recent_wins = (recent_matches['winner_clean'] == team_a).sum()
        recent_form = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0.5
        
        # Home/away splits
        h2h_matches['is_home'] = h2h_matches.apply(lambda x: is_home_match(x, team_a), axis=1)
        home_wins = ((h2h_matches['winner_clean'] == team_a) & (h2h_matches['is_home'])).sum()
        away_wins = ((h2h_matches['winner_clean'] == team_a) & (~h2h_matches['is_home'])).sum()
        
        # Dominance score (-1 to +1)
        # Combines win rate, margin, and recent form
        dominance = (win_rate - 0.5) * 2  # Scale to -1 to +1
        dominance += (recent_form - 0.5) * 0.5  # Weight recent form
        dominance = np.clip(dominance, -1, 1)
        
        h2h_records.append({
            'team_a': team_a,
            'team_b': team_b,
            'h2h_matches': total,
            'h2h_wins': wins,
            'h2h_losses': losses,
            'h2h_win_rate': round(win_rate, 3),
            'h2h_avg_margin': round(avg_margin, 1),
            'h2h_recent_form': round(recent_form, 3),
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_dominance': round(dominance, 3)
        })

h2h_df = pd.DataFrame(h2h_records)

print(f"  ✅ Calculated H2H stats for {len(h2h_df)} team pairs")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

output_file = OUTPUT_DIR / "team_h2h_matrix_2025.csv"
h2h_df.to_csv(output_file, index=False)

print(f"\n" + "=" * 80)
print(f"✅ TEAM H2H MATRIX GENERATED!")
print(f"=" * 80)
print(f"\nOutput: {output_file}")
print(f"Team pairs: {len(h2h_df)}")
print(f"Features: {len(h2h_df.columns)}")

# ============================================================================
# SAMPLE RIVALRIES
# ============================================================================

print(f"\n📊 Classic Rivalries:")

rivalries = [
    ('MI', 'CSK'),  # Most popular rivalry
    ('KKR', 'MI'),  # 2024 Finals matchup
    ('RCB', 'CSK'),  # South Indian derby
    ('DC', 'MI')    # Capital vs Financial capital
]

for team_a, team_b in rivalries:
    record_ab = h2h_df[(h2h_df['team_a'] == team_a) & (h2h_df['team_b'] == team_b)].iloc[0]
    record_ba = h2h_df[(h2h_df['team_a'] == team_b) & (h2h_df['team_b'] == team_a)].iloc[0]
    
    print(f"\n  {team_a} vs {team_b}:")
    print(f"    Matches: {record_ab['h2h_matches']}")
    print(f"    {team_a} wins: {record_ab['h2h_wins']} ({record_ab['h2h_win_rate']:.1%})")
    print(f"    {team_b} wins: {record_ba['h2h_wins']} ({record_ba['h2h_win_rate']:.1%})")
    print(f"    Recent form: {team_a} {record_ab['h2h_recent_form']:.2f} | {team_b} {record_ba['h2h_recent_form']:.2f}")
    
    # Use win_rate directly with refined thresholds
    wr_a = record_ab['h2h_win_rate']
    wr_b = record_ba['h2h_win_rate']
    
    if wr_a >= 0.70:  # 70%+ = dominance
        print(f"    ⚡ {team_a} dominates! ({wr_a:.1%} win rate)")
    elif wr_b >= 0.70:
        print(f"    ⚡ {team_b} dominates! ({wr_b:.1%} win rate)")
    elif 0.45 <= wr_a <= 0.55:  # 45-55% = evenly matched
        print(f"    ⚖️  Evenly matched ({wr_a:.1%} vs {wr_b:.1%})")
    elif 0.55 < wr_a < 0.70:  # 55-70% = upper hand
        print(f"    📊 {team_a} has upper hand ({wr_a:.1%} vs {wr_b:.1%})")
    elif 0.55 < wr_b < 0.70:
        print(f"    📊 {team_b} has upper hand ({wr_b:.1%} vs {wr_a:.1%})")
    else:
        # Edge cases (shouldn't happen with proper data)
        print(f"    🤔 Unclear advantage ({wr_a:.1%} vs {wr_b:.1%})")

print(f"\n" + "=" * 80)
print(f"✅ H2H stats ready for TabTransformer (15% weight)!")
print(f"=" * 80)