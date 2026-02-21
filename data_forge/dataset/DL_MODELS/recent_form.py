"""
================================================================================
TEAM RECENT FORM GENERATOR
================================================================================
Calculate rolling form metrics for all IPL 2025 teams from 2021-2024 data

Features Generated (per team):
  - last_5_win_rate: Win % in last 5 matches
  - last_5_avg_score: Average runs scored (last 5)
  - last_5_avg_conceded: Average runs conceded (last 5)
  - last_5_net_run_rate: NRR over last 5 matches
  - last_10_win_rate: Win % in last 10 matches
  - form_trend: Momentum (+1 improving, 0 stable, -1 declining)
  - wins_streak: Current winning streak (0 if on loss)
  - losses_streak: Current losing streak (0 if on win)
  - last_match_result: 1 = win, 0 = loss
  - days_since_last_match: Recency factor

Data Source: rajsengo all_season_summary.csv (2021-2024)
Output: team_recent_form_2025.csv (10 teams)

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
ALL_SEASONS_FILE = BASE_DIR / "all_season_summary.csv"  # 2021-2023
SEASON_2024_FILE = BASE_DIR / "2024" / "season_summary.csv"  # 2024

OUTPUT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPL 2025 teams
IPL_2025_TEAMS = [
    'CSK', 'DC', 'GT', 'KKR', 'LSG', 
    'MI', 'PK', 'RCB', 'RR', 'SRH'
]

# Team name mappings
TEAM_NAME_MAPPING = {
    'CSK': 'CSK',
    'DC': 'DC',
    'GT': 'GT',
    'KKR': 'KKR',
    'LSG': 'LSG',
    'MI': 'MI',
    'RCB': 'RCB',
    'RR': 'RR',
    'SRH': 'SRH',
    'PBKS': 'PK',
    'KXIP': 'PK',
    # Defunct teams
    'GL': None,
    'Kochi': None,
    'PWI': None,
    'RPS': None
}

print("=" * 80)
print("TEAM RECENT FORM GENERATOR")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n📁 Loading match data...")

all_dfs = []

# Load 2021-2023 data
if ALL_SEASONS_FILE.exists():
    df_hist = pd.read_csv(ALL_SEASONS_FILE)
    all_dfs.append(df_hist)
    print(f"  ✅ 2021-2023: {len(df_hist)} matches")
else:
    print(f"  ⚠️  all_season_summary.csv not found")

# Load 2024 data
if SEASON_2024_FILE.exists():
    df_2024 = pd.read_csv(SEASON_2024_FILE)
    all_dfs.append(df_2024)
    print(f"  ✅ 2024: {len(df_2024)} matches")
else:
    print(f"  ⚠️  2024/season_summary.csv not found")

if len(all_dfs) == 0:
    raise FileNotFoundError("No data files found!")

matches_df = pd.concat(all_dfs, ignore_index=True)
print(f"  ✅ Total loaded: {len(matches_df)} matches")

# ============================================================================
# PREPROCESS DATA
# ============================================================================

print(f"\n🔧 Preprocessing...")

# Convert season to numeric and filter 2021-2024
matches_df['season'] = pd.to_numeric(matches_df['season'], errors='coerce')
matches_df = matches_df[matches_df['season'].between(2021, 2024)]
print(f"  ✅ Filtered to 2021-2024: {len(matches_df)} matches")

# Parse dates for recency calculation
matches_df['match_date'] = pd.to_datetime(matches_df['start_date'], errors='coerce')

# Sort by date (chronological order)
matches_df = matches_df.sort_values('match_date').reset_index(drop=True)

# Standardize team names
matches_df['home_team_clean'] = matches_df['home_team'].map(TEAM_NAME_MAPPING)
matches_df['away_team_clean'] = matches_df['away_team'].map(TEAM_NAME_MAPPING)
matches_df['winner_clean'] = matches_df['winner'].map(TEAM_NAME_MAPPING)

# Filter to IPL 2025 teams only
matches_df = matches_df[
    matches_df['home_team_clean'].isin(IPL_2025_TEAMS) & 
    matches_df['away_team_clean'].isin(IPL_2025_TEAMS) &
    matches_df['home_team_clean'].notna() &
    matches_df['away_team_clean'].notna()
].copy()

print(f"  ✅ Filtered to IPL 2025 teams: {len(matches_df)} matches")

# Extract scores (handle different column formats)
if 'home_runs' in matches_df.columns:
    matches_df['home_score'] = matches_df['home_runs']
    matches_df['away_score'] = matches_df['away_runs']
elif 'home_score' not in matches_df.columns:
    # Parse from 1st_inning_score / 2nd_inning_score
    print("  ⚠️  Score columns not found, using inning scores")

print(f"  ✅ Data preprocessed")

# ============================================================================
# CALCULATE RECENT FORM
# ============================================================================

print(f"\n📊 Calculating recent form for each team...")

team_forms = []

for team in IPL_2025_TEAMS:
    # Get all matches for this team (home or away)
    team_matches = matches_df[
        (matches_df['home_team_clean'] == team) | 
        (matches_df['away_team_clean'] == team)
    ].copy()
    
    if len(team_matches) == 0:
        print(f"  ⚠️  {team}: No matches found, using neutral defaults")
        team_forms.append({
            'IPL_Team_2025': team,
            'last_5_win_rate': 0.5,
            'last_5_avg_score': 160.0,
            'last_5_avg_conceded': 160.0,
            'last_5_net_run_rate': 0.0,
            'last_10_win_rate': 0.5,
            'form_trend': 0.0,
            'wins_streak': 0,
            'losses_streak': 0,
            'last_match_result': 0.5,
            'days_since_last_match': 365
        })
        continue
    
    # Sort by date (most recent last)
    team_matches = team_matches.sort_values('match_date')
    
    # Determine results (1 = win, 0 = loss)
    team_matches['is_win'] = (team_matches['winner_clean'] == team).astype(int)
    
    # Get scores for/against
    team_matches['score_for'] = np.where(
        team_matches['home_team_clean'] == team,
        team_matches['home_score'],
        team_matches['away_score']
    )
    team_matches['score_against'] = np.where(
        team_matches['home_team_clean'] == team,
        team_matches['away_score'],
        team_matches['home_score']
    )
    
    # Last 5 matches
    last_5 = team_matches.tail(5)
    last_5_wins = last_5['is_win'].sum()
    last_5_win_rate = last_5_wins / len(last_5) if len(last_5) > 0 else 0.5
    last_5_avg_score = last_5['score_for'].mean() if len(last_5) > 0 else 160.0
    last_5_avg_conceded = last_5['score_against'].mean() if len(last_5) > 0 else 160.0
    
    # NRR calculation (runs per over difference)
    # Assuming 20 overs per innings (T20)
    last_5_nrr = (last_5_avg_score - last_5_avg_conceded) / 20.0 if len(last_5) > 0 else 0.0
    
    # Last 10 matches
    last_10 = team_matches.tail(10)
    last_10_wins = last_10['is_win'].sum()
    last_10_win_rate = last_10_wins / len(last_10) if len(last_10) > 0 else 0.5
    
    # Form trend: compare last 3 vs previous 3
    if len(team_matches) >= 6:
        last_3 = team_matches.tail(3)['is_win'].mean()
        prev_3 = team_matches.tail(6).head(3)['is_win'].mean()
        form_trend = last_3 - prev_3  # +1 to -1 scale
    else:
        form_trend = 0.0
    
    # Current streak
    wins_streak = 0
    losses_streak = 0
    
    for idx in range(len(team_matches) - 1, -1, -1):
        result = team_matches.iloc[idx]['is_win']
        if result == 1:
            if losses_streak > 0:
                break
            wins_streak += 1
        else:
            if wins_streak > 0:
                break
            losses_streak += 1
    
    # Last match result
    last_match_result = team_matches.iloc[-1]['is_win'] if len(team_matches) > 0 else 0.5
    
    # Days since last match
    last_match_date = team_matches.iloc[-1]['match_date']
    # Use end of 2024 season as reference (Nov 30, 2024)
    reference_date = pd.Timestamp('2024-11-30')
    
    if pd.notna(last_match_date):
        # Remove timezone info if present
        if hasattr(last_match_date, 'tz') and last_match_date.tz is not None:
            last_match_date = last_match_date.tz_localize(None)
        days_since = (reference_date - last_match_date).days
    else:
        days_since = 365
    
    team_forms.append({
        'IPL_Team_2025': team,
        'last_5_win_rate': round(last_5_win_rate, 3),
        'last_5_avg_score': round(last_5_avg_score, 1),
        'last_5_avg_conceded': round(last_5_avg_conceded, 1),
        'last_5_net_run_rate': round(last_5_nrr, 3),
        'last_10_win_rate': round(last_10_win_rate, 3),
        'form_trend': round(form_trend, 3),
        'wins_streak': wins_streak,
        'losses_streak': losses_streak,
        'last_match_result': int(last_match_result),
        'days_since_last_match': int(days_since)
    })
    
    print(f"  ✅ {team}: Last 5 = {last_5_win_rate:.1%} ({wins_streak}W / {losses_streak}L streak)")

form_df = pd.DataFrame(team_forms)

# ============================================================================
# SAVE OUTPUT
# ============================================================================

output_file = OUTPUT_DIR / "team_recent_form_2025.csv"
form_df.to_csv(output_file, index=False)

print(f"\n" + "=" * 80)
print(f"✅ TEAM RECENT FORM GENERATED!")
print(f"=" * 80)
print(f"\nOutput: {output_file}")
print(f"Teams: {len(form_df)}")
print(f"Features: {len(form_df.columns)}")

# ============================================================================
# TEAM RANKINGS BY FORM
# ============================================================================

print(f"\n📊 Team Rankings by Recent Form:")

# Sort by last_5_win_rate
form_ranked = form_df.sort_values('last_5_win_rate', ascending=False)

for idx, row in form_ranked.iterrows():
    team = row['IPL_Team_2025']
    wr5 = row['last_5_win_rate']
    trend = row['form_trend']
    streak_w = row['wins_streak']
    streak_l = row['losses_streak']
    
    # Trend indicator
    if trend > 0.3:
        trend_icon = "📈"
    elif trend < -0.3:
        trend_icon = "📉"
    else:
        trend_icon = "➡️"
    
    # Streak info
    if streak_w > 0:
        streak_info = f"🔥 {streak_w}W streak"
    elif streak_l > 0:
        streak_info = f"❄️ {streak_l}L streak"
    else:
        streak_info = "—"
    
    print(f"  {team}: {wr5:.1%} {trend_icon} (Trend: {trend:+.2f}) | {streak_info}")

print(f"\n" + "=" * 80)
print(f"✅ Recent form ready for TabTransformer (20% weight)!")
print(f"=" * 80)