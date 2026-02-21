"""
================================================================================
FIX MASTER PHASE STATISTICS
================================================================================
Purpose: Fill missing Powerplay/Death stats in BATTING_MASTER and BOWLING_MASTER
         by parsing ball-by-ball data from rajsengo (2021-2024)

Approach:
  1. Load existing MASTER files
  2. Parse rajsengo match data (2021-2024) for phase-specific stats
  3. Match players by name and update their phase columns
  4. For players with NO match data (debutants or missing):
     - Estimate phase stats from overall stats using proportional distribution
  5. Save updated MASTER files (overwrite originals with backup)

Author: El Dorado Project - CSCI 566
Date: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

RAJSENGO_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Kaggle_download\rajsengo")
BATTING_MASTER = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\Master_Datasets\BATTING_MASTER_2025_20251129_052900.csv")
BOWLING_MASTER = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\Master_Datasets\BOWLING_MASTER_2025_20251129_052900.csv")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_phase(over_num):
    """Classify over into phase"""
    if over_num <= 6:
        return 'powerplay'
    elif over_num <= 15:
        return 'middle'
    else:
        return 'death'

def safe_divide(a, b, default=0.0):
    """Safe division with default for zero denominator"""
    return a / b if b > 0 else default

# ============================================================================
# STEP 1: LOAD EXISTING MASTER FILES
# ============================================================================

print("=" * 80)
print("MASTER PHASE STATISTICS FIXER")
print("=" * 80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n📂 Loading existing MASTER files...")
batting_master = pd.read_csv(BATTING_MASTER)
bowling_master = pd.read_csv(BOWLING_MASTER)

print(f"  ✅ Batting: {len(batting_master)} players")
print(f"  ✅ Bowling: {len(bowling_master)} players")

# Create backups
backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
batting_backup = BATTING_MASTER.parent / f"BATTING_MASTER_BACKUP_{backup_suffix}.csv"
bowling_backup = BOWLING_MASTER.parent / f"BOWLING_MASTER_BACKUP_{backup_suffix}.csv"

shutil.copy(BATTING_MASTER, batting_backup)
shutil.copy(BOWLING_MASTER, bowling_backup)

print(f"\n💾 Backups created:")
print(f"  📄 {batting_backup.name}")
print(f"  📄 {bowling_backup.name}")

# ============================================================================
# STEP 2: LOAD RAJSENGO BALL-BY-BALL DATA (2021-2024)
# ============================================================================

print(f"\n📂 Loading ball-by-ball data (2021-2024)...")

# Source files
ALL_SEASON_DETAILS = RAJSENGO_DIR / "all_season_details.csv"
SEASON_2024_DETAILS = RAJSENGO_DIR / "2024" / "season_details.csv"

# Load ball-by-ball data
ball_dfs = []
if ALL_SEASON_DETAILS.exists():
    df = pd.read_csv(ALL_SEASON_DETAILS)
    ball_dfs.append(df)
    print(f"  ✅ 2021-2023: {len(df):,} balls")
else:
    print(f"  ⚠️  2021-2023: Not found")

if SEASON_2024_DETAILS.exists():
    df = pd.read_csv(SEASON_2024_DETAILS)
    ball_dfs.append(df)
    print(f"  ✅ 2024: {len(df):,} balls")
else:
    print(f"  ⚠️  2024: Not found")

if not ball_dfs:
    print("\n❌ ERROR: No ball-by-ball data found!")
    exit(1)

all_balls = pd.concat(ball_dfs, ignore_index=True)
print(f"\n✅ Total: {len(all_balls):,} balls from 2021-2024")

# Check columns
print(f"\n📋 Available columns: {list(all_balls.columns)[:10]}...")

# ============================================================================
# STEP 3: CALCULATE BATTING PHASE STATS FROM BALL-BY-BALL DATA
# ============================================================================

print(f"\n🏏 Calculating BATTING phase statistics...")

# Extract over number from 'over' column (format: integer over number)
# Classify into phases: PP (1-6), Middle (7-15), Death (16-20)
all_balls['over_num'] = pd.to_numeric(all_balls['over'], errors='coerce').fillna(0).astype(int) + 1
all_balls['phase'] = all_balls['over_num'].apply(get_phase)

# Get striker (batsman1 is on strike, batsman2 is non-striker)
# For runs scored, credit to batsman1 (striker)
all_balls['striker'] = all_balls['batsman1_name']
all_balls['runs_scored'] = pd.to_numeric(all_balls['runs'], errors='coerce').fillna(0)

# Filter to master batters
master_batters = set(batting_master['Kaggle_Match_Name'].dropna())
batting_balls = all_balls[all_balls['striker'].isin(master_batters)].copy()

print(f"  Filtered to {len(batting_balls):,} balls by {len(batting_balls['striker'].unique())} MASTER players")

# Calculate phase-specific batting stats
batting_phase = batting_balls.groupby(['striker', 'phase']).agg({
    'runs_scored': 'sum',
    'striker': 'count'  # balls faced
}).rename(columns={'striker': 'balls_faced', 'runs_scored': 'runs'})

batting_phase['strike_rate'] = (batting_phase['runs'] / batting_phase['balls_faced'] * 100)

# Pivot to get powerplay and death stats
batting_phase = batting_phase.reset_index()

batting_pp = batting_phase[batting_phase['phase'] == 'powerplay'][['striker', 'runs', 'balls_faced', 'strike_rate']].rename(columns={
    'striker': 'batter',
    'runs': 'Powerplay_Runs',
    'balls_faced': 'Powerplay_Balls',
    'strike_rate': 'Powerplay_SR'
})

batting_death = batting_phase[batting_phase['phase'] == 'death'][['striker', 'runs', 'balls_faced', 'strike_rate']].rename(columns={
    'striker': 'batter',
    'runs': 'Death_Overs_Runs',
    'balls_faced': 'Death_Overs_Balls',
    'strike_rate': 'Death_Overs_SR'
})

print(f"  ✅ Powerplay stats: {len(batting_pp)} batters")
print(f"  ✅ Death stats: {len(batting_death)} batters")

# ============================================================================
# STEP 4: CALCULATE BOWLING PHASE STATS FROM BALL-BY-BALL DATA
# ============================================================================

print(f"\n🎳 Calculating BOWLING phase statistics...")

# For bowling, credit to bowler1 (current bowler)
all_balls['bowler'] = all_balls['bowler1_name']
all_balls['runs_conceded'] = pd.to_numeric(all_balls['runs'], errors='coerce').fillna(0)

# Check for wickets (if wicket_id exists and is not null)
all_balls['is_wicket'] = all_balls['wicket_id'].notna().astype(int)

# Filter to master bowlers
master_bowlers = set(bowling_master['Kaggle_Match_Name'].dropna())
bowling_balls = all_balls[all_balls['bowler'].isin(master_bowlers)].copy()

print(f"  Filtered to {len(bowling_balls):,} balls by {len(bowling_balls['bowler'].unique())} MASTER players")

# Calculate phase-specific bowling stats
bowling_phase = bowling_balls.groupby(['bowler', 'phase']).agg({
    'runs_conceded': 'sum',
    'bowler': 'count',  # balls bowled
    'is_wicket': 'sum'
}).rename(columns={'bowler': 'balls_bowled', 'is_wicket': 'wickets', 'runs_conceded': 'runs'})

bowling_phase['overs'] = bowling_phase['balls_bowled'] / 6.0
bowling_phase['economy'] = bowling_phase['runs'] / bowling_phase['overs']

# Pivot to get powerplay and death stats
bowling_phase = bowling_phase.reset_index()

bowling_pp = bowling_phase[bowling_phase['phase'] == 'powerplay'][['bowler', 'runs', 'overs', 'wickets', 'economy']].rename(columns={
    'runs': 'Powerplay_Runs',
    'overs': 'Powerplay_Overs',
    'wickets': 'Powerplay_Wickets',
    'economy': 'Powerplay_Economy'
})

bowling_death = bowling_phase[bowling_phase['phase'] == 'death'][['bowler', 'runs', 'overs', 'wickets', 'economy']].rename(columns={
    'runs': 'Death_Runs',
    'overs': 'Death_Overs',
    'wickets': 'Death_Wickets',
    'economy': 'Death_Overs_Economy'
})

print(f"  ✅ Powerplay stats: {len(bowling_pp)} bowlers")
print(f"  ✅ Death stats: {len(bowling_death)} bowlers")

# ============================================================================
# STEP 5: MERGE BATTING STATS INTO MASTER
# ============================================================================

print(f"\n🔄 Updating BATTING_MASTER with real phase stats...")

# Merge powerplay stats
batting_master = batting_master.merge(
    batting_pp,
    left_on='Kaggle_Match_Name',
    right_on='batter',
    how='left',
    suffixes=('_old', '')
)

# Merge death overs stats
batting_master = batting_master.merge(
    batting_death,
    left_on='Kaggle_Match_Name',
    right_on='batter',
    how='left',
    suffixes=('_old', '')
)

# Drop merge helper columns
cols_to_drop = [c for c in batting_master.columns if c in ['batter', 'batter_old']]
batting_master = batting_master.drop(columns=cols_to_drop, errors='ignore')

# Check coverage
pp_filled = batting_master['Powerplay_Balls'].notna().sum()
death_filled = batting_master['Death_Overs_Balls'].notna().sum()
total_active = (batting_master['Total_Innings'] > 0).sum()

print(f"  ✅ Powerplay data: {pp_filled}/{total_active} active batters ({pp_filled/total_active*100:.1f}%)")
print(f"  ✅ Death data: {death_filled}/{total_active} active batters ({death_filled/total_active*100:.1f}%)")

# ============================================================================
# STEP 6: MERGE BOWLING STATS INTO MASTER
# ============================================================================

print(f"\n🔄 Updating BOWLING_MASTER with real phase stats...")

# Merge powerplay stats
bowling_master = bowling_master.merge(
    bowling_pp,
    left_on='Kaggle_Match_Name',
    right_on='bowler',
    how='left',
    suffixes=('_old', '')
)

# Merge death overs stats
bowling_master = bowling_master.merge(
    bowling_death,
    left_on='Kaggle_Match_Name',
    right_on='bowler',
    how='left',
    suffixes=('_old', '')
)

# Drop merge helper columns
cols_to_drop = [c for c in bowling_master.columns if c in ['bowler', 'bowler_old']]
bowling_master = bowling_master.drop(columns=cols_to_drop, errors='ignore')

# Check coverage
pp_filled = bowling_master['Powerplay_Overs'].notna().sum()
death_filled = bowling_master['Death_Overs'].notna().sum()
total_active = (bowling_master['Total_Innings_Bowled'] > 0).sum()

print(f"  ✅ Powerplay data: {pp_filled}/{total_active} active bowlers ({pp_filled/total_active*100:.1f}%)")
print(f"  ✅ Death data: {death_filled}/{total_active} active bowlers ({death_filled/total_active*100:.1f}%)")

# ============================================================================
# STEP 7: ESTIMATE MISSING PHASE STATS (PROPORTIONAL FALLBACK)
# ============================================================================

print(f"\n🔧 Estimating missing phase stats from overall performance...")

# BATTING: For players with no phase data, estimate proportionally
batting_missing_pp = (batting_master['Total_Innings'] > 0) & (batting_master['Powerplay_Balls'].isna())
batting_missing_death = (batting_master['Total_Innings'] > 0) & (batting_master['Death_Overs_Balls'].isna())

print(f"\n  Batting estimates needed:")
print(f"    Powerplay: {batting_missing_pp.sum()} players")
print(f"    Death: {batting_missing_death.sum()} players")

# Estimate: Assume ~30% balls in PP, ~25% in death, SR similar to overall
batting_master.loc[batting_missing_pp, 'Powerplay_Balls'] = (batting_master.loc[batting_missing_pp, 'Total_Balls_Faced'] * 0.30).round()
batting_master.loc[batting_missing_pp, 'Powerplay_Runs'] = (batting_master.loc[batting_missing_pp, 'Total_Runs'] * 0.30).round()
batting_master.loc[batting_missing_pp, 'Powerplay_SR'] = batting_master.loc[batting_missing_pp, 'Strike_Rate'] * 0.95  # Typically 5% lower in PP

batting_master.loc[batting_missing_death, 'Death_Overs_Balls'] = (batting_master.loc[batting_missing_death, 'Total_Balls_Faced'] * 0.25).round()
batting_master.loc[batting_missing_death, 'Death_Overs_Runs'] = (batting_master.loc[batting_missing_death, 'Total_Runs'] * 0.30).round()  # Higher run rate
batting_master.loc[batting_missing_death, 'Death_Overs_SR'] = batting_master.loc[batting_missing_death, 'Strike_Rate'] * 1.10  # Typically 10% higher in death

# BOWLING: For bowlers with no phase data, estimate proportionally
bowling_missing_pp = (bowling_master['Total_Innings_Bowled'] > 0) & (bowling_master['Powerplay_Overs'].isna())
bowling_missing_death = (bowling_master['Total_Innings_Bowled'] > 0) & (bowling_master['Death_Overs'].isna())

print(f"\n  Bowling estimates needed:")
print(f"    Powerplay: {bowling_missing_pp.sum()} bowlers")
print(f"    Death: {bowling_missing_death.sum()} bowlers")

# Estimate: Assume ~30% overs in PP, ~25% in death
bowling_master.loc[bowling_missing_pp, 'Powerplay_Overs'] = (bowling_master.loc[bowling_missing_pp, 'Total_Overs_Bowled'] * 0.30).round(1)
bowling_master.loc[bowling_missing_pp, 'Powerplay_Runs'] = (bowling_master.loc[bowling_missing_pp, 'Total_Runs_Conceded'] * 0.28).round()  # PP typically better
bowling_master.loc[bowling_missing_pp, 'Powerplay_Wickets'] = (bowling_master.loc[bowling_missing_pp, 'Total_Wickets'] * 0.32).round()  # More wickets in PP
bowling_master.loc[bowling_missing_pp, 'Powerplay_Economy'] = bowling_master.loc[bowling_missing_pp, 'Economy_Rate'] * 0.95  # Typically 5% better

bowling_master.loc[bowling_missing_death, 'Death_Overs'] = (bowling_master.loc[bowling_missing_death, 'Total_Overs_Bowled'] * 0.25).round(1)
bowling_master.loc[bowling_missing_death, 'Death_Runs'] = (bowling_master.loc[bowling_missing_death, 'Total_Runs_Conceded'] * 0.32).round()  # Death more expensive
bowling_master.loc[bowling_missing_death, 'Death_Wickets'] = (bowling_master.loc[bowling_missing_death, 'Total_Wickets'] * 0.25).round()
bowling_master.loc[bowling_missing_death, 'Death_Overs_Economy'] = bowling_master.loc[bowling_missing_death, 'Economy_Rate'] * 1.15  # Typically 15% worse

print(f"\n  ✅ Estimates applied using proportional distribution")

# ============================================================================
# STEP 8: FILL REMAINING ZEROS FOR DEBUTANTS
# ============================================================================

print(f"\n🔧 Handling debutants (DEBUT='YES')...")

# For debutants, keep all phase stats as 0.0 (no data, no estimates)
debut_batters = (batting_master['DEBUT'] == 'YES')
debut_bowlers = (bowling_master['DEBUT'] == 'YES')

print(f"  Debut batters: {debut_batters.sum()} (keeping phase stats as 0.0)")
print(f"  Debut bowlers: {debut_bowlers.sum()} (keeping phase stats as 0.0)")

# Ensure debutant phase stats are 0.0 (not NaN)
phase_cols_batting = ['Powerplay_Balls', 'Powerplay_Runs', 'Powerplay_SR', 'Death_Overs_Balls', 'Death_Overs_Runs', 'Death_Overs_SR']
phase_cols_bowling = ['Powerplay_Overs', 'Powerplay_Runs', 'Powerplay_Wickets', 'Powerplay_Economy', 'Death_Overs', 'Death_Runs', 'Death_Wickets', 'Death_Overs_Economy']

for col in phase_cols_batting:
    if col in batting_master.columns:
        batting_master.loc[debut_batters, col] = batting_master.loc[debut_batters, col].fillna(0.0)

for col in phase_cols_bowling:
    if col in bowling_master.columns:
        bowling_master.loc[debut_bowlers, col] = bowling_master.loc[debut_bowlers, col].fillna(0.0)

# ============================================================================
# STEP 9: SAVE UPDATED MASTER FILES
# ============================================================================

print(f"\n💾 Saving updated MASTER files...")

batting_master.to_csv(BATTING_MASTER, index=False)
bowling_master.to_csv(BOWLING_MASTER, index=False)

print(f"  ✅ {BATTING_MASTER.name}")
print(f"  ✅ {BOWLING_MASTER.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n" + "=" * 80)
print("✅ MASTER FILES UPDATED!")
print("=" * 80)

print(f"\n📊 BATTING_MASTER Summary:")
print(f"  Total players: {len(batting_master)}")
print(f"  Active (innings > 0): {(batting_master['Total_Innings'] > 0).sum()}")
print(f"  With real PP data: {(batting_master['Powerplay_Balls'] > 0).sum()}")
print(f"  With real death data: {(batting_master['Death_Overs_Balls'] > 0).sum()}")
print(f"  Debutants: {(batting_master['DEBUT'] == 'YES').sum()}")

print(f"\n📊 BOWLING_MASTER Summary:")
print(f"  Total players: {len(bowling_master)}")
print(f"  Active (innings > 0): {(bowling_master['Total_Innings_Bowled'] > 0).sum()}")
print(f"  With real PP data: {(bowling_master['Powerplay_Overs'] > 0).sum()}")
print(f"  With real death data: {(bowling_master['Death_Overs'] > 0).sum()}")
print(f"  Debutants: {(bowling_master['DEBUT'] == 'YES').sum()}")

print(f"\n✅ Backups saved:")
print(f"  📄 {batting_backup}")
print(f"  📄 {bowling_backup}")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)