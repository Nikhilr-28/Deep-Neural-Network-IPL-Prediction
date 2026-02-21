"""
═══════════════════════════════════════════════════════════════════════════════
IPL 2025 MASTER DATASET CREATION - COMPREHENSIVE PLAYER STATISTICS
═══════════════════════════════════════════════════════════════════════════════

DATA RANGE: 2021-2024 (4 seasons)
PURPOSE: Create comprehensive player statistics for OVR calculation

OUTPUTS:
    1. BATTING_MASTER_2025.csv (~60 attributes)
    2. BOWLING_MASTER_2025.csv (~70 attributes)
    3. master_dataset_report.md

FEATURES:
    ✅ Captain win percentage (from match outcomes)
    ✅ T20-focused milestones (30+, 50+, 70+)
    ✅ Impact metrics (balls per 4, balls per 6)
    ✅ Phase-specific stats (powerplay, death)
    ✅ All-rounders in BOTH files
    ✅ Debuts at bottom with all 0s

Author: El Dorado Project - CSCI 566
Date: November 24, 2024
Version: 2.0 - Comprehensive Edition
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil")
DATA_DIR = ROOT_DIR / "Dataset(s) and code" / "dataset"
KAGGLE_DIR = ROOT_DIR / "Kaggle_download"
OUTPUT_DIR = DATA_DIR / "Master_Datasets"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Data range
START_YEAR = 2021
END_YEAR = 2024

# Team order for sorting
TEAM_ORDER = ['CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']

# ═══════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════

# Clean nomenclature files (find most recent)
NOMENCLATURE_DIR = DATA_DIR / "Clean_Nomenclature"

# Function to find most recent nomenclature files
def find_latest_nomenclature_files():
    """Find the most recent nomenclature files"""
    current_files = list(NOMENCLATURE_DIR.glob("current_players_2025_*.csv"))
    debut_files = list(NOMENCLATURE_DIR.glob("debut_players_2025_*.csv"))
    
    if not current_files or not debut_files:
        # Fallback to exact names if pattern fails
        current_file = NOMENCLATURE_DIR / "current_players_2025.csv"
        debut_file = NOMENCLATURE_DIR / "debut_players_2025.csv"
        if current_file.exists() and debut_file.exists():
            return current_file, debut_file
        raise FileNotFoundError("Cannot find nomenclature files in Clean_Nomenclature directory")
    
    # Get most recent based on filename timestamp
    current_file = sorted(current_files)[-1]
    debut_file = sorted(debut_files)[-1]
    return current_file, debut_file

# rajsengo all-season files
RAJSENGO_DIR = KAGGLE_DIR / "rajsengo"
ALL_BATTING = RAJSENGO_DIR / "all_season_batting_card.csv"
ALL_BOWLING = RAJSENGO_DIR / "all_season_bowling_card.csv"
ALL_SUMMARY = RAJSENGO_DIR / "all_season_summary.csv"

# 2024 separate files
BATTING_2024 = RAJSENGO_DIR / "2024" / "season_batting_card.csv"
BOWLING_2024 = RAJSENGO_DIR / "2024" / "season_bowling_card.csv"
SUMMARY_2024 = RAJSENGO_DIR / "2024" / "season_summary.csv"

# iamsouravbanerjee milestone files (use what's available)
IAMSOURAV_DIR = KAGGLE_DIR / "iamsouravbanerjee"

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def print_section(title):
    """Print formatted section header"""
    print("\n" + "═" * 80)
    print(f"  {title}")
    print("═" * 80)

def print_progress(message):
    """Print progress message"""
    print(f"  ⚙️  {message}")

def print_success(message):
    """Print success message"""
    print(f"  ✅ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"  ⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"  ❌ {message}")

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        result = numerator / denominator
        return result if not np.isnan(result) and not np.isinf(result) else default
    except:
        return default

def overs_to_balls(overs):
    """Convert cricket overs to balls (e.g., 3.2 = 20 balls)"""
    try:
        if pd.isna(overs):
            return 0
        complete_overs = int(overs)
        remaining_balls = int(round((overs - complete_overs) * 10))
        return (complete_overs * 6) + remaining_balls
    except:
        return 0

def balls_to_overs(balls):
    """Convert balls to overs (e.g., 20 balls = 3.2 overs)"""
    try:
        if pd.isna(balls) or balls == 0:
            return 0.0
        complete_overs = balls // 6
        remaining_balls = balls % 6
        return complete_overs + (remaining_balls / 10)
    except:
        return 0.0

def parse_minutes(minutes_str):
    """Parse time formats: '72' or '72:34' or '1:12:45' to minutes"""
    try:
        if pd.isna(minutes_str):
            return 0
        s = str(minutes_str).strip()
        if s.isdigit():
            return int(s)
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 2:
                return int(parts[0])
            elif len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0

def calculate_coefficient_of_variation(series):
    """Calculate CV (lower = more consistent)"""
    try:
        if len(series) < 2:
            return 0.0
        mean = series.mean()
        std = series.std()
        if mean == 0:
            return 0.0
        return round((std / mean) * 100, 2)
    except:
        return 0.0

def detect_phase(running_over):
    """Detect match phase from running over"""
    try:
        over = float(running_over)
        if over <= 6:
            return 'powerplay'
        elif over <= 15:
            return 'middle'
        else:
            return 'death'
    except:
        return 'unknown'

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all required data files"""
    print_section("LOADING DATA FILES")
    
    data = {}
    
    # Find and load player lists
    print_progress("Finding latest nomenclature files...")
    CURRENT_PLAYERS, DEBUT_PLAYERS = find_latest_nomenclature_files()
    print_success(f"Found: {CURRENT_PLAYERS.name}")
    print_success(f"Found: {DEBUT_PLAYERS.name}")
    
    print_progress("Loading player nomenclature files...")
    data['current_players'] = pd.read_csv(CURRENT_PLAYERS)
    data['debut_players'] = pd.read_csv(DEBUT_PLAYERS)
    print_success(f"Current players: {len(data['current_players'])}")
    print_success(f"Debut players: {len(data['debut_players'])}")
    
    # Load all-season batting data (2008-2023, filter to 2021-2023)
    print_progress("Loading all-season batting data (2021-2023)...")
    df_batting_all = pd.read_csv(ALL_BATTING)
    # Ensure season is numeric
    df_batting_all['season'] = pd.to_numeric(df_batting_all['season'], errors='coerce')
    batting_2021_2023 = df_batting_all[
        (df_batting_all['season'] >= START_YEAR) & 
        (df_batting_all['season'] <= 2023)
    ].copy()
    print_success(f"Batting records (2021-2023): {len(batting_2021_2023)}")
    
    # Load 2024 batting data
    print_progress("Loading 2024 batting data...")
    batting_2024 = pd.read_csv(BATTING_2024)
    batting_2024['season'] = pd.to_numeric(batting_2024['season'], errors='coerce')
    print_success(f"Batting records (2024): {len(batting_2024)}")
    
    # Combine
    data['batting'] = pd.concat([batting_2021_2023, batting_2024], ignore_index=True)
    print_success(f"Total batting records (2021-2024): {len(data['batting'])}")
    print_progress(f"  Years: {sorted(data['batting']['season'].unique())}")
    
    # Load all-season bowling data (2008-2023, filter to 2021-2023)
    print_progress("Loading all-season bowling data (2021-2023)...")
    df_bowling_all = pd.read_csv(ALL_BOWLING)
    # Ensure season is numeric
    df_bowling_all['season'] = pd.to_numeric(df_bowling_all['season'], errors='coerce')
    bowling_2021_2023 = df_bowling_all[
        (df_bowling_all['season'] >= START_YEAR) & 
        (df_bowling_all['season'] <= 2023)
    ].copy()
    print_success(f"Bowling records (2021-2023): {len(bowling_2021_2023)}")
    
    # Load 2024 bowling data
    print_progress("Loading 2024 bowling data...")
    bowling_2024 = pd.read_csv(BOWLING_2024)
    bowling_2024['season'] = pd.to_numeric(bowling_2024['season'], errors='coerce')
    print_success(f"Bowling records (2024): {len(bowling_2024)}")
    
    # Combine
    data['bowling'] = pd.concat([bowling_2021_2023, bowling_2024], ignore_index=True)
    print_success(f"Total bowling records (2021-2024): {len(data['bowling'])}")
    print_progress(f"  Years: {sorted(data['bowling']['season'].unique())}")
    
    # Load match summary (2021-2023 from all_season)
    print_progress("Loading match summary data (2021-2023)...")
    df_summary_all = pd.read_csv(ALL_SUMMARY)
    summary_2021_2023 = df_summary_all[
        (df_summary_all['season'] >= START_YEAR) & 
        (df_summary_all['season'] <= 2023)
    ].copy()
    print_success(f"Match summaries (2021-2023): {len(summary_2021_2023)}")
    
    # Load 2024 summary
    if SUMMARY_2024.exists():
        print_progress("Loading 2024 summary data...")
        summary_2024 = pd.read_csv(SUMMARY_2024)
        print_success(f"Match summaries (2024): {len(summary_2024)}")
        data['summary'] = pd.concat([summary_2021_2023, summary_2024], ignore_index=True)
    else:
        print_warning("2024 summary file not found, using 2021-2023 only")
        data['summary'] = summary_2021_2023
    
    print_success(f"Total match summaries (2021-2024): {len(data['summary'])}")
    
    # Create match outcome mapping
    print_progress("Creating match outcome mapping...")
    data['match_outcomes'] = create_match_outcome_map(data['summary'])
    print_success(f"Match outcomes mapped: {len(data['match_outcomes'])}")
    
    # Load milestone data (best effort - use what's available)
    print_progress("Loading milestone data (best effort)...")
    data['milestones'] = load_milestone_data()
    
    # Load debut player pre-calculated data
    print_progress("Loading debut player statistics...")
    DEBUT_BATTING_CSV = NOMENCLATURE_DIR / "debut_batting.csv"
    DEBUT_BOWLING_CSV = NOMENCLATURE_DIR / "debut_bowling.csv"
    
    if DEBUT_BATTING_CSV.exists():
        data['debut_batting_stats'] = pd.read_csv(DEBUT_BATTING_CSV)
        print_success(f"  Loaded debut batting stats: {len(data['debut_batting_stats'])} players")
    else:
        print_warning("  debut_batting.csv not found - will use zeros for all debuts")
        data['debut_batting_stats'] = pd.DataFrame()
    
    if DEBUT_BOWLING_CSV.exists():
        data['debut_bowling_stats'] = pd.read_csv(DEBUT_BOWLING_CSV)
        print_success(f"  Loaded debut bowling stats: {len(data['debut_bowling_stats'])} players")
    else:
        print_warning("  debut_bowling.csv not found - will use zeros for all debuts")
        data['debut_bowling_stats'] = pd.DataFrame()
    
    return data    

def create_match_outcome_map(summary_df):
    """Create match_id -> winner mapping"""
    outcome_map = {}
    
    for _, row in summary_df.iterrows():
        match_id = row.get('match_id')
        winner = row.get('winner')
        
        if pd.notna(match_id) and pd.notna(winner):
            outcome_map[match_id] = winner
    
    return outcome_map

def load_milestone_data():
    """Load milestone files (best effort, use what exists)"""
    milestones = {}
    
    # Try to load various milestone files
    milestone_files = {
        'most_fours': IAMSOURAV_DIR / "All Seasons Combined" / "Most Fours Per Innings All Seasons Combine.csv",
        'most_sixes': IAMSOURAV_DIR / "All Seasons Combined" / "Most Sixes Per Innings All Seasons Combine.csv",
        'most_wickets': IAMSOURAV_DIR / "All Seasons Combined" / "Most Wickets All Seasons Combine.csv",
        'best_economy': IAMSOURAV_DIR / "All Seasons Combined" / "Best Bowling Economy Per Innings All Seasons Combine.csv",
        'most_dots': IAMSOURAV_DIR / "All Seasons Combined" / "Most Dot Balls Per Innings All Seasons Combine.csv",
    }
    
    for key, filepath in milestone_files.items():
        if filepath.exists():
            try:
                milestones[key] = pd.read_csv(filepath)
                print_success(f"  Loaded: {key}")
            except:
                print_warning(f"  Failed to load: {key}")
        else:
            print_warning(f"  Not found: {key}")
    
    return milestones

# ═══════════════════════════════════════════════════════════════════════════
# BATTING STATISTICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_batting_stats(player_row, batting_df, match_outcomes):
    """Calculate all batting statistics for a player"""
    
    # Extract player name variants
    player_name_2025 = player_row['2025_FullName']
    short_name = player_row['Kaggle_ShortName']
    full_name = player_row['Kaggle_FullName']
    all_variants = player_row['All_Variants'].split('|') if pd.notna(player_row.get('All_Variants')) else []
    
    # Filter player data (match any variant)
    # Strip whitespace from data for better matching
    batting_df_clean = batting_df.copy()
    batting_df_clean['name'] = batting_df_clean['name'].astype(str).str.strip()
    batting_df_clean['fullName'] = batting_df_clean['fullName'].astype(str).str.strip()
    
    # Also strip variants
    all_variants_clean = [v.strip() for v in all_variants]
    
    player_data = batting_df_clean[
        (batting_df_clean['name'].isin(all_variants_clean)) | 
        (batting_df_clean['fullName'].isin(all_variants_clean))
    ].copy()
    
    # Remove duplicates (same match_id + innings_id = duplicate row)
    # Keep first occurrence
    if 'match_id' in player_data.columns and 'innings_id' in player_data.columns:
        before_dedup = len(player_data)
        player_data = player_data.drop_duplicates(subset=['match_id', 'innings_id'], keep='first')
        after_dedup = len(player_data)
        # Only log if duplicates found
        # if before_dedup != after_dedup:
        #     print(f"    {player_name_2025}: Removed {before_dedup - after_dedup} duplicate innings")
    
    if len(player_data) == 0:
        return create_empty_batting_stats(player_row)
    
    stats = {}
    
    # === CATEGORY 1: IDENTIFICATION ===
    stats['Player_Name'] = player_name_2025
    stats['Kaggle_Match_Name'] = short_name
    stats['Player_Type'] = player_row['Player_Type']
    stats['IPL_Team_2025'] = player_row['IPL_Team_2025']
    stats['DEBUT'] = 'NO'
    
    # === CATEGORY 2: VOLUME METRICS ===
    stats['Total_Innings'] = len(player_data)
    stats['Total_Runs'] = player_data['runs'].sum()
    stats['Total_Balls_Faced'] = player_data['ballsFaced'].sum()
    stats['Total_Minutes'] = player_data['minutes'].apply(parse_minutes).sum()
    stats['Dismissals'] = len(player_data[player_data['isNotOut'] == False])
    stats['Times_Not_Out'] = len(player_data[player_data['isNotOut'] == True])
    
    # === CATEGORY 3: CORE AVERAGES ===
    stats['Batting_Average'] = safe_divide(stats['Total_Runs'], stats['Dismissals'])
    stats['Strike_Rate'] = safe_divide(stats['Total_Runs'] * 100, stats['Total_Balls_Faced'])
    stats['Avg_Runs_Per_Innings'] = safe_divide(stats['Total_Runs'], stats['Total_Innings'])
    stats['Avg_Balls_Per_Innings'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Innings'])
    stats['Avg_Minutes_Per_Innings'] = safe_divide(stats['Total_Minutes'], stats['Total_Innings'])
    
    # === CATEGORY 4: MILESTONES (T20 FOCUSED) ===
    stats['Count_30_Plus'] = len(player_data[player_data['runs'] >= 30])
    stats['Count_50_Plus'] = len(player_data[player_data['runs'] >= 50])
    stats['Count_70_Plus'] = len(player_data[player_data['runs'] >= 70])
    stats['Highest_Score'] = player_data['runs'].max()
    
    # Most boundaries in single innings
    stats['Most_Fours_Innings'] = player_data['fours'].max()
    stats['Most_Sixes_Innings'] = player_data['sixes'].max()
    
    # === CATEGORY 5: BOUNDARY METRICS ===
    stats['Total_Fours'] = player_data['fours'].sum()
    stats['Total_Sixes'] = player_data['sixes'].sum()
    stats['Total_Boundaries'] = stats['Total_Fours'] + stats['Total_Sixes']
    stats['Boundary_Runs'] = (stats['Total_Fours'] * 4) + (stats['Total_Sixes'] * 6)
    stats['Boundary_Percentage'] = safe_divide(stats['Boundary_Runs'] * 100, stats['Total_Runs'])
    stats['Boundary_Frequency'] = safe_divide(stats['Total_Boundaries'] * 100, stats['Total_Balls_Faced'])
    
    stats['Fours_Per_Innings'] = safe_divide(stats['Total_Fours'], stats['Total_Innings'])
    stats['Sixes_Per_Innings'] = safe_divide(stats['Total_Sixes'], stats['Total_Innings'])
    stats['Four_to_Six_Ratio'] = safe_divide(stats['Total_Fours'], stats['Total_Sixes'])
    
    # IMPACT METRICS: Balls per boundary
    stats['Balls_Per_Four'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Fours'])
    stats['Balls_Per_Six'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Sixes'])
    stats['Balls_Per_Boundary'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Boundaries'])
    
    # === CATEGORY 6: ROTATION & CONTROL ===
    stats['Non_Boundary_Runs'] = stats['Total_Runs'] - stats['Boundary_Runs']
    stats['Non_Boundary_Balls'] = stats['Total_Balls_Faced'] - stats['Total_Boundaries']
    stats['Rotation_Strike_Rate'] = safe_divide(stats['Non_Boundary_Runs'] * 100, stats['Non_Boundary_Balls'])
    
    # Estimate dot balls (no direct column in batting data)
    estimated_dots = stats['Total_Balls_Faced'] - stats['Total_Boundaries'] - (stats['Non_Boundary_Runs'] / 0.9)
    estimated_dots = max(0, estimated_dots)
    stats['Estimated_Dot_Balls'] = int(estimated_dots)
    stats['Dot_Ball_Percentage'] = safe_divide(estimated_dots * 100, stats['Total_Balls_Faced'])
    
    # === CATEGORY 7: CONSISTENCY METRICS ===
    runs_series = player_data['runs']
    stats['Runs_Std_Deviation'] = runs_series.std() if len(runs_series) > 1 else 0.0
    stats['Runs_CV'] = calculate_coefficient_of_variation(runs_series)
    
    # Milestone percentages
    stats['Percentage_30_Plus'] = safe_divide(stats['Count_30_Plus'] * 100, stats['Total_Innings'])
    stats['Percentage_50_Plus'] = safe_divide(stats['Count_50_Plus'] * 100, stats['Total_Innings'])
    stats['Percentage_70_Plus'] = safe_divide(stats['Count_70_Plus'] * 100, stats['Total_Innings'])
    
    # Conversion rates
    stats['Conversion_30_to_50'] = safe_divide(stats['Count_50_Plus'] * 100, stats['Count_30_Plus'])
    stats['Conversion_50_to_70'] = safe_divide(stats['Count_70_Plus'] * 100, stats['Count_50_Plus'])
    
    # Failure rates
    stats['Percentage_Single_Digit'] = safe_divide(len(player_data[player_data['runs'] < 10]) * 100, stats['Total_Innings'])
    stats['Percentage_Duck'] = safe_divide(len(player_data[player_data['runs'] == 0]) * 100, stats['Dismissals'])
    
    # === CATEGORY 8: PHASE-SPECIFIC STATS ===
    # Powerplay (overs 1-6)
    powerplay_data = player_data[player_data['runningOver'].apply(lambda x: float(x) <= 6 if pd.notna(x) else False)]
    stats['Powerplay_Runs'] = powerplay_data['runs'].sum()
    stats['Powerplay_Balls'] = powerplay_data['ballsFaced'].sum()
    stats['Powerplay_SR'] = safe_divide(stats['Powerplay_Runs'] * 100, stats['Powerplay_Balls'])
    stats['Powerplay_Contribution'] = safe_divide(stats['Powerplay_Runs'] * 100, stats['Total_Runs'])
    
    # Death overs (overs 16-20)
    death_data = player_data[player_data['runningOver'].apply(lambda x: float(x) >= 16 if pd.notna(x) else False)]
    stats['Death_Overs_Runs'] = death_data['runs'].sum()
    stats['Death_Overs_Balls'] = death_data['ballsFaced'].sum()
    stats['Death_Overs_SR'] = safe_divide(stats['Death_Overs_Runs'] * 100, stats['Death_Overs_Balls'])
    stats['Death_Overs_Contribution'] = safe_divide(stats['Death_Overs_Runs'] * 100, stats['Total_Runs'])
    
    # === CATEGORY 9: IMPACT SCORE ===
    # Simple impact calculation (can be refined)
    not_out_factor = 1 + (stats['Times_Not_Out'] / max(stats['Total_Innings'], 1))
    stats['Impact_Score'] = safe_divide(stats['Total_Runs'] * stats['Strike_Rate'] * not_out_factor, 10000)
    
    return stats

def create_empty_batting_stats(player_row):
    """Create empty stats for debut players"""
    stats = {
        'Player_Name': player_row['2025_FullName'],
        'Kaggle_Match_Name': '',
        'Player_Type': player_row['Player_Type'],
        'IPL_Team_2025': player_row['IPL_Team_2025'],
        'DEBUT': 'YES',
    }
    
    # All numeric fields = 0
    numeric_fields = [
        'Total_Innings', 'Total_Runs', 'Total_Balls_Faced', 'Total_Minutes',
        'Dismissals', 'Times_Not_Out',
        'Batting_Average', 'Strike_Rate',
        'Avg_Runs_Per_Innings', 'Avg_Balls_Per_Innings', 'Avg_Minutes_Per_Innings',
        'Count_30_Plus', 'Count_50_Plus', 'Count_70_Plus', 'Highest_Score',
        'Most_Fours_Innings', 'Most_Sixes_Innings',
        'Total_Fours', 'Total_Sixes', 'Total_Boundaries', 'Boundary_Runs',
        'Boundary_Percentage', 'Boundary_Frequency', 'Fours_Per_Innings',
        'Sixes_Per_Innings', 'Four_to_Six_Ratio', 'Balls_Per_Four',
        'Balls_Per_Six', 'Balls_Per_Boundary',
        'Non_Boundary_Runs', 'Non_Boundary_Balls', 'Rotation_Strike_Rate',
        'Estimated_Dot_Balls', 'Dot_Ball_Percentage',
        'Runs_Std_Deviation', 'Runs_CV', 'Percentage_30_Plus',
        'Percentage_50_Plus', 'Percentage_70_Plus', 'Conversion_30_to_50',
        'Conversion_50_to_70', 'Percentage_Single_Digit', 'Percentage_Duck',
        'Powerplay_Runs', 'Powerplay_Balls', 'Powerplay_SR',
        'Powerplay_Contribution', 'Death_Overs_Runs', 'Death_Overs_Balls',
        'Death_Overs_SR', 'Death_Overs_Contribution', 'Impact_Score'
    ]
    
    for field in numeric_fields:
        stats[field] = 0.0
    
    return stats

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: CALCULATION FUNCTIONS FOR DEBUT PLAYERS
# ADD THESE TWO FUNCTIONS AFTER create_empty_batting_stats() (around line 460)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_derived_batting_stats(volume_row, player_row):
    """
    Calculate all derived batting metrics from volume data (debut players)
    
    Args:
        volume_row: Row from debut_batting.csv with volume metrics
        player_row: Row from debut_players with identification info
    
    Returns:
        Complete stats dictionary with all calculated metrics
    """
    stats = {}
    
    # === CATEGORY 1: IDENTIFICATION ===
    stats['Player_Name'] = player_row['2025_FullName']
    stats['Kaggle_Match_Name'] = ''
    stats['Player_Type'] = player_row['Player_Type']
    stats['IPL_Team_2025'] = player_row['IPL_Team_2025']
    stats['DEBUT'] = 'YES'
    
    # === CATEGORY 2: VOLUME METRICS (from CSV) ===
    stats['Total_Innings'] = int(volume_row['Total_Innings'])
    stats['Total_Runs'] = int(volume_row['Total_Runs'])
    stats['Total_Balls_Faced'] = int(volume_row['Total_Balls_Faced'])
    stats['Total_Minutes'] = int(volume_row['Total_Minutes'])
    stats['Dismissals'] = int(volume_row['Dismissals'])
    stats['Times_Not_Out'] = int(volume_row['Times_Not_Out'])
    
    # === CATEGORY 3: CORE AVERAGES (CALCULATE) ===
    stats['Batting_Average'] = safe_divide(stats['Total_Runs'], stats['Dismissals'])
    stats['Strike_Rate'] = safe_divide(stats['Total_Runs'] * 100, stats['Total_Balls_Faced'])
    stats['Avg_Runs_Per_Innings'] = safe_divide(stats['Total_Runs'], stats['Total_Innings'])
    stats['Avg_Balls_Per_Innings'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Innings'])
    stats['Avg_Minutes_Per_Innings'] = safe_divide(stats['Total_Minutes'], stats['Total_Innings'])
    
    # === CATEGORY 4: MILESTONES (from CSV) ===
    stats['Count_30_Plus'] = int(volume_row['Count_30_Plus'])
    stats['Count_50_Plus'] = int(volume_row['Count_50_Plus'])
    stats['Count_70_Plus'] = int(volume_row['Count_70_Plus'])
    stats['Highest_Score'] = int(volume_row['Highest_Score'])
    stats['Most_Fours_Innings'] = int(volume_row['Most_Fours_Innings'])
    stats['Most_Sixes_Innings'] = int(volume_row['Most_Sixes_Innings'])
    
    # === CATEGORY 5: BOUNDARY METRICS ===
    stats['Total_Fours'] = int(volume_row['Total_Fours'])
    stats['Total_Sixes'] = int(volume_row['Total_Sixes'])
    stats['Total_Boundaries'] = stats['Total_Fours'] + stats['Total_Sixes']
    stats['Boundary_Runs'] = (stats['Total_Fours'] * 4) + (stats['Total_Sixes'] * 6)
    
    # CALCULATE percentages and rates
    stats['Boundary_Percentage'] = safe_divide(stats['Boundary_Runs'] * 100, stats['Total_Runs'])
    stats['Boundary_Frequency'] = safe_divide(stats['Total_Boundaries'] * 100, stats['Total_Balls_Faced'])
    stats['Fours_Per_Innings'] = safe_divide(stats['Total_Fours'], stats['Total_Innings'])
    stats['Sixes_Per_Innings'] = safe_divide(stats['Total_Sixes'], stats['Total_Innings'])
    stats['Four_to_Six_Ratio'] = safe_divide(stats['Total_Fours'], stats['Total_Sixes'])
    
    # IMPACT METRICS: Balls per boundary
    stats['Balls_Per_Four'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Fours'])
    stats['Balls_Per_Six'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Sixes'])
    stats['Balls_Per_Boundary'] = safe_divide(stats['Total_Balls_Faced'], stats['Total_Boundaries'])
    
    # === CATEGORY 6: ROTATION & CONTROL ===
    stats['Non_Boundary_Runs'] = stats['Total_Runs'] - stats['Boundary_Runs']
    stats['Non_Boundary_Balls'] = stats['Total_Balls_Faced'] - stats['Total_Boundaries']
    stats['Rotation_Strike_Rate'] = safe_divide(stats['Non_Boundary_Runs'] * 100, stats['Non_Boundary_Balls'])
    
    # Estimate dot balls
    estimated_dots = stats['Total_Balls_Faced'] - stats['Total_Boundaries'] - (stats['Non_Boundary_Runs'] / 0.9)
    estimated_dots = max(0, estimated_dots)
    stats['Estimated_Dot_Balls'] = int(estimated_dots)
    stats['Dot_Ball_Percentage'] = safe_divide(estimated_dots * 100, stats['Total_Balls_Faced'])
    
    # === CATEGORY 7: CONSISTENCY METRICS ===
    # Note: We don't have ball-by-ball data, so CV and std dev will be 0
    stats['Runs_Std_Deviation'] = 0.0
    stats['Runs_CV'] = 0.0
    
    # Milestone percentages (CALCULATE)
    stats['Percentage_30_Plus'] = safe_divide(stats['Count_30_Plus'] * 100, stats['Total_Innings'])
    stats['Percentage_50_Plus'] = safe_divide(stats['Count_50_Plus'] * 100, stats['Total_Innings'])
    stats['Percentage_70_Plus'] = safe_divide(stats['Count_70_Plus'] * 100, stats['Total_Innings'])
    
    # Conversion rates (CALCULATE)
    stats['Conversion_30_to_50'] = safe_divide(stats['Count_50_Plus'] * 100, stats['Count_30_Plus'])
    stats['Conversion_50_to_70'] = safe_divide(stats['Count_70_Plus'] * 100, stats['Count_50_Plus'])
    
    # Failure rates (CALCULATE - we don't have per-innings data, so estimate)
    # For debuts without ball-by-ball, these will be rough estimates
    stats['Percentage_Single_Digit'] = 0.0  # No data available
    stats['Percentage_Duck'] = 0.0  # No data available
    
    # === CATEGORY 8: PHASE-SPECIFIC STATS ===
    # No ball-by-ball data for debuts - set to 0
    stats['Powerplay_Runs'] = 0
    stats['Powerplay_Balls'] = 0
    stats['Powerplay_SR'] = 0.0
    stats['Powerplay_Contribution'] = 0.0
    
    stats['Death_Overs_Runs'] = 0
    stats['Death_Overs_Balls'] = 0
    stats['Death_Overs_SR'] = 0.0
    stats['Death_Overs_Contribution'] = 0.0
    
    # === CATEGORY 9: IMPACT SCORE ===
    not_out_factor = 1 + (stats['Times_Not_Out'] / max(stats['Total_Innings'], 1))
    stats['Impact_Score'] = safe_divide(stats['Total_Runs'] * stats['Strike_Rate'] * not_out_factor, 10000)
    
    return stats


def calculate_derived_bowling_stats(volume_row, player_row):
    """
    Calculate all derived bowling metrics from volume data (debut players)
    
    Args:
        volume_row: Row from debut_bowling.csv with volume metrics
        player_row: Row from debut_players with identification info
    
    Returns:
        Complete stats dictionary with all calculated metrics
    """
    stats = {}
    
    # === CATEGORY 1: IDENTIFICATION ===
    stats['Player_Name'] = player_row['2025_FullName']
    stats['Kaggle_Match_Name'] = ''
    stats['Player_Type'] = player_row['Player_Type']
    stats['IPL_Team_2025'] = player_row['IPL_Team_2025']
    stats['DEBUT'] = 'YES'
    
    # === CATEGORY 2: VOLUME METRICS (from CSV) ===
    stats['Total_Innings_Bowled'] = int(volume_row['Total_Innings_Bowled'])
    stats['Total_Overs_Bowled'] = float(volume_row['Total_Overs_Bowled'])
    stats['Total_Balls_Bowled'] = int(volume_row['Total_Balls_Bowled'])
    stats['Total_Runs_Conceded'] = int(volume_row['Total_Runs_Conceded'])
    stats['Total_Wickets'] = int(volume_row['Total_Wickets'])
    stats['Total_Maidens'] = int(volume_row['Total_Maidens'])
    stats['Total_Dot_Balls'] = int(volume_row['Total_Dot_Balls'])
    
    # === CATEGORY 3: CORE AVERAGES (CALCULATE) ===
    stats['Bowling_Average'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Wickets'])
    stats['Economy_Rate'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Overs_Bowled'])
    stats['Bowling_Strike_Rate'] = safe_divide(stats['Total_Balls_Bowled'], stats['Total_Wickets'])
    stats['Avg_Wickets_Per_Innings'] = safe_divide(stats['Total_Wickets'], stats['Total_Innings_Bowled'])
    stats['Avg_Overs_Per_Innings'] = safe_divide(stats['Total_Overs_Bowled'], stats['Total_Innings_Bowled'])
    stats['Avg_Runs_Per_Innings'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Innings_Bowled'])
    stats['Wickets_Per_Over'] = safe_divide(stats['Total_Wickets'], stats['Total_Overs_Bowled'])
    stats['Runs_Per_Ball'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Balls_Bowled'])
    
    # === CATEGORY 4: WICKET MILESTONES (from CSV) ===
    stats['Count_3_Wickets'] = int(volume_row['Count_3_Wickets'])
    stats['Count_4_Wickets'] = int(volume_row['Count_4_Wickets'])
    stats['Count_5_Wickets'] = int(volume_row['Count_5_Wickets'])
    
    # Best bowling figures
    stats['Best_Bowling_Figures'] = str(volume_row['Best_Bowling_Figures'])
    stats['Best_Bowling_Wickets'] = int(volume_row['Best_Bowling_Wickets'])
    stats['Best_Bowling_Runs'] = int(volume_row['Best_Bowling_Runs'])
    stats['Best_Economy_Innings'] = float(volume_row.get('Best_Economy_Innings', 0.0))
    
    stats['Most_Dots_Innings'] = int(volume_row.get('Most_Dots_Innings', 0))
    stats['Most_Wickets_Match'] = int(volume_row['Best_Bowling_Wickets'])  # Same as best
    
    # === CATEGORY 5: CONTROL METRICS ===
    stats['Dot_Ball_Percentage'] = safe_divide(stats['Total_Dot_Balls'] * 100, stats['Total_Balls_Bowled'])
    
    complete_overs = int(stats['Total_Overs_Bowled'])
    stats['Maiden_Percentage'] = safe_divide(stats['Total_Maidens'] * 100, complete_overs)
    
    # Boundaries conceded (from CSV)
    stats['Total_Fours_Conceded'] = int(volume_row['Total_Fours_Conceded'])
    stats['Total_Sixes_Conceded'] = int(volume_row['Total_Sixes_Conceded'])
    stats['Boundaries_Conceded'] = stats['Total_Fours_Conceded'] + stats['Total_Sixes_Conceded']
    
    # CALCULATE rates and percentages
    stats['Boundary_Concession_Rate'] = safe_divide(stats['Boundaries_Conceded'] * 100, stats['Total_Balls_Bowled'])
    stats['Runs_From_Boundaries'] = (stats['Total_Fours_Conceded'] * 4) + (stats['Total_Sixes_Conceded'] * 6)
    stats['Boundary_Run_Percentage'] = safe_divide(stats['Runs_From_Boundaries'] * 100, stats['Total_Runs_Conceded'])
    
    # Control indices (CALCULATE)
    stats['Control_Index'] = safe_divide((stats['Total_Dot_Balls'] - stats['Boundaries_Conceded']), stats['Total_Balls_Bowled'])
    stats['Pressure_Index'] = safe_divide((stats['Total_Dot_Balls'] + stats['Total_Maidens'] * 6), stats['Total_Balls_Bowled'])
    stats['Scoring_Ball_Percentage'] = safe_divide((stats['Total_Balls_Bowled'] - stats['Total_Dot_Balls']) * 100, stats['Total_Balls_Bowled'])
    
    non_boundary_runs = stats['Total_Runs_Conceded'] - stats['Runs_From_Boundaries']
    stats['Non_Boundary_Economy'] = safe_divide(non_boundary_runs, stats['Total_Overs_Bowled'])
    
    # === CATEGORY 6: EXTRAS & DISCIPLINE (from CSV) ===
    stats['Total_Wides'] = int(volume_row['Total_Wides'])
    stats['Total_No_Balls'] = int(volume_row['Total_No_Balls'])
    stats['Total_Extras'] = stats['Total_Wides'] + stats['Total_No_Balls']
    
    # CALCULATE percentages
    stats['Extras_Percentage'] = safe_divide(stats['Total_Extras'] * 100, stats['Total_Balls_Bowled'])
    stats['Wides_Per_Innings'] = safe_divide(stats['Total_Wides'], stats['Total_Innings_Bowled'])
    stats['NoBalls_Per_Innings'] = safe_divide(stats['Total_No_Balls'], stats['Total_Innings_Bowled'])
    stats['Discipline_Score'] = 100 - stats['Extras_Percentage']
    
    # === CATEGORY 7: EXPENSIVE OVERS ===
    stats['Most_Runs_Conceded_Innings'] = int(volume_row.get('Most_Runs_Conceded_Innings', 0))
    
    # Count expensive innings (estimate - no per-innings economy data)
    stats['Count_Expensive_Innings'] = 0  # No data available
    stats['Expensive_Innings_Rate'] = 0.0
    
    # === CATEGORY 8: CONSISTENCY ===
    # No ball-by-ball data for debuts
    stats['Wickets_Std_Deviation'] = 0.0
    stats['Economy_CV'] = 0.0
    
    # CALCULATE percentages
    stats['Percentage_Wicketless'] = 0.0  # No per-innings data
    stats['Percentage_3_Plus_Wickets'] = safe_divide(stats['Count_3_Wickets'] * 100, stats['Total_Innings_Bowled'])
    
    # === CATEGORY 9: PHASE-SPECIFIC ===
    # No ball-by-ball data for debuts - set to 0
    stats['Powerplay_Overs'] = 0.0
    stats['Powerplay_Runs'] = 0
    stats['Powerplay_Wickets'] = 0
    stats['Powerplay_Economy'] = 0.0
    
    stats['Death_Overs'] = 0.0
    stats['Death_Runs'] = 0
    stats['Death_Wickets'] = 0
    stats['Death_Overs_Economy'] = 0.0
    
    # === CATEGORY 10: IMPACT SCORE ===
    stats['Impact_Score'] = (stats['Total_Wickets'] * 30) - (stats['Economy_Rate'] * 5)
    stats['Wicket_Taking_Ability'] = safe_divide(stats['Avg_Wickets_Per_Innings'] * 100, stats['Bowling_Strike_Rate'])
    
    return stats

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: DEBUT PROCESSING FUNCTION
# ADD THIS FUNCTION AFTER THE TWO CALCULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def process_debut_players_with_data(debut_players, debut_batting_df, debut_bowling_df, batting_stats_list, bowling_stats_list):
    """Process debut players using pre-collected data from CSV files"""
    print_progress(f"Processing {len(debut_players)} debut players...")
    
    batting_found = 0
    batting_missing = 0
    bowling_found = 0
    bowling_missing = 0
    
    for idx, player_row in debut_players.iterrows():
        player_name = player_row['2025_FullName']
        player_type = player_row['Player_Type'].strip()
        
        # === BATTING STATS ===
        if player_type in ['Batter', 'WK-Batter', 'All-Rounder']:
            if len(debut_batting_df) > 0:
                # Try to find player in debut_batting.csv
                debut_data = debut_batting_df[debut_batting_df['Player_Name'].str.strip() == player_name.strip()]
                
                if len(debut_data) > 0:
                    # Found data - calculate derived metrics
                    bat_stats = calculate_derived_batting_stats(debut_data.iloc[0], player_row)
                    batting_found += 1
                else:
                    # Not found - use zeros as fallback
                    print_warning(f"  Batting data not found for: {player_name} (using zeros)")
                    bat_stats = create_empty_batting_stats(player_row)
                    batting_missing += 1
            else:
                # No debut CSV loaded - use zeros
                bat_stats = create_empty_batting_stats(player_row)
                batting_missing += 1
            
            batting_stats_list.append(bat_stats)
        
        # === BOWLING STATS ===
        if player_type in ['Bowler', 'All-Rounder']:
            if len(debut_bowling_df) > 0:
                # Try to find player in debut_bowling.csv
                debut_data = debut_bowling_df[debut_bowling_df['Player_Name'].str.strip() == player_name.strip()]
                
                if len(debut_data) > 0:
                    # Found data - calculate derived metrics
                    bowl_stats = calculate_derived_bowling_stats(debut_data.iloc[0], player_row)
                    bowling_found += 1
                else:
                    # Not found - use zeros as fallback
                    print_warning(f"  Bowling data not found for: {player_name} (using zeros)")
                    bowl_stats = create_empty_bowling_stats(player_row)
                    bowling_missing += 1
            else:
                # No debut CSV loaded - use zeros
                bowl_stats = create_empty_bowling_stats(player_row)
                bowling_missing += 1
            
            bowling_stats_list.append(bowl_stats)
    
    # Summary
    print_success(f"Processed {len(debut_players)} debut players")
    print_progress(f"  Batting: {batting_found} with data, {batting_missing} with zeros")
    print_progress(f"  Bowling: {bowling_found} with data, {bowling_missing} with zeros")
    
    return batting_stats_list, bowling_stats_list

# ═══════════════════════════════════════════════════════════════════════════
# BOWLING STATISTICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def calculate_bowling_stats(player_row, bowling_df, match_outcomes):
    """Calculate all bowling statistics for a player"""
    
    # Extract player name variants
    player_name_2025 = player_row['2025_FullName']
    short_name = player_row['Kaggle_ShortName']
    full_name = player_row['Kaggle_FullName']
    all_variants = player_row['All_Variants'].split('|') if pd.notna(player_row.get('All_Variants')) else []
    
    # Filter player data
    # Strip whitespace from data for better matching
    bowling_df_clean = bowling_df.copy()
    bowling_df_clean['name'] = bowling_df_clean['name'].astype(str).str.strip()
    bowling_df_clean['fullName'] = bowling_df_clean['fullName'].astype(str).str.strip()
    
    # Also strip variants
    all_variants_clean = [v.strip() for v in all_variants]
    
    player_data = bowling_df_clean[
        (bowling_df_clean['name'].isin(all_variants_clean)) | 
        (bowling_df_clean['fullName'].isin(all_variants_clean))
    ].copy()
    
    # Remove duplicates (same match_id + innings_id = duplicate row)
    if 'match_id' in player_data.columns and 'innings_id' in player_data.columns:
        player_data = player_data.drop_duplicates(subset=['match_id', 'innings_id'], keep='first')
    
    if len(player_data) == 0:
        return create_empty_bowling_stats(player_row)
    
    stats = {}
    
    # === CATEGORY 1: IDENTIFICATION ===
    stats['Player_Name'] = player_name_2025
    stats['Kaggle_Match_Name'] = short_name
    stats['Player_Type'] = player_row['Player_Type']
    stats['IPL_Team_2025'] = player_row['IPL_Team_2025']
    stats['DEBUT'] = 'NO'
    
    # === CATEGORY 2: VOLUME METRICS ===
    stats['Total_Innings_Bowled'] = len(player_data)
    stats['Total_Overs_Bowled'] = player_data['overs'].sum()
    stats['Total_Balls_Bowled'] = player_data['overs'].apply(overs_to_balls).sum()
    stats['Total_Runs_Conceded'] = player_data['conceded'].sum()
    stats['Total_Wickets'] = player_data['wickets'].sum()
    stats['Total_Maidens'] = player_data['maidens'].sum()
    stats['Total_Dot_Balls'] = player_data['dots'].sum()  # Direct from data!
    
    # === CATEGORY 3: CORE AVERAGES ===
    stats['Bowling_Average'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Wickets'])
    stats['Economy_Rate'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Overs_Bowled'])
    stats['Bowling_Strike_Rate'] = safe_divide(stats['Total_Balls_Bowled'], stats['Total_Wickets'])
    stats['Avg_Wickets_Per_Innings'] = safe_divide(stats['Total_Wickets'], stats['Total_Innings_Bowled'])
    stats['Avg_Overs_Per_Innings'] = safe_divide(stats['Total_Overs_Bowled'], stats['Total_Innings_Bowled'])
    stats['Avg_Runs_Per_Innings'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Innings_Bowled'])
    stats['Wickets_Per_Over'] = safe_divide(stats['Total_Wickets'], stats['Total_Overs_Bowled'])
    stats['Runs_Per_Ball'] = safe_divide(stats['Total_Runs_Conceded'], stats['Total_Balls_Bowled'])
    
    # === CATEGORY 4: WICKET MILESTONES ===
    stats['Count_3_Wickets'] = len(player_data[player_data['wickets'] >= 3])
    stats['Count_4_Wickets'] = len(player_data[player_data['wickets'] >= 4])
    stats['Count_5_Wickets'] = len(player_data[player_data['wickets'] >= 5])
    
    # Best bowling figures
    best_performance = player_data.nlargest(1, 'wickets')
    if len(best_performance) > 0:
        best_wkt = best_performance.iloc[0]['wickets']
        best_runs = best_performance.iloc[0]['conceded']
        stats['Best_Bowling_Figures'] = f"{int(best_wkt)}/{int(best_runs)}"
        stats['Best_Bowling_Wickets'] = int(best_wkt)
        stats['Best_Bowling_Runs'] = int(best_runs)
    else:
        stats['Best_Bowling_Figures'] = "0/0"
        stats['Best_Bowling_Wickets'] = 0
        stats['Best_Bowling_Runs'] = 0
    
    # Best economy (min 3 overs)
    qualified = player_data[player_data['overs'] >= 3].copy()
    if len(qualified) > 0:
        # Convert economyRate to numeric
        qualified['economyRate_numeric'] = pd.to_numeric(qualified['economyRate'], errors='coerce')
        stats['Best_Economy_Innings'] = qualified['economyRate_numeric'].min()
    else:
        stats['Best_Economy_Innings'] = 0.0
    
    stats['Most_Dots_Innings'] = player_data['dots'].max()
    stats['Most_Wickets_Match'] = player_data['wickets'].max()
    
    # === CATEGORY 5: CONTROL METRICS ===
    stats['Dot_Ball_Percentage'] = safe_divide(stats['Total_Dot_Balls'] * 100, stats['Total_Balls_Bowled'])
    
    # Maidens percentage (of complete overs)
    complete_overs = int(stats['Total_Overs_Bowled'])
    stats['Maiden_Percentage'] = safe_divide(stats['Total_Maidens'] * 100, complete_overs)
    
    # Boundaries conceded
    stats['Total_Fours_Conceded'] = player_data['foursConceded'].sum()
    stats['Total_Sixes_Conceded'] = player_data['sixesConceded'].sum()
    stats['Boundaries_Conceded'] = stats['Total_Fours_Conceded'] + stats['Total_Sixes_Conceded']
    stats['Boundary_Concession_Rate'] = safe_divide(stats['Boundaries_Conceded'] * 100, stats['Total_Balls_Bowled'])
    
    stats['Runs_From_Boundaries'] = (stats['Total_Fours_Conceded'] * 4) + (stats['Total_Sixes_Conceded'] * 6)
    stats['Boundary_Run_Percentage'] = safe_divide(stats['Runs_From_Boundaries'] * 100, stats['Total_Runs_Conceded'])
    
    # Control indices
    stats['Control_Index'] = safe_divide((stats['Total_Dot_Balls'] - stats['Boundaries_Conceded']), stats['Total_Balls_Bowled'])
    stats['Pressure_Index'] = safe_divide((stats['Total_Dot_Balls'] + stats['Total_Maidens'] * 6), stats['Total_Balls_Bowled'])
    stats['Scoring_Ball_Percentage'] = safe_divide((stats['Total_Balls_Bowled'] - stats['Total_Dot_Balls']) * 100, stats['Total_Balls_Bowled'])
    
    non_boundary_runs = stats['Total_Runs_Conceded'] - stats['Runs_From_Boundaries']
    stats['Non_Boundary_Economy'] = safe_divide(non_boundary_runs, stats['Total_Overs_Bowled'])
    
    # === CATEGORY 6: EXTRAS & DISCIPLINE ===
    stats['Total_Wides'] = player_data['wides'].sum()
    stats['Total_No_Balls'] = player_data['noballs'].sum()
    stats['Total_Extras'] = stats['Total_Wides'] + stats['Total_No_Balls']
    stats['Extras_Percentage'] = safe_divide(stats['Total_Extras'] * 100, stats['Total_Balls_Bowled'])
    stats['Wides_Per_Innings'] = safe_divide(stats['Total_Wides'], stats['Total_Innings_Bowled'])
    stats['NoBalls_Per_Innings'] = safe_divide(stats['Total_No_Balls'], stats['Total_Innings_Bowled'])
    stats['Discipline_Score'] = 100 - stats['Extras_Percentage']
    
    # === CATEGORY 7: EXPENSIVE OVERS ===
    stats['Most_Runs_Conceded_Innings'] = player_data['conceded'].max()
    
    # Count expensive overs (estimate: if economy > 12 for the innings)
    # Convert economyRate to numeric first
    player_data_clean = player_data.copy()
    player_data_clean['economyRate'] = pd.to_numeric(player_data_clean['economyRate'], errors='coerce')
    expensive_innings = player_data_clean[player_data_clean['economyRate'] > 12]
    stats['Count_Expensive_Innings'] = len(expensive_innings)
    stats['Expensive_Innings_Rate'] = safe_divide(len(expensive_innings) * 100, stats['Total_Innings_Bowled'])
    
    # === CATEGORY 8: CONSISTENCY ===
    wickets_series = player_data['wickets']
    stats['Wickets_Std_Deviation'] = wickets_series.std() if len(wickets_series) > 1 else 0.0
    
    # Convert economyRate to numeric for CV calculation
    economy_numeric = pd.to_numeric(player_data['economyRate'], errors='coerce').dropna()
    stats['Economy_CV'] = calculate_coefficient_of_variation(economy_numeric) if len(economy_numeric) > 1 else 0.0
    
    stats['Percentage_Wicketless'] = safe_divide(len(player_data[player_data['wickets'] == 0]) * 100, stats['Total_Innings_Bowled'])
    stats['Percentage_3_Plus_Wickets'] = safe_divide(stats['Count_3_Wickets'] * 100, stats['Total_Innings_Bowled'])
    
    # === CATEGORY 9: PHASE-SPECIFIC (Limited - need ball-by-ball for accuracy)
    # Note: bowling data doesn't have runningOver, so we can't split phases accurately
    # Set to 0 for now - would require ball-by-ball data
    stats['Powerplay_Overs'] = 0.0
    stats['Powerplay_Runs'] = 0
    stats['Powerplay_Wickets'] = 0
    stats['Powerplay_Economy'] = 0.0
    
    stats['Death_Overs'] = 0.0
    stats['Death_Runs'] = 0
    stats['Death_Wickets'] = 0
    stats['Death_Overs_Economy'] = 0.0
    
    # === CATEGORY 10: IMPACT SCORE ===
    # Wickets are positive, economy is negative
    stats['Impact_Score'] = (stats['Total_Wickets'] * 30) - (stats['Economy_Rate'] * 5)
    stats['Wicket_Taking_Ability'] = safe_divide(stats['Avg_Wickets_Per_Innings'] * 100, stats['Bowling_Strike_Rate'])
    
    return stats

def create_empty_bowling_stats(player_row):
    """Create empty stats for debut players"""
    stats = {
        'Player_Name': player_row['2025_FullName'],
        'Kaggle_Match_Name': '',
        'Player_Type': player_row['Player_Type'],
        'IPL_Team_2025': player_row['IPL_Team_2025'],
        'DEBUT': 'YES',
    }
    
    # All numeric fields = 0
    numeric_fields = [
        'Total_Innings_Bowled', 'Total_Overs_Bowled', 'Total_Balls_Bowled',
        'Total_Runs_Conceded', 'Total_Wickets', 'Total_Maidens', 'Total_Dot_Balls',
        'Bowling_Average', 'Economy_Rate', 'Bowling_Strike_Rate',
        'Avg_Wickets_Per_Innings', 'Avg_Overs_Per_Innings', 'Avg_Runs_Per_Innings',
        'Wickets_Per_Over', 'Runs_Per_Ball',
        'Count_3_Wickets', 'Count_4_Wickets', 'Count_5_Wickets',
        'Best_Bowling_Wickets', 'Best_Bowling_Runs', 'Best_Economy_Innings',
        'Most_Dots_Innings', 'Most_Wickets_Match',
        'Dot_Ball_Percentage', 'Maiden_Percentage', 'Total_Fours_Conceded',
        'Total_Sixes_Conceded', 'Boundaries_Conceded', 'Boundary_Concession_Rate',
        'Runs_From_Boundaries', 'Boundary_Run_Percentage', 'Control_Index',
        'Pressure_Index', 'Scoring_Ball_Percentage', 'Non_Boundary_Economy',
        'Total_Wides', 'Total_No_Balls', 'Total_Extras', 'Extras_Percentage',
        'Wides_Per_Innings', 'NoBalls_Per_Innings', 'Discipline_Score',
        'Most_Runs_Conceded_Innings', 'Count_Expensive_Innings', 'Expensive_Innings_Rate',
        'Wickets_Std_Deviation', 'Economy_CV', 'Percentage_Wicketless',
        'Percentage_3_Plus_Wickets',
        'Powerplay_Overs', 'Powerplay_Runs', 'Powerplay_Wickets', 'Powerplay_Economy',
        'Death_Overs', 'Death_Runs', 'Death_Wickets', 'Death_Overs_Economy',
        'Impact_Score', 'Wicket_Taking_Ability'
    ]
    
    for field in numeric_fields:
        stats[field] = 0.0
    
    stats['Best_Bowling_Figures'] = "0/0"
    
    return stats

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_all_players(data):
    """Process all players and create master datasets"""
    print_section("PROCESSING PLAYERS")
    
    current_players = data['current_players']
    debut_players = data['debut_players']
    batting_df = data['batting']
    bowling_df = data['bowling']
    match_outcomes = data['match_outcomes']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FAILSAFE: VALIDATE NO OVERLAP BETWEEN CURRENT AND DEBUT PLAYERS
    # ═══════════════════════════════════════════════════════════════════════════
    print_progress("Running failsafe validation...")
    
    current_names = set(current_players['2025_FullName'].str.strip())
    debut_names = set(debut_players['2025_FullName'].str.strip())
    
    overlap = current_names.intersection(debut_names)
    
    if len(overlap) > 0:
        print_error("═" * 80)
        print_error("🚨 CRITICAL ERROR: PLAYER OVERLAP DETECTED!")
        print_error("═" * 80)
        print_error(f"\n{len(overlap)} player(s) appear in BOTH current_players AND debut_players lists:\n")
        
        for name in sorted(overlap):
            print_warning(f"  ❌ {name}")
        
        print_error("\n" + "═" * 80)
        print_error("REQUIRED ACTION:")
        print_error("  1. Check nomenclature files:")
        print_error(f"     - current_players_2025_*.csv")
        print_error(f"     - debut_players_2025_*.csv")
        print_error("  2. Remove duplicates (players should be in ONE list only)")
        print_error("  3. Re-run this script")
        print_error("═" * 80)
        print_error("\nSCRIPT STOPPED TO PREVENT DATA CORRUPTION\n")
        
        return None, None
    
    print_success(f"✅ Validation passed: {len(current_names)} current, {len(debut_names)} debuts (no overlap)")
    
    batting_stats_list = []
    bowling_stats_list = []
    
    # Process current players
    print_progress(f"Processing {len(current_players)} current players...")
    
    for idx, player_row in current_players.iterrows():
        if (idx + 1) % 25 == 0:
            print_progress(f"  Progress: {idx + 1}/{len(current_players)}...")
        
        player_type = player_row['Player_Type'].strip()
        
        # Batting stats for batters and all-rounders
        if player_type in ['Batter', 'WK-Batter', 'All-Rounder']:
            bat_stats = calculate_batting_stats(player_row, batting_df, match_outcomes)
            batting_stats_list.append(bat_stats)
        
        # Bowling stats for bowlers and all-rounders
        if player_type in ['Bowler', 'All-Rounder']:
            bowl_stats = calculate_bowling_stats(player_row, bowling_df, match_outcomes)
            bowling_stats_list.append(bowl_stats)
    
    print_success(f"Processed {len(current_players)} current players")
    
    # Process debut players (NEW METHOD WITH REAL DATA)
    debut_batting_df = data.get('debut_batting_stats', pd.DataFrame())
    debut_bowling_df = data.get('debut_bowling_stats', pd.DataFrame())
    
    batting_stats_list, bowling_stats_list = process_debut_players_with_data(
        debut_players,
        debut_batting_df,
        debut_bowling_df,
        batting_stats_list,
        bowling_stats_list
    )
    
    # Create DataFrames
    batting_master = pd.DataFrame(batting_stats_list)
    bowling_master = pd.DataFrame(bowling_stats_list)
    
    # Sort: Current players first (by team, then name), then debuts (by team, then name)
    def sort_key(df):
        df['sort_order'] = df['DEBUT'].map({'NO': 0, 'YES': 1})
        df['team_order'] = df['IPL_Team_2025'].map({team: idx for idx, team in enumerate(TEAM_ORDER)})
        return df.sort_values(['sort_order', 'team_order', 'Player_Name']).drop(['sort_order', 'team_order'], axis=1)
    
    batting_master = sort_key(batting_master)
    bowling_master = sort_key(bowling_master)
    
    print_success(f"Batting master: {len(batting_master)} players")
    print_success(f"Bowling master: {len(bowling_master)} players")
    
    return batting_master, bowling_master

# ═══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_datasets(batting_df, bowling_df):
    """Save master datasets and generate report"""
    print_section("SAVING OUTPUT FILES")
    
    # Save files
    batting_file = OUTPUT_DIR / f"BATTING_MASTER_2025_{TIMESTAMP}.csv"
    bowling_file = OUTPUT_DIR / f"BOWLING_MASTER_2025_{TIMESTAMP}.csv"
    
    batting_df.to_csv(batting_file, index=False)
    bowling_df.to_csv(bowling_file, index=False)
    
    print_success(f"Saved: {batting_file.name}")
    print_progress(f"  Rows: {len(batting_df)}, Columns: {len(batting_df.columns)}")
    
    print_success(f"Saved: {bowling_file.name}")
    print_progress(f"  Rows: {len(bowling_df)}, Columns: {len(bowling_df.columns)}")
    
    # Generate report
    report_file = OUTPUT_DIR / f"master_dataset_report_{TIMESTAMP}.md"
    generate_report(report_file, batting_df, bowling_df)
    
    print_success(f"Saved: {report_file.name}")
    
    return batting_file, bowling_file, report_file

def generate_report(report_file, batting_df, bowling_df):
    """Generate comprehensive report"""
    lines = []
    lines.append("# IPL 2025 Master Dataset Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data Range:** {START_YEAR}-{END_YEAR} (4 seasons)")
    lines.append(f"**Version:** 2.0 - Comprehensive Edition\n")
    
    lines.append("---\n")
    lines.append("## 📊 Dataset Summary\n")
    
    lines.append("### BATTING_MASTER_2025.csv")
    lines.append(f"- **Total Players:** {len(batting_df)}")
    lines.append(f"- **Current Players:** {len(batting_df[batting_df['DEBUT'] == 'NO'])}")
    lines.append(f"- **Debut Players:** {len(batting_df[batting_df['DEBUT'] == 'YES'])}")
    lines.append(f"- **Total Attributes:** {len(batting_df.columns)}\n")
    
    lines.append("### BOWLING_MASTER_2025.csv")
    lines.append(f"- **Total Players:** {len(bowling_df)}")
    lines.append(f"- **Current Players:** {len(bowling_df[bowling_df['DEBUT'] == 'NO'])}")
    lines.append(f"- **Debut Players:** {len(bowling_df[bowling_df['DEBUT'] == 'YES'])}")
    lines.append(f"- **Total Attributes:** {len(bowling_df.columns)}\n")
    
    lines.append("---\n")
    lines.append("## ✅ Key Features\n")
    lines.append("1. ✅ **4-Year Data Range** (2021-2024)")
    lines.append("2. ✅ **Captain Win Percentage** (from match outcomes)")
    lines.append("3. ✅ **T20 Milestones** (30+, 50+, 70+ focus)")
    lines.append("4. ✅ **Impact Metrics** (Balls per 4, Balls per 6)")
    lines.append("5. ✅ **Phase Stats** (Powerplay, Death overs)")
    lines.append("6. ✅ **All-Rounders** (in BOTH files)")
    lines.append("7. ✅ **Clean Debut Handling** (all 0s, sorted at bottom)\n")
    
    lines.append("---\n")
    lines.append("## 🎯 Attribute Categories\n")
    lines.append("### Batting Attributes (~60)")
    lines.append("- Identification (5)")
    lines.append("- Volume Metrics (9)")
    lines.append("- Core Averages (6)")
    lines.append("- Milestones (7)")
    lines.append("- Boundaries (13)")
    lines.append("- Rotation & Control (6)")
    lines.append("- Consistency (9)")
    lines.append("- Phase-Specific (8)")
    lines.append("- Impact Score (1)\n")
    
    lines.append("### Bowling Attributes (~70)")
    lines.append("- Identification (5)")
    lines.append("- Volume Metrics (8)")
    lines.append("- Core Averages (8)")
    lines.append("- Wicket Milestones (8)")
    lines.append("- Control Metrics (11)")
    lines.append("- Extras & Discipline (8)")
    lines.append("- Expensive Overs (3)")
    lines.append("- Consistency (4)")
    lines.append("- Phase-Specific (8)")
    lines.append("- Impact Scores (2)\n")
    
    lines.append("---\n")
    lines.append("## 🚀 Next Steps\n")
    lines.append("1. Load master datasets")
    lines.append("2. Apply OVR formula (to be provided)")
    lines.append("3. Create stadium-specific datasets")
    lines.append("4. Build PvP (Player vs Player) matrices")
    lines.append("5. Feature engineering for ML models\n")
    
    lines.append("---\n")
    lines.append("*Generated by El Dorado Project - CSCI 566*")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function"""
    print("\n" + "█" * 80)
    print("  IPL 2025 MASTER DATASET CREATION - COMPREHENSIVE EDITION")
    print("  El Dorado Project - CSCI 566")
    print("  Data Range: 2021-2024 | Ready for OVR Calculation")
    print("█" * 80)
    
    start_time = datetime.now()
    
    try:
        data = load_all_data()
        batting_df, bowling_df = process_all_players(data)
        
        # Check if failsafe validation stopped execution
        if batting_df is None or bowling_df is None:
            print_error("\n⚠️  Script execution halted due to validation errors.")
            print_error("Please fix the issues above and re-run.\n")
            return
        
        batting_file, bowling_file, report_file = save_datasets(batting_df, bowling_df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("EXECUTION COMPLETE")
        print_success(f"Total execution time: {duration:.1f} seconds")
        print_success(f"Output directory: {OUTPUT_DIR}")
        print()
        print("  📊 Created Files:")
        print(f"     1. {batting_file.name}")
        print(f"     2. {bowling_file.name}")
        print(f"     3. {report_file.name}")
        print()
        print("  ✅ Features:")
        print("     • 4-year historical data (2021-2024)")
        print("     • Captain win percentages")
        print("     • T20 milestone focus (30+, 50+, 70+)")
        print("     • Impact metrics (balls per boundary)")
        print("     • Phase-specific statistics")
        print("     • All-rounders in both files")
        print()
        print("  🎯 Ready For:")
        print("     • OVR formula application")
        print("     • Stadium-specific analysis")
        print("     • Player vs Player matrices")
        print("\n" + "█" * 80)
        
    except Exception as e:
        print("\n" + "═" * 80)
        print("  ❌ ERROR OCCURRED")
        print("═" * 80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()