"""
════════════════════════════════════════════════════════════════════════════════
CLEAN PLAYER NOMENCLATURE MATCHING SYSTEM - FIXED FOR 2021-2024
════════════════════════════════════════════════════════════════════════════════

PURPOSE:
Create a clean, unambiguous player reference mapping system that:
1. Uses full canonical names from IPL_2025_Players_list.csv
2. Maps to ALL name variants in Kaggle historical data (2021-2024) ✅ FIXED
3. Separates current players (with history) from debut players (no history)
4. Applies strict matching rules (95%+ similarity) with multiple validation layers
5. Handles disambiguation for players with similar names

OUTPUT FILES:
1. current_players_2025.csv - Players with historical data + all name variants
2. debut_players_2025.csv - Players with no historical data
3. matching_report.md - Detailed report of all matches and decisions

AUTHOR: El Dorado Project - CSCI 566
DATE: November 24, 2024
VERSION: 1.1 - Fixed 2021 Data Loading
════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil")
DATA_DIR = ROOT_DIR / "Dataset(s) and code" / "dataset"
KAGGLE_DIR = ROOT_DIR / "Kaggle_download"
OUTPUT_DIR = DATA_DIR / "Clean_Nomenclature"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Input files
PLAYERS_2025 = DATA_DIR / "IPL_2025_Players_list.csv"

# Historical data files (FIXED FOR 2021-2024)
RAJSENGO_DIR = KAGGLE_DIR / "rajsengo"

# Use all_season files for 2021-2023, separate file for 2024
ALL_SEASON_BATTING = RAJSENGO_DIR / "all_season_batting_card.csv"
ALL_SEASON_BOWLING = RAJSENGO_DIR / "all_season_bowling_card.csv"
BATTING_2024 = RAJSENGO_DIR / "2024" / "season_batting_card.csv"
BOWLING_2024 = RAJSENGO_DIR / "2024" / "season_bowling_card.csv"

# Years to extract (2021-2024)
START_YEAR = 2021
END_YEAR = 2024

# Matching parameters
SIMILARITY_THRESHOLD = 95  # Very strict (95%+)
EXACT_MATCH_THRESHOLD = 100
HIGH_CONFIDENCE_THRESHOLD = 97

# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

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

def calculate_similarity(name1, name2):
    """Calculate similarity between two names"""
    return fuzz.ratio(name1.lower().strip(), name2.lower().strip())

def normalize_name(name):
    """Normalize name for comparison"""
    return name.lower().strip().replace("  ", " ")

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING - FIXED FOR 2021-2024
# ════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all required data files (2021-2024)"""
    print_section("LOADING DATA FILES (2021-2024)")
    
    data = {}
    
    # Load 2025 player list
    print_progress("Loading IPL 2025 Players List...")
    data['players_2025'] = pd.read_csv(PLAYERS_2025)
    print_success(f"Loaded {len(data['players_2025'])} players")
    
    # Check for duplicate names
    duplicates = data['players_2025']['PLAYER_NAME'].duplicated()
    if duplicates.any():
        dup_names = data['players_2025'][duplicates]['PLAYER_NAME'].tolist()
        print_warning(f"Found {len(dup_names)} duplicate names: {dup_names}")
        print_warning("Will add suffixes for disambiguation")
    else:
        print_success("All player names are unique ✓")
    
    # Load batting data (2021-2024)
    print_progress("Loading historical batting data (2021-2024)...")
    data['batting'] = {}
    data['batting_names'] = {}
    
    # Load all_season file and filter for 2021-2023
    if ALL_SEASON_BATTING.exists():
        print_progress(f"  Loading all_season_batting_card.csv...")
        df_all = pd.read_csv(ALL_SEASON_BATTING)
        df_all['season'] = pd.to_numeric(df_all['season'], errors='coerce')
        
        # Split by year (2021-2023)
        for year in range(START_YEAR, 2024):
            df_year = df_all[df_all['season'] == year].copy()
            if len(df_year) > 0:
                data['batting'][year] = df_year
                
                # Extract unique names
                names_short = set(df_year['name'].dropna().unique())
                names_full = set(df_year['fullName'].dropna().unique()) if 'fullName' in df_year.columns else set()
                data['batting_names'][year] = {
                    'short': names_short,
                    'full': names_full,
                    'all': names_short | names_full
                }
                
                print_success(f"  {year}: {len(df_year)} records, {len(names_short)} unique players")
            else:
                print_warning(f"  {year}: No data found")
    else:
        print_error(f"  all_season_batting_card.csv not found!")
    
    # Load 2024 batting data
    if BATTING_2024.exists():
        print_progress(f"  Loading 2024 batting data...")
        df_2024 = pd.read_csv(BATTING_2024)
        df_2024['season'] = pd.to_numeric(df_2024['season'], errors='coerce')
        
        data['batting'][2024] = df_2024
        
        # Extract unique names
        names_short = set(df_2024['name'].dropna().unique())
        names_full = set(df_2024['fullName'].dropna().unique()) if 'fullName' in df_2024.columns else set()
        data['batting_names'][2024] = {
            'short': names_short,
            'full': names_full,
            'all': names_short | names_full
        }
        
        print_success(f"  2024: {len(df_2024)} records, {len(names_short)} unique players")
    else:
        print_error(f"  2024 batting file not found!")
    
    # Load bowling data (2021-2024)
    print_progress("Loading historical bowling data (2021-2024)...")
    data['bowling'] = {}
    data['bowling_names'] = {}
    
    # Load all_season file and filter for 2021-2023
    if ALL_SEASON_BOWLING.exists():
        print_progress(f"  Loading all_season_bowling_card.csv...")
        df_all = pd.read_csv(ALL_SEASON_BOWLING)
        df_all['season'] = pd.to_numeric(df_all['season'], errors='coerce')
        
        # Split by year (2021-2023)
        for year in range(START_YEAR, 2024):
            df_year = df_all[df_all['season'] == year].copy()
            if len(df_year) > 0:
                data['bowling'][year] = df_year
                
                # Extract unique names
                names_short = set(df_year['name'].dropna().unique())
                names_full = set(df_year['fullName'].dropna().unique()) if 'fullName' in df_year.columns else set()
                data['bowling_names'][year] = {
                    'short': names_short,
                    'full': names_full,
                    'all': names_short | names_full
                }
                
                print_success(f"  {year}: {len(df_year)} records, {len(names_short)} unique players")
            else:
                print_warning(f"  {year}: No data found")
    else:
        print_error(f"  all_season_bowling_card.csv not found!")
    
    # Load 2024 bowling data
    if BOWLING_2024.exists():
        print_progress(f"  Loading 2024 bowling data...")
        df_2024 = pd.read_csv(BOWLING_2024)
        df_2024['season'] = pd.to_numeric(df_2024['season'], errors='coerce')
        
        data['bowling'][2024] = df_2024
        
        # Extract unique names
        names_short = set(df_2024['name'].dropna().unique())
        names_full = set(df_2024['fullName'].dropna().unique()) if 'fullName' in df_2024.columns else set()
        data['bowling_names'][2024] = {
            'short': names_short,
            'full': names_full,
            'all': names_short | names_full
        }
        
        print_success(f"  2024: {len(df_2024)} records, {len(names_short)} unique players")
    else:
        print_error(f"  2024 bowling file not found!")
    
    # Create combined historical name sets
    all_historical_names = set()
    for year_data in data['batting_names'].values():
        all_historical_names.update(year_data['all'])
    for year_data in data['bowling_names'].values():
        all_historical_names.update(year_data['all'])
    
    data['all_historical_names'] = all_historical_names
    print_success(f"Total unique historical names (2021-2024): {len(all_historical_names)}")
    
    # Show year coverage
    batting_years = sorted(data['batting'].keys())
    bowling_years = sorted(data['bowling'].keys())
    print_progress(f"Batting data years: {batting_years}")
    print_progress(f"Bowling data years: {bowling_years}")
    
    return data

# ════════════════════════════════════════════════════════════════════════════
# PLAYER TYPE VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def validate_player_type(player_type_2025, found_in_batting, found_in_bowling):
    """
    Validate if player type is consistent with where data was found
    
    Returns: (is_valid, confidence_adjustment)
    """
    if player_type_2025 in ['Batter', 'WK-Batter']:
        # Should primarily be in batting data
        if found_in_batting and not found_in_bowling:
            return True, 0  # Perfect match
        elif found_in_batting and found_in_bowling:
            return True, -2  # Okay, might be occasional bowler
        else:
            return False, -10  # Wrong type
    
    elif player_type_2025 == 'Bowler':
        # Should primarily be in bowling data
        if found_in_bowling and not found_in_batting:
            return True, 0  # Perfect match
        elif found_in_bowling and found_in_batting:
            return True, -2  # Okay, might bat occasionally
        else:
            return False, -10  # Wrong type
    
    elif player_type_2025 == 'All-Rounder':
        # Could be in either or both
        if found_in_batting or found_in_bowling:
            return True, 0  # Any data is good
        else:
            return False, -10  # No data
    
    return True, 0  # Default: accept

# ════════════════════════════════════════════════════════════════════════════
# TEMPORAL VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def get_first_appearance_year(player_name, batting_data, bowling_data):
    """Find the earliest year this player appeared in data"""
    years = []
    
    for year, df in batting_data.items():
        if player_name in df['name'].values or \
           (player_name in df['fullName'].values if 'fullName' in df.columns else False):
            years.append(year)
    
    for year, df in bowling_data.items():
        if player_name in df['name'].values or \
           (player_name in df['fullName'].values if 'fullName' in df.columns else False):
            years.append(year)
    
    return min(years) if years else None

# ════════════════════════════════════════════════════════════════════════════
# NAME MATCHING ENGINE
# ════════════════════════════════════════════════════════════════════════════

def find_exact_match(player_name_2025, all_historical_names):
    """
    Strategy 1: Exact match (100% confidence)
    """
    normalized_2025 = normalize_name(player_name_2025)
    
    for hist_name in all_historical_names:
        if normalize_name(hist_name) == normalized_2025:
            return hist_name, 100, "EXACT"
    
    return None, 0, None

def find_fuzzy_matches(player_name_2025, all_historical_names, threshold=95):
    """
    Strategy 2: Fuzzy matching with confidence scoring
    Returns list of (name, similarity, match_type) tuples
    """
    matches = []
    
    for hist_name in all_historical_names:
        similarity = calculate_similarity(player_name_2025, hist_name)
        if similarity >= threshold:
            match_type = "EXACT" if similarity == 100 else "HIGH_SIMILARITY"
            matches.append((hist_name, similarity, match_type))
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches

def collect_all_variants(matched_name, batting_names, bowling_names):
    """
    Collect all name variants for a matched player across all years
    Returns: set of all name variants
    """
    variants = set()
    
    # Check batting data
    for year_data in batting_names.values():
        if matched_name in year_data['all']:
            # Find which field it was in
            if matched_name in year_data['short']:
                # Also get the corresponding fullName if exists
                variants.add(matched_name)
            if matched_name in year_data['full']:
                variants.add(matched_name)
    
    # Check bowling data
    for year_data in bowling_names.values():
        if matched_name in year_data['all']:
            if matched_name in year_data['short']:
                variants.add(matched_name)
            if matched_name in year_data['full']:
                variants.add(matched_name)
    
    return variants

# ════════════════════════════════════════════════════════════════════════════
# MAIN MATCHING PROCESS
# ════════════════════════════════════════════════════════════════════════════

def match_all_players(data):
    """Match all 2025 players to historical data"""
    print_section("MATCHING PLAYERS (2021-2024 DATA)")
    
    players_2025 = data['players_2025']
    all_historical_names = data['all_historical_names']
    batting_data = data['batting']
    bowling_data = data['bowling']
    batting_names = data['batting_names']
    bowling_names = data['bowling_names']
    
    results = []
    
    for idx, row in players_2025.iterrows():
        player_name = row['PLAYER_NAME']
        player_type = row['PLAYER_TYPE']
        team = row['IPL_TEAM']
        
        if (idx + 1) % 25 == 0:
            print_progress(f"Progress: {idx + 1}/{len(players_2025)} players...")
        
        # Try exact match first
        matched_name, confidence, match_type = find_exact_match(player_name, all_historical_names)
        
        # If no exact match, try fuzzy
        if matched_name is None:
            fuzzy_matches = find_fuzzy_matches(player_name, all_historical_names, SIMILARITY_THRESHOLD)
            if fuzzy_matches:
                # Take best match
                matched_name, confidence, match_type = fuzzy_matches[0]
        
        # If we found a match
        if matched_name:
            # Collect all variants
            all_variants = collect_all_variants(matched_name, batting_names, bowling_names)
            all_variants.add(player_name)  # Include 2025 name too
            
            # Determine where player was found
            found_in_batting = any(matched_name in year_data['all'] for year_data in batting_names.values())
            found_in_bowling = any(matched_name in year_data['all'] for year_data in bowling_names.values())
            
            # Validate player type
            is_valid, conf_adj = validate_player_type(player_type, found_in_batting, found_in_bowling)
            final_confidence = min(100, confidence + conf_adj)
            
            # Get first appearance
            first_year = get_first_appearance_year(matched_name, batting_data, bowling_data)
            
            # Determine short name (usually from 'name' column) and full name
            short_names = set()
            full_names = set()
            
            for year_data in batting_names.values():
                if matched_name in year_data['short']:
                    short_names.add(matched_name)
                if matched_name in year_data['full']:
                    full_names.add(matched_name)
            
            for year_data in bowling_names.values():
                if matched_name in year_data['short']:
                    short_names.add(matched_name)
                if matched_name in year_data['full']:
                    full_names.add(matched_name)
            
            # Pick canonical names
            kaggle_short = list(short_names)[0] if short_names else matched_name
            kaggle_full = list(full_names)[0] if full_names else matched_name
            
            results.append({
                '2025_FullName': player_name,
                'Kaggle_ShortName': kaggle_short,
                'Kaggle_FullName': kaggle_full,
                'All_Variants': '|'.join(sorted(all_variants)),
                'Confidence': final_confidence,
                'Match_Type': match_type,
                'Player_Type': player_type,
                'IPL_Team_2025': team,
                'Found_In_Batting': 'Yes' if found_in_batting else 'No',
                'Found_In_Bowling': 'Yes' if found_in_bowling else 'No',
                'First_Appearance': first_year if first_year else 'N/A',
                'Status': 'MATCHED'
            })
        else:
            # No match found - debut player
            results.append({
                '2025_FullName': player_name,
                'Kaggle_ShortName': '',
                'Kaggle_FullName': '',
                'All_Variants': player_name,
                'Confidence': 0,
                'Match_Type': 'NO_MATCH',
                'Player_Type': player_type,
                'IPL_Team_2025': team,
                'Found_In_Batting': 'No',
                'Found_In_Bowling': 'No',
                'First_Appearance': 'DEBUT',
                'Status': 'DEBUT'
            })
    
    print_success(f"Processed all {len(players_2025)} players")
    
    return pd.DataFrame(results)

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ════════════════════════════════════════════════════════════════════════════

def save_results(results_df):
    """Save all output files"""
    print_section("SAVING RESULTS")
    
    # Split into current and debut
    current_players = results_df[results_df['Status'] == 'MATCHED'].copy()
    debut_players = results_df[results_df['Status'] == 'DEBUT'].copy()
    
    # Save current players
    current_file = OUTPUT_DIR / f"current_players_2025_{TIMESTAMP}.csv"
    current_players.to_csv(current_file, index=False)
    print_success(f"Saved: {current_file.name}")
    print_progress(f"  {len(current_players)} players with historical data (2021-2024)")
    
    # Save debut players
    debut_file = OUTPUT_DIR / f"debut_players_2025_{TIMESTAMP}.csv"
    debut_players.to_csv(debut_file, index=False)
    print_success(f"Saved: {debut_file.name}")
    print_progress(f"  {len(debut_players)} debut players")
    
    # Generate report
    report_file = OUTPUT_DIR / f"matching_report_{TIMESTAMP}.md"
    generate_report(report_file, results_df, current_players, debut_players)
    print_success(f"Saved: {report_file.name}")
    
    return current_file, debut_file, report_file

def generate_report(report_file, results_df, current_players, debut_players):
    """Generate markdown report"""
    lines = []
    lines.append("# IPL 2025 Player Nomenclature Matching Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data Range:** 2021-2024 (4 seasons) ✅")
    lines.append(f"**Match Threshold:** {SIMILARITY_THRESHOLD}% similarity\n")
    
    lines.append("---\n")
    lines.append("## 📊 Summary Statistics\n")
    lines.append(f"- **Total Players:** {len(results_df)}")
    lines.append(f"- **Matched (with history):** {len(current_players)} ({len(current_players)/len(results_df)*100:.1f}%)")
    lines.append(f"- **Debut (no history):** {len(debut_players)} ({len(debut_players)/len(results_df)*100:.1f}%)\n")
    
    # Match type breakdown
    lines.append("### Match Quality Breakdown\n")
    exact = len(current_players[current_players['Match_Type'] == 'EXACT'])
    high_sim = len(current_players[current_players['Match_Type'] == 'HIGH_SIMILARITY'])
    lines.append(f"- **Exact Matches (100%):** {exact}")
    lines.append(f"- **High Similarity (95-99%):** {high_sim}\n")
    
    # Year coverage
    lines.append("### Historical Data Coverage (2021-2024)\n")
    year_dist = current_players['First_Appearance'].value_counts().sort_index()
    for year, count in year_dist.items():
        if year != 'N/A':
            lines.append(f"- **First appeared in {year}:** {count} players")
    lines.append("")
    
    lines.append("---\n")
    lines.append("## ✅ Matched Players Sample\n")
    sample = current_players.head(10)
    for _, row in sample.iterrows():
        lines.append(f"- **{row['2025_FullName']}** → `{row['Kaggle_ShortName']}` ({row['Confidence']}% match)")
    
    lines.append("\n---\n")
    lines.append("## 🆕 Debut Players\n")
    for _, row in debut_players.iterrows():
        lines.append(f"- **{row['2025_FullName']}** ({row['Player_Type']}, {row['IPL_Team_2025']})")
    
    lines.append("\n---\n")
    lines.append("*Generated by El Dorado Project - CSCI 566*")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution"""
    print("\n" + "█" * 80)
    print("  CLEAN PLAYER NOMENCLATURE MATCHING")
    print("  Data Range: 2021-2024 (4 seasons) ✅ FIXED")
    print("  El Dorado Project - CSCI 566")
    print("█" * 80)
    
    start_time = datetime.now()
    
    try:
        # Load data
        data = load_all_data()
        
        # Match players
        results = match_all_players(data)
        
        # Save results
        current_file, debut_file, report_file = save_results(results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_section("EXECUTION COMPLETE")
        print_success(f"Total execution time: {duration:.1f} seconds")
        print_success(f"Output directory: {OUTPUT_DIR}")
        print()
        print("  📊 Created Files:")
        print(f"     1. {current_file.name}")
        print(f"     2. {debut_file.name}")
        print(f"     3. {report_file.name}")
        print()
        print("  ✅ Data Coverage: 2021-2024 (4 full seasons)")
        print()
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