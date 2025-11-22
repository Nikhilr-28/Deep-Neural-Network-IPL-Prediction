"""
STEP 1 ULTIMATE: Enhanced Player Matching with Comprehensive Strategies
==========================================================================

This version handles ALL name variations:
1. Exact matching
2. FullName matching
3. Reversed name matching (Vyshak Vijaykumar vs Vijaykumar Vyshak)
4. Last name matching
5. First name + Last name combinations
6. Initial variations (V Vyshak vs Vyshak V)
7. Fuzzy matching on all combinations
8. Partial name matching (for foreign players)

Goal: Maximize matching rate to 85-90%+ (200+ players)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(r"A:\DL\Dataset_Nikhl")
KAGGLE_DIR = ROOT_DIR / "Kaggle_download"
PLAYER_LIST_2025 = ROOT_DIR / "IPL_2025_Players_list.csv"

BATTING_FILES = {
    2022: KAGGLE_DIR / "rajsengo" / "2022" / "season_batting_card.csv",
    2023: KAGGLE_DIR / "rajsengo" / "2023" / "batting_card.csv",
    2024: KAGGLE_DIR / "rajsengo" / "2024" / "season_batting_card.csv"
}

BOWLING_FILES = {
    2022: KAGGLE_DIR / "rajsengo" / "2022" / "season_bowling_card.csv",
    2023: KAGGLE_DIR / "rajsengo" / "2023" / "bowling_card.csv",
    2024: KAGGLE_DIR / "rajsengo" / "2024" / "season_bowling_card.csv"
}

OUTPUT_DIR = ROOT_DIR / "Step1_Ultimate_Outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# ENHANCED NAME MATCHING FUNCTIONS
# ============================================================================

def clean_name(name):
    """Clean and normalize name"""
    if pd.isna(name):
        return ""
    # Remove extra spaces, special characters
    name = re.sub(r'[^\w\s-]', '', str(name))
    name = ' '.join(name.split()).strip()
    return name

def get_name_parts(full_name):
    """Extract first name, last name, and all parts"""
    if pd.isna(full_name):
        return None, None, []
    
    parts = clean_name(full_name).split()
    if not parts:
        return None, None, []
    
    first_name = parts[0]
    last_name = parts[-1] if len(parts) > 1 else parts[0]
    
    return first_name, last_name, parts

def normalize_name(name):
    """Normalize player name with common replacements"""
    if pd.isna(name):
        return ""
    
    name = clean_name(name)
    
    replacements = {
        'Mohammed': 'Mohammad',
        'Md': 'Mohammad',
        'Mohd': 'Mohammad',
        'Mohammad': 'Mohammad',
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name

def similarity_ratio(str1, str2):
    """Calculate similarity ratio"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def create_name_variations(full_name):
    """Create all possible variations of a name"""
    variations = set()
    
    if pd.isna(full_name):
        return variations
    
    normalized = normalize_name(full_name)
    variations.add(normalized)
    
    first, last, parts = get_name_parts(normalized)
    
    if not parts:
        return variations
    
    # Original
    variations.add(' '.join(parts))
    
    # Reversed (Vyshak Vijaykumar -> Vijaykumar Vyshak)
    if len(parts) >= 2:
        variations.add(' '.join(parts[::-1]))
    
    # First + Last only
    if first and last and len(parts) > 2:
        variations.add(f"{first} {last}")
        variations.add(f"{last} {first}")
    
    # With initials
    if len(parts) >= 2:
        # First initial + Last (V Vyshak)
        variations.add(f"{first[0]} {last}")
        # Last + First initial (Vyshak V)
        variations.add(f"{last} {first[0]}")
        # First + Last initial
        variations.add(f"{first} {last[0]}")
    
    if len(parts) >= 3:
        # Middle initials
        initials = ''.join([p[0] for p in parts[:-1]])
        variations.add(f"{initials} {last}")
        
        # First + middle initial + Last
        variations.add(f"{first} {parts[1][0]} {last}")
    
    # All initials + last name
    if len(parts) > 1:
        initials = ''.join([p[0] for p in parts[:-1]])
        variations.add(f"{initials} {last}")
    
    return variations

def match_player_ultimate(target_name, candidate_dict, threshold=0.75):
    """
    Ultimate matching algorithm with all strategies
    
    Args:
        target_name: Player name from 2025 list
        candidate_dict: {hist_name: hist_fullname} from historical data
        threshold: Minimum similarity score
    
    Returns:
        (matched_name, confidence_score, match_type, matched_fullname)
    """
    
    target_normalized = normalize_name(target_name)
    target_variations = create_name_variations(target_name)
    target_first, target_last, target_parts = get_name_parts(target_normalized)
    
    # Strategy 1: Exact match on any variation
    for hist_name, hist_fullname in candidate_dict.items():
        hist_normalized = normalize_name(hist_name)
        
        # Check against all target variations
        if hist_normalized.lower() in [v.lower() for v in target_variations]:
            return hist_name, 1.0, "exact_variation", hist_fullname
        
        # Check if target variation matches historical name
        for variation in target_variations:
            if variation.lower() == hist_normalized.lower():
                return hist_name, 1.0, "exact_variation", hist_fullname
    
    # Strategy 2: Exact match on fullName with variations
    for hist_name, hist_fullname in candidate_dict.items():
        if pd.notna(hist_fullname):
            hist_full_normalized = normalize_name(hist_fullname)
            hist_variations = create_name_variations(hist_fullname)
            
            # Check all combinations
            for t_var in target_variations:
                for h_var in hist_variations:
                    if t_var.lower() == h_var.lower():
                        return hist_name, 1.0, "exact_fullname_variation", hist_fullname
    
    # Strategy 3: Last name exact match (unique surnames)
    if target_last and len(target_last) > 2:
        last_name_matches = []
        
        for hist_name, hist_fullname in candidate_dict.items():
            _, hist_last, _ = get_name_parts(hist_name)
            _, hist_full_last, _ = get_name_parts(hist_fullname)
            
            # Check both name and fullName fields
            if (hist_last and hist_last.lower() == target_last.lower()) or \
               (hist_full_last and hist_full_last.lower() == target_last.lower()):
                last_name_matches.append((hist_name, hist_fullname))
        
        # If only one last name match, use it
        if len(last_name_matches) == 1:
            hist_name, hist_fullname = last_name_matches[0]
            return hist_name, 0.95, "lastname_unique", hist_fullname
        
        # If multiple, use additional fuzzy matching
        if len(last_name_matches) > 1:
            best_match = None
            best_score = 0
            
            for hist_name, hist_fullname in last_name_matches:
                # Try all variations
                for t_var in target_variations:
                    # Check against hist_name
                    score1 = similarity_ratio(t_var, hist_name)
                    
                    # Check against hist_fullname
                    score2 = 0
                    if pd.notna(hist_fullname):
                        score2 = similarity_ratio(t_var, hist_fullname)
                    
                    # Also check reversed
                    hist_variations = create_name_variations(hist_fullname if pd.notna(hist_fullname) else hist_name)
                    for h_var in hist_variations:
                        score3 = similarity_ratio(t_var, h_var)
                        score = max(score1, score2, score3)
                        
                        if score > best_score:
                            best_score = score
                            best_match = (hist_name, hist_fullname)
            
            if best_match and best_score >= 0.70:
                return best_match[0], best_score, "lastname_fuzzy", best_match[1]
    
    # Strategy 4: First name + Last name combination match
    if target_first and target_last:
        for hist_name, hist_fullname in candidate_dict.items():
            hist_first, hist_last, _ = get_name_parts(hist_name)
            hist_full_first, hist_full_last, _ = get_name_parts(hist_fullname)
            
            # Check if both first and last match (order independent)
            if hist_first and hist_last:
                if (target_first.lower() == hist_first.lower() and target_last.lower() == hist_last.lower()) or \
                   (target_first.lower() == hist_last.lower() and target_last.lower() == hist_first.lower()):
                    return hist_name, 0.93, "first_last_match", hist_fullname
            
            if hist_full_first and hist_full_last:
                if (target_first.lower() == hist_full_first.lower() and target_last.lower() == hist_full_last.lower()) or \
                   (target_first.lower() == hist_full_last.lower() and target_last.lower() == hist_full_first.lower()):
                    return hist_name, 0.93, "first_last_match_full", hist_fullname
    
    # Strategy 5: Comprehensive fuzzy matching on all variations
    best_match = None
    best_score = 0
    
    for hist_name, hist_fullname in candidate_dict.items():
        hist_variations = create_name_variations(hist_fullname if pd.notna(hist_fullname) else hist_name)
        
        # Try all combinations of variations
        for t_var in target_variations:
            for h_var in hist_variations:
                score = similarity_ratio(t_var, h_var)
                if score > best_score:
                    best_score = score
                    best_match = (hist_name, hist_fullname)
            
            # Also try against original hist_name and hist_fullname
            score1 = similarity_ratio(t_var, hist_name)
            if score1 > best_score:
                best_score = score1
                best_match = (hist_name, hist_fullname)
            
            if pd.notna(hist_fullname):
                score2 = similarity_ratio(t_var, hist_fullname)
                if score2 > best_score:
                    best_score = score2
                    best_match = (hist_name, hist_fullname)
    
    if best_score >= threshold:
        return best_match[0], best_score, "fuzzy_comprehensive", best_match[1]
    
    return None, 0.0, "no_match", None

# ============================================================================
# MAIN FUNCTIONS (Same structure as before)
# ============================================================================

def load_player_list_2025():
    """Load IPL 2025 player list"""
    print("=" * 80)
    print("LOADING IPL 2025 PLAYER LIST")
    print("=" * 80)
    
    df = pd.read_csv(PLAYER_LIST_2025)
    df_clean = df.drop_duplicates(subset=['PLAYER_NAME'], keep='first')
    
    print(f"\nTotal players: {len(df_clean)}")
    print(f"\nPlayer type breakdown:")
    print(df_clean['PLAYER_TYPE'].value_counts())
    
    return df_clean

def load_historical_data():
    """Load 2022-2024 data"""
    print("\n" + "=" * 80)
    print("LOADING HISTORICAL DATA (2022-2024)")
    print("=" * 80)
    
    batting_data = {}
    bowling_data = {}
    
    print("\n📊 Loading Batting Files:")
    for year, filepath in BATTING_FILES.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            batting_data[year] = df
            print(f"  ✓ {year}: {len(df)} records, {df['name'].nunique()} unique players")
        else:
            print(f"  ✗ {year}: FILE NOT FOUND")
    
    print("\n🎯 Loading Bowling Files:")
    for year, filepath in BOWLING_FILES.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            bowling_data[year] = df
            print(f"  ✓ {year}: {len(df)} records, {df['name'].nunique()} unique players")
        else:
            print(f"  ✗ {year}: FILE NOT FOUND")
    
    return batting_data, bowling_data

def build_candidate_dict(batting_data, bowling_data):
    """Build dictionary of {name: fullName}"""
    candidate_dict = {}
    
    for year, df in batting_data.items():
        for _, row in df[['name', 'fullName']].drop_duplicates().iterrows():
            if pd.notna(row['name']):
                candidate_dict[row['name']] = row.get('fullName', row['name'])
    
    for year, df in bowling_data.items():
        for _, row in df[['name', 'fullName']].drop_duplicates().iterrows():
            if pd.notna(row['name']):
                candidate_dict[row['name']] = row.get('fullName', row['name'])
    
    return candidate_dict

def match_all_players_ultimate(players_2025, candidate_dict):
    """Match players using ultimate algorithm"""
    print("\n" + "=" * 80)
    print("MATCHING PLAYERS (ULTIMATE ALGORITHM)")
    print("=" * 80)
    print(f"\nTotal historical names: {len(candidate_dict)}")
    print("Using: Exact, Reversed, Initials, First+Last, Fuzzy variations")
    
    results = []
    match_types_count = {}
    
    for idx, row in players_2025.iterrows():
        player_name = row['PLAYER_NAME']
        player_type = row['PLAYER_TYPE']
        team = row['IPL_TEAM']
        
        matched_name, confidence, match_type, matched_fullname = match_player_ultimate(
            player_name, candidate_dict, threshold=0.72
        )
        
        results.append({
            '2025_Name': player_name,
            'Matched_Historical_Name': matched_name,
            'Matched_FullName': matched_fullname,
            'Confidence': round(confidence, 3),
            'Match_Type': match_type,
            'Player_Type': player_type,
            'IPL_Team_2025': team,
            'Status': 'MATCHED' if matched_name else 'UNMATCHED'
        })
        
        match_types_count[match_type] = match_types_count.get(match_type, 0) + 1
        
        # Print matches
        if confidence >= 0.95:
            print(f"  ✓ {player_name:30s} → {matched_name:20s} ({match_type})")
        elif 0.80 <= confidence < 0.95:
            print(f"  ? {player_name:30s} → {matched_name:20s} ({confidence:.2f}, {match_type})")
        elif matched_name:
            print(f"  ~ {player_name:30s} → {matched_name:20s} ({confidence:.2f}, {match_type})")
    
    print(f"\n📊 Match Type Distribution:")
    for mtype, count in sorted(match_types_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mtype}: {count}")
    
    match_df = pd.DataFrame(results)
    return match_df

def extract_filtered_data(match_df, batting_data, bowling_data):
    """Extract data for matched players"""
    print("\n" + "=" * 80)
    print("EXTRACTING FILTERED DATA")
    print("=" * 80)
    
    matched_players = match_df[match_df['Status'] == 'MATCHED']
    matched_names = matched_players['Matched_Historical_Name'].tolist()
    
    print(f"\nPlayers to extract: {len(matched_names)}")
    
    filtered_batting = {}
    total_batting = 0
    
    print("\n📊 Filtering Batting Data:")
    for year, df in batting_data.items():
        filtered = df[df['name'].isin(matched_names)].copy()
        filtered_batting[year] = filtered
        total_batting += len(filtered)
        print(f"  {year}: {len(filtered)} records ({filtered['name'].nunique()} players)")
    
    filtered_bowling = {}
    total_bowling = 0
    
    print("\n🎯 Filtering Bowling Data:")
    for year, df in bowling_data.items():
        filtered = df[df['name'].isin(matched_names)].copy()
        filtered_bowling[year] = filtered
        total_bowling += len(filtered)
        print(f"  {year}: {len(filtered)} records ({filtered['name'].nunique()} players)")
    
    print(f"\n✓ Total batting records: {total_batting}")
    print(f"✓ Total bowling records: {total_bowling}")
    
    return filtered_batting, filtered_bowling

def generate_match_report(match_df, filtered_batting, filtered_bowling):
    """Generate match report"""
    print("\n" + "=" * 80)
    print("GENERATING MATCH REPORT")
    print("=" * 80)
    
    matched = match_df[match_df['Status'] == 'MATCHED']
    unmatched = match_df[match_df['Status'] == 'UNMATCHED']
    
    print(f"\n📊 STATISTICS:")
    print(f"  Total 2025 players: {len(match_df)}")
    print(f"  Matched: {len(matched)} ({len(matched)/len(match_df)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched)} ({len(unmatched)/len(match_df)*100:.1f}%)")
    print(f"  Average confidence: {matched['Confidence'].mean():.3f}")
    
    all_batting = pd.concat(filtered_batting.values(), ignore_index=True)
    all_bowling = pd.concat(filtered_bowling.values(), ignore_index=True)
    
    batting_counts = all_batting['name'].value_counts()
    bowling_counts = all_bowling['name'].value_counts()
    
    print(f"\n📋 DATA AVAILABILITY:")
    print(f"  Players with batting: {len(batting_counts)}")
    print(f"  Players with bowling: {len(bowling_counts)}")
    
    return {
        'matched': matched,
        'unmatched': unmatched,
        'batting_counts': batting_counts,
        'bowling_counts': bowling_counts
    }

def save_outputs(match_df, filtered_batting, filtered_bowling, report):
    """Save outputs"""
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save files
    match_df.to_csv(OUTPUT_DIR / f"player_match_report_ULTIMATE_{timestamp}.csv", index=False)
    report['matched'].to_csv(OUTPUT_DIR / f"matched_players_ULTIMATE_{timestamp}.csv", index=False)
    report['unmatched'].to_csv(OUTPUT_DIR / f"unmatched_players_ULTIMATE_{timestamp}.csv", index=False)
    
    all_batting = pd.concat(filtered_batting.values(), ignore_index=True)
    all_batting.to_csv(OUTPUT_DIR / f"filtered_batting_2022_2024_ULTIMATE_{timestamp}.csv", index=False)
    
    all_bowling = pd.concat(filtered_bowling.values(), ignore_index=True)
    all_bowling.to_csv(OUTPUT_DIR / f"filtered_bowling_2022_2024_ULTIMATE_{timestamp}.csv", index=False)
    
    # Summary
    summary_data = []
    for player in report['matched']['Matched_Historical_Name']:
        batting_innings = report['batting_counts'].get(player, 0)
        bowling_innings = report['bowling_counts'].get(player, 0)
        player_info = report['matched'][report['matched']['Matched_Historical_Name'] == player].iloc[0]
        
        summary_data.append({
            'Player_Name': player,
            '2025_Name': player_info['2025_Name'],
            'Matched_FullName': player_info['Matched_FullName'],
            'Player_Type': player_info['Player_Type'],
            'IPL_Team_2025': player_info['IPL_Team_2025'],
            'Batting_Innings': batting_innings,
            'Bowling_Innings': bowling_innings,
            'Total_Innings': batting_innings + bowling_innings,
            'Match_Confidence': player_info['Confidence'],
            'Match_Type': player_info['Match_Type']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Total_Innings', ascending=False)
    summary_df.to_csv(OUTPUT_DIR / f"player_data_summary_ULTIMATE_{timestamp}.csv", index=False)
    
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")

def main():
    """Main execution"""
    print("\n" + "█" * 80)
    print("STEP 1 ULTIMATE: MAXIMUM PLAYER MATCHING")
    print("█" * 80)
    
    try:
        players_2025 = load_player_list_2025()
        batting_data, bowling_data = load_historical_data()
        
        if not batting_data and not bowling_data:
            print("\n❌ ERROR: No data found!")
            return
        
        candidate_dict = build_candidate_dict(batting_data, bowling_data)
        print(f"\n✓ Built candidate dictionary: {len(candidate_dict)} unique players")
        
        match_df = match_all_players_ultimate(players_2025, candidate_dict)
        
        filtered_batting, filtered_bowling = extract_filtered_data(
            match_df, batting_data, bowling_data
        )
        
        report = generate_match_report(match_df, filtered_batting, filtered_bowling)
        
        save_outputs(match_df, filtered_batting, filtered_bowling, report)
        
        matched_count = len(report['matched'])
        total_count = len(match_df)
        
        print("\n" + "█" * 80)
        print("✅ STEP 1 ULTIMATE COMPLETE!")
        print("█" * 80)
        print(f"\nMatched: {matched_count}/{total_count} ({matched_count/total_count*100:.1f}%)")
        print(f"Expected: 200-210 players (87-91%)")
        print(f"\nOutputs in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()