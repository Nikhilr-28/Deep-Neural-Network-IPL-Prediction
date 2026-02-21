"""
================================================================================
CITY DEVIATION CALCULATION - BOWLING
================================================================================
Project: El Dorado - CSCI 566
Purpose: Calculate city-specific bowling performance deviations (2022-2024)

Command: python city_deviation_bowling.py
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
print("CITY DEVIATION CALCULATION - BOWLING")
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
ALL_SUMMARY_FILE = KAGGLE_DIR / "all_season_summary.csv"
SUMMARY_2024_FILE = KAGGLE_DIR / "2024" / "season_summary.csv"
BOWLING_MASTER_FILE = MASTER_DIR / "BOWLING_MASTER_2025_20251129_052900.csv"

# Output files
VENUE_MAPPING_FILE = OUTPUT_DIR / "VENUE_TO_CITY_MAPPING_BOWLING.csv"
OUTPUT_FILE = OUTPUT_DIR / "CITY_DEVIATION_BOWLING_2022_2024.csv"

print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# PART 1: CREATE VENUE MAPPING FROM SUMMARY
# ============================================================================

print("\n[1/6] Creating Venue → City Mapping from Summary Files...")

# Load summary files
summary_chunks = []

for file in [ALL_SUMMARY_FILE, SUMMARY_2024_FILE]:
    if file.exists():
        df = pd.read_csv(file)
        summary_chunks.append(df)
        print(f"  ✓ Loaded: {file.name} ({len(df)} matches)")

summary_df = pd.concat(summary_chunks, ignore_index=True)
print(f"  ✓ Total matches: {len(summary_df)}")

# Extract venue mapping
venue_mapping = summary_df[['venue_id', 'venue_name']].drop_duplicates().copy()
venue_mapping = venue_mapping.dropna(subset=['venue_name'])

print(f"  ✓ Unique venues: {len(venue_mapping)}")

# Extract city from venue_name
def extract_city_from_venue(venue_name):
    """Extract city from venue name."""
    if pd.isna(venue_name):
        return 'Unknown'
    
    # Manual mapping for known IPL venues
    venue_city_map = {
        'Wankhede': 'Mumbai',
        'Eden Gardens': 'Kolkata',
        'M Chinnaswamy': 'Bengaluru',
        'Chinnaswamy': 'Bengaluru',
        'Chepauk': 'Chennai',
        'MA Chidambaram': 'Chennai',
        'Feroz Shah Kotla': 'Delhi',
        'Arun Jaitley': 'Delhi',
        'Sawai Mansingh': 'Jaipur',
        'Rajiv Gandhi': 'Hyderabad',
        'DY Patil': 'Navi Mumbai',
        'Brabourne': 'Mumbai',
        'Narendra Modi': 'Ahmedabad',
        'Punjab Cricket': 'Mohali',
        'HPCA': 'Dharamshala',
        'Holkar': 'Indore',
        'Green Park': 'Kanpur',
        'Dubai International': 'Dubai',
        'Sheikh Zayed': 'Abu Dhabi',
        'Sharjah Cricket': 'Sharjah',
        'Dubai': 'Dubai',
        'Abu Dhabi': 'Abu Dhabi',
        'Sharjah': 'Sharjah',
        # South Africa venues (IPL 2009)
        'Kingsmead': 'Durban',
        'SuperSport Park': 'Centurion',
        'Newlands': 'Cape Town',
        'St George': 'Port Elizabeth',
        'Wanderers': 'Johannesburg',
    }
    
    venue_lower = str(venue_name).lower()
    for key, city in venue_city_map.items():
        if key.lower() in venue_lower:
            return city
    
    # Fallback: split by comma or use first word
    if ',' in venue_name:
        return venue_name.split(',')[-1].strip()
    
    # Use first meaningful word
    words = venue_name.split()
    if len(words) > 0:
        return words[0]
    
    return venue_name

venue_mapping['city'] = venue_mapping['venue_name'].apply(extract_city_from_venue)

# Metro aggregation
metro_rules = {
    'Navi Mumbai': 'Mumbai',
    'Mumbai': 'Mumbai',
}

venue_mapping['metro_area'] = venue_mapping['city'].replace(metro_rules)

# Add country (India, UAE, or South Africa)
uae_cities = ['Dubai', 'Abu Dhabi', 'Sharjah']
sa_cities = ['Durban', 'Centurion', 'Cape Town', 'Port Elizabeth', 'Johannesburg']

def assign_country(city):
    if city in uae_cities:
        return 'UAE'
    elif city in sa_cities:
        return 'South Africa'
    else:
        return 'India'

venue_mapping['country'] = venue_mapping['metro_area'].apply(assign_country)
venue_mapping['is_uae'] = venue_mapping['country'] == 'UAE'

print(f"  ✓ Cities after metro merge: {venue_mapping['metro_area'].nunique()}")
print(f"  ✓ UAE venues: {venue_mapping['is_uae'].sum()}")
print(f"  ✓ South Africa venues: {(venue_mapping['country'] == 'South Africa').sum()}")

print("\n  Sample venue mappings:")
print(venue_mapping[['venue_name', 'metro_area', 'country']].head(10).to_string(index=False))

# Save mapping
venue_mapping.to_csv(VENUE_MAPPING_FILE, index=False)
print(f"\n  ✓ Saved: {VENUE_MAPPING_FILE.name}")

# ============================================================================
# PART 2: LOAD DETAILS AND MERGE WITH VENUE INFO
# ============================================================================

print("\n[2/6] Loading Ball-by-Ball Data...")

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

# ============================================================================
# PART 3: MERGE DETAILS WITH VENUE MAPPING VIA SUMMARY
# ============================================================================

print("\n[3/6] Merging Details with Venue Information...")

# Get match_id → venue_id mapping from summary
match_venue = summary_df[['id', 'venue_id', 'season']].copy()
match_venue.rename(columns={'id': 'match_id'}, inplace=True)

# Merge details with match_venue
details_df = details_df.merge(match_venue, on=['match_id', 'season'], how='left')
print(f"  ✓ Merged match-venue mapping")

# Merge with venue_mapping to get city
details_df = details_df.merge(
    venue_mapping[['venue_id', 'metro_area', 'country', 'is_uae']], 
    on='venue_id', 
    how='left'
)
print(f"  ✓ Merged venue-city mapping")

# Filter: 2022-2024 only
before = len(details_df)
details_df = details_df[details_df['season'].isin([2022, 2023, 2024])]
print(f"  ✓ Filtered to 2022-2024: {len(details_df):,} rows (removed {before-len(details_df):,})")

# Filter: India only
before = len(details_df)
details_df = details_df[details_df['country'] == 'India']
print(f"  ✓ Filtered to India only: {len(details_df):,} rows")
print(f"    (removed {before-len(details_df):,} non-India: UAE + South Africa)")

cities = sorted(details_df['metro_area'].dropna().unique())
print(f"\n  Cities in dataset ({len(cities)}): {', '.join(cities)}")

# ============================================================================
# PART 4: CALCULATE CITY BOWLING STATS
# ============================================================================

print("\n[4/6] Calculating City-Level Bowling Statistics...")

# Identify bowler (bowler1 is the active bowler)
details_df['bowler'] = details_df['bowler1_name']
details_df['runs_conceded'] = details_df['runs']

# Calculate balls bowled (exclude wides and no-balls from ball count)
details_df['legal_ball'] = (~details_df['isWide']) & (~details_df['isNoball'])

print(f"  Aggregating by bowler + metro_area...")

city_stats = details_df.groupby(['bowler', 'metro_area']).agg({
    'runs_conceded': 'sum',
    'legal_ball': 'sum',  # Count only legal balls
    'wicket_id': lambda x: x.notna().sum()  # Count wickets
}).reset_index()

city_stats.columns = ['bowler', 'city', 'runs', 'balls', 'wickets']

# Calculate overs (6 balls = 1 over)
city_stats['overs'] = (city_stats['balls'] / 6).round(1)

# Calculate Economy Rate and Average
city_stats['economy'] = np.where(
    city_stats['overs'] > 0,
    (city_stats['runs'] / city_stats['overs']).round(2),
    np.nan
)

city_stats['avg'] = np.where(
    city_stats['wickets'] > 0,
    (city_stats['runs'] / city_stats['wickets']).round(2),
    np.nan
)

city_stats['sr'] = np.where(
    city_stats['wickets'] > 0,
    (city_stats['balls'] / city_stats['wickets']).round(2),
    np.nan
)

# Count innings (unique matches per bowler-city)
innings_count = details_df.groupby(['bowler', 'metro_area'])['match_id'].nunique().reset_index()
innings_count.columns = ['bowler', 'city', 'innings']

city_stats = city_stats.merge(innings_count, on=['bowler', 'city'], how='left')

print(f"  ✓ Calculated stats for {len(city_stats):,} bowler-city combinations")

# ============================================================================
# PART 5: CALCULATE DEVIATIONS
# ============================================================================

print("\n[5/6] Calculating Deviations and Merging with Master...")

# Load bowling master
bowling_master = pd.read_csv(BOWLING_MASTER_FILE)
print(f"  ✓ Loaded {len(bowling_master)} players from BOWLING_MASTER")

def calculate_confidence_weight(balls, min_threshold=30, full_confidence=120):
    """Sigmoid curve for confidence weight."""
    if balls < min_threshold:
        return 0.0
    if balls >= full_confidence:
        return min(0.95, 0.85 + (balls - full_confidence) / 1000)
    
    x = (balls - min_threshold) / (full_confidence - min_threshold)
    sigmoid = 1 / (1 + np.exp(-10 * (x - 0.5)))
    return 0.3 + (sigmoid * 0.55)

def calculate_city_deviation_bowling(bowler_econ_city, bowler_overall_econ, city_baseline_econ):
    """
    Calculate city deviation for bowling (INVERTED - lower economy = better).
    
    Formula:
      personal_delta = bowler_overall_econ - bowler_econ_city  # FLIPPED!
      city_difficulty_factor = city_baseline_econ / 8.5  # 8.5 = IPL avg economy
      normalized_delta = personal_delta / (city_baseline_econ * city_difficulty_factor)
      return clip(normalized_delta, -1.0, 1.0)
    """
    if pd.isna(bowler_overall_econ) or bowler_overall_econ == 0:
        return 0.0
    if pd.isna(city_baseline_econ) or city_baseline_econ == 0:
        return 0.0
    
    # INVERTED: overall - city (lower city economy = better = positive deviation)
    personal_delta = bowler_overall_econ - bowler_econ_city
    city_difficulty_factor = city_baseline_econ / 8.5  # 8.5 = IPL avg economy
    
    if city_difficulty_factor == 0:
        return 0.0
    
    normalized_delta = personal_delta / (city_baseline_econ * city_difficulty_factor)
    return np.clip(normalized_delta, -1.0, 1.0)

# Calculate city baselines (average economy per city)
city_baselines = city_stats.groupby('city')['economy'].mean().to_dict()

print(f"\n  City Baseline Economy Rates:")
for city, econ in sorted(city_baselines.items(), key=lambda x: x[1]):
    print(f"    {city:20s}: {econ:>6.2f}")

# Validation: Track unmatched players
unique_bowlers_in_data = set(city_stats['bowler'].unique())
unique_bowlers_in_master = set(bowling_master['Kaggle_Match_Name'].unique())

print(f"\n  Validation:")
print(f"    - Unique bowlers in rajsengo data: {len(unique_bowlers_in_data)}")
print(f"    - Unique bowlers in master: {len(unique_bowlers_in_master)}")

# Track unmatched for reporting
unmatched_bowlers = {}
matched_count = 0

# Calculate deviations
results = []

for _, row in city_stats.iterrows():
    bowler = row['bowler']
    city = row['city']
    city_econ = row['economy']
    city_avg = row['avg']
    city_sr = row['sr']
    balls = row['balls']
    innings = row['innings']
    overs = row['overs']
    
    # Match with master
    master_row = bowling_master[bowling_master['Kaggle_Match_Name'] == bowler]
    
    if len(master_row) == 0:
        # Track unmatched
        if bowler not in unmatched_bowlers:
            unmatched_bowlers[bowler] = {
                'total_balls': 0,
                'cities': set()
            }
        unmatched_bowlers[bowler]['total_balls'] += balls
        unmatched_bowlers[bowler]['cities'].add(city)
        continue
    
    matched_count += 1
    
    master_row = master_row.iloc[0]
    overall_econ = master_row['Economy_Rate']
    overall_avg = master_row['Bowling_Average']
    overall_sr = master_row['Bowling_Strike_Rate']
    
    if pd.isna(overall_econ) or overall_econ == 0 or pd.isna(city_econ):
        continue
    
    city_baseline_econ = city_baselines.get(city, 8.5)
    deviation = calculate_city_deviation_bowling(city_econ, overall_econ, city_baseline_econ)
    confidence = calculate_confidence_weight(balls)
    
    # CRITICAL FIX: Force deviation to 0 if insufficient sample size
    if balls < 30:
        deviation = 0.0
    
    results.append({
        'Player_Name': master_row['Player_Name'],
        'Kaggle_Match_Name': bowler,
        'City': city,
        'Sample_Innings': int(innings) if not pd.isna(innings) else 0,
        'Sample_Balls': int(balls),
        'Confidence_Weight': round(confidence, 3),
        'Overall_Econ': round(overall_econ, 2),
        'City_Econ': round(city_econ, 2),
        'City_Baseline_Econ': round(city_baseline_econ, 2),
        'City_Deviation': round(deviation, 3),
        'Overall_Avg': round(overall_avg, 2) if not pd.isna(overall_avg) else 0.0,
        'City_Avg': round(city_avg, 2) if not pd.isna(city_avg) else 0.0,
        'Overall_SR': round(overall_sr, 2) if not pd.isna(overall_sr) else 0.0,
        'City_SR': round(city_sr, 2) if not pd.isna(city_sr) else 0.0,
    })

output_df = pd.DataFrame(results)
print(f"\n  ✓ Calculated deviations for {len(output_df):,} bowler-city combinations")
print(f"  ✓ Successfully matched: {matched_count:,} player-city pairs")

# Report unmatched players
if unmatched_bowlers:
    print(f"\n  ⚠️  WARNING: {len(unmatched_bowlers)} bowlers in rajsengo NOT FOUND in master:")
    
    # Sort by total balls bowled
    sorted_unmatched = sorted(
        unmatched_bowlers.items(),
        key=lambda x: x[1]['total_balls'],
        reverse=True
    )
    
    print(f"\n  Top unmatched bowlers (by balls bowled):")
    for bowler, info in sorted_unmatched[:20]:
        cities_count = len(info['cities'])
        print(f"    {bowler:30s} - {info['total_balls']:4d} balls, {cities_count:2d} cities")
    
    if len(sorted_unmatched) > 20:
        print(f"    ... and {len(sorted_unmatched) - 20} more")
    
    # Save unmatched to CSV
    unmatched_df = pd.DataFrame([
        {'Bowler_Name': name, 'Total_Balls': info['total_balls'], 'Num_Cities': len(info['cities'])}
        for name, info in sorted_unmatched
    ])
    unmatched_file = OUTPUT_DIR / "UNMATCHED_BOWLERS_2022_2024.csv"
    unmatched_df.to_csv(unmatched_file, index=False)
    print(f"\n  ✓ Saved unmatched list: {unmatched_file.name}")
else:
    print(f"\n  ✓ All bowlers matched successfully!")

# ============================================================================
# PART 6: SAVE AND VALIDATE
# ============================================================================

print("\n[6/6] Saving Output...")

output_df = output_df.sort_values(['Player_Name', 'City'])
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  ✓ Saved: {OUTPUT_FILE.name}")
print(f"  ✓ Total rows: {len(output_df):,}")
print(f"  ✓ Unique players: {output_df['Player_Name'].nunique()}")
print(f"  ✓ Unique cities: {output_df['City'].nunique()}")
print(f"\n  Player coverage:")
print(f"    - Total bowlers in BOWLING_MASTER: {len(bowling_master)}")
print(f"    - Bowlers with city data (2022-24): {output_df['Player_Name'].nunique()}")
print(f"    - Coverage: {output_df['Player_Name'].nunique() / len(bowling_master) * 100:.1f}%")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION - FAMOUS BOWLER-CITY COMBINATIONS")
print("=" * 80)

famous_combos = [
    ('Jasprit Bumrah', 'Mumbai'),
    ('Yuzvendra Chahal', 'Bengaluru'),
    ('Rashid Khan', 'Ahmedabad'),
    ('Mohammed Shami', 'Ahmedabad'),
]

for player, city in famous_combos:
    subset = output_df[
        (output_df['Player_Name'].str.contains(player, case=False, na=False)) & 
        (output_df['City'] == city)
    ]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n{player} at {city}:")
        print(f"  City Econ: {row['City_Econ']:.2f} vs Overall: {row['Overall_Econ']:.2f} (Baseline: {row['City_Baseline_Econ']:.2f})")
        print(f"  Deviation: {row['City_Deviation']:+.3f} | Confidence: {row['Confidence_Weight']:.2f}")
        print(f"  Sample: {row['Sample_Innings']} innings, {row['Sample_Balls']} balls")

print("\n" + "-" * 80)
print("TOP 10 POSITIVE DEVIATIONS (Best City Performance, Min 60 balls)")
print("-" * 80)
top_pos = output_df[output_df['Sample_Balls'] >= 60].nlargest(10, 'City_Deviation')
print(top_pos[['Player_Name', 'City', 'City_Econ', 'Overall_Econ', 'City_Deviation', 'Sample_Balls']].to_string(index=False))

print("\n" + "-" * 80)
print("TOP 10 NEGATIVE DEVIATIONS (Struggles, Min 60 balls)")
print("-" * 80)
top_neg = output_df[output_df['Sample_Balls'] >= 60].nsmallest(10, 'City_Deviation')
print(top_neg[['Player_Name', 'City', 'City_Econ', 'Overall_Econ', 'City_Deviation', 'Sample_Balls']].to_string(index=False))

print("\n" + "=" * 80)
print("✅ COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutputs: {OUTPUT_DIR}")
print(f"  1. {VENUE_MAPPING_FILE.name}")
print(f"  2. {OUTPUT_FILE.name}")
print("\nNext: Create pace/spin matchup scores (pace_spin_matchup.py)")
print("=" * 80)