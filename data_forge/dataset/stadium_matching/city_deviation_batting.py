"""
================================================================================
CITY DEVIATION CALCULATION - BATTING
================================================================================
Project: El Dorado - CSCI 566
Purpose: Calculate city-specific batting performance deviations (2022-2024)

Command: python city_deviation_batting.py
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
print("CITY DEVIATION CALCULATION - BATTING")
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
BATTING_MASTER_FILE = MASTER_DIR / "BATTING_MASTER_2025_20251129_052900.csv"

# Output files
VENUE_MAPPING_FILE = OUTPUT_DIR / "VENUE_TO_CITY_MAPPING.csv"
OUTPUT_FILE = OUTPUT_DIR / "CITY_DEVIATION_BATTING_2022_2024.csv"

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
# Columns: venue_id, venue_name
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

# Filter: India only (exclude UAE and South Africa)
before = len(details_df)
details_df = details_df[details_df['country'] == 'India']
uae_removed = len(details_df[details_df['country'] == 'UAE']) if 'country' in details_df.columns else 0
sa_removed = before - len(details_df) - uae_removed
print(f"  ✓ Filtered to India only: {len(details_df):,} rows")
print(f"    (removed {before-len(details_df):,} non-India: UAE + South Africa)")

cities = sorted(details_df['metro_area'].dropna().unique())
print(f"\n  Cities in dataset ({len(cities)}): {', '.join(cities)}")

# ============================================================================
# PART 4: CALCULATE CITY BATTING STATS
# ============================================================================

print("\n[4/6] Calculating City-Level Batting Statistics...")

# Identify striker (batsman on strike)
# Use batsman1 as striker (they faced the ball)
details_df['striker'] = details_df['batsman1_name']
details_df['runs_off_bat'] = details_df['runs']

# Group by striker + city
print(f"  Aggregating by striker + metro_area...")

city_stats = details_df.groupby(['striker', 'metro_area']).agg({
    'runs_off_bat': ['sum', 'count'],
    'wicket_id': lambda x: x.notna().sum()  # Count dismissals (wicket_id is not null when out)
}).reset_index()

city_stats.columns = ['batter', 'city', 'runs', 'balls', 'dismissals']

# Calculate SR and Average
city_stats['sr'] = (city_stats['runs'] / city_stats['balls'] * 100).round(2)
city_stats['avg'] = np.where(
    city_stats['dismissals'] > 0,
    (city_stats['runs'] / city_stats['dismissals']).round(2),
    np.nan
)

# Count innings (unique matches per batter-city)
innings_count = details_df.groupby(['striker', 'metro_area'])['match_id'].nunique().reset_index()
innings_count.columns = ['batter', 'city', 'innings']

city_stats = city_stats.merge(innings_count, on=['batter', 'city'], how='left')

print(f"  ✓ Calculated stats for {len(city_stats):,} batter-city combinations")

# ============================================================================
# PART 5: CALCULATE DEVIATIONS
# ============================================================================

print("\n[5/6] Calculating Deviations and Merging with Master...")

# Load batting master
batting_master = pd.read_csv(BATTING_MASTER_FILE)
print(f"  ✓ Loaded {len(batting_master)} players from BATTING_MASTER")

def calculate_confidence_weight(balls, min_threshold=30, full_confidence=120):
    """Sigmoid curve for confidence weight."""
    if balls < min_threshold:
        return 0.0
    if balls >= full_confidence:
        return min(0.95, 0.85 + (balls - full_confidence) / 1000)
    
    x = (balls - min_threshold) / (full_confidence - min_threshold)
    sigmoid = 1 / (1 + np.exp(-10 * (x - 0.5)))
    return 0.3 + (sigmoid * 0.55)

def calculate_city_deviation_batting(player_sr_city, player_overall_sr, city_baseline_sr):
    """Calculate city deviation for batting."""
    if pd.isna(player_overall_sr) or player_overall_sr == 0:
        return 0.0
    if pd.isna(city_baseline_sr) or city_baseline_sr == 0:
        return 0.0
    
    personal_delta = player_sr_city - player_overall_sr
    city_difficulty_factor = city_baseline_sr / 145.0  # 145 = IPL avg SR
    
    if city_difficulty_factor == 0:
        return 0.0
    
    normalized_delta = personal_delta / (city_baseline_sr * city_difficulty_factor)
    return np.clip(normalized_delta, -1.0, 1.0)

# Calculate city baselines
city_baselines = city_stats.groupby('city')['sr'].mean().to_dict()

print(f"\n  City Baseline Strike Rates:")
for city, sr in sorted(city_baselines.items(), key=lambda x: x[1], reverse=True):
    print(f"    {city:20s}: {sr:>6.2f}")

# Validation: Track unmatched players
unique_batters_in_data = set(city_stats['batter'].unique())
unique_batters_in_master = set(batting_master['Kaggle_Match_Name'].unique())

print(f"\n  Validation:")
print(f"    - Unique batters in rajsengo data: {len(unique_batters_in_data)}")
print(f"    - Unique batters in master: {len(unique_batters_in_master)}")

# Track unmatched for reporting
unmatched_batters = {}
matched_count = 0

# Calculate deviations
results = []

for _, row in city_stats.iterrows():
    batter = row['batter']
    city = row['city']
    city_sr = row['sr']
    city_avg = row['avg']
    balls = row['balls']
    innings = row['innings']
    
    # Match with master
    master_row = batting_master[batting_master['Kaggle_Match_Name'] == batter]
    
    if len(master_row) == 0:
        # Track unmatched
        if batter not in unmatched_batters:
            unmatched_batters[batter] = {
                'total_balls': 0,
                'cities': set()
            }
        unmatched_batters[batter]['total_balls'] += balls
        unmatched_batters[batter]['cities'].add(city)
        continue
    
    matched_count += 1
    
    master_row = master_row.iloc[0]
    overall_sr = master_row['Strike_Rate']
    overall_avg = master_row['Batting_Average']
    overall_boundary_pct = master_row.get('Boundary_Percentage', 0.0)
    
    if pd.isna(overall_sr) or overall_sr == 0:
        continue
    
    city_baseline_sr = city_baselines.get(city, 145.0)
    deviation = calculate_city_deviation_batting(city_sr, overall_sr, city_baseline_sr)
    confidence = calculate_confidence_weight(balls)
    
    # CRITICAL FIX: Force deviation to 0 if insufficient sample size
    if balls < 30:
        deviation = 0.0
    
    results.append({
        'Player_Name': master_row['Player_Name'],
        'Kaggle_Match_Name': batter,
        'City': city,
        'Sample_Innings': int(innings) if not pd.isna(innings) else 0,
        'Sample_Balls': int(balls),
        'Confidence_Weight': round(confidence, 3),
        'Overall_SR': round(overall_sr, 2),
        'City_SR': round(city_sr, 2),
        'City_Baseline_SR': round(city_baseline_sr, 2),
        'City_Deviation': round(deviation, 3),
        'Overall_Avg': round(overall_avg, 2) if not pd.isna(overall_avg) else 0.0,
        'City_Avg': round(city_avg, 2) if not pd.isna(city_avg) else 0.0,
        'Overall_Boundary_Pct': round(overall_boundary_pct, 2) if not pd.isna(overall_boundary_pct) else 0.0,
    })

output_df = pd.DataFrame(results)
print(f"\n  ✓ Calculated deviations for {len(output_df):,} player-city combinations")
print(f"  ✓ Successfully matched: {matched_count:,} player-city pairs")

# Report unmatched players
if unmatched_batters:
    print(f"\n  ⚠️  WARNING: {len(unmatched_batters)} batters in rajsengo NOT FOUND in master:")
    
    # Sort by total balls faced
    sorted_unmatched = sorted(
        unmatched_batters.items(),
        key=lambda x: x[1]['total_balls'],
        reverse=True
    )
    
    print(f"\n  Top unmatched batters (by balls faced):")
    for batter, info in sorted_unmatched[:20]:
        cities_count = len(info['cities'])
        print(f"    {batter:30s} - {info['total_balls']:4d} balls, {cities_count:2d} cities")
    
    if len(sorted_unmatched) > 20:
        print(f"    ... and {len(sorted_unmatched) - 20} more")
    
    # Save unmatched to CSV
    unmatched_df = pd.DataFrame([
        {'Batter_Name': name, 'Total_Balls': info['total_balls'], 'Num_Cities': len(info['cities'])}
        for name, info in sorted_unmatched
    ])
    unmatched_file = OUTPUT_DIR / "UNMATCHED_BATTERS_2022_2024.csv"
    unmatched_df.to_csv(unmatched_file, index=False)
    print(f"\n  ✓ Saved unmatched list: {unmatched_file.name}")
else:
    print(f"\n  ✓ All batters matched successfully!")

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
print(f"    - Total batters in BATTING_MASTER: {len(batting_master)}")
print(f"    - Players with city data (2022-24): {output_df['Player_Name'].nunique()}")
print(f"    - Coverage: {output_df['Player_Name'].nunique() / len(batting_master) * 100:.1f}%")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

unique_players = output_df['Player_Name'].nunique()
unique_cities = output_df['City'].nunique()

print(f"\n✓ Unique players: {unique_players} (out of {len(batting_master)} in BATTING_MASTER)")
print(f"✓ Unique cities: {unique_cities}")
print(f"✓ Average cities per player: {len(output_df) / unique_players:.1f}")

# Coverage stats
players_5plus = (output_df.groupby('Player_Name')['City'].count() >= 5).sum()
players_10plus = (output_df.groupby('Player_Name')['City'].count() >= 10).sum()

print(f"\n✓ Players with 5+ cities: {players_5plus}")
print(f"✓ Players with 10+ cities: {players_10plus}")

# Confidence distribution
high_conf = (output_df['Sample_Balls'] >= 120).sum()
med_conf = ((output_df['Sample_Balls'] >= 60) & (output_df['Sample_Balls'] < 120)).sum()
low_conf = (output_df['Sample_Balls'] < 60).sum()

print(f"\n✓ High confidence (≥120 balls): {high_conf} ({high_conf/len(output_df)*100:.1f}%)")
print(f"✓ Medium confidence (60-119 balls): {med_conf} ({med_conf/len(output_df)*100:.1f}%)")
print(f"✓ Low confidence (<60 balls): {low_conf} ({low_conf/len(output_df)*100:.1f}%)")

# Top coverage
print("\nTop 10 Players by City Coverage:")
top_coverage = output_df.groupby('Player_Name')['City'].count().sort_values(ascending=False).head(10)
for player, count in top_coverage.items():
    print(f"  {player:30s}: {count:2d} cities")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION - FAMOUS PLAYER-CITY COMBINATIONS")
print("=" * 80)

famous_combos = [
    ('Rohit Sharma', 'Mumbai'),
    ('Virat Kohli', 'Bengaluru'),
    ('MS Dhoni', 'Chennai'),
    ('Shubman Gill', 'Ahmedabad'),
]

for player, city in famous_combos:
    subset = output_df[
        (output_df['Player_Name'].str.contains(player, case=False, na=False)) & 
        (output_df['City'] == city)
    ]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f"\n{player} at {city}:")
        print(f"  City SR: {row['City_SR']:.2f} vs Overall: {row['Overall_SR']:.2f} (Baseline: {row['City_Baseline_SR']:.2f})")
        print(f"  Deviation: {row['City_Deviation']:+.3f} | Confidence: {row['Confidence_Weight']:.2f}")
        print(f"  Sample: {row['Sample_Innings']} innings, {row['Sample_Balls']} balls")

print("\n" + "-" * 80)
print("TOP 10 POSITIVE DEVIATIONS (Min 60 balls)")
print("-" * 80)
top_pos = output_df[output_df['Sample_Balls'] >= 60].nlargest(10, 'City_Deviation')
print(top_pos[['Player_Name', 'City', 'City_SR', 'Overall_SR', 'City_Deviation', 'Sample_Balls']].to_string(index=False))

print("\n" + "-" * 80)
print("TOP 10 NEGATIVE DEVIATIONS (Min 60 balls)")
print("-" * 80)
top_neg = output_df[output_df['Sample_Balls'] >= 60].nsmallest(10, 'City_Deviation')
print(top_neg[['Player_Name', 'City', 'City_SR', 'Overall_SR', 'City_Deviation', 'Sample_Balls']].to_string(index=False))

print("\n" + "=" * 80)
print("✅ COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutputs: {OUTPUT_DIR}")
print(f"  1. {VENUE_MAPPING_FILE.name}")
print(f"  2. {OUTPUT_FILE.name}")
print("=" * 80)