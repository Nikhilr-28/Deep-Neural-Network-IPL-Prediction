"""
Construct Training Data for Player Attention Model
FINAL VERSION - current_innings is TEAM NAME, not innings number!
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

RAJSENGO_BASE = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Kaggle_download\rajsengo"
OUTPUT_DIR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs"

def extract_playing_xi(match_id: int, team_name: str, batting_card: pd.DataFrame, 
                      bowling_card: pd.DataFrame) -> list:
    """
    Extract 11 players for a specific team in a match
    
    KEY INSIGHT: current_innings contains TEAM NAME, not innings number!
    """
    # Get batters from this team (current_innings = TEAM NAME!)
    team_batters = batting_card[
        (batting_card['match_id'] == match_id) & 
        (batting_card['current_innings'] == team_name)
    ]['fullName'].unique().tolist()
    
    # Get bowlers from this team
    team_bowlers = bowling_card[
        (bowling_card['match_id'] == match_id) & 
        (bowling_card['bowling_team'] == team_name)
    ]['fullName'].unique().tolist()
    
    # Combine (union) - some players are both batters and bowlers
    playing_xi = list(set(team_batters + team_bowlers))
    
    if len(playing_xi) > 11:
        print(f"  ⚠️ {team_name} has {len(playing_xi)} players, taking first 11")
        playing_xi = playing_xi[:11]
    
    return playing_xi

def process_season(year: int) -> list:
    """Process one season"""
    print(f"\n{'='*60}")
    print(f"Processing {year} Season...")
    print(f"{'='*60}")
    
    season_path = Path(RAJSENGO_BASE) / str(year)
    
    # Load files
    if year == 2023:
        summary = pd.read_csv(season_path / 'summary.csv')
        batting = pd.read_csv(season_path / 'batting_card.csv')
        bowling = pd.read_csv(season_path / 'bowling_card.csv')
    else:
        summary = pd.read_csv(season_path / 'season_summary.csv')
        batting = pd.read_csv(season_path / 'season_batting_card.csv')
        bowling = pd.read_csv(season_path / 'season_bowling_card.csv')
    
    print(f"✅ Loaded {len(summary)} matches")
    
    matches = []
    skipped = 0
    
    for idx in range(len(summary)):
        row = summary.iloc[idx]
        
        # Extract match info
        match_id = int(row['id'])
        home_team = str(row['home_team'])
        away_team = str(row['away_team'])
        venue = str(row['venue_name'])
        winner = str(row['winner'])
        toss_winner = str(row['toss_won'])
        toss_decision = str(row['decision'])
        
        # Determine who batted first
        if toss_decision == 'BAT FIRST':
            bat_first = toss_winner
        elif toss_decision == 'BOWL FIRST':
            bat_first = away_team if toss_winner == home_team else home_team
        else:
            bat_first = home_team
        
        # Extract playing XIs (FIXED!)
        home_xi = extract_playing_xi(match_id, home_team, batting, bowling)
        away_xi = extract_playing_xi(match_id, away_team, batting, bowling)
        
        # Skip if insufficient players
        if len(home_xi) < 9 or len(away_xi) < 9:
            print(f"  ⚠️ Match {match_id}: {home_team} ({len(home_xi)}) vs {away_team} ({len(away_xi)}) - SKIPPED")
            skipped += 1
            continue
        
        matches.append({
            'match_id': str(match_id),
            'year': year,
            'team_a': home_team,
            'team_b': away_team,
            'team_a_xi': home_xi,
            'team_b_xi': away_xi,
            'venue': venue,
            'winner': winner,
            'toss_winner': toss_winner,
            'bat_first': bat_first
        })
        
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(summary)}... ({len(matches)} valid, {skipped} skipped)")
    
    print(f"✅ Extracted {len(matches)} valid matches from {year} (skipped {skipped})")
    return matches

def main():
    """Main execution"""
    print("="*80)
    print("CONSTRUCTING TRAINING DATA FOR PLAYER ATTENTION MODEL")
    print("="*80)
    
    all_matches = []
    
    # Process each season
    for year in [2022, 2023, 2024]:
        try:
            season_matches = process_season(year)
            all_matches.extend(season_matches)
        except Exception as e:
            print(f"❌ Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total matches: {len(all_matches)}")
    print(f"  2022: {sum(1 for m in all_matches if m['year'] == 2022)}")
    print(f"  2023: {sum(1 for m in all_matches if m['year'] == 2023)}")
    print(f"  2024: {sum(1 for m in all_matches if m['year'] == 2024)}")
    
    # Statistics
    xi_sizes = [len(m['team_a_xi']) + len(m['team_b_xi']) for m in all_matches]
    print(f"\nXI Statistics:")
    print(f"  Average players per match: {sum(xi_sizes)/len(xi_sizes):.1f}")
    print(f"  Min players: {min(xi_sizes)}")
    print(f"  Max players: {max(xi_sizes)}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV (lists as pipe-separated)
    df = pd.DataFrame(all_matches)
    df_export = df.copy()
    df_export['team_a_xi'] = df_export['team_a_xi'].apply(lambda x: '|'.join(x))
    df_export['team_b_xi'] = df_export['team_b_xi'].apply(lambda x: '|'.join(x))
    
    output_csv = Path(OUTPUT_DIR) / f"training_matches_with_xi_{timestamp}.csv"
    df_export.to_csv(output_csv, index=False)
    print(f"\n✅ CSV saved: {output_csv}")
    
    # JSON (preserves lists)
    output_json = Path(OUTPUT_DIR) / f"training_matches_with_xi_{timestamp}.json"
    with open(output_json, 'w') as f:
        json.dump(all_matches, f, indent=2)
    print(f"✅ JSON saved: {output_json}")
    
    # Sample
    if all_matches:
        print(f"\n{'='*80}")
        print("SAMPLE MATCH:")
        print(f"{'='*80}")
        sample = all_matches[0]
        print(f"Match ID: {sample['match_id']}")
        print(f"Year: {sample['year']}")
        print(f"Teams: {sample['team_a']} vs {sample['team_b']}")
        print(f"Venue: {sample['venue']}")
        print(f"Winner: {sample['winner']}")
        print(f"\n{sample['team_a']} XI ({len(sample['team_a_xi'])} players):")
        for i, p in enumerate(sample['team_a_xi'], 1):
            print(f"  {i}. {p}")
        print(f"\n{sample['team_b']} XI ({len(sample['team_b_xi'])} players):")
        for i, p in enumerate(sample['team_b_xi'], 1):
            print(f"  {i}. {p}")
    
    print(f"\n{'='*80}")
    print("✅ TRAINING DATA CONSTRUCTION COMPLETE!")
    print(f"{'='*80}")
    
    return output_csv, output_json

if __name__ == "__main__":
    main()