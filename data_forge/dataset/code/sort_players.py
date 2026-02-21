"""
═══════════════════════════════════════════════════════════════════════════════
IPL 2025 PLAYERS LIST SORTER
═══════════════════════════════════════════════════════════════════════════════

PURPOSE: Sort players alphabetically by:
    1. IPL_TEAM (ascending alphabetical)
    2. PLAYER_TYPE (ascending alphabetical)
    3. PLAYER_NAME (ascending alphabetical)

INPUT:  IPL_2025_Players_list.csv
OUTPUT: IPL_2025_Players_list.csv (overwritten, sorted)

Author: El Dorado Project - CSCI 566
Date: November 24, 2024
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil")
DATA_DIR = ROOT_DIR / "Dataset(s) and code" / "dataset"
PLAYERS_FILE = DATA_DIR / "IPL_2025_Players_list.csv"

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Sort IPL 2025 players list"""
    
    print("\n" + "=" * 80)
    print("  IPL 2025 PLAYERS LIST SORTER")
    print("=" * 80)
    
    # Check file exists
    if not PLAYERS_FILE.exists():
        print(f"\n❌ ERROR: File not found!")
        print(f"   Expected: {PLAYERS_FILE}")
        return
    
    print(f"\n📂 Loading: {PLAYERS_FILE.name}")
    
    # Load data
    df = pd.read_csv(PLAYERS_FILE)
    original_count = len(df)
    
    print(f"✅ Loaded: {original_count} players")
    print(f"\n📊 Original order sample:")
    print(df.head(10).to_string(index=False))
    
    # Create backup
    backup_file = PLAYERS_FILE.parent / f"IPL_2025_Players_list_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False)
    print(f"\n💾 Backup created: {backup_file.name}")
    
    # Sort by IPL_TEAM, PLAYER_TYPE, PLAYER_NAME (all alphabetically ascending)
    print(f"\n🔄 Sorting by:")
    print(f"   1. IPL_TEAM (alphabetical)")
    print(f"   2. PLAYER_TYPE (alphabetical)")
    print(f"   3. PLAYER_NAME (alphabetical)")
    
    df_sorted = df.sort_values(
        by=['IPL_TEAM', 'PLAYER_TYPE', 'PLAYER_NAME'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # Verify no data lost
    sorted_count = len(df_sorted)
    
    if sorted_count != original_count:
        print(f"\n❌ ERROR: Row count mismatch!")
        print(f"   Original: {original_count}")
        print(f"   Sorted: {sorted_count}")
        return
    
    print(f"✅ Sorting complete: {sorted_count} players")
    
    # Show sample of sorted data
    print(f"\n📊 Sorted order sample:")
    print(df_sorted.head(10).to_string(index=False))
    
    # Save sorted file (overwrite original)
    df_sorted.to_csv(PLAYERS_FILE, index=False)
    print(f"\n💾 Saved: {PLAYERS_FILE.name}")
    
    # Summary by team
    print(f"\n📋 Players by Team:")
    team_counts = df_sorted['IPL_TEAM'].value_counts().sort_index()
    for team, count in team_counts.items():
        print(f"   {team}: {count} players")
    
    # Summary by type
    print(f"\n📋 Players by Type:")
    type_counts = df_sorted['PLAYER_TYPE'].value_counts().sort_index()
    for ptype, count in type_counts.items():
        print(f"   {ptype}: {count} players")
    
    print("\n" + "=" * 80)
    print("  ✅ SORTING COMPLETE")
    print("=" * 80)
    print(f"\n🎯 File ready for nomenclature matching!")
    print(f"   Location: {PLAYERS_FILE}")
    print(f"   Backup: {backup_file.name}")
    print()

if __name__ == "__main__":
    main()