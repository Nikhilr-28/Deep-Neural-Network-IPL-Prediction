"""
STEP 1 TEST VERSION: Quick verification (First 10 players only)
=================================================================

This is a TEST version that processes only the first 10 players
to verify your setup before running the full script.

Use this to:
- Check if file paths are correct
- Verify data can be loaded
- Test matching algorithm
- Ensure output directory is writable

If this works, run the full version: step1_player_matching.py
"""

import pandas as pd
from pathlib import Path

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

OUTPUT_DIR = ROOT_DIR / "Step1_Test_Output"

# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_test():
    print("=" * 80)
    print("STEP 1 TEST VERSION - VERIFICATION RUN")
    print("=" * 80)
    
    # Test 1: Check root directory
    print(f"\n[TEST 1] Checking root directory...")
    if ROOT_DIR.exists():
        print(f"  ✓ Root directory exists: {ROOT_DIR}")
    else:
        print(f"  ✗ ERROR: Root directory not found: {ROOT_DIR}")
        return False
    
    # Test 2: Check player list
    print(f"\n[TEST 2] Loading player list...")
    if PLAYER_LIST_2025.exists():
        df_players = pd.read_csv(PLAYER_LIST_2025)
        print(f"  ✓ Player list loaded: {len(df_players)} players")
        print(f"  ✓ First 10 players:")
        for i, name in enumerate(df_players['PLAYER_NAME'].head(10), 1):
            print(f"     {i:2d}. {name}")
    else:
        print(f"  ✗ ERROR: Player list not found: {PLAYER_LIST_2025}")
        return False
    
    # Test 3: Check batting files
    print(f"\n[TEST 3] Checking batting files...")
    batting_ok = True
    for year, filepath in BATTING_FILES.items():
        if filepath.exists():
            df = pd.read_csv(filepath, nrows=5)
            print(f"  ✓ {year}: {filepath.name} ({len(df.columns)} columns)")
        else:
            print(f"  ✗ {year}: NOT FOUND - {filepath}")
            batting_ok = False
    
    # Test 4: Check bowling files
    print(f"\n[TEST 4] Checking bowling files...")
    bowling_ok = True
    for year, filepath in BOWLING_FILES.items():
        if filepath.exists():
            df = pd.read_csv(filepath, nrows=5)
            print(f"  ✓ {year}: {filepath.name} ({len(df.columns)} columns)")
        else:
            print(f"  ✗ {year}: NOT FOUND - {filepath}")
            bowling_ok = False
    
    # Test 5: Create output directory
    print(f"\n[TEST 5] Creating output directory...")
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        test_file = OUTPUT_DIR / "test.txt"
        test_file.write_text("Test write successful")
        test_file.unlink()  # Delete test file
        print(f"  ✓ Output directory ready: {OUTPUT_DIR}")
    except Exception as e:
        print(f"  ✗ ERROR: Cannot create output directory: {e}")
        return False
    
    # Test 6: Quick matching test
    print(f"\n[TEST 6] Testing player matching (first 3 players)...")
    try:
        # Load first batting file
        df_batting = pd.read_csv(BATTING_FILES[2022])
        historical_names = df_batting['name'].unique()
        
        # Try to match first 3 players
        for player_name in df_players['PLAYER_NAME'].head(3):
            if player_name in historical_names:
                print(f"  ✓ {player_name}: EXACT MATCH found")
            else:
                print(f"  ? {player_name}: No exact match (fuzzy matching needed)")
    except Exception as e:
        print(f"  ✗ ERROR during matching test: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if batting_ok and bowling_ok:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou are ready to run the full script:")
        print("  → python step1_player_matching.py")
        return True
    else:
        print("\n⚠️ SOME TESTS FAILED!")
        print("\nPlease fix the errors above before running the full script.")
        return False

if __name__ == "__main__":
    success = run_test()
    
    if not success:
        print("\n" + "=" * 80)
        print("TROUBLESHOOTING TIPS:")
        print("=" * 80)
        print("\n1. Check that all file paths are correct")
        print("2. Verify Kaggle data is properly extracted")
        print("3. Ensure you have write permissions")
        print("4. Run PowerShell as Administrator if needed")
