#!/usr/bin/env python3
"""
IPL 2025 Monte Carlo Simulation Evaluation
==========================================
Pure simulation-based prediction using OVR + GAT PvP
No neural network required - fastest evaluation

Author: El Dorado Team - CSCI 566
Date: December 1, 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#                           LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Setup logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/mnt/user-data/outputs/simulation_eval_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("IPL 2025 MONTE CARLO SIMULATION EVALUATION")
    logger.info("="*80)
    return logger

logger = setup_logging()

# ═══════════════════════════════════════════════════════════════════════════
#                           FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════

PATHS = {
    'test_matches': Path(r'A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\2025\test_matches_10.json'),
    'batting_ovr': Path(r'A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BATTING_20251129_154628.csv'),
    'bowling_ovr': Path(r'A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR\PRE_OVR_BOWLING_20251129_154628.csv'),
    'gat_pvp': Path(r'A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\DL_MODELS\outputs\gat_enhanced_pvp.json')
}

# ═══════════════════════════════════════════════════════════════════════════
#                      MATCH SIMULATOR (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════

class MatchSimulator:
    """Ball-by-ball Monte Carlo match simulator"""
    
    def __init__(self, batting_ovr: pd.DataFrame, bowling_ovr: pd.DataFrame, gat_pvp: List[Dict]):
        self.batting_ovr = batting_ovr
        self.bowling_ovr = bowling_ovr
        self.gat_pvp = gat_pvp
        self.missing_ovr_players = []
        
        logger.info(f"Simulator initialized: {len(batting_ovr)} batters, {len(bowling_ovr)} bowlers, {len(gat_pvp)} PvP edges")
    
    def simulate_match(self, team_a_xi: List[str], team_b_xi: List[str], 
                      num_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulations"""
        
        logger.info(f"Running {num_simulations} simulations...")
        
        team_a_wins = 0
        team_a_scores = []
        team_b_scores = []
        
        progress_points = [int(num_simulations * p) for p in [0.25, 0.5, 0.75, 1.0]]
        
        for i in range(num_simulations):
            # Progress tracking
            if i in progress_points:
                pct = int((i / num_simulations) * 100)
                logger.info(f"  Progress: {pct}% ({i}/{num_simulations})")
            
            # Team A bats first
            team_a_score = self._simulate_innings(team_a_xi, team_b_xi)
            team_a_scores.append(team_a_score)
            
            # Team B chases
            team_b_score = self._simulate_innings(team_b_xi, team_a_xi)
            team_b_scores.append(team_b_score)
            
            if team_b_score < team_a_score:
                team_a_wins += 1
        
        team_a_win_prob = team_a_wins / num_simulations
        
        results = {
            'win_probs': {
                'team_a': team_a_win_prob,
                'team_b': 1 - team_a_win_prob
            },
            'avg_scores': {
                'team_a': np.mean(team_a_scores),
                'team_b': np.mean(team_b_scores)
            },
            'std_scores': {
                'team_a': np.std(team_a_scores),
                'team_b': np.std(team_b_scores)
            }
        }
        
        logger.info(f"Simulation complete: Team A win prob = {team_a_win_prob:.3f}")
        return results
    
    def _simulate_innings(self, batting_xi: List[str], bowling_xi: List[str]) -> int:
        """Simulate one innings"""
        
        total_runs = 0
        wickets = 0
        balls_faced = 0
        max_balls = 120  # 20 overs
        
        striker_idx = 0
        non_striker_idx = 1
        bowler_idx = 0
        
        while balls_faced < max_balls and wickets < 10:
            # Get phase
            phase = self._get_phase(balls_faced)
            
            # Get current batter and bowler
            batter = batting_xi[striker_idx] if striker_idx < len(batting_xi) else batting_xi[0]
            bowler = bowling_xi[bowler_idx % min(5, len(bowling_xi))]  # Rotate 5 bowlers
            
            # Simulate ball
            runs, is_wicket = self._simulate_ball(batter, bowler, phase)
            
            total_runs += runs
            balls_faced += 1
            
            if is_wicket:
                wickets += 1
                striker_idx += 2  # Next batter
                if striker_idx >= 11:
                    striker_idx = 10
            else:
                # Rotate strike on odd runs or end of over
                if runs % 2 == 1:
                    striker_idx, non_striker_idx = non_striker_idx, striker_idx
            
            # Change bowler at end of over
            if balls_faced % 6 == 0:
                bowler_idx += 1
                striker_idx, non_striker_idx = non_striker_idx, striker_idx
        
        return total_runs
    
    def _simulate_ball(self, batter: str, bowler: str, phase: str) -> Tuple[int, bool]:
        """Simulate one ball with GAT PvP and OVR"""
        
        # Get OVRs
        batter_ovr = self._get_batter_ovr(batter, phase)
        bowler_ovr = self._get_bowler_ovr(bowler, phase)
        
        # Get GAT advantage
        gat_advantage = self._get_gat_advantage(batter, bowler)
        
        # Calculate effective advantage
        ovr_diff = batter_ovr - bowler_ovr
        effective_advantage = ovr_diff + (gat_advantage * 5.0)
        
        # Wicket probability (5% base ± 0.2% per OVR point)
        wicket_prob = 0.05 - (effective_advantage * 0.002)
        wicket_prob = np.clip(wicket_prob, 0.01, 0.15)
        
        if np.random.random() < wicket_prob:
            return 0, True  # Wicket!
        
        # Boundary probability (15% base ± 0.3% per OVR point)
        boundary_prob = 0.15 + (effective_advantage * 0.003)
        boundary_prob = np.clip(boundary_prob, 0.05, 0.30)
        
        if np.random.random() < boundary_prob:
            # Six or four?
            if np.random.random() < 0.4:  # 40% six, 60% four
                return 6, False
            else:
                return 4, False
        
        # Dot ball probability (30% base)
        if np.random.random() < 0.30:
            return 0, False
        
        # Singles, twos, threes
        runs_dist = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3]  # 60% single, 30% two, 10% three
        return np.random.choice(runs_dist), False
    
    def _get_phase(self, ball_number: int) -> str:
        """Get match phase based on ball number"""
        over = (ball_number // 6) + 1
        if over <= 6:
            return 'powerplay'
        elif over <= 15:
            return 'middle'
        else:
            return 'death'
    
    def _get_batter_ovr(self, name: str, phase: str) -> float:
        """Get phase-specific batting OVR"""
        
        player_row = self.batting_ovr[self.batting_ovr['Kaggle_Match_Name'] == name]
        
        if player_row.empty:
            self.missing_ovr_players.append(('batter', name))
            return 65.0
        
        phase_map = {
            'powerplay': 'TOP_ORDER_OVR',
            'middle': 'MIDDLE_ORDER_OVR',
            'death': 'FINISHER_OVR'
        }
        
        ovr = player_row[phase_map[phase]].values[0]
        return ovr if not pd.isna(ovr) else 65.0
    
    def _get_bowler_ovr(self, name: str, phase: str) -> float:
        """Get phase-specific bowling OVR"""
        
        player_row = self.bowling_ovr[self.bowling_ovr['Kaggle_Match_Name'] == name]
        
        if player_row.empty:
            self.missing_ovr_players.append(('bowler', name))
            return 70.0
        
        phase_map = {
            'powerplay': 'POWERPLAY_OVR',
            'middle': 'MIDDLE_OVERS_OVR',
            'death': 'DEATH_OVERS_OVR'
        }
        
        ovr = player_row[phase_map[phase]].values[0]
        return ovr if not pd.isna(ovr) else 70.0
    
    def _get_gat_advantage(self, batter: str, bowler: str) -> float:
        """Get GAT PvP advantage (-1 to +1 scale)"""
        
        # Search for matchup in GAT PvP data
        for edge in self.gat_pvp:
            if ((edge['batter'] == batter and edge['bowler'] == bowler) or
                (edge['bowler'] == batter and edge['batter'] == bowler)):
                
                # Convert 0-1 scale to -1 to +1 scale
                gat_score = edge['gat_advantage']
                return (gat_score - 0.5) * 2.0
        
        return 0.0  # Neutral if no historical matchup

# ═══════════════════════════════════════════════════════════════════════════
#                          MAIN EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution"""
    
    try:
        # Load data
        logger.info("\nLoading data...")
        
        with open(PATHS['test_matches'], 'r') as f:
            test_matches = json.load(f)
        logger.info(f"[OK] Loaded {len(test_matches)} test matches")
        
        batting_ovr = pd.read_csv(PATHS['batting_ovr'])
        logger.info(f"[OK] Loaded {len(batting_ovr)} batting OVRs")
        
        bowling_ovr = pd.read_csv(PATHS['bowling_ovr'])
        logger.info(f"[OK] Loaded {len(bowling_ovr)} bowling OVRs")
        
        with open(PATHS['gat_pvp'], 'r') as f:
            gat_data = json.load(f)
            # Extract pvp_edges from the dictionary structure
            if isinstance(gat_data, dict) and 'pvp_edges' in gat_data:
                gat_pvp = gat_data['pvp_edges']
            else:
                gat_pvp = gat_data
        logger.info(f"[OK] Loaded {len(gat_pvp)} GAT PvP edges")
        
        # Initialize simulator
        simulator = MatchSimulator(batting_ovr, bowling_ovr, gat_pvp)
        
        # Evaluate all matches
        logger.info("\n" + "="*80)
        logger.info(f"STARTING EVALUATION OF {len(test_matches)} MATCHES")
        logger.info("="*80)
        
        results = []
        correct = 0
        brier_scores = []
        
        for match_id, match_data in test_matches.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"EVALUATING {match_id.upper()}")
            logger.info(f"{match_data['team_a']} vs {match_data['team_b']}")
            logger.info(f"Venue: {match_data['venue']}, Date: {match_data['date']}")
            logger.info(f"Actual Winner: {match_data['winner']}")
            logger.info(f"{'='*80}")
            
            # Simulate
            sim_result = simulator.simulate_match(
                match_data['team_a_xi'],
                match_data['team_b_xi'],
                num_simulations=10000
            )
            
            # Determine winner
            team_a = match_data['team_a']
            team_b = match_data['team_b']
            actual_winner = match_data['winner']
            
            predicted_winner = team_a if sim_result['win_probs']['team_a'] > 0.5 else team_b
            is_correct = (predicted_winner == actual_winner)
            
            if is_correct:
                correct += 1
            
            # Calculate Brier score
            actual = 1.0 if actual_winner == team_a else 0.0
            predicted = sim_result['win_probs']['team_a']
            brier = (predicted - actual) ** 2
            brier_scores.append(brier)
            
            # Log results
            logger.info(f"\nPredicted Winner: {predicted_winner}")
            logger.info(f"  {team_a}: {sim_result['win_probs']['team_a']:.1%}")
            logger.info(f"  {team_b}: {sim_result['win_probs']['team_b']:.1%}")
            logger.info(f"  Expected Scores: {team_a} {sim_result['avg_scores']['team_a']:.0f}±{sim_result['std_scores']['team_a']:.0f}, "
                       f"{team_b} {sim_result['avg_scores']['team_b']:.0f}±{sim_result['std_scores']['team_b']:.0f}")
            logger.info(f"  Result: {'[OK] CORRECT' if is_correct else '[X] INCORRECT'}")
            logger.info(f"  Brier Score: {brier:.4f}")
            
            results.append({
                'match_id': match_id,
                'team_a': team_a,
                'team_b': team_b,
                'actual_winner': actual_winner,
                'predicted_winner': predicted_winner,
                'correct': is_correct,
                'prob_a': sim_result['win_probs']['team_a'],
                'prob_b': sim_result['win_probs']['team_b'],
                'avg_score_a': sim_result['avg_scores']['team_a'],
                'avg_score_b': sim_result['avg_scores']['team_b'],
                'brier': brier
            })
        
        # Final report
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        
        accuracy = correct / len(test_matches) * 100
        avg_brier = np.mean(brier_scores)
        
        logger.info(f"\nTotal Matches:     {len(test_matches)}")
        logger.info(f"Correct:           {correct}/{len(test_matches)} ({accuracy:.1f}%)")
        logger.info(f"Average Brier:     {avg_brier:.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        csv_path = "/mnt/user-data/outputs/simulation_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n[OK] Results saved to: {csv_path}")
        
        # Missing players report
        unique_missing = list(set(simulator.missing_ovr_players))
        if unique_missing:
            logger.info(f"\n[!] WARNING: {len(unique_missing)} missing OVR players:")
            for role, name in unique_missing[:20]:  # Show first 20
                logger.info(f"  - {role}: {name}")
        
        logger.info("\n" + "="*80)
        logger.info("[OK][OK][OK] EVALUATION COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n[X][X][X] CRITICAL ERROR: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()