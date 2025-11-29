"""
================================================================================
PRE-OVR CALCULATION ENGINE
================================================================================
Generates position-based OVR scores from MASTER datasets.

Input:  BATTING_MASTER_2025.csv, BOWLING_MASTER_2025.csv
Output: PRE_OVR_BATTING_2025.csv, PRE_OVR_BOWLING_2025.csv

OVR Range: 55-97
- 55 = Baseline (zero data or SR ≤114)
- 97 = Elite maximum

Author: El Dorado Project - CSCI 566
Date: 2025-11-29
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths - UPDATE THESE TO MATCH YOUR ACTUAL LOCATIONS
BATTING_MASTER_PATH = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\Master_Datasets\BATTING_MASTER_2025_*.csv"
BOWLING_MASTER_PATH = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\Master_Datasets\BOWLING_MASTER_2025_*.csv"

OUTPUT_DIR = r"A:\DL\Dataset_Nikhl\CSCI_566\Nikhil\Dataset(s) and code\dataset\OVR"  # Save outputs here

# ============================================================================
# STRIKE RATE NORMALIZATION
# ============================================================================

def normalize_strike_rate(sr):
    """
    Normalize Strike Rate to OVR score (55-97 range)
    
    Benchmarks:
    - 0.0 or ≤114 → 55
    - 115 → 59
    - 130 → 66
    - 145 → 75
    - 160 → 85
    - 180 → 95
    - 200+ → 97 (cap)
    
    Linear interpolation between benchmarks
    """
    
    # Handle zero/missing/low SR
    if pd.isna(sr) or sr == 0.0 or sr <= 114:
        return 55.0
    
    # Benchmarks (SR: OVR)
    benchmarks = [
        (115, 59),
        (130, 66),
        (145, 75),
        (160, 85),
        (180, 95),
        (200, 97)
    ]
    
    # Find bracket and interpolate
    for i in range(len(benchmarks) - 1):
        sr_low, ovr_low = benchmarks[i]
        sr_high, ovr_high = benchmarks[i + 1]
        
        if sr_low <= sr < sr_high:
            # Linear interpolation
            ratio = (sr - sr_low) / (sr_high - sr_low)
            ovr = ovr_low + (ovr_high - ovr_low) * ratio
            return round(ovr, 1)
    
    # Beyond 200 SR
    return 97.0


def normalize_metric_to_100(value, max_threshold):
    """
    Normalize any metric to 0-100 scale
    
    Args:
        value: Actual value
        max_threshold: Value that equals 100 score
    """
    if pd.isna(value) or value == 0:
        return 0.0
    
    score = min(100, (value / max_threshold) * 100)
    return round(score, 1)


def get_effective_sr(phase_sr, phase_balls, overall_sr, min_sample=60):
    """
    Use phase SR if sufficient data, else blend with overall SR
    
    Args:
        phase_sr: Phase-specific SR (powerplay/death)
        phase_balls: Balls faced in that phase
        overall_sr: Overall career SR
        min_sample: Minimum balls to trust phase SR (default 60)
    
    Returns:
        Effective SR to use for OVR calculation
    """
    
    if pd.isna(phase_sr) or phase_sr == 0:
        return overall_sr
    
    if pd.isna(phase_balls) or phase_balls == 0:
        return overall_sr
    
    if phase_balls >= min_sample:
        # Sufficient data - use phase SR
        return phase_sr
    else:
        # Insufficient data - weighted blend
        weight = phase_balls / min_sample  # 0 to 1
        blended_sr = (phase_sr * weight) + (overall_sr * (1 - weight))
        return blended_sr


def get_experience_bonus(total_innings):
    """
    Reward proven players with large sample sizes
    
    Returns bonus OVR points based on innings played
    """
    
    if pd.isna(total_innings) or total_innings == 0:
        return 0.0
    
    if total_innings >= 50:
        return 5.0  # Increased from 4
    elif total_innings >= 30:
        return 4.0  # Increased from 2
    elif total_innings >= 20:
        return 3.0  # New tier
    elif total_innings >= 15:
        return 2.0  # Increased from 1
    else:
        return 0.0


def get_debut_potential_bonus(player):
    """
    Reward debut players with impressive limited data
    
    Criteria:
    - 100+ balls faced AND SR > 150 → High potential
    - Compensates for missing conversion/average data
    
    Returns: 0-3 bonus OVR points
    """
    
    total_balls = player.get('Total_Balls_Faced', 0)
    sr = player.get('Strike_Rate', 0)
    total_innings = player.get('Total_Innings', 0)
    debut = player.get('DEBUT', 'NO')
    
    # Only for debut players
    if debut != 'YES':
        return 0.0
    
    # Check if impressive limited sample
    if total_balls >= 100:
        if sr >= 170:
            return 3.0  # Elite SR
        elif sr >= 150:
            return 2.0  # Excellent SR
        elif sr >= 140:
            return 1.0  # Very good SR
    
    return 0.0


def get_explosive_batsman_bonus(player):
    """
    Reward explosive T20 batsmen who provide match-winning firepower
    
    Criteria:
    - SR ≥ 150 AND Boundary % ≥ 60 → True power hitter
    - SR ≥ 160 AND Boundary % ≥ 65 → Elite explosiveness
    
    Returns: 0-5 bonus OVR points
    """
    
    sr = player.get('Strike_Rate', 0)
    boundary_pct = player.get('Boundary_Percentage', 0)
    sixes_per_innings = player.get('Sixes_Per_Innings', 0)
    
    bonus = 0.0
    
    # Elite explosive (SR 160+, Boundary 65+)
    if sr >= 160 and boundary_pct >= 65:
        bonus = 5.0
    # High explosive (SR 155+, Boundary 62+)
    elif sr >= 155 and boundary_pct >= 62:
        bonus = 4.0
    # Explosive (SR 150+, Boundary 60+)
    elif sr >= 150 and boundary_pct >= 60:
        bonus = 3.0
    # Power hitter (SR 145+, Boundary 58+)
    elif sr >= 145 and boundary_pct >= 58:
        bonus = 2.0
    
    # Additional bonus for six-hitting prowess
    if sixes_per_innings >= 2.5 and bonus > 0:
        bonus += 1.0
    
    return min(5.0, bonus)  # Cap at +5


def get_low_sample_bonus(player):
    """
    Reward players with promising limited data (100+ balls, <15 innings)
    Compensates for missing conversion/volume data
    
    Returns: 0-4 bonus OVR points
    """
    
    innings = player.get('Total_Innings', 0)
    balls = player.get('Total_Balls_Faced', 0)
    sr = player.get('Strike_Rate', 0)
    boundary_pct = player.get('Boundary_Percentage', 0)
    
    # Only apply if low sample but meaningful data
    if innings >= 15 or balls < 100:
        return 0.0
    
    bonus = 0.0
    
    # Elite limited sample (SR 140+, Boundary 70+)
    if sr >= 140 and boundary_pct >= 70:
        bonus = 4.0
    # Very good limited sample (SR 135+, Boundary 65+)
    elif sr >= 135 and boundary_pct >= 65:
        bonus = 3.5
    # Good limited sample (SR 130+, Boundary 60+)
    elif sr >= 130 and boundary_pct >= 60:
        bonus = 3.0
    # Decent limited sample (SR 120+, Boundary 55+)
    elif sr >= 120 and boundary_pct >= 55:
        bonus = 2.0
    
    return bonus


def normalize_bowling_average(avg):
    """
    Normalize bowling average to 0-100 score (IPL-tuned)
    
    Thresholds (dynamic interpolation):
    - ≤20 → 97 (Elite)
    - ≤24 → 91
    - ≤28 → 86 (Very Good - Noor/Khaleel tier)
    - ≤32 → 77
    - ≤36 → 68
    - ≤40 → 60
    - ≥44 → 55 (floor)
    
    Linear interpolation between all points
    """
    
    if pd.isna(avg) or avg == 0:
        return 0.0
    
    # Benchmarks (avg: score)
    benchmarks = [
        (20, 97),
        (24, 91),
        (28, 86),
        (32, 77),
        (36, 68),
        (40, 60),
        (44, 55)
    ]
    
    # Beyond 44 = floor
    if avg >= 44:
        return 55.0
    
    # Find bracket and interpolate
    for i in range(len(benchmarks) - 1):
        avg_low, score_high = benchmarks[i]
        avg_high, score_low = benchmarks[i + 1]
        
        if avg_low <= avg < avg_high:
            # Linear interpolation
            ratio = (avg - avg_low) / (avg_high - avg_low)
            score = score_high - (ratio * (score_high - score_low))
            return round(score, 1)
    
    # Elite (<20)
    return 97.0


def normalize_bowling_strike_rate(sr):
    """
    Normalize bowling strike rate to 0-100 score (IPL-tuned)
    
    Thresholds (dynamic interpolation):
    - ≤15 → 97 (Elite)
    - ≤18 → 91
    - ≤21 → 86 (Very Good - Noor/Khaleel tier)
    - ≤24 → 77
    - ≤27 → 68
    - ≤30 → 60
    - ≥33 → 55 (floor)
    
    Linear interpolation between all points
    """
    
    if pd.isna(sr) or sr == 0:
        return 0.0
    
    # Benchmarks (sr: score)
    benchmarks = [
        (15, 97),
        (18, 91),
        (21, 86),
        (24, 77),
        (27, 68),
        (30, 60),
        (33, 55)
    ]
    
    # Beyond 33 = floor
    if sr >= 33:
        return 55.0
    
    # Find bracket and interpolate
    for i in range(len(benchmarks) - 1):
        sr_low, score_high = benchmarks[i]
        sr_high, score_low = benchmarks[i + 1]
        
        if sr_low <= sr < sr_high:
            # Linear interpolation
            ratio = (sr - sr_low) / (sr_high - sr_low)
            score = score_high - (ratio * (score_high - score_low))
            return round(score, 1)
    
    # Elite (<15)
    return 97.0


def get_bowling_consistency_score(player):
    """
    Calculate consistency score for bowlers
    
    Components:
    - Wicketless % (lower = better)
    - Economy CV (lower = better)
    - Wickets Standard Deviation (lower = better)
    
    Returns: 0-100 score
    """
    
    wicketless_pct = player.get('Percentage_Wicketless', 50)
    economy_cv = player.get('Economy_CV', 50)
    wickets_sd = player.get('Wickets_Std_Deviation', 1.5)
    
    # Invert scores (lower is better)
    wicketless_score = max(0, 100 - wicketless_pct)
    cv_score = max(0, 100 - economy_cv)
    sd_score = max(0, 100 - (wickets_sd * 20))  # SD typically 0-5
    
    # Weighted combination
    consistency = (
        wicketless_score * 0.5 +
        cv_score * 0.3 +
        sd_score * 0.2
    )
    
    return round(min(100, max(0, consistency)), 1)


def get_multi_wicket_bonus(player):
    """
    Calculate multi-wicket haul bonus
    
    Rewards match-winning performances
    
    Returns: 0-100 score
    """
    
    pct_3plus = player.get('Percentage_3_Plus_Wickets', 0)
    
    # Adjusted scaling: 10% 3+ wickets = 100 score
    # (1 in 10 innings is excellent in T20)
    multi_wicket_score = min(100, pct_3plus * 10)
    
    return round(multi_wicket_score, 1)


def get_boundary_control_score(player):
    """
    Calculate boundary control for death overs
    
    Components:
    - Boundary Concession Rate (lower = better)
    - Non-Boundary Economy (lower = better)
    
    Returns: 0-100 score
    """
    
    boundary_rate = player.get('Boundary_Concession_Rate', 25)
    non_boundary_econ = player.get('Non_Boundary_Economy', 6.0)
    
    # Boundary rate: 10% = 100, 25% = 0
    boundary_score = max(0, 100 - (boundary_rate * 4))
    
    # Non-boundary economy: 3.0 = 100, 6.0 = 0
    nb_econ_score = normalize_metric_to_100(non_boundary_econ, 3.0)
    nb_econ_score = max(0, 100 - nb_econ_score)  # Invert (lower is better)
    
    # Weighted
    boundary_control = (
        boundary_score * 0.6 +
        nb_econ_score * 0.4
    )
    
    return round(min(100, max(0, boundary_control)), 1)


def get_bowling_debut_bonus(player):
    """
    Reward debut bowlers with elite limited data
    
    Criteria:
    - 100+ balls bowled AND economy < 8.5 → Potential
    - Compensates for missing phase/consistency data
    
    Returns: 0-3 bonus OVR points
    """
    
    total_balls = player.get('Total_Balls_Bowled', 0)
    econ = player.get('Economy_Rate', 0)
    debut = player.get('DEBUT', 'NO')
    
    # Only for debut players
    if debut != 'YES':
        return 0.0
    
    # Check if impressive limited sample
    if total_balls >= 100 and econ > 0:
        if econ <= 7.5:
            return 3.0  # Elite economy
        elif econ <= 8.5:
            return 2.0  # Very good economy
        elif econ <= 9.0:
            return 1.0  # Good economy
    
    return 0.0


def get_match_winner_bonus(player):
    """
    Identify match-winners based on clutch traits
    
    Awards bonus OVR for:
    - High conversion rates
    - Finishing ability (not outs)
    - Consistency
    - Volume of big scores
    
    Returns: 0-5 bonus OVR points
    """
    
    bonus = 0.0
    
    # Big-innings conversion
    conv_30_50 = player.get('Conversion_30_to_50', 0)
    conv_50_70 = player.get('Conversion_50_to_70', 0)
    
    if conv_30_50 > 60 and conv_50_70 > 40:
        bonus += 3.0
    elif conv_30_50 > 50 and conv_50_70 > 30:
        bonus += 1.5
    
    # Finishing ability (not-out %)
    not_outs = player.get('Times_Not_Out', 0)
    innings = player.get('Total_Innings', 1)
    not_out_pct = (not_outs / max(innings, 1)) * 100
    
    if not_out_pct > 40:
        bonus += 2.0
    elif not_out_pct > 25:
        bonus += 1.0
    
    # Consistency (low coefficient of variation)
    cv = player.get('Runs_CV', 999)
    if cv < 0.7:
        bonus += 2.0
    elif cv < 1.0:
        bonus += 1.0
    
    # Volume of big scores
    count_50_plus = player.get('Count_50_Plus', 0)
    if count_50_plus >= 10:
        bonus += 2.0
    elif count_50_plus >= 5:
        bonus += 1.0
    
    return min(5.0, bonus)  # Cap at +5


def get_volume_consistency_score(player):
    """
    Reward players for proven track record
    
    Components:
    - Total runs (volume/experience)
    - Consistency (low CV = reliable)
    - Quality (batting average)
    
    Returns: 0-100 score
    """
    
    total_runs = player.get('Total_Runs', 0)
    cv = player.get('Runs_CV', 999)
    avg = player.get('Batting_Average', 0)
    
    # Volume score (2000 runs = 100, 1000 = 50, etc.)
    volume_score = normalize_metric_to_100(total_runs, 2000)
    
    # Consistency score (lower CV = better, inverted scale)
    if cv < 0.7:
        consistency_score = 100
    elif cv < 1.0:
        consistency_score = 70
    elif cv < 1.3:
        consistency_score = 40
    else:
        consistency_score = 10
    
    # Quality score (avg 40 = 100)
    quality_score = normalize_metric_to_100(avg, 40)
    
    # Weighted combination
    vc_score = (
        volume_score * 0.4 +
        consistency_score * 0.3 +
        quality_score * 0.3
    )
    
    return round(vc_score, 1)


# ============================================================================
# BATTING OVR CALCULATIONS
# ============================================================================

def calculate_base_batting_ovr(player):
    """
    Calculate BASE OVR - Universal batting score
    
    Components (normal):
    - Strike Rate (30%)
    - Batting Average (20%)
    - Boundary Percentage (15%)
    - Conversion Ability (15%)
    - Rotation Strike Rate (10%)
    - Volume+Consistency (10%)
    
    Smart redistribution if low sample (<15 innings):
    - Strike Rate (40%) +10%
    - Batting Average (25%) +5%
    - Boundary Percentage (25%) +10%
    - Rotation Strike Rate (10%)
    (Skip conversion and volume - not enough data)
    
    Bonuses:
    - Experience Bonus (up to +5)
    - Match Winner Bonus (up to +5)
    - Debut Potential Bonus (up to +3)
    - Explosive Batsman Bonus (up to +5)
    - Low Sample Bonus (up to +4)
    """
    
    # Extract metrics
    sr = player.get('Strike_Rate', 0)
    avg = player.get('Batting_Average', 0)
    boundary_pct = player.get('Boundary_Percentage', 0)
    rotation_sr = player.get('Rotation_Strike_Rate', 0)
    conv_30_50 = player.get('Conversion_30_to_50', 0)
    conv_50_70 = player.get('Conversion_50_to_70', 0)
    total_innings = player.get('Total_Innings', 0)
    debut = player.get('DEBUT', 'NO')
    
    # Normalize components
    sr_score = normalize_strike_rate(sr)
    boundary_score = normalize_metric_to_100(boundary_pct, 67)
    rotation_score = normalize_metric_to_100(rotation_sr, 100)
    
    # For debuts: Use lower threshold for average (less penalty)
    if debut == 'YES':
        avg_score = normalize_metric_to_100(avg, 35)
    else:
        avg_score = normalize_metric_to_100(avg, 50)
    
    # SMART REDISTRIBUTION for low sample size
    if total_innings < 15:
        # Missing conversion/volume data - redistribute to core metrics
        base_ovr = (
            sr_score * 0.40 +         # +10%
            avg_score * 0.25 +        # +5%
            boundary_score * 0.25 +   # +10%
            rotation_score * 0.10
        )
    else:
        # Normal weights with all metrics
        conv_score = (conv_30_50 + conv_50_70) / 2 if (conv_30_50 > 0 or conv_50_70 > 0) else 0
        conv_score = min(100, conv_score)
        vc_score = get_volume_consistency_score(player)
        
        base_ovr = (
            sr_score * 0.30 +
            avg_score * 0.20 +
            boundary_score * 0.15 +
            conv_score * 0.15 +
            rotation_score * 0.10 +
            vc_score * 0.10
        )
    
    # Add bonuses
    exp_bonus = get_experience_bonus(total_innings)
    match_winner_bonus = get_match_winner_bonus(player) if total_innings >= 15 else 0
    debut_bonus = get_debut_potential_bonus(player)
    explosive_bonus = get_explosive_batsman_bonus(player)
    low_sample_bonus = get_low_sample_bonus(player)
    
    base_ovr += exp_bonus + match_winner_bonus + debut_bonus + explosive_bonus + low_sample_bonus
    
    return round(min(97, max(55, base_ovr)), 1)


def calculate_top_order_ovr(player):
    """
    Calculate TOP ORDER OVR (Positions 1-3)
    
    Components:
    - Powerplay SR (25%)
    - Overall SR (10%)
    - [SR AGGREGATE: 35%]
    - Boundary Percentage (25%)
    - Conversion 50→70 (20%)
    - Volume+Consistency (10%)
    - Batting Average (10%)
    - Experience Bonus (up to +4)
    
    BONUS: +3 OVR if Powerplay SR ≈ Death SR (±15 range) WITH sufficient data
    """
    
    pp_sr = player.get('Powerplay_SR', 0)
    pp_balls = player.get('Powerplay_Balls', 0)
    death_sr = player.get('Death_Overs_SR', 0)
    death_balls = player.get('Death_Overs_Balls', 0)
    boundary_pct = player.get('Boundary_Percentage', 0)
    conv_50_70 = player.get('Conversion_50_to_70', 0)
    sr = player.get('Strike_Rate', 0)
    avg = player.get('Batting_Average', 0)
    total_innings = player.get('Total_Innings', 0)
    
    # SMART SR SELECTION
    effective_pp_sr = get_effective_sr(pp_sr, pp_balls, sr, min_sample=60)
    
    # Normalize
    pp_sr_score = normalize_strike_rate(effective_pp_sr)
    sr_score = normalize_strike_rate(sr)
    boundary_score = normalize_metric_to_100(boundary_pct, 67)
    conv_score = min(100, conv_50_70)
    avg_score = normalize_metric_to_100(avg, 50)
    vc_score = get_volume_consistency_score(player)
    
    # Weighted (SR aggregate = 35%)
    top_ovr = (
        pp_sr_score * 0.25 +
        sr_score * 0.10 +
        boundary_score * 0.25 +
        conv_score * 0.20 +
        vc_score * 0.10 +
        avg_score * 0.10
    )
    
    # Experience bonus
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_debut_potential_bonus(player)
    explosive_bonus = get_explosive_batsman_bonus(player)
    low_sample_bonus = get_low_sample_bonus(player)
    top_ovr += exp_bonus + debut_bonus + explosive_bonus + low_sample_bonus
    
    # OPENER BONUS: Consistent aggression throughout innings
    if pp_balls >= 60 and death_balls >= 60:
        if pp_sr > 0 and death_sr > 0:
            sr_difference = abs(pp_sr - death_sr)
            if sr_difference <= 15:
                top_ovr += 3.0
    
    return round(min(97, max(55, top_ovr)), 1)


def calculate_middle_order_ovr(player):
    """
    Calculate MIDDLE ORDER OVR (Positions 4-5)
    
    Components:
    - Overall SR (30%)
    - [SR AGGREGATE: 30%]
    - Rotation SR (25%)
    - Conversion 30→50 (20%)
    - Batting Average (15%)
    - Volume+Consistency (10%)
    - Experience Bonus (up to +4)
    
    VERSATILITY BONUS: +4 OVR if all position OVRs within ±5 range
                       (indicates true all-position player)
    """
    
    rotation_sr = player.get('Rotation_Strike_Rate', 0)
    conv_30_50 = player.get('Conversion_30_to_50', 0)
    sr = player.get('Strike_Rate', 0)
    avg = player.get('Batting_Average', 0)
    total_innings = player.get('Total_Innings', 0)
    
    # Normalize
    sr_score = normalize_strike_rate(sr)
    rotation_score = normalize_metric_to_100(rotation_sr, 100)
    conv_score = min(100, conv_30_50)
    avg_score = normalize_metric_to_100(avg, 40)
    vc_score = get_volume_consistency_score(player)
    
    # Weighted (SR aggregate = 30%)
    middle_ovr = (
        sr_score * 0.30 +
        rotation_score * 0.25 +
        conv_score * 0.20 +
        avg_score * 0.15 +
        vc_score * 0.10
    )
    
    # Experience bonus
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_debut_potential_bonus(player)
    explosive_bonus = get_explosive_batsman_bonus(player)
    low_sample_bonus = get_low_sample_bonus(player)
    middle_ovr += exp_bonus + debut_bonus + explosive_bonus + low_sample_bonus
    
    return round(min(97, max(55, middle_ovr)), 1)


def calculate_finisher_ovr(player):
    """
    Calculate FINISHER OVR (Positions 6-8)
    
    Components:
    - Death Overs SR (25%)
    - Overall SR (10%)
    - [SR AGGREGATE: 35%]
    - Boundary Percentage (25%)
    - Not Out Ratio (10%)
    - Sixes Per Innings (15%)
    - Volume+Consistency (10%)
    - Batting Average (5%)
    - Experience Bonus (up to +4)
    
    PENALTY: -3 OVR if rotation SR > 100 (strong middle order trait)
    """
    
    death_sr = player.get('Death_Overs_SR', 0)
    death_balls = player.get('Death_Overs_Balls', 0)
    overall_sr = player.get('Strike_Rate', 0)
    not_outs = player.get('Times_Not_Out', 0)
    boundary_pct = player.get('Boundary_Percentage', 0)
    sixes_per_innings = player.get('Sixes_Per_Innings', 0)
    rotation_sr = player.get('Rotation_Strike_Rate', 0)
    avg = player.get('Batting_Average', 0)
    total_innings = player.get('Total_Innings', 1)
    
    # SMART SR SELECTION
    effective_death_sr = get_effective_sr(death_sr, death_balls, overall_sr, min_sample=60)
    
    # Normalize
    death_sr_score = normalize_strike_rate(effective_death_sr)
    overall_sr_score = normalize_strike_rate(overall_sr)
    
    # Not outs: Use ratio instead of absolute count
    not_out_ratio = (not_outs / max(total_innings, 1)) * 100
    not_out_score = min(100, not_out_ratio)
    
    boundary_score = normalize_metric_to_100(boundary_pct, 67)
    sixes_score = normalize_metric_to_100(sixes_per_innings, 2.5)
    avg_score = normalize_metric_to_100(avg, 50)
    vc_score = get_volume_consistency_score(player)
    
    # Weighted (SR aggregate = 35%)
    finisher_ovr = (
        death_sr_score * 0.25 +
        overall_sr_score * 0.10 +
        boundary_score * 0.25 +
        not_out_score * 0.10 +
        sixes_score * 0.15 +
        vc_score * 0.10 +
        avg_score * 0.05
    )
    
    # Experience bonus
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_debut_potential_bonus(player)
    explosive_bonus = get_explosive_batsman_bonus(player)
    low_sample_bonus = get_low_sample_bonus(player)
    finisher_ovr += exp_bonus + debut_bonus + explosive_bonus + low_sample_bonus
    
    # PENALTY: Very high rotation SR = strong middle order trait
    if rotation_sr > 100:
        finisher_ovr -= 3.0
    
    return round(min(97, max(55, finisher_ovr)), 1)


# ============================================================================
# BOWLING OVR CALCULATIONS
# ============================================================================

def normalize_economy_rate(econ):
    """
    Normalize Economy Rate to OVR score (IPL-tuned for T20 batting paradise)
    Lower economy = higher score
    
    Benchmarks:
    - ≤7.0 → 97 (Elite: Bumrah tier)
    - ≤7.5 → 93
    - ≤8.0 → 89
    - ≤8.5 → 85
    - ≤9.0 → 81
    - ≤9.3 → 77 (Penalty threshold)
    - ≤10.0 → 70
    - ≤11.0 → 62
    - ≥12.0 → 55 (Floor)
    
    Linear interpolation between benchmarks
    """
    
    if pd.isna(econ) or econ == 0.0 or econ >= 12.0:
        return 55.0
    
    # Benchmarks (Economy: OVR) - inverted scale
    benchmarks = [
        (7.0, 97),
        (7.5, 93),
        (8.0, 89),
        (8.5, 85),
        (9.0, 81),
        (9.3, 77),
        (10.0, 70),
        (11.0, 62),
        (12.0, 55)
    ]
    
    # Elite (<7.0) - allow phase OVRs to exceed 97
    if econ < 7.0:
        bonus = (7.0 - econ) * 0.5
        return min(97.0, 97.0 + bonus)
    
    # Find bracket and interpolate (reverse order since lower is better)
    for i in range(len(benchmarks) - 1):
        econ_low, ovr_high = benchmarks[i]
        econ_high, ovr_low = benchmarks[i + 1]
        
        if econ_low <= econ < econ_high:
            # Linear interpolation
            ratio = (econ - econ_low) / (econ_high - econ_low)
            ovr = ovr_high - (ratio * (ovr_high - ovr_low))
            return round(ovr, 1)
    
    # Beyond 12.0 economy
    return 55.0


def calculate_base_bowling_ovr(player):
    """
    Calculate BASE BOWLING OVR - Universal bowling score
    
    Components (normal):
    - Economy Rate (25%)
    - Bowling Average (20%)
    - Bowling Strike Rate (15%)
    - Dot Ball Percentage (18%)
    - Wicket Taking Ability (10%)
    - Consistency Score (10%)
    - Multi-Wicket Bonus (2%)
    - Experience Bonus (up to +4)
    - Debut Bonus (up to +3)
    
    Smart redistribution if consistency data missing (<10 innings):
    - Economy Rate (30%)
    - Bowling Average (25%)
    - Bowling Strike Rate (20%)
    - Dot Ball Percentage (18%)
    - Wicket Taking Ability (5%)
    - Multi-Wicket Bonus (2%)
    """
    
    econ = player.get('Economy_Rate', 0)
    avg = player.get('Bowling_Average', 0)
    strike_rate = player.get('Bowling_Strike_Rate', 0)
    dot_pct = player.get('Dot_Ball_Percentage', 0)
    wicket_ability = player.get('Wicket_Taking_Ability', 0)
    total_innings = player.get('Total_Innings_Bowled', 0)
    
    # Normalize
    econ_score = normalize_economy_rate(econ)
    avg_score = normalize_bowling_average(avg)
    sr_score = normalize_bowling_strike_rate(strike_rate)
    
    # IPL-tuned: 40% dots = 100 (not 50%)
    dot_score = normalize_metric_to_100(dot_pct, 40)
    wicket_score = normalize_metric_to_100(wicket_ability, 100)
    
    # Advanced metrics
    consistency_score = get_bowling_consistency_score(player)
    multi_wicket_score = get_multi_wicket_bonus(player)
    
    # SMART REDISTRIBUTION for low sample size
    if total_innings < 10:
        # Missing consistency data - redistribute to core metrics
        base_ovr = (
            econ_score * 0.30 +      # +5%
            avg_score * 0.25 +       # +5%
            sr_score * 0.20 +        # +5%
            dot_score * 0.18 +
            wicket_score * 0.05 +    # -5%
            multi_wicket_score * 0.02
        )
    else:
        # Normal weights with consistency
        base_ovr = (
            econ_score * 0.25 +
            avg_score * 0.20 +
            sr_score * 0.15 +
            dot_score * 0.18 +
            wicket_score * 0.10 +
            consistency_score * 0.10 +
            multi_wicket_score * 0.02
        )
    
    # Bonuses
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_bowling_debut_bonus(player)
    
    # IPL T20 BUFF: Bowling is harder in batting paradise (+7 OVR)
    ipl_buff = 7.0
    
    base_ovr += exp_bonus + debut_bonus + ipl_buff
    
    return round(min(97, max(55, base_ovr)), 1)


def calculate_powerplay_bowling_ovr(player):
    """
    Calculate POWERPLAY BOWLING OVR
    
    Components (normal with PP data):
    - Powerplay Economy (35%)
    - Powerplay Wickets (30%)
    - Dot Ball Percentage (20%)
    - Control Index (15%)
    
    Smart redistribution if PP data missing:
    - Overall Economy (50%)
    - Dot Ball Percentage (30%)
    - Control Index (20%)
    
    Experience Bonus (up to +4)
    Debut Bonus (up to +3)
    """
    
    pp_econ = player.get('Powerplay_Economy', 0)
    pp_wickets = player.get('Powerplay_Wickets', 0)
    pp_overs = player.get('Powerplay_Overs', 0)
    dot_pct = player.get('Dot_Ball_Percentage', 0)
    control = player.get('Control_Index', 0)
    total_innings = player.get('Total_Innings_Bowled', 0)
    
    # Check if PP data exists
    has_pp_data = pp_overs > 0 and (pp_econ > 0 or pp_wickets > 0)
    
    # Normalize
    dot_score = normalize_metric_to_100(dot_pct, 40)
    control_score = normalize_metric_to_100(control, 100)
    
    if has_pp_data:
        # Normal weights with PP-specific data
        pp_econ_score = normalize_economy_rate(pp_econ)
        pp_wickets_score = normalize_metric_to_100(pp_wickets, 15)
        
        pp_ovr = (
            pp_econ_score * 0.35 +
            pp_wickets_score * 0.30 +
            dot_score * 0.20 +
            control_score * 0.15
        )
    else:
        # Missing PP data - use overall economy
        overall_econ = player.get('Economy_Rate', 0)
        econ_score = normalize_economy_rate(overall_econ)
        
        pp_ovr = (
            econ_score * 0.50 +      # Redistributed
            dot_score * 0.30 +       # Increased
            control_score * 0.20     # Increased
        )
    
    # Bonuses
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_bowling_debut_bonus(player)
    
    # IPL T20 BUFF: Bowling is harder in batting paradise (+7 OVR)
    ipl_buff = 7.0
    
    pp_ovr += exp_bonus + debut_bonus + ipl_buff
    
    return round(min(97, max(55, pp_ovr)), 1)


def calculate_middle_overs_bowling_ovr(player):
    """
    Calculate MIDDLE OVERS BOWLING OVR
    
    Components:
    - Economy Rate (35%)
    - Dot Ball Percentage (25%) - IPL-tuned
    - Bowling Strike Rate (25%)
    - Control Index (15%)
    - Experience Bonus (up to +4)
    - Debut Bonus (up to +3)
    """
    
    econ = player.get('Economy_Rate', 0)
    dot_pct = player.get('Dot_Ball_Percentage', 0)
    strike_rate = player.get('Bowling_Strike_Rate', 0)
    control = player.get('Control_Index', 0)
    total_innings = player.get('Total_Innings_Bowled', 0)
    
    # Normalize
    econ_score = normalize_economy_rate(econ)
    
    # IPL-tuned: 40% dots = 100 (not 50%)
    dot_score = normalize_metric_to_100(dot_pct, 40)
    sr_score = normalize_bowling_strike_rate(strike_rate)
    control_score = normalize_metric_to_100(control, 100)
    
    # Weighted
    middle_ovr = (
        econ_score * 0.35 +
        dot_score * 0.25 +
        sr_score * 0.25 +
        control_score * 0.15
    )
    
    # Bonuses
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_bowling_debut_bonus(player)
    
    # IPL T20 BUFF: Bowling is harder in batting paradise (+7 OVR)
    ipl_buff = 7.0
    
    middle_ovr += exp_bonus + debut_bonus + ipl_buff
    
    return round(min(97, max(55, middle_ovr)), 1)


def calculate_death_bowling_ovr(player):
    """
    Calculate DEATH OVERS BOWLING OVR
    
    Components (normal with death data):
    - Death Overs Economy (30%)
    - Death Wickets (25%)
    - Boundary Control (22%)
    - Discipline Score (13%)
    - Pressure Index (10%)
    
    Smart redistribution if death data missing:
    - Overall Economy (40%)
    - Boundary Control (30%)
    - Discipline Score (20%)
    - Pressure Index (10%)
    
    Experience Bonus (up to +4)
    Debut Bonus (up to +3)
    """
    
    death_econ = player.get('Death_Overs_Economy', 0)
    death_wickets = player.get('Death_Wickets', 0)
    death_overs = player.get('Death_Overs', 0)
    discipline = player.get('Discipline_Score', 0)
    pressure = player.get('Pressure_Index', 0)
    total_innings = player.get('Total_Innings_Bowled', 0)
    
    # Check if death data exists
    has_death_data = death_overs > 0 and (death_econ > 0 or death_wickets > 0)
    
    # Normalize
    discipline_score = normalize_metric_to_100(discipline, 100)
    pressure_score = normalize_metric_to_100(pressure, 100)
    boundary_control_score = get_boundary_control_score(player)
    
    if has_death_data:
        # Normal weights with death-specific data
        death_econ_score = normalize_economy_rate(death_econ)
        death_wickets_score = normalize_metric_to_100(death_wickets, 12)
        
        death_ovr = (
            death_econ_score * 0.30 +
            death_wickets_score * 0.25 +
            boundary_control_score * 0.22 +
            discipline_score * 0.13 +
            pressure_score * 0.10
        )
    else:
        # Missing death data - use overall economy
        overall_econ = player.get('Economy_Rate', 0)
        econ_score = normalize_economy_rate(overall_econ)
        
        death_ovr = (
            econ_score * 0.40 +             # Redistributed
            boundary_control_score * 0.30 + # Increased
            discipline_score * 0.20 +       # Increased
            pressure_score * 0.10
        )
    
    # Bonuses
    exp_bonus = get_experience_bonus(total_innings)
    debut_bonus = get_bowling_debut_bonus(player)
    
    # IPL T20 BUFF: Bowling is harder in batting paradise (+7 OVR)
    ipl_buff = 7.0
    
    death_ovr += exp_bonus + debut_bonus + ipl_buff
    
    return round(min(97, max(55, death_ovr)), 1)


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_batting_master():
    """
    Process batting master CSV and generate PRE-OVR dataset
    """
    
    print("\n" + "="*80)
    print("PROCESSING BATTING MASTER DATASET")
    print("="*80)
    
    # Find latest batting master file
    import glob
    batting_files = glob.glob(BATTING_MASTER_PATH)
    if not batting_files:
        print(f"❌ ERROR: No batting master file found matching '{BATTING_MASTER_PATH}'")
        return None
    
    batting_file = sorted(batting_files)[-1]  # Get latest
    print(f"📂 Loading: {batting_file}")
    
    # Load data
    df = pd.read_csv(batting_file)
    print(f"✅ Loaded {len(df)} players")
    
    # Calculate OVRs
    print("\n⚙️  Calculating OVR scores...")
    
    pre_ovr_data = []
    for idx, (_, player) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            print(f"   Processing player {idx+1}/{len(df)}...")
        
        # Calculate all position OVRs first
        base_ovr = calculate_base_batting_ovr(player)
        top_ovr = calculate_top_order_ovr(player)
        middle_ovr = calculate_middle_order_ovr(player)
        finisher_ovr = calculate_finisher_ovr(player)
        
        # DEBUT NERF: Reduce all OVRs by 7% for debut batters
        debut = player.get('DEBUT', 'NO')
        if debut == 'YES':
            debut_nerf = 0.93  # 7% reduction
            base_ovr *= debut_nerf
            top_ovr *= debut_nerf
            middle_ovr *= debut_nerf
            finisher_ovr *= debut_nerf
        
        # VERSATILITY BONUS: Check if all OVRs within ±5 of each other
        all_ovrs = [top_ovr, middle_ovr, finisher_ovr]
        max_ovr = max(all_ovrs)
        min_ovr = min(all_ovrs)
        
        # Check if every OVR is within ±5 of every other OVR
        is_versatile = True
        for i in range(len(all_ovrs)):
            for j in range(i + 1, len(all_ovrs)):
                if abs(all_ovrs[i] - all_ovrs[j]) > 5:
                    is_versatile = False
                    break
            if not is_versatile:
                break
        
        if is_versatile:
            # Versatile player - bonus to middle order
            middle_ovr += 4.0
            middle_ovr = min(97, middle_ovr)
        
        # WK-BATTER BONUS: +3.5 to highest OVR (flexibility)
        player_type = player.get('Player_Type', '')
        if player_type == 'WK-Batter':
            # Find highest position OVR
            position_ovrs = {'top': top_ovr, 'middle': middle_ovr, 'finisher': finisher_ovr}
            highest_position = max(position_ovrs, key=lambda k: position_ovrs[k])
            
            if highest_position == 'top':
                top_ovr += 3.5
                top_ovr = min(97, top_ovr)
            elif highest_position == 'middle':
                middle_ovr += 3.5
                middle_ovr = min(97, middle_ovr)
            elif highest_position == 'finisher':
                finisher_ovr += 3.5
                finisher_ovr = min(97, finisher_ovr)
        
        # CAPTAIN BONUS: +2 to base OVR and best position OVR
        is_captain = player.get('captain', False)
        if is_captain:
            base_ovr += 2.0
            base_ovr = min(97, base_ovr)
            
            # Add to best position OVR
            position_ovrs = {'top': top_ovr, 'middle': middle_ovr, 'finisher': finisher_ovr}
            highest_position = max(position_ovrs, key=lambda k: position_ovrs[k])
            
            if highest_position == 'top':
                top_ovr += 2.0
                top_ovr = min(97, top_ovr)
            elif highest_position == 'middle':
                middle_ovr += 2.0
                middle_ovr = min(97, middle_ovr)
            elif highest_position == 'finisher':
                finisher_ovr += 2.0
                finisher_ovr = min(97, finisher_ovr)
        
        ovr_record = {
            'Player_Name': player['Player_Name'],
            'Kaggle_Match_Name': player.get('Kaggle_Match_Name', ''),
            'Player_Type': player['Player_Type'],
            'IPL_Team_2025': player['IPL_Team_2025'],
            'DEBUT': player.get('DEBUT', 'NO'),
            'BASE_OVR': round(base_ovr, 1),
            'TOP_ORDER_OVR': round(top_ovr, 1),
            'MIDDLE_ORDER_OVR': round(middle_ovr, 1),
            'FINISHER_OVR': round(finisher_ovr, 1)
        }
        
        pre_ovr_data.append(ovr_record)
    
    # Create DataFrame
    pre_ovr_df = pd.DataFrame(pre_ovr_data)
    
    # SORT: Team → Player Type → Name (merge debuts with all players)
    pre_ovr_df = pre_ovr_df.sort_values(
        by=['IPL_Team_2025', 'Player_Type', 'Player_Name'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"PRE_OVR_BATTING_{timestamp}.csv")
    pre_ovr_df.to_csv(output_file, index=False)
    
    print(f"\n✅ BATTING PRE-OVR COMPLETE!")
    print(f"📁 Saved: {output_file}")
    print(f"📊 Total Players: {len(pre_ovr_df)}")
    
    # Summary stats
    print("\n" + "="*80)
    print("BATTING OVR SUMMARY STATISTICS")
    print("="*80)
    for col in ['BASE_OVR', 'TOP_ORDER_OVR', 'MIDDLE_ORDER_OVR', 'FINISHER_OVR']:
        print(f"\n{col}:")
        print(f"  Mean:   {pre_ovr_df[col].mean():.1f}")
        print(f"  Median: {pre_ovr_df[col].median():.1f}")
        print(f"  Min:    {pre_ovr_df[col].min():.1f}")
        print(f"  Max:    {pre_ovr_df[col].max():.1f}")
    
    return pre_ovr_df


def process_bowling_master():
    """
    Process bowling master CSV and generate PRE-OVR dataset
    """
    
    print("\n" + "="*80)
    print("PROCESSING BOWLING MASTER DATASET")
    print("="*80)
    
    # Find latest bowling master file
    import glob
    bowling_files = glob.glob(BOWLING_MASTER_PATH)
    if not bowling_files:
        print(f"❌ ERROR: No bowling master file found matching '{BOWLING_MASTER_PATH}'")
        return None
    
    bowling_file = sorted(bowling_files)[-1]  # Get latest
    print(f"📂 Loading: {bowling_file}")
    
    # Load data
    df = pd.read_csv(bowling_file)
    print(f"✅ Loaded {len(df)} players")
    
    # Calculate OVRs
    print("\n⚙️  Calculating OVR scores...")
    
    pre_ovr_data = []
    for idx, (_, player) in enumerate(df.iterrows()):
        if idx % 30 == 0:
            print(f"   Processing player {idx+1}/{len(df)}...")
        
        # Calculate all OVRs first
        base_ovr_original = calculate_base_bowling_ovr(player)
        pp_ovr = calculate_powerplay_bowling_ovr(player)
        middle_ovr = calculate_middle_overs_bowling_ovr(player)
        death_ovr = calculate_death_bowling_ovr(player)
        
        # CORRECTION FACTOR: Base OVR = Average of top 2 phase OVRs
        # This prevents over-inflation while rewarding versatility
        phase_ovrs = [pp_ovr, middle_ovr, death_ovr]
        phase_ovrs_sorted = sorted(phase_ovrs, reverse=True)
        corrected_base_ovr = round((phase_ovrs_sorted[0] + phase_ovrs_sorted[1]) / 2, 1)
        
        ovr_record = {
            'Player_Name': player['Player_Name'],
            'Kaggle_Match_Name': player.get('Kaggle_Match_Name', ''),
            'Player_Type': player['Player_Type'],
            'IPL_Team_2025': player['IPL_Team_2025'],
            'DEBUT': player.get('DEBUT', 'NO'),
            'BASE_OVR': corrected_base_ovr,  # Use corrected value
            'POWERPLAY_OVR': pp_ovr,
            'MIDDLE_OVERS_OVR': middle_ovr,
            'DEATH_OVERS_OVR': death_ovr
        }
        
        pre_ovr_data.append(ovr_record)
    
    # Create DataFrame
    pre_ovr_df = pd.DataFrame(pre_ovr_data)
    
    # SORT: Team → Player Type → Name (merge debuts with all players)
    pre_ovr_df = pre_ovr_df.sort_values(
        by=['IPL_Team_2025', 'Player_Type', 'Player_Name'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"PRE_OVR_BOWLING_{timestamp}.csv")
    pre_ovr_df.to_csv(output_file, index=False)
    
    print(f"\n✅ BOWLING PRE-OVR COMPLETE!")
    print(f"📁 Saved: {output_file}")
    print(f"📊 Total Players: {len(pre_ovr_df)}")
    
    # Summary stats
    print("\n" + "="*80)
    print("BOWLING OVR SUMMARY STATISTICS")
    print("="*80)
    for col in ['BASE_OVR', 'POWERPLAY_OVR', 'MIDDLE_OVERS_OVR', 'DEATH_OVERS_OVR']:
        print(f"\n{col}:")
        print(f"  Mean:   {pre_ovr_df[col].mean():.1f}")
        print(f"  Median: {pre_ovr_df[col].median():.1f}")
        print(f"  Min:    {pre_ovr_df[col].min():.1f}")
        print(f"  Max:    {pre_ovr_df[col].max():.1f}")
    
    return pre_ovr_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("PRE-OVR CALCULATION ENGINE")
    print("El Dorado Project - CSCI 566")
    print("="*80)
    
    # Process batting
    batting_pre_ovr = process_batting_master()
    
    # Process bowling
    bowling_pre_ovr = process_bowling_master()
    
    print("\n" + "="*80)
    print("✅ ALL PROCESSING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - PRE_OVR_BATTING_[timestamp].csv")
    print("  - PRE_OVR_BOWLING_[timestamp].csv")
    print("\nNext Steps:")
    print("  1. Review OVR distributions in summary stats")
    print("  2. Use these files for dynamic OVR calculations")
    print("  3. Integrate with Streamlit dashboard")
    print("="*80 + "\n")