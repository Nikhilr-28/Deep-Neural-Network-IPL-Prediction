import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set global style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ============================================================================
# VISUAL 1: COMPACT MULTI-LEAGUE DATA INTEGRATION PIPELINE
# ============================================================================

def create_visual_1():
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.7, 'MULTI-LEAGUE DATA INTEGRATION PIPELINE', 
            ha='center', va='top', fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(5, 9.35, 'Transforming 13 cricket leagues into unified player ratings',
            ha='center', va='top', fontsize=9, color='#555555', style='italic')
    
    # Color palette
    color_process = '#FF6B35'  # Process Orange
    color_output = '#FFC107'   # Output Gold
    
    # ========== LAYER 1: INPUT SOURCES (COMPACT) ==========
    y_start = 8.5
    ax.text(5, y_start, 'INPUT SOURCES', 
            ha='center', fontsize=10, fontweight='bold', color='#2C3E50')
    
    # Single compact box
    input_box = FancyBboxPatch((1.5, y_start-0.6), 7, 0.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='#5C2E7E', facecolor='#EDE7F6',
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, y_start-0.4, '13 Leagues: IPL, T20I, BBL, CPL, SA20, ILT20, PSL, SMAT, VH, Ranji, Duleep, STATE, ODI',
            ha='center', fontsize=7, color='#333333')
    
    # ========== LAYER 2: QUALITY WEIGHTING ==========
    y_weight = 7.3
    
    # Arrow
    ax.annotate('', xy=(5, y_weight+0.35), xytext=(5, y_start-0.65),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_process))
    
    # Box
    weight_box = FancyBboxPatch((1.5, y_weight-0.35), 7, 0.6,
                                boxstyle="round,pad=0.08",
                                edgecolor=color_process, facecolor='#FFF3E0', 
                                linewidth=2)
    ax.add_patch(weight_box)
    
    ax.text(5, y_weight+0.15, 'QUALITY WEIGHTING', 
            ha='center', fontsize=9, fontweight='bold', color=color_process)
    ax.text(5, y_weight-0.15, 'IPL/T20I: 1.0  |  BBL/SA20: 0.9  |  CPL/ILT20: 0.8  |  Domestic: 0.6-0.7',
            ha='center', fontsize=7, color='#555555')
    
    # ========== LAYER 3: RECENCY DECAY ==========
    y_recency = 6.2
    
    # Arrow
    ax.annotate('', xy=(5, y_recency+0.35), xytext=(5, y_weight-0.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_process))
    
    # Box
    recency_box = FancyBboxPatch((1.5, y_recency-0.35), 7, 0.6,
                                 boxstyle="round,pad=0.08",
                                 edgecolor=color_process, facecolor='#E8F5E9',
                                 linewidth=2)
    ax.add_patch(recency_box)
    
    ax.text(5, y_recency+0.15, 'RECENCY DECAY', 
            ha='center', fontsize=9, fontweight='bold', color=color_process)
    ax.text(5, y_recency-0.15, '2024: 1.0  |  2023: 0.75  |  2022: 0.5',
            ha='center', fontsize=7, color='#555555')
    
    # ========== LAYER 4: PHASE DECOMPOSITION ==========
    y_phase = 5.1
    
    # Arrow
    ax.annotate('', xy=(5, y_phase+0.35), xytext=(5, y_recency-0.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_process))
    
    # Box
    phase_box = FancyBboxPatch((1.5, y_phase-0.35), 7, 0.6,
                               boxstyle="round,pad=0.08",
                               edgecolor=color_process, facecolor='#E3F2FD',
                               linewidth=2)
    ax.add_patch(phase_box)
    
    ax.text(5, y_phase+0.15, 'PHASE DECOMPOSITION', 
            ha='center', fontsize=9, fontweight='bold', color=color_process)
    ax.text(5, y_phase-0.15, 'Powerplay (1-6)  |  Middle (7-15)  |  Death (16-20)',
            ha='center', fontsize=7, color='#555555')
    
    # ========== LAYER 5: OUTPUT OVR SCORES ==========
    y_output = 3.7
    
    # Arrow
    ax.annotate('', xy=(5, y_output+0.55), xytext=(5, y_phase-0.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_process))
    
    # Output box (highlighted)
    output_box = FancyBboxPatch((1, y_output-0.5), 8, 0.9,
                                boxstyle="round,pad=0.12",
                                edgecolor=color_output, facecolor='#FFFDE7',
                                linewidth=2.5)
    ax.add_patch(output_box)
    
    ax.text(5, y_output+0.3, 'NORMALIZED OVR SCORES (55-97)', 
            ha='center', fontsize=10, fontweight='bold', color=color_output)
    ax.text(2.8, y_output-0.05, 'BATTING:', ha='right', fontsize=8, fontweight='bold', color='#D84315')
    ax.text(2.9, y_output-0.05, 'BASE | TOP | MIDDLE | FINISHER', ha='left', fontsize=7, color='#555555')
    ax.text(2.8, y_output-0.3, 'BOWLING:', ha='right', fontsize=8, fontweight='bold', color='#1565C0')
    ax.text(2.9, y_output-0.3, 'BASE | POWERPLAY | MIDDLE | DEATH', ha='left', fontsize=7, color='#555555')
    
    plt.tight_layout()
    return fig

# ============================================================================
# VISUAL 2: OVR FORMULA ANATOMY (unchanged)
# ============================================================================

def create_visual_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), facecolor='white')
    
    # Title
    fig.suptitle('OVR CALCULATION FORMULA ANATOMY', 
                 fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')
    fig.text(0.5, 0.94, 'IPL-tuned weighting scheme with role-specific bonuses and smart adjustments',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # ========== LEFT: BATTING OVR ==========
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('BATTING OVR FORMULA', fontsize=14, fontweight='bold', 
                  color='#D84315', pad=20)
    
    # Base components
    batting_components = [
        ('Strike Rate', 30, '#E53935'),
        ('Batting Average', 20, '#FB8C00'),
        ('Boundary %', 15, '#FDD835'),
        ('Conversion Rate', 15, '#9CCC65'),
        ('Rotation %', 10, '#26A69A'),
        ('Venue Consistency', 10, '#42A5F5')
    ]
    
    y_pos = 10.5
    ax1.text(50, y_pos, 'BASE FORMULA WEIGHTS (≥15 innings)', 
             ha='center', fontsize=11, fontweight='bold', color='#2C3E50')
    
    y_pos = 9.5
    for component, weight, color in batting_components:
        # Bar
        bar_width = weight * 2
        rect = Rectangle((5, y_pos-0.3), bar_width, 0.5, 
                         facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        # Label
        ax1.text(3, y_pos, component, ha='right', va='center', 
                fontsize=9, color='#333333', fontweight='bold')
        # Percentage
        ax1.text(bar_width+7, y_pos, f'{weight}%', ha='left', va='center',
                fontsize=10, color=color, fontweight='bold')
        y_pos -= 0.7
    
    # Bonuses section
    y_pos -= 0.5
    ax1.text(50, y_pos, 'BONUSES (Additive)', 
             ha='center', fontsize=11, fontweight='bold', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#43A047'))
    
    bonuses = [
        ('Experience', '+0 to +5', '15-50+ innings'),
        ('Explosive', '+0 to +5', 'SR≥160, Boundary≥65%'),
        ('Match Winner', '+0 to +5', '70+ RUNS ≥3 times'),
        ('Low Sample', '+0 to +4', '<15 inn, ≥100 balls'),
        ('Debut Potential', '+0 to +3', 'DEBUT=YES'),
        ('Wicket-Keeper', '+3.5', 'WK=YES'),
        ('Captain', '+2.0', 'CAPTAIN=YES'),
        ('Versatility', '+4.0', 'Multi-position')
    ]
    
    y_pos -= 0.8
    for bonus, value, condition in bonuses:
        ax1.text(8, y_pos, f'• {bonus}', ha='left', va='center',
                fontsize=8, color='#333333')
        ax1.text(35, y_pos, value, ha='left', va='center',
                fontsize=8, color='#43A047', fontweight='bold')
        ax1.text(50, y_pos, condition, ha='left', va='center',
                fontsize=7, color='#666666', style='italic')
        y_pos -= 0.5
    
    # Debut nerf
    y_pos -= 0.3
    ax1.text(50, y_pos, 'DEBUT NERF (Multiplicative)', 
             ha='center', fontsize=10, fontweight='bold', color='#C62828',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828'))
    y_pos -= 0.5
    ax1.text(50, y_pos, 'Final OVR × 0.93 (-7%) for DEBUT=YES batters only',
             ha='center', fontsize=8, color='#C62828')
    
    # ========== RIGHT: BOWLING OVR ==========
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('BOWLING OVR FORMULA', fontsize=14, fontweight='bold',
                  color='#1565C0', pad=20)
    
    # Base components
    bowling_components = [
        ('Economy Rate', 35, '#1565C0'),
        ('Dot Ball %', 18, '#0277BD'),
        ('Bowling Average', 15, '#0288D1'),
        ('Bowling Strike Rate', 14, '#039BE5'),
        ('Wicket Consistency', 10, '#03A9F4'),
        ('Control Index', 8, '#29B6F6')
    ]
    
    y_pos = 10.5
    ax2.text(50, y_pos, 'BASE FORMULA WEIGHTS (≥10 innings)', 
             ha='center', fontsize=11, fontweight='bold', color='#2C3E50')
    
    y_pos = 9.5
    for component, weight, color in bowling_components:
        # Bar
        bar_width = weight * 2
        rect = Rectangle((5, y_pos-0.3), bar_width, 0.5,
                         facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.8)
        ax2.add_patch(rect)
        # Label
        ax2.text(3, y_pos, component, ha='right', va='center',
                fontsize=9, color='#333333', fontweight='bold')
        # Percentage
        ax2.text(bar_width+7, y_pos, f'{weight}%', ha='left', va='center',
                fontsize=10, color=color, fontweight='bold')
        y_pos -= 0.7
    
    # IPL Buff
    y_pos -= 0.5
    ax2.text(50, y_pos, 'IPL T20 BUFF', 
             ha='center', fontsize=11, fontweight='bold', color='#5C2E7E',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE7F6', edgecolor='#5C2E7E'))
    y_pos -= 0.6
    ax2.text(50, y_pos, '+7 OVR to ALL bowling scores',
             ha='center', fontsize=9, color='#5C2E7E', fontweight='bold')
    y_pos -= 0.4
    ax2.text(50, y_pos, '(Accounts for batting-friendly T20 conditions)',
             ha='center', fontsize=7, color='#666666', style='italic')
    
    # Base correction
    y_pos -= 0.8
    ax2.text(50, y_pos, 'BASE OVR CORRECTION', 
             ha='center', fontsize=11, fontweight='bold', color='#FF6B35',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#FF6B35'))
    y_pos -= 0.6
    ax2.text(50, y_pos, 'BASE = Average(Top 2 Phase OVRs)',
             ha='center', fontsize=9, color='#FF6B35', fontweight='bold')
    y_pos -= 0.4
    ax2.text(50, y_pos, 'Example: PP:88, Mid:92, Death:85 → BASE = (92+88)/2 = 90',
             ha='center', fontsize=7, color='#666666', style='italic')
    
    # Bonuses
    y_pos -= 0.8
    ax2.text(50, y_pos, 'BONUSES', 
             ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#43A047'))
    
    bowling_bonuses = [
        ('Experience', '+0 to +5', '15-50+ innings'),
        ('Debut Potential', '+0 to +3', 'DEBUT=YES (NO nerf)')
    ]
    
    y_pos -= 0.7
    for bonus, value, condition in bowling_bonuses:
        ax2.text(15, y_pos, f'• {bonus}', ha='left', va='center',
                fontsize=8, color='#333333')
        ax2.text(45, y_pos, value, ha='left', va='center',
                fontsize=8, color='#43A047', fontweight='bold')
        ax2.text(60, y_pos, condition, ha='left', va='center',
                fontsize=7, color='#666666', style='italic')
        y_pos -= 0.5
    
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    return fig

# ============================================================================
# VISUAL 4: PHASE-SPECIFIC PERFORMANCE HEATMAP (FIXED OVERLAP)
# ============================================================================

def create_visual_4():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
    
    # Title
    fig.suptitle('PHASE-SPECIFIC PERFORMANCE PATTERNS', 
                 fontsize=18, fontweight='bold', y=0.97, color='#2C3E50')
    fig.text(0.5, 0.93, 'IPL 2024 winning vs losing teams show distinct phase-specific signatures',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # Data
    phases = ['Powerplay\n(1-6)', 'Middle\n(7-15)', 'Death\n(16-20)']
    
    # Batting Strike Rates
    wins_sr = [186.4, 145.2, 195.1]
    loss_sr = [142.7, 127.8, 167.9]
    
    # Bowling Economy
    wins_econ = [7.8, 7.2, 8.9]
    loss_econ = [9.4, 8.8, 11.2]
    
    # ========== TOP: BATTING STRIKE RATE ==========
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(0, 220)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(phases, fontsize=11)
    ax1.set_ylabel('Strike Rate', fontsize=12, fontweight='bold')
    ax1.set_title('BATTING STRIKE RATE BY PHASE', fontsize=13, fontweight='bold',
                  color='#D84315', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    x = np.arange(len(phases))
    width = 0.35
    
    # Color mapping function
    def get_sr_color(sr):
        if sr >= 180:
            return '#D32F2F', 'HOT'  # Elite
        elif sr >= 150:
            return '#FB8C00', 'WARM'  # Good
        else:
            return '#1976D2', 'COLD'  # Poor
    
    # Wins bars
    for i, sr in enumerate(wins_sr):
        color, label = get_sr_color(sr)
        bar = ax1.bar(x[i] - width/2, sr, width, color=color, alpha=0.85,
                     edgecolor='#333333', linewidth=1.5, label='Wins' if i == 0 else '')
        # Value label on top
        ax1.text(x[i] - width/2, sr + 8, f'{sr:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
    
    # Loss bars
    for i, sr in enumerate(loss_sr):
        color, label = get_sr_color(sr)
        bar = ax1.bar(x[i] + width/2, sr, width, color=color, alpha=0.85,
                     edgecolor='#333333', linewidth=1.5, label='Losses' if i == 0 else '')
        # Value label on top
        ax1.text(x[i] + width/2, sr + 8, f'{sr:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
    
    ax1.legend(['Wins', 'Losses'], loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add significance annotation
    ax1.text(0, 205, 'p = 0.003***', ha='center', fontsize=9,
            color='#C62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828'))
    
    # ========== BOTTOM: BOWLING ECONOMY ==========
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(0, 13)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(phases, fontsize=11)
    ax2.set_ylabel('Economy Rate (RPO)', fontsize=12, fontweight='bold')
    ax2.set_title('BOWLING ECONOMY BY PHASE', fontsize=13, fontweight='bold',
                  color='#1565C0', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Color mapping (inverted for economy)
    def get_econ_color(econ):
        if econ <= 8.0:
            return '#2E7D32', 'GOOD'  # Elite
        elif econ <= 9.5:
            return '#FB8C00', 'WARM'  # Average
        else:
            return '#C62828', 'BAD'  # Poor
    
    # Wins bars
    for i, econ in enumerate(wins_econ):
        color, label = get_econ_color(econ)
        bar = ax2.bar(x[i] - width/2, econ, width, color=color, alpha=0.85,
                     edgecolor='#333333', linewidth=1.5)
        # Value label on top
        ax2.text(x[i] - width/2, econ + 0.4, f'{econ:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
    
    # Loss bars
    for i, econ in enumerate(loss_econ):
        color, label = get_econ_color(econ)
        bar = ax2.bar(x[i] + width/2, econ, width, color=color, alpha=0.85,
                     edgecolor='#333333', linewidth=1.5)
        # Value label on top
        ax2.text(x[i] + width/2, econ + 0.4, f'{econ:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E7D32', edgecolor='#333333', label='Elite'),
        mpatches.Patch(facecolor='#FB8C00', edgecolor='#333333', label='Average'),
        mpatches.Patch(facecolor='#C62828', edgecolor='#333333', label='Poor')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9,
              title='Economy Rating')
    
    # Key insight box (moved higher to avoid overlap)
    fig.text(0.5, 0.01, 
            '💡 Powerplay & Death phases show largest performance gaps',
            ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.92))  # Adjusted bottom margin
    return fig

# ============================================================================
# GENERATE ALL VISUALS
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    import os
    output_dir = "/mnt/user-data/outputs/poster_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Generating Methods Visuals...")
    
    # Visual 1
    print("\n📊 Creating Visual 1: Compact Data Integration Pipeline...")
    fig1 = create_visual_1()
    fig1.savefig(f"{output_dir}/Visual_1_Data_Pipeline.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_1_Data_Pipeline.png")
    
    # Visual 2
    print("\n📊 Creating Visual 2: OVR Formula Anatomy...")
    fig2 = create_visual_2()
    fig2.savefig(f"{output_dir}/Visual_2_OVR_Formula.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_2_OVR_Formula.png")
    
    # Visual 4
    print("\n📊 Creating Visual 4: Phase-Specific Performance Heatmap...")
    fig4 = create_visual_4()
    fig4.savefig(f"{output_dir}/Visual_4_Phase_Heatmap.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_4_Phase_Heatmap.png")
    
    print("\n✨ ALL VISUALS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {output_dir}")
    print("\n📋 Generated files:")
    print("  1. Visual_1_Data_Pipeline.png (COMPACT VERSION)")
    print("  2. Visual_2_OVR_Formula.png")
    print("  3. Visual_4_Phase_Heatmap.png (FIXED OVERLAP)")
    
    plt.show()

    import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set global style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ============================================================================
# VISUAL 6: SMART WEIGHT REDISTRIBUTION LOGIC (Scenarios 1 & 2)
# ============================================================================

def create_visual_6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    
    # Main title
    fig.suptitle('SMART WEIGHT REDISTRIBUTION LOGIC', 
                 fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')
    fig.text(0.5, 0.93, 'Adaptive feature weighting system for complete vs incomplete player data',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # ========== LEFT: SCENARIO 1 - COMPLETE DATA ==========
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('SCENARIO 1: Complete Data (≥15 innings)', 
                  fontsize=13, fontweight='bold', color='#2E7D32', pad=20)
    
    # Components with weights
    components_complete = [
        ('Strike Rate', 30, '#E53935'),
        ('Batting Average', 20, '#FB8C00'),
        ('Boundary %', 15, '#FDD835'),
        ('Conversion Rate', 15, '#9CCC65'),
        ('Rotation %', 10, '#26A69A'),
        ('Venue Consistency', 10, '#42A5F5')
    ]
    
    y_pos = 8.5
    
    for component, weight, color in components_complete:
        # Bar
        bar_width = weight * 2  # Scale for visibility
        rect = Rectangle((10, y_pos-0.35), bar_width, 0.6,
                         facecolor=color, edgecolor='#333333', linewidth=2, alpha=0.85)
        ax1.add_patch(rect)
        
        # Label
        ax1.text(8, y_pos, component, ha='right', va='center',
                fontsize=10, color='#333333', fontweight='bold')
        
        # Percentage
        ax1.text(bar_width+12, y_pos, f'{weight}%', ha='left', va='center',
                fontsize=11, color=color, fontweight='bold')
        
        y_pos -= 1.1
    
    # Category annotations
    ax1.text(85, 6.8, '} Consistency', ha='left', va='center',
            fontsize=9, color='#666666', style='italic')
    ax1.text(85, 5.7, '} Metrics', ha='left', va='center',
            fontsize=9, color='#666666', style='italic')
    
    # Total weight box
    ax1.text(50, 1.5, 'TOTAL: 100%', ha='center', fontsize=11, fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', 
                     edgecolor='#2E7D32', linewidth=2))
    
    ax1.text(50, 0.5, 'All metrics available\nFull information utilized',
            ha='center', fontsize=9, color='#555555', style='italic')
    
    # ========== RIGHT: SCENARIO 2 - LOW SAMPLE ==========
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('SCENARIO 2: Low Sample (<15 innings, ≥100 balls)', 
                  fontsize=13, fontweight='bold', color='#D84315', pad=20)
    
    # Components with redistributed weights
    components_low = [
        ('Strike Rate', 40, '#E53935', True),      # ⬆ Increased
        ('Batting Average', 25, '#FB8C00', True),  # ⬆ Increased
        ('Boundary %', 25, '#FDD835', True),       # ⬆ Increased
        ('Rotation %', 10, '#26A69A', False),      # Same
        ('Conversion Rate', 0, '#9CCC65', 'skip'), # ✗ Skipped
        ('Venue Consistency', 0, '#42A5F5', 'skip') # ✗ Skipped
    ]
    
    y_pos = 8.5
    
    for component, weight, color, status in components_low:
        if status == 'skip':
            # Skipped metric - grayed out
            rect = Rectangle((10, y_pos-0.35), 5, 0.6,
                           facecolor='#E0E0E0', edgecolor='#999999', 
                           linewidth=1.5, alpha=0.5, linestyle='--')
            ax2.add_patch(rect)
            
            ax2.text(8, y_pos, component, ha='right', va='center',
                    fontsize=10, color='#999999', style='italic')
            
            ax2.text(17, y_pos, '✗ SKIPPED', ha='left', va='center',
                    fontsize=9, color='#C62828', fontweight='bold')
            
            ax2.text(35, y_pos, '(insufficient data)', ha='left', va='center',
                    fontsize=8, color='#999999', style='italic')
        else:
            # Active metric
            bar_width = weight * 2
            rect = Rectangle((10, y_pos-0.35), bar_width, 0.6,
                           facecolor=color, edgecolor='#333333', linewidth=2, alpha=0.85)
            ax2.add_patch(rect)
            
            ax2.text(8, y_pos, component, ha='right', va='center',
                    fontsize=10, color='#333333', fontweight='bold')
            
            # Show percentage with arrow if increased
            if status == True:
                ax2.text(bar_width+12, y_pos, f'{weight}% ⬆', ha='left', va='center',
                        fontsize=11, color=color, fontweight='bold')
            else:
                ax2.text(bar_width+12, y_pos, f'{weight}%', ha='left', va='center',
                        fontsize=11, color=color, fontweight='bold')
        
        y_pos -= 1.1
    
    # Total weight box
    ax2.text(50, 1.5, 'TOTAL: 100%', ha='center', fontsize=11, fontweight='bold',
            color='#D84315',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', 
                     edgecolor='#D84315', linewidth=2))
    
    ax2.text(50, 0.5, 'Prioritize immediate impact\n+ Low Sample Bonus: 0 to +4 OVR',
            ha='center', fontsize=9, color='#555555', style='italic')
    
    # ========== REDISTRIBUTION ARROW ==========
    # Arrow annotation between scenarios
    fig.text(0.5, 0.52, '▼ AUTOMATIC REDISTRIBUTION ▼',
            ha='center', fontsize=12, fontweight='bold', color='#FF6B35',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', 
                     edgecolor='#FF6B35', linewidth=2.5))
    
    # Key insight box
    fig.text(0.5, 0.02,
            '💡 LOGIC: When consistency metrics unavailable, boost weights for immediate impact stats (SR, Boundaries)',
            ha='center', fontsize=11, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', 
                     edgecolor='#1976D2', linewidth=2))
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.91))
    return fig

# ============================================================================
# GENERATE VISUAL
# ============================================================================

if __name__ == "__main__":
    import os
    output_dir = "/mnt/user-data/outputs/poster_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Generating Visual 6: Smart Weight Redistribution...")
    
    fig = create_visual_6()
    fig.savefig(f"{output_dir}/Visual_6_Weight_Redistribution.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Saved: {output_dir}/Visual_6_Weight_Redistribution.png")
    print("\n📊 Visual shows:")
    print("  • Scenario 1: Complete data (≥15 innings) - All 6 metrics at normal weights")
    print("  • Scenario 2: Low sample (<15 innings) - 4 active metrics, 2 skipped")
    print("  • Automatic weight redistribution with ⬆ arrows")
    print("  • ✗ markers for skipped metrics")
    print("  • Low sample bonus annotation")
    
    plt.show()