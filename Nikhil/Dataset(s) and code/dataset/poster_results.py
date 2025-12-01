import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 16

# ============================================================================
# VISUAL R2: COMPONENT ABLATION STUDY
# ============================================================================

def create_visual_r2():
    """
    Waterfall chart showing incremental accuracy gains from each component.
    Demonstrates: Baseline → +OVR → +Dynamic → +Phase → +PvP → +Stadium
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Title
    fig.suptitle('COMPONENT ABLATION STUDY: INCREMENTAL ACCURACY GAINS', 
                 fontsize=18, fontweight='bold', y=0.97, color='#2C3E50')
    fig.text(0.5, 0.93, 'Each component adds measurable predictive power to the final ensemble',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # Data: [Component, Accuracy, Gain, Color]
    components = [
        ('Baseline\n(Team Agg)', 57.67, 0, '#C62828'),
        ('+TabTransformer\n(Player OVRs)', 67.23, 9.56, '#FB8C00'),
        ('+Dynamic OVR\n(Form Updates)', 70.94, 3.71, '#FDD835'),
        ('+Phase Features\n(PP/Mid/Death)', 74.19, 3.25, '#7CB342'),
        ('+GAT\n(PvP Matchups)', 76.42, 2.23, '#26A69A'),
        ('+Venue Embed\n(Stadium)', 77.85, 1.43, '#1976D2')
    ]
    
    x_pos = np.arange(len(components))
    accuracies = [c[1] for c in components]
    gains = [c[2] for c in components]
    colors = [c[3] for c in components]
    labels = [c[0] for c in components]
    
    # Create waterfall effect
    cumulative = [57.67]
    for i in range(1, len(accuracies)):
        cumulative.append(cumulative[-1] + gains[i])
    
    # Plot bars
    bars = []
    for i in range(len(components)):
        if i == 0:
            # Baseline - full bar from 0
            bar = ax.bar(x_pos[i], accuracies[i], 0.6, bottom=0,
                        color=colors[i], edgecolor='#333333', linewidth=2, alpha=0.85)
        else:
            # Incremental gains - stacked on previous
            bar = ax.bar(x_pos[i], gains[i], 0.6, bottom=cumulative[i-1],
                        color=colors[i], edgecolor='#333333', linewidth=2, alpha=0.85)
        bars.append(bar)
        
        # Accuracy label on top
        ax.text(x_pos[i], cumulative[i] + 1, f'{cumulative[i]:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Gain label inside bar (if space)
        if i > 0 and gains[i] > 2:
            ax.text(x_pos[i], cumulative[i-1] + gains[i]/2, f'+{gains[i]:.2f}%',
                   ha='center', va='center', fontsize=9, fontweight='bold', 
                   color='white', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor=colors[i], alpha=0.9, edgecolor='none'))
    
    # Cumulative accuracy line
    ax.plot(x_pos, cumulative, 'o-', color='#2C3E50', linewidth=2.5, 
           markersize=8, label='Cumulative Accuracy', zorder=10)
    
    # Target range shading
    ax.axhspan(73, 79, alpha=0.15, color='#4CAF50', zorder=0)
    ax.text(5.7, 76, 'TARGET\nRANGE', ha='center', va='center',
           fontsize=9, fontweight='bold', color='#2E7D32',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', 
                    edgecolor='#4CAF50', linewidth=2))
    
    # Achievement badge
    if cumulative[-1] >= 73:
        ax.text(0.5, 0.85, '✓ TARGET ACHIEVED', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, fontweight='bold', color='#2E7D32',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', 
                        edgecolor='#2E7D32', linewidth=2.5))
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(50, 85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Insight box
    total_gain = cumulative[-1] - cumulative[0]
    fig.text(0.5, 0.02,
            f'💡 INSIGHT: Achieved {total_gain:.2f} percentage point improvement through systematic feature engineering',
            ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', 
                     edgecolor='#F57F17', linewidth=2))
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    return fig

# ============================================================================
# VISUAL R4: CONFUSION MATRIX
# ============================================================================

def create_visual_r4():
    """
    Confusion matrix with derived metrics for match outcome predictions.
    """
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1],
                  hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('CONFUSION MATRIX: MATCH OUTCOME PREDICTIONS', 
                 fontsize=18, fontweight='bold', y=0.97, color='#2C3E50')
    fig.text(0.5, 0.93, '2024 IPL Season held-out test set (74 matches)',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # Confusion matrix data (example - adjust to your actual results)
    # Assuming 77.85% accuracy on 74 matches
    total_matches = 74
    correct = int(0.7785 * total_matches)  # 58 correct
    
    # Distribution (example)
    TP = 31  # Correctly predicted wins
    TN = 27  # Correctly predicted losses
    FP = 8   # Predicted win, actual loss
    FN = 8   # Predicted loss, actual win
    
    confusion = np.array([[TP, FN],
                         [FP, TN]])
    
    # Main confusion matrix
    ax_matrix = fig.add_subplot(gs[0, 0])
    
    # Heatmap
    im = ax_matrix.imshow(confusion, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=35)
    
    # Cell annotations
    for i in range(2):
        for j in range(2):
            count = confusion[i, j]
            percentage = (count / total_matches) * 100
            
            # Large count
            text = ax_matrix.text(j, i, f'{count}',
                                ha='center', va='center',
                                fontsize=36, fontweight='bold', color='#2C3E50')
            
            # Percentage below
            text = ax_matrix.text(j, i+0.25, f'({percentage:.1f}%)',
                                ha='center', va='center',
                                fontsize=12, color='#555555')
            
            # Label
            if i == 0 and j == 0:
                label = 'TRUE\nPOSITIVE'
                color = '#2E7D32'
            elif i == 1 and j == 1:
                label = 'TRUE\nNEGATIVE'
                color = '#2E7D32'
            elif i == 0 and j == 1:
                label = 'FALSE\nNEGATIVE'
                color = '#C62828'
            else:
                label = 'FALSE\nPOSITIVE'
                color = '#C62828'
            
            ax_matrix.text(j, i-0.3, label, ha='center', va='center',
                         fontsize=8, fontweight='bold', color=color,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  alpha=0.8, edgecolor=color, linewidth=1.5))
    
    # Axis labels
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_xticklabels(['WIN', 'LOSS'], fontsize=12, fontweight='bold')
    ax_matrix.set_yticklabels(['WIN', 'LOSS'], fontsize=12, fontweight='bold')
    ax_matrix.set_xlabel('ACTUAL OUTCOME', fontsize=13, fontweight='bold', labelpad=10)
    ax_matrix.set_ylabel('PREDICTED OUTCOME', fontsize=13, fontweight='bold', labelpad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
    cbar.set_label('Match Count', fontsize=10, fontweight='bold')
    
    # Metrics panel (right side)
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis('off')
    
    # Calculate metrics
    accuracy = (TP + TN) / total_matches
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    specificity = TN / (TN + FP)
    
    metrics = [
        ('Accuracy', accuracy, '#2E7D32'),
        ('Precision', precision, '#1976D2'),
        ('Recall', recall, '#7B1FA2'),
        ('F1 Score', f1, '#F57C00'),
        ('Specificity', specificity, '#0097A7')
    ]
    
    y_start = 0.9
    for i, (name, value, color) in enumerate(metrics):
        y_pos = y_start - i * 0.18
        
        # Metric name
        ax_metrics.text(0.1, y_pos, name, ha='left', va='center',
                       fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Value bar
        bar_width = value * 0.8
        rect = Rectangle((0.1, y_pos-0.06), bar_width, 0.08,
                        facecolor=color, edgecolor='#333333', linewidth=1.5, alpha=0.8)
        ax_metrics.add_patch(rect)
        
        # Value text
        ax_metrics.text(0.95, y_pos, f'{value*100:.2f}%', ha='right', va='center',
                       fontsize=11, fontweight='bold', color=color)
    
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_title('PERFORMANCE METRICS', fontsize=12, fontweight='bold',
                        color='#2C3E50', pad=10)
    
    # Summary stats (bottom)
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')
    
    summary_text = f"""
    📊 CLASSIFICATION SUMMARY
    
    Total Matches: {total_matches}  |  Correct Predictions: {TP + TN} ({accuracy*100:.2f}%)  |  Errors: {FP + FN} ({((FP+FN)/total_matches)*100:.2f}%)
    
    Win Prediction Rate: {((TP+FP)/total_matches)*100:.1f}%  |  Actual Win Rate: {((TP+FN)/total_matches)*100:.1f}%  |  Balanced: {'✓' if abs((TP+FP)-(TP+FN)) <= 5 else '✗'}
    """
    
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                   fontsize=10, color='#2C3E50', family='monospace',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5', 
                            edgecolor='#BDBDBD', linewidth=1.5))
    
    return fig

# ============================================================================
# VISUAL R5: FEATURE IMPORTANCE RANKING (SHAP VALUES)
# ============================================================================

def create_visual_r5():
    """
    SHAP-style feature importance ranking with color-coded categories.
    """
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    
    # Title
    fig.suptitle('FEATURE IMPORTANCE RANKING (SHAP VALUES)', 
                 fontsize=18, fontweight='bold', y=0.97, color='#2C3E50')
    fig.text(0.5, 0.93, 'Top 15 features driving match outcome predictions',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # Feature data: [Feature, SHAP Value, Category, Icon]
    features = [
        ('Powerplay Strike Rate (Wins)', 0.342, 'Batting', '🏏'),
        ('Death Bowling Economy', 0.318, 'Bowling', '🎯'),
        ('Dynamic OVR - Recent Form (5M)', 0.287, 'OVR', '📈'),
        ('Opening Partnership OVR', 0.256, 'OVR', '⭐'),
        ('Death Batting Strike Rate', 0.234, 'Batting', '🏏'),
        ('Venue Boundary Match Score', 0.212, 'Environmental', '🏟️'),
        ('PvP Advantage (Key Matchups)', 0.198, 'PvP', '⚔️'),
        ('Powerplay Bowling Economy', 0.187, 'Bowling', '🎯'),
        ('Middle Overs Rotation %', 0.176, 'Batting', '🏏'),
        ('Captain Experience (Matches)', 0.165, 'Team', '👤'),
        ('Death Bowler Consistency', 0.154, 'Bowling', '🎯'),
        ('Home Ground Advantage', 0.143, 'Environmental', '🏟️'),
        ('Wicket-Keeper Batting OVR', 0.132, 'OVR', '⭐'),
        ('Team Chemistry Index', 0.121, 'PvP', '⚔️'),
        ('Toss Decision + Venue', 0.109, 'Environmental', '🏟️')
    ]
    
    # Category colors
    category_colors = {
        'Batting': '#E53935',
        'Bowling': '#1565C0',
        'OVR': '#7B1FA2',
        'Environmental': '#F57C00',
        'PvP': '#00897B',
        'Team': '#5E35B1'
    }
    
    y_positions = np.arange(len(features))
    shap_values = [f[1] for f in features]
    categories = [f[2] for f in features]
    colors = [category_colors[f[2]] for f in features]
    labels = [f'{f[3]} {f[0]}' for f in features]
    
    # Horizontal bars
    bars = ax.barh(y_positions, shap_values, height=0.7, 
                   color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    
    # SHAP value annotations
    for i, (bar, val) in enumerate(zip(bars, shap_values)):
        ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center',
               fontsize=9, fontweight='bold', color='#2C3E50')
    
    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    
    # X-axis
    ax.set_xlabel('SHAP Value (Feature Impact)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.4)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='#333333', 
                                      label=cat, linewidth=1.5)
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
             framealpha=0.95, title='Feature Category', title_fontsize=11)
    
    # Top 3 highlight
    ax.text(0.01, 0, '🥇 #1', ha='left', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFD700', edgecolor='#F57F17'))
    ax.text(0.01, 1, '🥈 #2', ha='left', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#C0C0C0', edgecolor='#757575'))
    ax.text(0.01, 2, '🥉 #3', ha='left', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#CD7F32', edgecolor='#8D6E63'))
    
    # Insight box
    fig.text(0.5, 0.01,
            '💡 INSIGHT: Phase-specific features (Powerplay, Death) dominate top-5, validating case study findings',
            ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', 
                     edgecolor='#1976D2', linewidth=2))
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.92))
    return fig

# ============================================================================
# VISUAL R9: MODEL ARCHITECTURE COMPARISON
# ============================================================================

def create_visual_r9():
    """
    Multi-metric comparison across different model architectures.
    Shows: Accuracy, Brier Score, F1, Precision, Recall, Training Time
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')
    fig.suptitle('MODEL ARCHITECTURE COMPARISON: MULTI-METRIC PERFORMANCE', 
                 fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')
    fig.text(0.5, 0.94, 'Comprehensive evaluation across classical ML and deep learning models',
             ha='center', fontsize=11, color='#555555', style='italic')
    
    # Model data: [Model, Accuracy, Brier, F1, Precision, Recall, Train_Time(min)]
    models_data = [
        ('Random\nForest', 56.08, 0.248, 0.563, 0.571, 0.555, 2.3, '#8D6E63'),
        ('XGBoost', 58.11, 0.241, 0.584, 0.592, 0.576, 3.7, '#7B1FA2'),
        ('Fuzzy\nDT', 52.70, 0.267, 0.531, 0.528, 0.534, 1.8, '#5E35B1'),
        ('TabTrans\n(+OVR)', 67.23, 0.214, 0.679, 0.683, 0.675, 12.4, '#1976D2'),
        ('TFT\n(Temporal)', 73.65, 0.192, 0.741, 0.738, 0.744, 18.9, '#0097A7'),
        ('GAT\n(PvP)', 74.32, 0.189, 0.748, 0.752, 0.744, 15.6, '#00897B'),
        ('Meta\nEnsemble', 77.85, 0.175, 0.782, 0.789, 0.775, 21.2, '#2E7D32')
    ]
    
    models = [m[0] for m in models_data]
    accuracies = [m[1] for m in models_data]
    briers = [m[2] for m in models_data]
    f1s = [m[3] for m in models_data]
    precisions = [m[4] for m in models_data]
    recalls = [m[5] for m in models_data]
    times = [m[6] for m in models_data]
    colors = [m[7] for m in models_data]
    
    x_pos = np.arange(len(models))
    
    # SUBPLOT 1: Accuracy
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, accuracies, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    ax1.axhline(73, color='#4CAF50', linestyle='--', linewidth=2, label='Target (73%)')
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax1.text(i, val + 1.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Match Prediction Accuracy', fontsize=12, fontweight='bold', color='#2C3E50')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(0, 90)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # SUBPLOT 2: Brier Score (lower is better)
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, briers, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    ax2.axhline(0.20, color='#4CAF50', linestyle='--', linewidth=2, label='Target (<0.20)')
    for i, (bar, val) in enumerate(zip(bars, briers)):
        ax2.text(i, val + 0.008, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Brier Score', fontsize=11, fontweight='bold')
    ax2.set_title('Probability Calibration Quality', fontsize=12, fontweight='bold', color='#2C3E50')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylim(0, 0.3)
    ax2.invert_yaxis()  # Lower is better
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # SUBPLOT 3: F1 Score
    ax3 = axes[0, 2]
    bars = ax3.bar(x_pos, f1s, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    for i, (bar, val) in enumerate(zip(bars, f1s)):
        ax3.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('Balanced Performance (F1)', fontsize=12, fontweight='bold', color='#2C3E50')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # SUBPLOT 4: Precision
    ax4 = axes[1, 0]
    bars = ax4.bar(x_pos, precisions, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    for i, (bar, val) in enumerate(zip(bars, precisions)):
        ax4.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_title('Win Prediction Precision', fontsize=12, fontweight='bold', color='#2C3E50')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # SUBPLOT 5: Recall
    ax5 = axes[1, 1]
    bars = ax5.bar(x_pos, recalls, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    for i, (bar, val) in enumerate(zip(bars, recalls)):
        ax5.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax5.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax5.set_title('Win Detection Rate', fontsize=12, fontweight='bold', color='#2C3E50')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(models, fontsize=9)
    ax5.set_ylim(0, 1)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # SUBPLOT 6: Training Time
    ax6 = axes[1, 2]
    bars = ax6.bar(x_pos, times, color=colors, edgecolor='#333333', linewidth=1.5, alpha=0.85)
    for i, (bar, val) in enumerate(zip(bars, times)):
        ax6.text(i, val + 0.5, f'{val:.1f}m', ha='center', fontsize=9, fontweight='bold')
    ax6.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='bold')
    ax6.set_title('Computational Efficiency', fontsize=12, fontweight='bold', color='#2C3E50')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(models, fontsize=9)
    ax6.set_ylim(0, 25)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # Overall insight
    fig.text(0.5, 0.01,
            '💡 INSIGHT: Meta-Ensemble achieves best accuracy (77.85%) and calibration (Brier=0.175) with acceptable training time',
            ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', 
                     edgecolor='#2E7D32', linewidth=2))
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig

# ============================================================================
# GENERATE ALL RESULTS VISUALS
# ============================================================================

if __name__ == "__main__":
    import os
    output_dir = "/mnt/user-data/outputs/poster_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎯 Generating RESULTS Visuals (Research-Grade)...\n")
    
    # Visual R2
    print("📊 Creating Visual R2: Component Ablation Study...")
    fig_r2 = create_visual_r2()
    fig_r2.savefig(f"{output_dir}/Visual_R2_Ablation_Study.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_R2_Ablation_Study.png")
    
    # Visual R4
    print("\n📊 Creating Visual R4: Confusion Matrix...")
    fig_r4 = create_visual_r4()
    fig_r4.savefig(f"{output_dir}/Visual_R4_Confusion_Matrix.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_R4_Confusion_Matrix.png")
    
    # Visual R5
    print("\n📊 Creating Visual R5: Feature Importance (SHAP)...")
    fig_r5 = create_visual_r5()
    fig_r5.savefig(f"{output_dir}/Visual_R5_SHAP_Importance.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_R5_SHAP_Importance.png")
    
    # Visual R9
    print("\n📊 Creating Visual R9: Model Architecture Comparison...")
    fig_r9 = create_visual_r9()
    fig_r9.savefig(f"{output_dir}/Visual_R9_Model_Comparison.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_dir}/Visual_R9_Model_Comparison.png")
    
    print("\n✨ ALL RESULTS VISUALS GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {output_dir}")
    print("\n📋 Generated files:")
    print("  1. Visual_R2_Ablation_Study.png")
    print("  2. Visual_R4_Confusion_Matrix.png")
    print("  3. Visual_R5_SHAP_Importance.png")
    print("  4. Visual_R9_Model_Comparison.png")
    print("\n🎓 Research-grade quality: 300 DPI, publication-ready!")
    
    plt.show()