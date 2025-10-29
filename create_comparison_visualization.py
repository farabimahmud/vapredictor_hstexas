import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_comparison_visualization():
    """Create comprehensive comparison visualization between manual and algorithmic approaches."""
    
    # Data for comparison
    treatments = ['Mental Health\n(q26)', 'Ever Cigarette\n(q32)', 'Current Alcohol\n(q42)', 
                 'Ever Marijuana\n(q46)', 'Adequate Sleep\n(q85)']
    
    # Manual approach results (single estimates)
    manual_effects = [0.0716, 0.2289, -0.0509, -0.0795, 0.2339]
    
    # Algorithmic approach results
    algorithmic_effects = [0.0424, -0.0795, -0.0344, -0.0954, 0.1181]
    algorithmic_std = [0.0173, 0.1292, 0.0175, 0.0757, 0.1052]
    robustness = ['High', 'Low', 'High', 'Low', 'Low']
    
    # Effect ranges from sensitivity analysis
    effect_ranges = {
        'Mental Health': (0.0338, 0.0725),
        'Ever Cigarette': (-0.0795, 0.2349),
        'Current Alcohol': (-0.0661, -0.0280),
        'Ever Marijuana': (-0.1042, 0.0735),
        'Adequate Sleep': (0.1181, 0.3373)
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Effect Size Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(treatments))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, manual_effects, width, label='Manual Approach', 
                   color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, algorithmic_effects, width, label='Algorithmic Approach',
                   color='skyblue', alpha=0.8)
    
    # Add error bars for algorithmic approach
    ax1.errorbar(x + width/2, algorithmic_effects, yerr=algorithmic_std, 
                fmt='none', color='black', capsize=5)
    
    ax1.set_xlabel('Treatment Variables')
    ax1.set_ylabel('Causal Effect Size')
    ax1.set_title('Causal Effect Comparison:\nManual vs Algorithmic Approach', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(treatments, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, (manual, algo) in enumerate(zip(manual_effects, algorithmic_effects)):
        ax1.text(i - width/2, manual + 0.01, f'{manual:.3f}', ha='center', fontsize=9)
        ax1.text(i + width/2, algo + algorithmic_std[i] + 0.01, f'{algo:.3f}', ha='center', fontsize=9)
    
    # 2. Robustness Assessment
    ax2 = plt.subplot(2, 3, 2)
    
    robustness_colors = {'High': 'green', 'Low': 'red', 'Medium': 'orange'}
    colors = [robustness_colors[r] for r in robustness]
    
    bars = ax2.bar(treatments, [abs(e) for e in algorithmic_effects], color=colors, alpha=0.7)
    ax2.set_xlabel('Treatment Variables')
    ax2.set_ylabel('Absolute Effect Size')
    ax2.set_title('Effect Robustness Assessment\n(Green=High, Red=Low)', fontweight='bold')
    ax2.set_xticklabels(treatments, rotation=45, ha='right')
    
    # Add robustness labels
    for i, (bar, rob, std) in enumerate(zip(bars, robustness, algorithmic_std)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{rob}\n(±{std:.3f})', ha='center', va='bottom', fontsize=9)
    
    # 3. Sensitivity Analysis Ranges
    ax3 = plt.subplot(2, 3, 3)
    
    treatment_names = ['Mental Health', 'Ever Cigarette', 'Current Alcohol', 'Ever Marijuana', 'Adequate Sleep']
    
    for i, treatment in enumerate(treatment_names):
        min_effect, max_effect = effect_ranges[treatment]
        ax3.errorbar(i, algorithmic_effects[i], 
                    yerr=[[algorithmic_effects[i] - min_effect], [max_effect - algorithmic_effects[i]]], 
                    fmt='o', capsize=10, capthick=2, markersize=8)
        
        # Add range text
        ax3.text(i + 0.1, algorithmic_effects[i], 
                f'[{min_effect:.3f}, {max_effect:.3f}]', 
                rotation=90, fontsize=8, va='center')
    
    ax3.set_xlabel('Treatment Variables')
    ax3.set_ylabel('Effect Size Range')
    ax3.set_title('Sensitivity Analysis:\nEffect Ranges Across Methods', fontweight='bold')
    ax3.set_xticks(range(len(treatment_names)))
    ax3.set_xticklabels(treatments, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 4. Agreement vs Disagreement
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate agreement
    agreement_data = []
    for i, treatment in enumerate(treatments):
        manual_val = manual_effects[i]
        algo_val = algorithmic_effects[i]
        
        # Check if same direction
        same_direction = (manual_val * algo_val) > 0
        # Check if within confidence interval
        within_ci = abs(manual_val - algo_val) <= 2 * algorithmic_std[i]
        
        if same_direction and within_ci:
            agreement = 'High Agreement'
        elif same_direction:
            agreement = 'Same Direction'
        else:
            agreement = 'Disagreement'
        
        agreement_data.append(agreement)
    
    agreement_counts = pd.Series(agreement_data).value_counts()
    colors = ['green', 'orange', 'red']
    
    wedges, texts, autotexts = ax4.pie(agreement_counts.values, labels=agreement_counts.index, 
                                      autopct='%1.0f%%', colors=colors[:len(agreement_counts)])
    ax4.set_title('Method Agreement Summary', fontweight='bold')
    
    # 5. Confounder Set Sizes
    ax5 = plt.subplot(2, 3, 5)
    
    # Manual approach uses same 6 confounders for all
    manual_confounder_counts = [6] * 5
    
    # Algorithmic approach uses optimized sets
    algorithmic_confounder_counts = [6, 6, 6, 6, 6]  # From the analysis results
    
    x = np.arange(len(treatments))
    width = 0.35
    
    ax5.bar(x - width/2, manual_confounder_counts, width, label='Manual (Fixed)', 
           color='lightcoral', alpha=0.8)
    ax5.bar(x + width/2, algorithmic_confounder_counts, width, label='Algorithmic (Optimized)',
           color='skyblue', alpha=0.8)
    
    ax5.set_xlabel('Treatment Variables')
    ax5.set_ylabel('Number of Confounders')
    ax5.set_title('Confounder Set Optimization', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(treatments, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Confidence Assessment Summary
    ax6 = plt.subplot(2, 3, 6)
    
    # Create confidence data
    confidence_data = {
        'High Confidence': 2,  # Mental Health, Current Alcohol
        'Medium Confidence': 0,
        'Low Confidence': 3   # Ever Cigarette, Ever Marijuana, Adequate Sleep
    }
    
    bars = ax6.bar(confidence_data.keys(), confidence_data.values(), 
                  color=['green', 'orange', 'red'], alpha=0.7)
    ax6.set_ylabel('Number of Effects')
    ax6.set_title('Algorithmic Confidence Assessment', fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/algorithmic_vs_manual_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed methods comparison table
    create_methods_comparison_table()

def create_methods_comparison_table():
    """Create a detailed comparison table of methods."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create comparison data
    comparison_data = [
        ['Aspect', 'Manual Approach', 'Algorithmic Approach'],
        ['Confounder Selection', 'Fixed theory-based set', 'Data-driven optimization per treatment'],
        ['Cross-Validation', 'None', 'Systematic CV for confounder selection'],
        ['Sensitivity Analysis', 'Basic refutation test', 'Multiple model specifications'],
        ['Robustness Assessment', 'None', 'Quantified uncertainty for each estimate'],
        ['Treatment Specificity', 'One-size-fits-all', 'Optimized for each treatment'],
        ['Effect Validation', 'Single estimate', 'Range of estimates across methods'],
        ['Confidence Rating', 'Not provided', 'High/Medium/Low classification'],
        ['Causal Discovery', 'Theory-only', 'Algorithmic graph discovery'],
        ['Bias Detection', 'Limited', 'Systematic sensitivity testing'],
        ['Computational Cost', 'Low', 'High'],
        ['Interpretability', 'High', 'Moderate'],
        ['Statistical Rigor', 'Moderate', 'High'],
        ['Evidence Quality', 'Single point estimate', 'Robustness-assessed ranges']
    ]
    
    # Create table
    table = ax.table(cellText=comparison_data[1:], 
                    colLabels=comparison_data[0],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.35, 0.35])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color coding
    for i in range(len(comparison_data)):
        if i == 0:  # Header
            for j in range(3):
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#E8F5E8')  # Aspect column
            table[(i, 1)].set_facecolor('#FFE8E8')  # Manual approach
            table[(i, 2)].set_facecolor('#E8F0FF')  # Algorithmic approach
    
    plt.title('Comprehensive Methodological Comparison:\nManual vs Algorithmic Causal Analysis', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('output/methods_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    create_comparison_visualization()