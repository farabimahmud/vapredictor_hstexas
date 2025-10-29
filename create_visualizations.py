import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import networkx as nx

def create_causal_visualizations():
    """Create visualizations for the causal analysis results."""
    
    # Load data
    df = pd.read_csv("data/cleaned_hstexas_full.csv")
    
    # Create target variable
    df['ever_vaped'] = (df['q35'] == 1).astype(int) if 'q35' in df.columns else 0
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Vaping rates by demographics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # By grade
    if 'grade' in df.columns:
        grade_rates = df.groupby('grade')['ever_vaped'].mean()
        axes[0,0].bar(grade_rates.index, grade_rates.values)
        axes[0,0].set_title('Vaping Rate by Grade Level', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Grade (1=9th, 2=10th, 3=11th, 4=12th)')
        axes[0,0].set_ylabel('Ever Vaping Rate')
        axes[0,0].set_ylim(0, 0.5)
        for i, v in enumerate(grade_rates.values):
            axes[0,0].text(grade_rates.index[i], v + 0.01, f'{v:.1%}', ha='center')
    
    # By sex
    if 'sex' in df.columns:
        sex_rates = df.groupby('sex')['ever_vaped'].mean()
        sex_labels = ['Female', 'Male']
        axes[0,1].bar(range(len(sex_rates)), sex_rates.values, color=['pink', 'lightblue'])
        axes[0,1].set_title('Vaping Rate by Sex', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Sex')
        axes[0,1].set_ylabel('Ever Vaping Rate')
        axes[0,1].set_xticks(range(len(sex_rates)))
        axes[0,1].set_xticklabels(sex_labels)
        axes[0,1].set_ylim(0, 0.4)
        for i, v in enumerate(sex_rates.values):
            axes[0,1].text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    # Mental health associations
    mh_vars = {'q26': 'Sad/Hopeless', 'q27': 'Considered Suicide', 
               'q28': 'Suicide Plan', 'q29': 'Attempted Suicide'}
    
    mh_rates_no = []
    mh_rates_yes = []
    mh_labels = []
    
    for var, label in mh_vars.items():
        if var in df.columns:
            no_mh = df[df[var] != 1]['ever_vaped'].mean()
            yes_mh = df[df[var] == 1]['ever_vaped'].mean()
            mh_rates_no.append(no_mh)
            mh_rates_yes.append(yes_mh)
            mh_labels.append(label)
    
    x = np.arange(len(mh_labels))
    width = 0.35
    
    axes[1,0].bar(x - width/2, mh_rates_no, width, label='No', alpha=0.8)
    axes[1,0].bar(x + width/2, mh_rates_yes, width, label='Yes', alpha=0.8)
    axes[1,0].set_title('Vaping Rate by Mental Health Status', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Mental Health Indicator')
    axes[1,0].set_ylabel('Ever Vaping Rate')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(mh_labels, rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 0.6)
    
    # Substance use associations
    substance_vars = {'q32': 'Ever Cigarette', 'q42': 'Current Alcohol', 
                     'q46': 'Ever Marijuana', 'q48': 'Current Marijuana'}
    
    sub_rates_no = []
    sub_rates_yes = []
    sub_labels = []
    
    for var, label in substance_vars.items():
        if var in df.columns:
            no_sub = df[df[var] != 1]['ever_vaped'].mean()
            yes_sub = df[df[var] == 1]['ever_vaped'].mean()
            sub_rates_no.append(no_sub)
            sub_rates_yes.append(yes_sub)
            sub_labels.append(label)
    
    x = np.arange(len(sub_labels))
    
    axes[1,1].bar(x - width/2, sub_rates_no, width, label='No', alpha=0.8)
    axes[1,1].bar(x + width/2, sub_rates_yes, width, label='Yes', alpha=0.8)
    axes[1,1].set_title('Vaping Rate by Other Substance Use', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Substance Use')
    axes[1,1].set_ylabel('Ever Vaping Rate')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(sub_labels, rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig('output/vaping_demographics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Causal Pathways Diagram
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Create a conceptual causal diagram
    G = nx.DiGraph()
    
    # Define node positions manually for better layout
    pos = {
        'Demographics\n(Age, Sex, Race)': (2, 8),
        'Mental Health\n(Depression,\nSuicidal Ideation)': (0, 6),
        'Peer Influence\n(Social Norms,\nFriend Behaviors)': (4, 6),
        'Risk-Taking\nPropensity': (6, 6),
        'Family\nEnvironment': (1, 4),
        'School\nEnvironment': (5, 4),
        'Traditional\nSubstance Use\n(Tobacco, Alcohol)': (2, 4),
        'Other Drug Use\n(Marijuana,\nPrescription)': (3, 2),
        'VAPING\nINITIATION': (3, 0)
    }
    
    # Add nodes
    for node in pos.keys():
        G.add_node(node)
    
    # Define causal relationships
    edges = [
        ('Demographics\n(Age, Sex, Race)', 'Mental Health\n(Depression,\nSuicidal Ideation)'),
        ('Demographics\n(Age, Sex, Race)', 'Peer Influence\n(Social Norms,\nFriend Behaviors)'),
        ('Demographics\n(Age, Sex, Race)', 'Traditional\nSubstance Use\n(Tobacco, Alcohol)'),
        ('Mental Health\n(Depression,\nSuicidal Ideation)', 'Traditional\nSubstance Use\n(Tobacco, Alcohol)'),
        ('Mental Health\n(Depression,\nSuicidal Ideation)', 'VAPING\nINITIATION'),
        ('Peer Influence\n(Social Norms,\nFriend Behaviors)', 'Traditional\nSubstance Use\n(Tobacco, Alcohol)'),
        ('Peer Influence\n(Social Norms,\nFriend Behaviors)', 'VAPING\nINITIATION'),
        ('Risk-Taking\nPropensity', 'Traditional\nSubstance Use\n(Tobacco, Alcohol)'),
        ('Risk-Taking\nPropensity', 'Other Drug Use\n(Marijuana,\nPrescription)'),
        ('Risk-Taking\nPropensity', 'VAPING\nINITIATION'),
        ('Family\nEnvironment', 'Mental Health\n(Depression,\nSuicidal Ideation)'),
        ('Family\nEnvironment', 'Traditional\nSubstance Use\n(Tobacco, Alcohol)'),
        ('School\nEnvironment', 'Mental Health\n(Depression,\nSuicidal Ideation)'),
        ('School\nEnvironment', 'Peer Influence\n(Social Norms,\nFriend Behaviors)'),
        ('Traditional\nSubstance Use\n(Tobacco, Alcohol)', 'Other Drug Use\n(Marijuana,\nPrescription)'),
        ('Traditional\nSubstance Use\n(Tobacco, Alcohol)', 'VAPING\nINITIATION'),
        ('Other Drug Use\n(Marijuana,\nPrescription)', 'VAPING\nINITIATION')
    ]
    
    G.add_edges_from(edges)
    
    # Define node colors based on categories
    node_colors = []
    for node in G.nodes():
        if 'VAPING' in node:
            node_colors.append('#ff4444')  # Red for outcome
        elif any(word in node for word in ['Demographics', 'Family', 'School']):
            node_colors.append('#4444ff')  # Blue for background factors
        elif any(word in node for word in ['Mental Health', 'Peer', 'Risk-Taking']):
            node_colors.append('#44ff44')  # Green for mediating factors
        else:
            node_colors.append('#ffaa44')  # Orange for substance use
    
    # Draw the graph
    nx.draw(G, pos, ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=3000,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            arrowstyle='->',
            node_shape='o')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', 
                   markersize=15, label='Background Factors'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44', 
                   markersize=15, label='Mediating Factors'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaa44', 
                   markersize=15, label='Substance Use'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                   markersize=15, label='Outcome')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    ax.set_title('Conceptual Causal Pathways to Vaping Initiation\n(Based on Texas Youth Risk Behavior Survey Analysis)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/causal_pathways_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Risk Profile Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create risk profile matrix
    risk_factors = ['Ever Cigarette', 'Current Alcohol', 'Ever Marijuana', 
                   'Sad/Hopeless', 'Considered Suicide', 'Violence', 'Unsafe Driving']
    
    risk_data = []
    risk_vars = ['q32', 'q42', 'q46', 'q26', 'q27', 'q16', 'q10']
    
    for i, var in enumerate(risk_vars):
        if var in df.columns:
            risk_indicator = (df[var] == 1).astype(int)
            vaping_by_risk = []
            
            for j, other_var in enumerate(risk_vars):
                if other_var in df.columns and i != j:
                    other_risk = (df[other_var] == 1).astype(int)
                    # Vaping rate when both risk factors present
                    both_risks = ((risk_indicator == 1) & (other_risk == 1))
                    if both_risks.sum() > 10:  # Minimum sample size
                        vaping_rate = df[both_risks]['ever_vaped'].mean()
                    else:
                        vaping_rate = np.nan
                    vaping_by_risk.append(vaping_rate)
                else:
                    vaping_by_risk.append(np.nan)
            
            risk_data.append(vaping_by_risk)
    
    # Create heatmap
    risk_matrix = np.array(risk_data)
    mask = np.isnan(risk_matrix)
    
    im = ax.imshow(risk_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(risk_factors)))
    ax.set_yticks(range(len(risk_factors)))
    ax.set_xticklabels(risk_factors, rotation=45, ha='right')
    ax.set_yticklabels(risk_factors)
    
    # Add text annotations
    for i in range(len(risk_factors)):
        for j in range(len(risk_factors)):
            if not mask[i, j] and i != j:
                text = ax.text(j, i, f'{risk_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if risk_matrix[i, j] < 0.5 else "white")
    
    ax.set_title('Vaping Rates When Multiple Risk Factors Are Present\n(Co-occurrence Matrix)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Vaping Rate', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('output/risk_factor_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    create_causal_visualizations()