import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_vaping_data():
    """Explore the vaping dataset to understand structure and relationships."""
    
    # Load data
    df = pd.read_csv("data/cleaned_hstexas_full.csv")
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Check for vaping variables
    vaping_cols = []
    for col in df.columns:
        if any(term in col.lower() for term in ['vap', '35', '36']):
            vaping_cols.append(col)
    
    print(f"\nPotential vaping columns: {vaping_cols}")
    
    # Analyze q35 and q36 (vaping questions)
    if 'q35' in df.columns:
        print(f"\nq35 (Ever used electronic vapor) distribution:")
        print(df['q35'].value_counts(dropna=False))
        ever_vaping_rate = (df['q35'] == 1).mean()
        print(f"Ever vaping rate: {ever_vaping_rate:.2%}")
    
    if 'q36' in df.columns:
        print(f"\nq36 (Current electronic vapor use) distribution:")
        print(df['q36'].value_counts(dropna=False))
        # Note: In YRBS, 1=Yes for current use, but check the actual coding
        current_vaping_rate = (df['q36'] == 1).mean()
        print(f"Current vaping rate (1=Yes): {current_vaping_rate:.2%}")
        
        # Alternative: if 1 means "never" then current users might be 2-7
        alt_current_rate = (df['q36'] > 1).mean()
        print(f"Alternative current vaping rate (>1): {alt_current_rate:.2%}")
    
    # Create target variable
    if 'q35' in df.columns:
        df['ever_vaped'] = (df['q35'] == 1).astype(int)
        
    # Analyze missing data pattern
    print(f"\nMissing data analysis:")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 20]
    print(f"Variables with >20% missing: {len(high_missing)}")
    
    # Analyze key demographics
    print(f"\nKey demographics:")
    demographic_vars = ['age', 'sex', 'grade', 'race4']
    for var in demographic_vars:
        if var in df.columns:
            print(f"{var}: {df[var].value_counts(dropna=False).to_dict()}")
    
    # Correlation with other substance use
    if 'ever_vaped' in df.columns:
        substance_vars = ['q32', 'q33', 'q42', 'q46', 'q48']  # tobacco, alcohol, marijuana
        correlations = []
        
        for var in substance_vars:
            if var in df.columns:
                # Convert to binary for correlation
                binary_var = (df[var] == 1).astype(int)
                corr = df['ever_vaped'].corr(binary_var)
                correlations.append((var, corr))
        
        print(f"\nCorrelations with vaping:")
        for var, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {var}: {corr:.3f}")
    
    # Basic visualization setup
    plt.style.use('default')
    
    # Plot vaping rates by demographics
    if 'ever_vaped' in df.columns and 'grade' in df.columns:
        plt.figure(figsize=(10, 6))
        
        grade_vaping = df.groupby('grade')['ever_vaped'].mean()
        plt.subplot(1, 2, 1)
        grade_vaping.plot(kind='bar')
        plt.title('Vaping Rate by Grade')
        plt.ylabel('Ever Vaping Rate')
        plt.xticks(rotation=0)
        
        if 'sex' in df.columns:
            plt.subplot(1, 2, 2)
            sex_vaping = df.groupby('sex')['ever_vaped'].mean()
            sex_vaping.plot(kind='bar')
            plt.title('Vaping Rate by Sex')
            plt.ylabel('Ever Vaping Rate')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('output/vaping_demographics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    df = explore_vaping_data()