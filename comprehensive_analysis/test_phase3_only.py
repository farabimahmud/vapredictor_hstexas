#!/usr/bin/env python3
"""
Test just the interpretability analysis phase
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Load the processed data and model from previous run
try:
    # Try to load saved model and data
    with open('output/best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    best_model = model_data['model']
    X_test = pd.read_csv('output/X_test.csv', index_col=0)
    y_test = pd.read_csv('output/y_test.csv', index_col=0).iloc[:, 0]
    
    print("Loaded saved model and test data")
    print(f"Model type: {type(best_model)}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target prevalence: {y_test.mean():.3f}")
    
except FileNotFoundError:
    print("Could not find saved model files. Please run main_analysis.py first.")
    exit(1)

# Setup interpretability analysis
from interpretability_analysis import InterpretabilityAnalysis
from enhanced_vaping_analysis import ComprehensiveVapingAnalysis

# Load feature domains and variable mapping
analysis = ComprehensiveVapingAnalysis()
analysis.load_and_explore_data()
analysis.create_target_variables()
analysis.prepare_features_for_modeling()

print("\n" + "="*60)
print("TESTING IMPROVED INTERACTION ANALYSIS")
print("="*60)

interp = InterpretabilityAnalysis()
interp.setup_analysis(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    feature_names=X_test.columns.tolist(),
    feature_domains=analysis.feature_domains,
    variable_mapping=analysis.get_variable_mapping()
)

# Run just the interaction analysis
interactions_df = interp.analyze_sociodemographic_interactions()

print(f"\nResults summary:")
print(f"- Total interactions: {len(interactions_df)}")
print(f"- Non-zero interactions: {(interactions_df['interaction_strength'] > 0).sum()}")
print(f"- Max strength: {interactions_df['interaction_strength'].max():.6f}")
print(f"- Mean strength: {interactions_df['interaction_strength'].mean():.6f}")
print(f"- Std strength: {interactions_df['interaction_strength'].std():.6f}")

print("\nTop 5 interactions:")
for i, (_, row) in enumerate(interactions_df.head(5).iterrows(), 1):
    print(f"  {i}. {row['variable_1']} × {row['variable_2']}: {row['interaction_strength']:.6f}")