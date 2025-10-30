#!/usr/bin/env python3
"""
Quick test of the improved interaction analysis
"""

import pandas as pd
import numpy as np
from enhanced_vaping_analysis import ComprehensiveVapingAnalysis
from advanced_modeling import AdvancedVapingModeling
from interpretability_analysis import InterpretabilityAnalysis

def test_interactions():
    print("="*80)
    print("TESTING IMPROVED INTERACTION ANALYSIS")
    print("="*80)
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    analysis = ComprehensiveVapingAnalysis()
    analysis.load_and_explore_data()
    analysis.create_target_variables()
    X, y = analysis.prepare_features_for_modeling(target_variable='ever_vaped')
    analysis.handle_missing_data(method='simple')
    
    # Train a quick model
    print("\n2. Training model...")
    modeling = AdvancedVapingModeling(analysis)
    modeling.prepare_data(test_size=0.3)
    
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(modeling.X_train, modeling.y_train)
    
    # Test interactions
    print("\n3. Testing interaction analysis...")
    interp = InterpretabilityAnalysis()
    interp.setup_analysis(
        model=rf_model,
        X_test=modeling.X_test,
        y_test=modeling.y_test,
        feature_names=modeling.X_test.columns.tolist(),
        feature_domains=analysis.feature_domains,
        variable_mapping=analysis.get_variable_mapping()
    )
    
    # Run interaction analysis
    interactions_df = interp.analyze_sociodemographic_interactions()
    
    print(f"\n4. Results summary:")
    print(f"   - Total interactions analyzed: {len(interactions_df)}")
    print(f"   - Non-zero interactions: {(interactions_df['interaction_strength'] > 0).sum()}")
    print(f"   - Max interaction strength: {interactions_df['interaction_strength'].max():.6f}")
    print(f"   - Mean interaction strength: {interactions_df['interaction_strength'].mean():.6f}")
    
    return interactions_df

if __name__ == "__main__":
    interactions = test_interactions()