#!/usr/bin/env python3
"""
Quick test of interaction calculation only
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create synthetic data for quick test
np.random.seed(42)
n_samples = 1000
n_features = 9

# Create sociodemographic-like data
X = pd.DataFrame({
    'age': np.random.choice([14, 15, 16, 17, 18], n_samples),
    'sex': np.random.choice([1, 2], n_samples),
    'grade': np.random.choice([9, 10, 11, 12], n_samples),
    'race4': np.random.choice([1, 2, 3, 4], n_samples),
    'race7': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_samples),
    'stheight': np.random.normal(65, 5, n_samples),
    'stweight': np.random.normal(140, 20, n_samples),
    'bmi': np.random.normal(22, 3, n_samples),
    'bmipct': np.random.normal(50, 20, n_samples)
})

# Create synthetic target with some interactions
y = (
    (X['age'] > 16).astype(int) + 
    (X['sex'] == 1).astype(int) + 
    (X['age'] > 16) & (X['sex'] == 1)  # interaction effect
).astype(int)
y = (y > 1).astype(int)  # binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

print("Testing interaction calculation...")
print(f"Data shape: {X_test.shape}")
print(f"Target prevalence: {y_test.mean():.3f}")

# Test the interaction method directly
from interpretability_analysis import InterpretabilityAnalysis

interp = InterpretabilityAnalysis()
interp.model = rf
interp.X_test = X_test
interp.feature_names = X_test.columns.tolist()

# Test a few interactions
test_pairs = [('age', 'sex'), ('stheight', 'stweight'), ('bmi', 'bmipct')]

print("\nTesting individual interactions:")
for var1, var2 in test_pairs:
    strength = interp._calculate_interaction_strength(var1, var2)
    print(f"  {var1} × {var2}: {strength:.6f}")

print("\nDone!")