"""
Quick Demo Version of the Comprehensive Analysis
This version uses simplified parameters for faster execution
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_vaping_analysis import ComprehensiveVapingAnalysis

warnings.filterwarnings('ignore')

def run_quick_demo():
    """Run a quick demonstration version of the analysis"""
    
    print("="*80)
    print("QUICK DEMO: COMPREHENSIVE VAPING PREDICTION ANALYSIS")
    print("Simplified Version for Fast Execution")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ComprehensiveVapingAnalysis()
    
    # Phase 1: Data preparation
    print("\n🔄 Phase 1: Data Preparation")
    print("-" * 40)
    
    df_raw = analyzer.load_and_explore_data()
    df_processed = analyzer.create_target_variables()
    X, y = analyzer.prepare_features_for_modeling()
    X_imputed = analyzer.handle_missing_data(X, method='simple')
    
    print(f"✅ Dataset prepared: {len(X)} observations, {len(X.columns)} features")
    print(f"✅ Target prevalence: {y.mean():.2%}")
    print(f"✅ Feature domains: {len(analyzer.feature_domains)}")
    
    # Phase 2: Quick modeling
    print("\n🤖 Phase 2: Quick Modeling")
    print("-" * 40)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from imblearn.over_sampling import SMOTE
    
    # Split data (60/40)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"✅ Train/test split: {len(X_train)}/{len(X_test)}")
    print(f"✅ SMOTE applied: {len(X_train_balanced)} balanced training samples")
    
    # Train models
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=10,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_balanced, y_train_balanced)
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'auc': auc_rf,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf
    }
    
    print(f"    Random Forest AUC: {auc_rf:.4f}")
    
    # 2. Logistic Regression
    print("  Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train_balanced)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'auc': auc_lr,
        'y_pred': y_pred_lr,
        'y_proba': y_proba_lr
    }
    
    print(f"    Logistic Regression AUC: {auc_lr:.4f}")
    
    # Phase 3: Feature Importance
    print("\n🔍 Phase 3: Feature Importance")
    print("-" * 40)
    
    # Get feature importance from Random Forest
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        readable_name = analyzer.variable_mapping.get(row['feature'], row['feature'])
        domain = analyzer._get_feature_domain(row['feature'])
        print(f"  {i:2d}. {row['feature']} ({domain}): {row['importance']:.4f}")
        print(f"      {readable_name}")
    
    # Domain-wise importance
    print("\nImportance by Domain:")
    domain_importance = {}
    for domain, features in analyzer.feature_domains.items():
        domain_features = [f for f in features if f in X.columns]
        if domain_features:
            domain_indices = [X.columns.get_loc(f) for f in domain_features]
            domain_importance[domain] = rf.feature_importances_[domain_indices].sum()
    
    sorted_domains = sorted(domain_importance.items(), key=lambda x: x[1], reverse=True)
    for domain, importance in sorted_domains:
        print(f"  {domain}: {importance:.4f}")
    
    # Phase 4: Simple Visualization
    print("\n📊 Phase 4: Quick Results Summary")
    print("-" * 40)
    
    # Model comparison
    print("Model Performance Comparison:")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name:20s}: AUC = {result['auc']:.4f}")
    
    # Best model confusion matrix
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    best_result = results[best_model]
    
    print(f"\nBest Model: {best_model}")
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Yes  {cm[1,0]:6d}  {cm[1,1]:5d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, best_result['y_pred'], 
                              target_names=['No Vaping', 'Vaping']))
    
    # Summary
    print("\n" + "="*80)
    print("QUICK DEMO COMPLETE!")
    print("="*80)
    print(f"✅ Successfully analyzed {len(X):,} observations")
    print(f"✅ Used {len(X.columns)} features across {len(analyzer.feature_domains)} domains")
    print(f"✅ Achieved {results[best_model]['auc']:.1%} AUC with {best_model}")
    print(f"✅ Most important domain: {sorted_domains[0][0]}")
    print("\n🚀 Ready for full comprehensive analysis!")
    print("   Run: python main_analysis.py (with faster parameters)")
    print("="*80)

if __name__ == "__main__":
    run_quick_demo()