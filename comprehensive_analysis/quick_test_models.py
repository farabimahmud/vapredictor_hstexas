"""
Quick test script for enhanced models (XGBoost and AdaBoost)
Uses pre-built models for faster testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_vaping_analysis import ComprehensiveVapingAnalysis

warnings.filterwarnings('ignore')

def quick_model_test():
    """
    Quick test of XGBoost and AdaBoost models with minimal tuning
    """
    print("="*80)
    print("QUICK TEST: XGBoost and AdaBoost Models")
    print("="*80)
    
    # Check for processed data
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("Processing fresh data...")
    analyzer = ComprehensiveVapingAnalysis("../hstexas.csv", "../variable_names.csv")
    print("Loading data...")
    analyzer.load_and_explore_data()
    print("Creating target variables...")
    analyzer.create_target_variables()
    print("Preparing features...")
    X, y = analyzer.prepare_features_for_modeling('ever_vaped')
    print("Handling missing data...")
    X = analyzer.handle_missing_data(X, method='simple')
    
    print(f"Data ready: {len(X)} observations, {len(X.columns)} features")
    print(f"Target prevalence: {y.mean():.2%}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")
    
    # Apply SMOTE for class balance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {len(X_train_balanced)} training observations")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    
    # Initialize results storage
    results = {}
    
    # 1. Random Forest (quick)
    print("\n" + "="*60)
    print("1. TRAINING RANDOM FOREST")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    
    rf_model.fit(X_train_balanced, y_train_balanced)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    results['Random Forest'] = {
        'auc': rf_auc,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'model': rf_model
    }
    
    print(f"Random Forest AUC: {rf_auc:.4f}")
    
    # 2. XGBoost
    print("\n" + "="*60)
    print("2. TRAINING XGBOOST")
    print("="*60)
    
    try:
        import xgboost as xgb
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train_balanced, y_train_balanced)
        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        
        results['XGBoost'] = {
            'auc': xgb_auc,
            'predictions': xgb_pred,
            'probabilities': xgb_proba,
            'model': xgb_model
        }
        
        print(f"XGBoost AUC: {xgb_auc:.4f}")
        
    except ImportError:
        print("XGBoost not available, skipping...")
        results['XGBoost'] = None
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        results['XGBoost'] = None
    
    # 3. AdaBoost
    print("\n" + "="*60)
    print("3. TRAINING ADABOOST")
    print("="*60)
    
    try:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        ada_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        
        ada_model.fit(X_train_balanced, y_train_balanced)
        ada_pred = ada_model.predict(X_test)
        ada_proba = ada_model.predict_proba(X_test)[:, 1]
        ada_auc = roc_auc_score(y_test, ada_proba)
        
        results['AdaBoost'] = {
            'auc': ada_auc,
            'predictions': ada_pred,
            'probabilities': ada_proba,
            'model': ada_model
        }
        
        print(f"AdaBoost AUC: {ada_auc:.4f}")
        
    except Exception as e:
        print(f"AdaBoost training failed: {e}")
        results['AdaBoost'] = None
    
    # 4. Logistic Regression (baseline)
    print("\n" + "="*60)
    print("4. TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train_balanced)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    results['Logistic Regression'] = {
        'auc': lr_auc,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'model': lr_model
    }
    
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    
    # Model Comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    comparison_data = []
    for model_name, result in results.items():
        if result is not None:
            y_pred = result['predictions']
            y_proba = result['probabilities']
            
            metrics = {
                'Model': model_name,
                'AUC': roc_auc_score(y_test, y_proba),
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }
            comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    
    print(comparison_df.round(4).to_string(index=False))
    
    # Feature Importance (for tree-based models)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load variable mapping
    try:
        var_df = pd.read_csv("../variable_names.csv")
        variable_mapping = dict(zip(var_df['variable'], var_df['description']))
    except:
        variable_mapping = {}
    
    tree_models = ['Random Forest', 'XGBoost', 'AdaBoost']
    
    for model_name in tree_models:
        if model_name in results and results[model_name] is not None:
            print(f"\n{model_name} - Top 10 Important Features:")
            print("-" * 50)
            
            model = results[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    feature_name = variable_mapping.get(row['feature'], row['feature'])
                    print(f"  {i:2d}. {feature_name}: {row['importance']:.4f}")
                    if feature_name != row['feature']:
                        print(f"      Code: {row['feature']}")
    
    # ROC Curve Plot
    print("\n" + "="*60)
    print("CREATING ROC CURVE VISUALIZATION")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(results.items()):
            if result is not None:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc_score = result['auc']
                
                plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                        label=f'{model_name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quick_model_comparison_roc.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ROC curve saved to output/quick_model_comparison_roc.png")
        
    except Exception as e:
        print(f"Plot creation failed: {e}")
    
    # Save results
    comparison_df.to_csv(output_dir / 'quick_model_comparison.csv', index=False)
    print(f"Results saved to {output_dir / 'quick_model_comparison.csv'}")
    
    print("\n" + "="*80)
    print("QUICK MODEL TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    quick_model_test()