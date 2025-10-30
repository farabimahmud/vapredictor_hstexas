"""
Test script for enhanced models (XGBoost and AdaBoost)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from advanced_modeling import AdvancedVapingModeling
from enhanced_vaping_analysis import ComprehensiveVapingAnalysis

warnings.filterwarnings('ignore')

def test_enhanced_models():
    """
    Test the enhanced modeling pipeline with XGBoost and AdaBoost
    """
    print("="*80)
    print("TESTING ENHANCED MODELS (XGBoost and AdaBoost)")
    print("="*80)
    
    try:
        # Load processed data if it exists, otherwise process fresh
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        processed_data_path = output_dir / "final_analysis_data.csv"
        
        print("Checking for processed data...")
        if processed_data_path.exists():
            print("Loading processed data...")
            df = pd.read_csv(processed_data_path)
            
            # Separate features and target
            target_col = 'ever_vaped'  # or whichever target was used
            if target_col not in df.columns:
                # Try to find the target column
                possible_targets = ['ever_vaped', 'frequent_vaping', 'vaping_target']
                for target in possible_targets:
                    if target in df.columns:
                        target_col = target
                        break
            
            if target_col not in df.columns:
                print("Error: No target variable found in processed data")
                print(f"Available columns: {list(df.columns)}")
                return
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            feature_names = list(X.columns)
            
        else:
            print("Processing fresh data...")
            # Use the analysis pipeline to get data
            analyzer = ComprehensiveVapingAnalysis("../hstexas.csv", "../variable_names.csv")
            print("Loading data...")
            analyzer.load_and_explore_data()
            print("Creating target variables...")
            analyzer.create_target_variables()
            print("Preparing features...")
            X, y = analyzer.prepare_features_for_modeling('ever_vaped')
            print("Handling missing data...")
            X = analyzer.handle_missing_data(X, method='simple')
            feature_names = list(X.columns)
        
        print(f"Data loaded: {len(X)} observations, {len(X.columns)} features")
        print(f"Target prevalence: {y.mean():.2%}")
        
    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize modeling pipeline
    modeling = AdvancedVapingModeling(output_dir="output")
    
    # Prepare training data - set the data in the modeling object first
    modeling.prepare_training_data(X, y, feature_domains={}, test_size=0.4, use_smote=True)
    
    print(f"Training set: {len(modeling.X_train)} observations")
    print(f"Test set: {len(modeling.X_test)} observations")
    
    # 1. Train Random Forest (tuned)
    print("\n" + "="*60)
    print("1. TRAINING RANDOM FOREST (TUNED)")
    print("="*60)
    
    # Quick tuning for demo (fewer iterations)
    best_rf = modeling.tune_random_forest(cv_folds=5)
    
    # 2. Train XGBoost (tuned)
    print("\n" + "="*60)
    print("2. TRAINING XGBOOST (TUNED)")
    print("="*60)
    
    try:
        best_xgb = modeling.tune_xgboost(cv_folds=5)
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        print("Continuing without XGBoost...")
    
    # 3. Train AdaBoost (tuned)
    print("\n" + "="*60)
    print("3. TRAINING ADABOOST (TUNED)")
    print("="*60)
    
    try:
        best_ada = modeling.tune_adaboost(cv_folds=5)
    except Exception as e:
        print(f"AdaBoost training failed: {e}")
        print("Continuing without AdaBoost...")
    
    # 4. Train comparison models
    print("\n" + "="*60)
    print("4. TRAINING COMPARISON MODELS")
    print("="*60)
    
    comparison_models = modeling.train_comparison_models()
    
    # 5. Evaluate all models
    print("\n" + "="*60)
    print("5. EVALUATING ALL MODELS")
    print("="*60)
    
    results = modeling.evaluate_all_models()
    
    # 6. Create model comparison
    print("\n" + "="*60)
    print("6. MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Sort models by AUC
    sorted_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    print(f"{'Model':<25} {'AUC':<8} {'Sensitivity':<12} {'Specificity':<12} {'PPV':<8} {'NPV':<8}")
    print("-" * 80)
    
    for model_name, metrics in sorted_models:
        print(f"{model_name:<25} {metrics['auc']:<8.4f} {metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f} {metrics['ppv']:<8.4f} {metrics['npv']:<8.4f}")
    
    # 7. Feature importance analysis
    print("\n" + "="*60)
    print("7. FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load variable mapping if available
    try:
        var_mapping_path = Path("../variable_names.csv")
        if var_mapping_path.exists():
            var_df = pd.read_csv(var_mapping_path)
            variable_mapping = dict(zip(var_df['variable'], var_df['description']))
        else:
            variable_mapping = None
    except:
        variable_mapping = None
    
    # Analyze feature importance for tree-based models
    tree_models = ['Random_Forest_Tuned', 'XGBoost_Tuned', 'AdaBoost_Tuned']
    
    for model_name in tree_models:
        if model_name in modeling.models:
            print(f"\n{model_name} Feature Importance:")
            print("-" * 40)
            modeling.analyze_feature_importance(model_name, top_n=15, variable_mapping=variable_mapping)
    
    # 8. Create visualizations
    print("\n" + "="*60)
    print("8. CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        modeling.plot_model_comparison()
        modeling.plot_roc_curves()
        print("Visualizations saved to output directory")
    except Exception as e:
        print(f"Visualization creation failed: {e}")
    
    print("\n" + "="*80)
    print("ENHANCED MODEL TESTING COMPLETED")
    print("="*80)
    
    # Save results summary
    results_summary = pd.DataFrame({
        model: {
            'AUC': metrics['auc'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'PPV': metrics['ppv'],
            'NPV': metrics['npv'],
            'Avg_Precision': metrics['avg_precision'],
            'Brier_Score': metrics['brier_score']
        }
        for model, metrics in results.items()
    }).T
    
    results_summary.to_csv(output_dir / 'enhanced_models_comparison.csv')
    print(f"Results summary saved to {output_dir / 'enhanced_models_comparison.csv'}")


if __name__ == "__main__":
    test_enhanced_models()