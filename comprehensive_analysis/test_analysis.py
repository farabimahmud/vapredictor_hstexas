"""
Test and Demo Script for Comprehensive Vaping Analysis
This script tests the basic functionality and provides a quick demo
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

def test_data_loading():
    """Test basic data loading functionality"""
    print("Testing data loading...")
    
    try:
        from enhanced_vaping_analysis import ComprehensiveVapingAnalysis
        
        analyzer = ComprehensiveVapingAnalysis()
        
        # Test data loading
        df = analyzer.load_and_explore_data()
        print(f"✓ Data loaded successfully: {df.shape}")
        
        # Test variable mapping
        mapping = analyzer.load_variable_mapping()
        print(f"✓ Variable mapping loaded: {len(mapping)} variables")
        
        # Test feature categorization
        domains = analyzer.categorize_features_by_domain()
        print(f"✓ Feature domains created: {len(domains)} domains")
        
        # Test target creation
        df_processed = analyzer.create_target_variables()
        print(f"✓ Target variables created")
        
        # Test feature preparation
        X, y = analyzer.prepare_features_for_modeling()
        print(f"✓ Features prepared: {X.shape}, target: {y.shape}")
        print(f"  Target prevalence: {y.mean():.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {str(e)}")
        return False

def test_modeling_setup():
    """Test modeling functionality"""
    print("\nTesting modeling setup...")
    
    try:
        from advanced_modeling import AdvancedVapingModeling
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 50
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.binomial(1, 0.3, n_samples))
        
        modeler = AdvancedVapingModeling()
        
        # Test data preparation
        modeler.prepare_training_data(X, y, test_size=0.4, use_smote=False)
        print(f"✓ Training data prepared: {modeler.X_train.shape}")
        
        # Test basic model training (without hyperparameter tuning)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(modeler.X_train, modeler.y_train)
        
        modeler.models['test_rf'] = rf
        print(f"✓ Basic model training successful")
        
        # Test evaluation
        results = modeler.evaluate_all_models()
        print(f"✓ Model evaluation completed: {len(results)} models")
        
        return True
        
    except Exception as e:
        print(f"✗ Modeling test failed: {str(e)}")
        return False

def test_interpretability_setup():
    """Test interpretability functionality"""
    print("\nTesting interpretability setup...")
    
    try:
        from interpretability_analysis import InterpretabilityAnalysis
        from sklearn.ensemble import RandomForestClassifier
        
        # Create sample data and model
        np.random.seed(42)
        n_samples, n_features = 500, 20
        X_test = pd.DataFrame(np.random.randn(n_samples, n_features), 
                             columns=[f'feature_{i}' for i in range(n_features)])
        y_test = pd.Series(np.random.binomial(1, 0.3, n_samples))
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)  # Using test data for simplicity
        
        interpreter = InterpretabilityAnalysis()
        
        # Test setup
        interpreter.setup_analysis(model, X_test, y_test)
        print(f"✓ Interpretability analysis setup successful")
        
        # Test feature importance analysis
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"✓ Feature importance calculated: {len(importance_df)} features")
        
        # Test interaction analysis setup
        interactions = interpreter.analyze_sociodemographic_interactions()
        print(f"✓ Interaction analysis completed: {len(interactions)} pairs")
        
        return True
        
    except Exception as e:
        print(f"✗ Interpretability test failed: {str(e)}")
        return False

def run_quick_demo():
    """Run a quick demonstration with actual data"""
    print("\n" + "="*60)
    print("RUNNING QUICK DEMONSTRATION")
    print("="*60)
    
    try:
        from enhanced_vaping_analysis import ComprehensiveVapingAnalysis
        
        # Initialize analyzer
        analyzer = ComprehensiveVapingAnalysis()
        
        # Quick data preparation
        print("1. Loading and preparing data...")
        df = analyzer.load_and_explore_data()
        df_processed = analyzer.create_target_variables()
        X, y = analyzer.prepare_features_for_modeling()
        X_imputed = analyzer.handle_missing_data(X, method='simple')
        
        print(f"   Dataset: {len(X)} observations, {len(X.columns)} features")
        print(f"   Target prevalence: {y.mean():.2%}")
        print(f"   Feature domains: {len(analyzer.feature_domains)}")
        
        # Quick modeling
        print("\n2. Quick modeling demonstration...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, classification_report
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"   Model trained successfully")
        print(f"   Test AUC: {auc:.4f}")
        
        # Feature importance
        print("\n3. Top 10 most important features:")
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            readable_name = analyzer.variable_mapping.get(row['feature'], row['feature'])
            print(f"   {i:2d}. {row['feature']} ({readable_name}): {row['importance']:.4f}")
        
        print(f"\n✓ Quick demonstration completed successfully!")
        print(f"   Ready to run full comprehensive analysis.")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick demo failed: {str(e)}")
        return False

def check_requirements():
    """Check if required packages are available"""
    print("Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages available")
        return True

def main():
    """Run all tests and demo"""
    print("="*60)
    print("COMPREHENSIVE VAPING ANALYSIS - TEST SUITE")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Install missing packages first.")
        return
    
    # Run tests
    test_results = []
    
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Modeling Setup", test_modeling_setup()))
    test_results.append(("Interpretability", test_interpretability_setup()))
    
    # Run demo if tests pass
    if all(result for _, result in test_results):
        print("\n🎉 All tests passed!")
        demo_success = run_quick_demo()
        test_results.append(("Quick Demo", demo_success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(result for _, result in test_results)
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🚀 System ready for comprehensive analysis!")
        print("Run: python main_analysis.py")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")

if __name__ == "__main__":
    main()