"""
Advanced Modeling and Interpretability Analysis Module
Implements sophisticated modeling approaches with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, classification_report, 
    confusion_matrix, precision_recall_curve, average_precision_score,
    brier_score_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class AdvancedVapingModeling:
    """
    Advanced modeling with focus on Random Forest following literature best practices
    """
    
    def __init__(self, output_dir: str = "output", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Model containers
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.interaction_analysis = {}
        
        # Data containers  
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.feature_domains = None
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def prepare_training_data(self, X: pd.DataFrame, y: pd.Series, 
                            feature_domains: Dict[str, List[str]] = None,
                            test_size: float = 0.4, use_smote: bool = True) -> None:
        """
        Prepare training and test data with optional SMOTE oversampling
        """
        print(f"Preparing training data (test_size={test_size})...")
        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.feature_domains = feature_domains or {}
        
        # Train-test split (60/40 as in reference study)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Initial split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"Train prevalence: {self.y_train.mean():.2%}")
        print(f"Test prevalence: {self.y_test.mean():.2%}")
        
        # Apply SMOTE to training data if requested
        if use_smote:
            print("Applying SMOTE oversampling to training data...")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"After SMOTE - Train: {len(self.X_train)}")
            print(f"SMOTE train prevalence: {self.y_train.mean():.2%}")
    
    def tune_random_forest(self, cv_folds: int = 10) -> RandomForestClassifier:
        """
        Tune Random Forest hyperparameters using 10-fold CV as in reference study
        """
        print(f"Tuning Random Forest hyperparameters with {cv_folds}-fold CV...")
        
        # Define parameter grid focusing on key parameters from literature
        # Simplified for faster execution
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        # Base model
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Grid search with stratified CV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            rf_base, param_grid, 
            cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Store best model
        best_rf = grid_search.best_estimator_
        self.models['Random_Forest_Tuned'] = best_rf
        
        return best_rf
    
    def train_comparison_models(self) -> Dict[str, Any]:
        """
        Train comparison models as in sensitivity analysis
        """
        print("Training comparison models...")
        
        models = {}
        
        # 1. Simple logistic regression (sociodemographics only)
        if self.feature_domains and 'demographics' in self.feature_domains:
            demo_features = [f for f in self.feature_domains['demographics'] if f in self.feature_names]
            if demo_features:
                X_train_demo = self.X_train[demo_features]
                X_test_demo = self.X_test[demo_features]
                
                # Scale features for logistic regression
                scaler = StandardScaler()
                X_train_demo_scaled = scaler.fit_transform(X_train_demo)
                X_test_demo_scaled = scaler.transform(X_test_demo)
                
                lr_demo = LogisticRegression(random_state=self.random_state, max_iter=1000)
                lr_demo.fit(X_train_demo_scaled, self.y_train)
                
                models['Logistic_Demographics'] = {
                    'model': lr_demo,
                    'scaler': scaler,
                    'features': demo_features,
                    'X_test': X_test_demo_scaled
                }
        
        # 2. Full logistic regression
        scaler_full = StandardScaler()
        X_train_scaled = scaler_full.fit_transform(self.X_train)
        X_test_scaled = scaler_full.transform(self.X_test)
        
        lr_full = LogisticRegression(random_state=self.random_state, max_iter=1000)
        lr_full.fit(X_train_scaled, self.y_train)
        
        models['Logistic_Full'] = {
            'model': lr_full,
            'scaler': scaler_full,
            'features': self.feature_names,
            'X_test': X_test_scaled
        }
        
        # 3. Default Random Forest (for comparison)
        rf_default = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        rf_default.fit(self.X_train, self.y_train)
        
        models['Random_Forest_Default'] = {
            'model': rf_default,
            'features': self.feature_names,
            'X_test': self.X_test
        }
        
        # Store comparison models
        self.models.update(models)
        
        return models
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of all models
        """
        print("Evaluating all models...")
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\nEvaluating {name}...")
            
            if isinstance(model_info, dict):
                model = model_info['model']
                X_test = model_info['X_test']
            else:
                model = model_info
                X_test = self.X_test
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_proba)
            avg_precision = average_precision_score(self.y_test, y_proba)
            brier = brier_score_loss(self.y_test, y_proba)
            
            # Classification metrics
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # C-index (same as AUC for binary classification)
            c_index = auc_score
            
            # Store results
            results[name] = {
                'auc': auc_score,
                'c_index': c_index,
                'avg_precision': avg_precision,
                'brier_score': brier,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"  AUC: {auc_score:.4f}")
            print(f"  C-index: {c_index:.4f}")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Brier Score: {brier:.4f}")
        
        self.results = results
        return results
    
    def analyze_feature_importance(self, model_name: str = 'Random_Forest_Tuned', top_n: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance with mean decrease in accuracy approach
        """
        print(f"Analyzing feature importance for {model_name}...")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return pd.DataFrame()
        
        model_info = self.models[model_name]
        if isinstance(model_info, dict):
            model = model_info['model']
            features = model_info['features']
        else:
            model = model_info
            features = self.feature_names
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Add domain information if available
        if self.feature_domains:
            importance_df['domain'] = importance_df['feature'].apply(self._get_feature_domain)
        
        # Store importance
        self.feature_importance[model_name] = importance_df
        
        print(f"Top {top_n} most important features:")
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            domain = f" ({row['domain']})" if 'domain' in row else ""
            print(f"  {i:2d}. {row['feature']}{domain}: {row['importance']:.4f}")
        
        return importance_df
    
    def _get_feature_domain(self, feature: str) -> str:
        """Get domain for a feature"""
        for domain, features in self.feature_domains.items():
            if feature in features:
                return domain
        return 'other'
    
    def plot_model_comparison(self) -> None:
        """
        Create comprehensive model comparison plots
        """
        print("Creating model comparison visualizations...")
        
        if not self.results:
            print("No results available. Run evaluate_all_models() first.")
            return
        
        # Setup plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['auc', 'avg_precision', 'sensitivity', 'specificity', 'ppv', 'npv']
        metric_labels = ['AUC', 'Avg Precision', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        
        # Plot each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i // 3, i % 3]
            
            values = [self.results[model][metric] for model in models]
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        
    
    def plot_roc_curves(self) -> None:
        """Plot ROC curves for all models"""
        print("Creating ROC curve comparison...")
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for i, (name, results) in enumerate(self.results.items()):
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, results['y_proba'])
            auc_score = results['auc']
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i], linewidth=2.5, 
                    label=f'{name.replace("_", " ")} (AUC = {auc_score:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, 
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves Comparison - Enhanced Vaping Prediction Models', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves_enhanced.png', 
                   dpi=300, bbox_inches='tight')
        
    
    def plot_calibration_curves(self) -> None:
        """Plot calibration curves for model reliability assessment"""
        print("Creating calibration curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plot
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for i, (name, results) in enumerate(self.results.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, results['y_proba'], n_bins=10
            )
            
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    color=colors[i], label=name.replace("_", " "), linewidth=2)
        
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Brier score comparison
        names = [n.replace("_", " ") for n in self.results.keys()]
        brier_scores = [self.results[n]['brier_score'] for n in self.results.keys()]
        
        bars = ax2.bar(range(len(names)), brier_scores, color=colors[:len(names)], alpha=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Brier Score (lower is better)')
        ax2.set_title('Brier Score Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, brier_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_analysis.png', 
                   dpi=300, bbox_inches='tight')
        
    
    def create_performance_summary_table(self) -> pd.DataFrame:
        """Create comprehensive performance summary table"""
        print("Creating performance summary table...")
        
        # Prepare summary data
        summary_data = []
        for name, results in self.results.items():
            row = {
                'Model': name.replace('_', ' '),
                'AUC': f"{results['auc']:.4f}",
                'C-Index': f"{results['c_index']:.4f}",
                'Avg Precision': f"{results['avg_precision']:.4f}",
                'Sensitivity': f"{results['sensitivity']:.4f}",
                'Specificity': f"{results['specificity']:.4f}",
                'PPV': f"{results['ppv']:.4f}",
                'NPV': f"{results['npv']:.4f}",
                'Brier Score': f"{results['brier_score']:.4f}"
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv(self.output_dir / 'model_performance_summary.csv', index=False)
        
        print("Performance Summary:")
        print(summary_df.to_string(index=False))
        
        return summary_df


if __name__ == "__main__":
    print("Advanced Modeling Module")
    print("Use this module in conjunction with enhanced_vaping_analysis.py")