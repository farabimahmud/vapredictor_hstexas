"""
Interpretability and Interaction Analysis Module
Implements SHAP analysis, partial dependence plots, and intersectionality analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

class InterpretabilityAnalysis:
    """
    Advanced interpretability analysis including:
    - SHAP values for model explanation
    - Partial dependence plots for top predictors
    - Interaction analysis for intersectionality
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis containers
        self.shap_values = {}
        self.pdp_data = {}
        self.interaction_strengths = {}
        
        # Model and data references
        self.model = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.feature_domains = None
        
    def setup_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      feature_domains: Dict[str, List[str]] = None) -> None:
        """Setup analysis with model and test data"""
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = X_test.columns.tolist()
        self.feature_domains = feature_domains or {}
        
        print(f"Interpretability analysis setup complete")
        print(f"- Model type: {type(model).__name__}")
        print(f"- Test data: {X_test.shape}")
        print(f"- Feature domains: {len(self.feature_domains)}")
    
    def calculate_shap_values(self, sample_size: int = 1000) -> np.ndarray:
        """
        Calculate SHAP values for model interpretability
        
        Note: This is a placeholder implementation. In practice, you would use:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        """
        print(f"Calculating SHAP values (sample size: {sample_size})...")
        
        # Sample data for efficiency
        if len(self.X_test) > sample_size:
            sample_idx = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_idx]
        else:
            X_sample = self.X_test
            sample_idx = np.arange(len(self.X_test))
        
        # Placeholder: In real implementation, use SHAP library
        # For now, create mock SHAP values based on feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            # Mock SHAP values: random values weighted by importance
            np.random.seed(42)
            mock_shap = np.random.randn(len(X_sample), len(self.feature_names))
            # Weight by feature importance
            mock_shap = mock_shap * importance.reshape(1, -1)
            # Scale to reasonable range
            mock_shap = mock_shap * 0.1
            
            self.shap_values['mock'] = {
                'values': mock_shap,
                'data': X_sample,
                'feature_names': self.feature_names
            }
            
            print("SHAP values calculated (mock implementation)")
            print("Note: Replace with actual SHAP library for production use")
            
            return mock_shap
        else:
            print("Model does not have feature_importances_ attribute")
            return None
    
    def plot_shap_summary(self, max_display: int = 20) -> None:
        """Plot SHAP summary plots"""
        if 'mock' not in self.shap_values:
            print("SHAP values not calculated. Run calculate_shap_values() first.")
            return
        
        print("Creating SHAP summary plots...")
        
        shap_data = self.shap_values['mock']
        shap_vals = shap_data['values']
        feature_names = shap_data['feature_names']
        
        # Calculate mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Create importance dataframe
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=True)
        
        # Plot top features
        top_features = shap_importance.tail(max_display)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['mean_abs_shap'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value| (Feature Importance)')
        plt.title(f'SHAP Feature Importance - Top {max_display} Features', 
                 fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['mean_abs_shap'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shap_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return shap_importance
    
    def create_partial_dependence_plots(self, top_n_features: int = 10) -> None:
        """
        Create partial dependence plots for top N most important features
        """
        print(f"Creating partial dependence plots for top {top_n_features} features...")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(top_n_features)['feature'].tolist()
        else:
            print("Model does not have feature importance. Using first 10 features.")
            top_features = self.feature_names[:top_n_features]
        
        # Create PDP plots
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(top_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get feature index
            feature_idx = self.feature_names.index(feature)
            
            # Create simple PDP (mock implementation)
            feature_values = self.X_test[feature].values
            unique_values = np.percentile(feature_values, np.linspace(0, 100, 20))
            
            # Mock PDP calculation (replace with sklearn.inspection.partial_dependence)
            pdp_values = self._mock_partial_dependence(feature_idx, unique_values)
            
            # Plot
            ax.plot(unique_values, pdp_values, 'b-', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel(feature)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'PDP: {feature}', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(top_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Partial Dependence Plots - Top Predictors', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'partial_dependence_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _mock_partial_dependence(self, feature_idx: int, feature_values: np.ndarray) -> np.ndarray:
        """
        Mock partial dependence calculation
        Replace with actual sklearn.inspection.partial_dependence in production
        """
        # Create copy of test data
        X_pdp = self.X_test.copy()
        pdp_values = []
        
        for value in feature_values:
            # Set all instances to this feature value
            X_pdp.iloc[:, feature_idx] = value
            
            # Get predictions
            y_pred = self.model.predict_proba(X_pdp)[:, 1]
            
            # Average prediction is the partial dependence
            pdp_values.append(y_pred.mean())
        
        return np.array(pdp_values)
    
    def analyze_sociodemographic_interactions(self) -> pd.DataFrame:
        """
        Analyze pairwise interactions between sociodemographic variables
        for intersectionality analysis
        """
        print("Analyzing sociodemographic interactions for intersectionality...")
        
        # Define sociodemographic variables
        sociodem_vars = []
        
        # Get demographics from feature domains
        if 'demographics' in self.feature_domains:
            sociodem_vars.extend([v for v in self.feature_domains['demographics'] 
                                if v in self.feature_names])
        
        # Add other relevant sociodemographic variables
        additional_sociodem = ['sexid', 'sexid2', 'race4', 'race7', 'grade']
        sociodem_vars.extend([v for v in additional_sociodem 
                            if v in self.feature_names and v not in sociodem_vars])
        
        # Limit to first 9 for computational efficiency (as in reference study)
        sociodem_vars = sociodem_vars[:9]
        
        print(f"Analyzing interactions among {len(sociodem_vars)} sociodemographic variables:")
        for var in sociodem_vars:
            print(f"  - {var}")
        
        # Calculate pairwise interaction strengths
        interaction_results = []
        
        for var1, var2 in combinations(sociodem_vars, 2):
            interaction_strength = self._calculate_interaction_strength(var1, var2)
            interaction_results.append({
                'variable_1': var1,
                'variable_2': var2,
                'interaction_strength': interaction_strength
            })
        
        # Create results dataframe
        interactions_df = pd.DataFrame(interaction_results)
        interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
        
        # Store results
        self.interaction_strengths['sociodemographic'] = interactions_df
        
        print(f"\nTop 10 strongest interactions:")
        for i, (_, row) in enumerate(interactions_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['variable_1']} × {row['variable_2']}: "
                  f"{row['interaction_strength']:.4f}")
        
        return interactions_df
    
    def _calculate_interaction_strength(self, var1: str, var2: str) -> float:
        """
        Calculate variance-based interaction strength between two variables
        """
        # Get feature indices
        idx1 = self.feature_names.index(var1)
        idx2 = self.feature_names.index(var2)
        
        # Get unique values for each variable
        vals1 = np.unique(self.X_test.iloc[:, idx1])
        vals2 = np.unique(self.X_test.iloc[:, idx2])
        
        # Limit combinations for computational efficiency
        if len(vals1) > 5:
            vals1 = np.percentile(vals1, [0, 25, 50, 75, 100])
        if len(vals2) > 5:
            vals2 = np.percentile(vals2, [0, 25, 50, 75, 100])
        
        # Calculate predictions for all combinations
        predictions = []
        X_interaction = self.X_test.copy()
        
        for v1 in vals1:
            for v2 in vals2:
                # Set variables to specific values
                X_interaction.iloc[:, idx1] = v1
                X_interaction.iloc[:, idx2] = v2
                
                # Get mean prediction
                y_pred = self.model.predict_proba(X_interaction)[:, 1]
                predictions.append(y_pred.mean())
        
        # Calculate variance as interaction strength
        interaction_strength = np.var(predictions)
        
        return interaction_strength
    
    def plot_strongest_interactions(self, n_interactions: int = 2) -> None:
        """
        Create two-way partial dependence plots for strongest interactions
        """
        if 'sociodemographic' not in self.interaction_strengths:
            print("Interaction analysis not completed. Run analyze_sociodemographic_interactions() first.")
            return
        
        print(f"Creating two-way PDP for {n_interactions} strongest interactions...")
        
        interactions_df = self.interaction_strengths['sociodemographic']
        top_interactions = interactions_df.head(n_interactions)
        
        fig, axes = plt.subplots(1, n_interactions, figsize=(8*n_interactions, 6))
        if n_interactions == 1:
            axes = [axes]
        
        for i, (_, interaction) in enumerate(top_interactions.iterrows()):
            var1, var2 = interaction['variable_1'], interaction['variable_2']
            strength = interaction['interaction_strength']
            
            # Create 2D interaction plot
            self._plot_2d_interaction(var1, var2, axes[i])
            axes[i].set_title(f'{var1} × {var2}\nInteraction Strength: {strength:.4f}',
                            fontweight='bold')
        
        plt.suptitle('Two-Way Partial Dependence Plots - Strongest Interactions',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'interaction_pdp_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_2d_interaction(self, var1: str, var2: str, ax: plt.Axes) -> None:
        """Plot 2D interaction effect"""
        # Get feature indices
        idx1 = self.feature_names.index(var1)
        idx2 = self.feature_names.index(var2)
        
        # Create grid of values
        vals1 = np.percentile(self.X_test.iloc[:, idx1], np.linspace(0, 100, 10))
        vals2 = np.percentile(self.X_test.iloc[:, idx2], np.linspace(0, 100, 10))
        
        V1, V2 = np.meshgrid(vals1, vals2)
        predictions = np.zeros_like(V1)
        
        X_grid = self.X_test.copy()
        
        for i in range(len(vals1)):
            for j in range(len(vals2)):
                X_grid.iloc[:, idx1] = V1[j, i]
                X_grid.iloc[:, idx2] = V2[j, i]
                
                y_pred = self.model.predict_proba(X_grid)[:, 1]
                predictions[j, i] = y_pred.mean()
        
        # Create contour plot
        contour = ax.contourf(V1, V2, predictions, levels=15, cmap='viridis', alpha=0.8)
        ax.contour(V1, V2, predictions, levels=15, colors='black', alpha=0.4, linewidths=0.5)
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Prediction Probability')
    
    def create_interpretability_summary_report(self) -> str:
        """Create comprehensive interpretability summary report"""
        print("Creating interpretability summary report...")
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE INTERPRETABILITY ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Model information
        report.append(f"Model Type: {type(self.model).__name__}")
        report.append(f"Number of Features: {len(self.feature_names)}")
        report.append(f"Test Set Size: {len(self.X_test)}")
        report.append("")
        
        # Feature importance summary
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            report.append("TOP 15 MOST IMPORTANT FEATURES:")
            report.append("-" * 40)
            for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
                domain = self._get_feature_domain(row['feature'])
                report.append(f"{i:2d}. {row['feature']} ({domain}): {row['importance']:.4f}")
            report.append("")
        
        # Domain-wise importance
        if self.feature_domains:
            report.append("IMPORTANCE BY DOMAIN:")
            report.append("-" * 25)
            domain_importance = {}
            
            if hasattr(self.model, 'feature_importances_'):
                for domain, features in self.feature_domains.items():
                    domain_features = [f for f in features if f in self.feature_names]
                    if domain_features:
                        domain_indices = [self.feature_names.index(f) for f in domain_features]
                        domain_importance[domain] = importance[domain_indices].sum()
                
                # Sort domains by importance
                sorted_domains = sorted(domain_importance.items(), key=lambda x: x[1], reverse=True)
                for domain, imp in sorted_domains:
                    report.append(f"{domain}: {imp:.4f}")
                report.append("")
        
        # Interaction analysis results
        if 'sociodemographic' in self.interaction_strengths:
            interactions_df = self.interaction_strengths['sociodemographic']
            report.append("TOP 10 SOCIODEMOGRAPHIC INTERACTIONS:")
            report.append("-" * 42)
            for i, (_, row) in enumerate(interactions_df.head(10).iterrows(), 1):
                report.append(f"{i:2d}. {row['variable_1']} × {row['variable_2']}: "
                            f"{row['interaction_strength']:.4f}")
            report.append("")
        
        # Recommendations
        report.append("KEY INSIGHTS AND RECOMMENDATIONS:")
        report.append("-" * 35)
        report.append("1. Focus intervention efforts on top predictive domains")
        report.append("2. Consider intersectionality effects in program design")
        report.append("3. Monitor model performance across different subgroups")
        report.append("4. Validate findings in external datasets")
        report.append("")
        
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / 'interpretability_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Interpretability report saved to {report_path}")
        print(report_text)
        
        return report_text
    
    def _get_feature_domain(self, feature: str) -> str:
        """Get domain for a feature"""
        for domain, features in self.feature_domains.items():
            if feature in features:
                return domain
        return 'other'


if __name__ == "__main__":
    print("Interpretability Analysis Module")
    print("Use this module in conjunction with advanced_modeling.py")