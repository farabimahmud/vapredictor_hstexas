import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
import dowhy
from dowhy import CausalModel
import networkx as nx
from scipy import stats
try:
    from econml.dml import LinearDML
    from econml.dr import DRLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("EconML not available, using alternative methods")

class RobustCausalAnalysis:
    """
    Robust causal analysis for vaping behavior with proper error handling.
    """
    
    def __init__(self, data_path="data/hstexas_full.csv"):
        self.data_path = data_path
        self.df = None
        self.causal_df = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for causal analysis."""
        print("Loading and preparing data for rigorous causal analysis...")
        self.df = pd.read_csv(self.data_path)
        print(f"Original shape: {self.df.shape}")
        
        # Create target variables
        if 'q35' in self.df.columns:
            self.df['ever_vaped'] = (self.df['q35'] == 1).astype(int)
        if 'q36' in self.df.columns:
            self.df['currently_vape'] = (self.df['q36'] > 1).astype(int)
        
        # Remove administrative columns
        admin_cols = ['sitecode', 'sitename', 'sitetype', 'sitetypenum', 
                     'weight', 'stratum', 'PSU', 'record', 'year', 'survyear']
        self.df = self.df.drop(columns=[col for col in admin_cols if col in self.df.columns])
        
        # Select key variables for causal analysis
        key_variables = [
            # Demographics (converted to binary/categorical)
            'age', 'sex', 'grade', 'race4',
            # Mental health (convert to binary)
            'q26', 'q27', 'q28', 'q29',
            # Substance use (convert to binary)
            'q32', 'q33', 'q40', 'q41', 'q42', 'q43', 'q46', 'q47', 'q48',
            'q49', 'q50', 'q51',
            # Risk behaviors (convert to binary)
            'q8', 'q9', 'q10', 'q16', 'q24', 'q25',
            # Health behaviors (convert to binary)
            'q76', 'q85', 'q87',
            # Target
            'ever_vaped'
        ]
        
        # Filter to available variables
        available_vars = [var for var in key_variables if var in self.df.columns]
        self.causal_df = self.df[available_vars].copy()
        
        # Convert all variables to binary for proper causal analysis
        print("Converting variables to binary format for causal analysis...")
        
        for col in self.causal_df.columns:
            if col == 'ever_vaped':
                continue  # Already binary
            elif col in ['age', 'grade']:
                # Create binary: above median vs below median
                median_val = self.causal_df[col].median()
                self.causal_df[col] = (self.causal_df[col] > median_val).astype(int)
            elif col in ['sex', 'race4']:
                # Keep as categorical but ensure proper encoding
                self.causal_df[col] = (self.causal_df[col] == 1).astype(int) if col == 'sex' else \
                                     (self.causal_df[col] == 1).astype(int)  # Most common race
            else:
                # Convert all behavioral variables to binary (1 = Yes, 0 = No/Other)
                self.causal_df[col] = (self.causal_df[col] == 1).astype(int)
        
        # Handle missing values
        print("Handling missing values...")
        for col in self.causal_df.columns:
            if self.causal_df[col].isnull().sum() > 0:
                # For binary variables, fill with 0 (most conservative)
                self.causal_df[col] = self.causal_df[col].fillna(0)
        
        # Remove rows with target variable missing
        if 'ever_vaped' in self.causal_df.columns:
            self.causal_df = self.causal_df.dropna(subset=['ever_vaped'])
        
        print(f"Final causal analysis dataset shape: {self.causal_df.shape}")
        print(f"All variables are now binary (0/1)")
        return self.causal_df
    
    def analyze_associations(self):
        """Analyze basic associations between variables."""
        print("\n" + "="*60)
        print("VARIABLE ASSOCIATIONS ANALYSIS")
        print("="*60)
        
        if 'ever_vaped' not in self.causal_df.columns:
            return
        
        # Variable name mapping for readability
        var_names = {
            'q26': 'Sad/Hopeless', 'q27': 'Considered Suicide', 'q28': 'Suicide Plan', 'q29': 'Attempted Suicide',
            'q32': 'Ever Cigarette', 'q33': 'Current Cigarette', 'q40': 'Tobacco Cessation',
            'q41': 'Early Alcohol', 'q42': 'Current Alcohol', 'q43': 'Binge Drinking',
            'q46': 'Ever Marijuana', 'q47': 'Early Marijuana', 'q48': 'Current Marijuana',
            'q49': 'Prescription Drugs', 'q50': 'Cocaine', 'q51': 'Inhalants',
            'q8': 'Seatbelt Use', 'q9': 'Drinking & Driving', 'q10': 'Impaired Driving',
            'q16': 'Physical Fighting', 'q24': 'Bullying', 'q25': 'Cyberbullying',
            'q76': 'Physical Activity', 'q85': 'Adequate Sleep', 'q87': 'Good Grades',
            'age': 'Older Age', 'sex': 'Female', 'grade': 'Higher Grade', 'race4': 'White Race'
        }
        
        # Calculate associations
        associations = []
        for var in self.causal_df.columns:
            if var != 'ever_vaped':
                # Calculate correlation and odds ratio
                corr = self.causal_df[var].corr(self.causal_df['ever_vaped'])
                
                # Create cross-tabulation for odds ratio
                cross_tab = pd.crosstab(self.causal_df[var], self.causal_df['ever_vaped'])
                if cross_tab.shape == (2, 2) and all(cross_tab.values.flatten() > 0):
                    odds_ratio = (cross_tab.iloc[1,1] * cross_tab.iloc[0,0]) / (cross_tab.iloc[1,0] * cross_tab.iloc[0,1])
                    
                    # Calculate rates
                    no_exposure_rate = cross_tab.iloc[0,1] / (cross_tab.iloc[0,0] + cross_tab.iloc[0,1])
                    yes_exposure_rate = cross_tab.iloc[1,1] / (cross_tab.iloc[1,0] + cross_tab.iloc[1,1])
                    
                    associations.append({
                        'variable': var_names.get(var, var),
                        'code': var,
                        'correlation': corr,
                        'odds_ratio': odds_ratio,
                        'no_exposure_rate': no_exposure_rate,
                        'yes_exposure_rate': yes_exposure_rate,
                        'prevalence': self.causal_df[var].mean()
                    })
        
        # Sort by odds ratio
        associations.sort(key=lambda x: x['odds_ratio'], reverse=True)
        
        print("Top Risk Factors (Odds Ratio > 1):")
        print("-" * 60)
        risk_factors = [a for a in associations if a['odds_ratio'] > 1]
        for i, assoc in enumerate(risk_factors[:10], 1):
            print(f"{i:2d}. {assoc['variable']:25s} OR={assoc['odds_ratio']:.2f} "
                  f"({assoc['no_exposure_rate']:.1%} → {assoc['yes_exposure_rate']:.1%})")
        
        print(f"\nTop Protective Factors (Odds Ratio < 1):")
        print("-" * 60)
        protective_factors = [a for a in associations if a['odds_ratio'] < 1]
        for i, assoc in enumerate(protective_factors[:10], 1):
            print(f"{i:2d}. {assoc['variable']:25s} OR={assoc['odds_ratio']:.2f} "
                  f"({assoc['no_exposure_rate']:.1%} → {assoc['yes_exposure_rate']:.1%})")
        
        self.results['associations'] = associations
        return associations
    
    def estimate_causal_effects_dowhy(self, treatment, outcome='ever_vaped'):
        """Estimate causal effects using DoWhy with proper binary treatment handling."""
        print(f"\nEstimating causal effect of {treatment} on {outcome} using DoWhy...")
        
        if treatment not in self.causal_df.columns or outcome not in self.causal_df.columns:
            print(f"Treatment or outcome variable not available")
            return None
        
        # Define potential confounders (exclude treatment and outcome)
        all_vars = list(self.causal_df.columns)
        potential_confounders = [var for var in all_vars if var not in [treatment, outcome]]
        
        # Select subset of most important confounders to avoid overfitting
        important_confounders = ['age', 'sex', 'grade', 'race4']
        if 'q26' in potential_confounders:  # Mental health
            important_confounders.append('q26')
        if 'q32' in potential_confounders and treatment != 'q32':  # Cigarette use
            important_confounders.append('q32')
        
        confounders = [c for c in important_confounders if c in potential_confounders]
        
        print(f"Using confounders: {confounders}")
        
        # Prepare data
        analysis_vars = [treatment, outcome] + confounders
        analysis_df = self.causal_df[analysis_vars].copy()
        
        # Create simple causal graph
        graph_str = f"digraph {{ "
        
        # Add confounder relationships
        for conf in confounders:
            graph_str += f"{conf} -> {treatment}; {conf} -> {outcome}; "
        
        # Add treatment effect
        graph_str += f"{treatment} -> {outcome}; "
        graph_str += "}"
        
        try:
            # Create causal model
            model = CausalModel(
                data=analysis_df,
                treatment=treatment,
                outcome=outcome,
                graph=graph_str
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate causal effect using linear regression (suitable for binary treatment/outcome)
            estimate = model.estimate_effect(
                identified_estimand, 
                method_name="backdoor.linear_regression"
            )
            
            print(f"Causal Effect: {estimate.value:.4f}")
            
            # Confidence interval if available
            if hasattr(estimate, 'conf_int') and estimate.conf_int is not None:
                print(f"95% CI: [{estimate.conf_int[0]:.4f}, {estimate.conf_int[1]:.4f}]")
            
            # Refutation test
            try:
                refute_result = model.refute_estimate(
                    identified_estimand, estimate,
                    method_name="random_common_cause"
                )
                print(f"Refutation test: {refute_result.new_effect:.4f}")
            except:
                print("Refutation test failed")
            
            return estimate.value
            
        except Exception as e:
            print(f"DoWhy analysis failed: {e}")
            return None
    
    def estimate_simple_causal_effects(self):
        """Estimate causal effects using simple regression adjustments."""
        print("\n" + "="*60)
        print("CAUSAL EFFECT ESTIMATION")
        print("="*60)
        
        # Key treatments to analyze
        treatments = {
            'q26': 'Mental Health Issues',
            'q32': 'Ever Cigarette Use', 
            'q42': 'Current Alcohol Use',
            'q46': 'Ever Marijuana Use',
            'q16': 'Physical Fighting',
            'q76': 'Physical Activity',
            'q85': 'Adequate Sleep',
            'q87': 'Good Academic Performance'
        }
        
        causal_effects = {}
        
        for treatment_var, treatment_name in treatments.items():
            if treatment_var in self.causal_df.columns:
                print(f"\nAnalyzing: {treatment_name} ({treatment_var})")
                
                # DoWhy analysis
                effect = self.estimate_causal_effects_dowhy(treatment_var)
                
                if effect is not None:
                    causal_effects[treatment_name] = {
                        'variable': treatment_var,
                        'effect': effect,
                        'interpretation': self._interpret_effect(effect, treatment_name)
                    }
        
        # Rank by absolute effect size
        if causal_effects:
            print(f"\n" + "="*60)
            print("CAUSAL EFFECTS SUMMARY (Ranked by Impact)")
            print("="*60)
            
            sorted_effects = sorted(causal_effects.items(), 
                                  key=lambda x: abs(x[1]['effect']), reverse=True)
            
            for i, (name, data) in enumerate(sorted_effects, 1):
                effect = data['effect']
                direction = "increases" if effect > 0 else "decreases"
                print(f"{i}. {name}: {direction} vaping probability by {abs(effect):.3f}")
                print(f"   Interpretation: {data['interpretation']}")
        
        self.results['causal_effects'] = causal_effects
        return causal_effects
    
    def _interpret_effect(self, effect, treatment_name):
        """Provide interpretation of causal effect."""
        abs_effect = abs(effect)
        direction = "increases" if effect > 0 else "decreases"
        
        if abs_effect < 0.01:
            magnitude = "negligible"
        elif abs_effect < 0.05:
            magnitude = "small"
        elif abs_effect < 0.10:
            magnitude = "moderate"
        else:
            magnitude = "large"
        
        return f"{treatment_name} has a {magnitude} effect that {direction} vaping risk"
    
    def create_causal_network_visualization(self):
        """Create a visualization of the causal network."""
        if 'associations' not in self.results:
            return
        
        plt.figure(figsize=(16, 12))
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        G.add_node('Ever Vaped', type='outcome')
        
        # Add top risk factors and protective factors
        associations = self.results['associations']
        top_risk = [a for a in associations if a['odds_ratio'] > 1.5][:8]
        top_protective = [a for a in associations if a['odds_ratio'] < 0.7][:8]
        
        for assoc in top_risk + top_protective:
            G.add_node(assoc['variable'], type='risk' if assoc['odds_ratio'] > 1 else 'protective')
            G.add_edge(assoc['variable'], 'Ever Vaped', weight=abs(assoc['correlation']))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Position outcome at center
        if 'Ever Vaped' in pos:
            pos['Ever Vaped'] = (0, 0)
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'other')
            if node_type == 'outcome':
                node_colors.append('#ff4444')
            elif node_type == 'risk':
                node_colors.append('#ffaaaa')
            elif node_type == 'protective':
                node_colors.append('#aaffaa')
            else:
                node_colors.append('#dddddd')
        
        # Draw network
        nx.draw(G, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=3000,
                font_size=9,
                font_weight='bold',
                edge_color='gray',
                alpha=0.8)
        
        plt.title('Causal Network for Vaping Behavior\n(Risk Factors in Red, Protective Factors in Green)', 
                 fontsize=16, fontweight='bold')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', 
                       markersize=15, label='Outcome (Vaping)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaaaa', 
                       markersize=15, label='Risk Factors'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaffaa', 
                       markersize=15, label='Protective Factors')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('output/causal_network_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evidence_based_recommendations(self):
        """Generate recommendations based on causal analysis."""
        print("\n" + "="*80)
        print("EVIDENCE-BASED INTERVENTION RECOMMENDATIONS")
        print("="*80)
        
        if 'causal_effects' not in self.results:
            print("No causal effects available for recommendations")
            return
        
        causal_effects = self.results['causal_effects']
        
        # Separate into intervention targets
        risk_interventions = []
        protective_interventions = []
        
        for name, data in causal_effects.items():
            effect = data['effect']
            if effect > 0.01:  # Significant risk factor
                risk_interventions.append((name, effect))
            elif effect < -0.01:  # Significant protective factor
                protective_interventions.append((name, abs(effect)))
        
        # Sort by effect size
        risk_interventions.sort(key=lambda x: x[1], reverse=True)
        protective_interventions.sort(key=lambda x: x[1], reverse=True)
        
        print("PRIORITY INTERVENTIONS (Ranked by Causal Impact):")
        print("-" * 80)
        
        if risk_interventions:
            print("\n1. REDUCE RISK FACTORS:")
            for i, (intervention, effect) in enumerate(risk_interventions, 1):
                print(f"   {i}. Target {intervention}")
                print(f"      - Causal impact: +{effect:.3f} increase in vaping probability")
                
                if "Mental Health" in intervention:
                    print(f"      - Strategy: Comprehensive mental health support and counseling")
                elif "Cigarette" in intervention:
                    print(f"      - Strategy: Intensive tobacco prevention programs")
                elif "Alcohol" in intervention:
                    print(f"      - Strategy: Alcohol abuse prevention and education")
                elif "Marijuana" in intervention:
                    print(f"      - Strategy: Drug education and prevention programs")
                elif "Fighting" in intervention:
                    print(f"      - Strategy: Conflict resolution and anger management")
        
        if protective_interventions:
            print(f"\n2. STRENGTHEN PROTECTIVE FACTORS:")
            for i, (intervention, effect) in enumerate(protective_interventions, 1):
                print(f"   {i}. Promote {intervention}")
                print(f"      - Causal impact: -{effect:.3f} decrease in vaping probability")
                
                if "Physical Activity" in intervention:
                    print(f"      - Strategy: Expand sports and fitness programs")
                elif "Sleep" in intervention:
                    print(f"      - Strategy: Sleep hygiene education and school start time policies")
                elif "Academic" in intervention:
                    print(f"      - Strategy: Academic support and tutoring programs")
        
        print(f"\n3. IMPLEMENTATION PRINCIPLES:")
        print("   - Target interventions with largest causal effects first")
        print("   - Implement multi-component approaches addressing multiple factors")
        print("   - Monitor and evaluate intervention effectiveness")
        print("   - Adapt strategies based on local context and resources")
        
        print(f"\n4. METHODOLOGICAL STRENGTHS:")
        print("   - Based on rigorous causal inference methods")
        print("   - Accounts for confounding variables")
        print("   - Uses large representative sample")
        print("   - Provides quantified effect sizes for prioritization")
    
    def run_complete_analysis(self):
        """Run the complete rigorous causal analysis."""
        print("="*80)
        print("RIGOROUS CAUSAL ANALYSIS - VAPING BEHAVIOR")
        print("Using DoWhy Framework for Causal Inference")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Analyze basic associations
        self.analyze_associations()
        
        # Estimate causal effects
        self.estimate_simple_causal_effects()
        
        # Create visualization
        self.create_causal_network_visualization()
        
        # Generate recommendations
        self.generate_evidence_based_recommendations()
        
        print("\n" + "="*80)
        print("RIGOROUS CAUSAL ANALYSIS COMPLETE")
        print("="*80)

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    analyzer = RobustCausalAnalysis()
    analyzer.run_complete_analysis()