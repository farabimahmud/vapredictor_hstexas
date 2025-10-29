import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
import dowhy
from dowhy import CausalModel
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
import networkx as nx

# Statistical libraries
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class VapingCausalAnalysis:
    """
    Comprehensive causal analysis for vaping behavior using Texas Youth Risk Behavior Survey data.
    """
    
    def __init__(self, data_path="data/cleaned_hstexas_full.csv"):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.variable_categories = {}
        self.causal_model = None
        self.dag = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for causal analysis."""
        print("Loading and preparing data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Original dataset shape: {self.df.shape}")
        
        # Create target variable mapping
        self.create_target_variables()
        
        # Remove administrative columns
        admin_cols = ['sitecode', 'sitename', 'sitetype', 'sitetypenum', 'year', 
                     'survyear', 'weight', 'stratum', 'PSU', 'record']
        self.df = self.df.drop(columns=[col for col in admin_cols if col in self.df.columns])
        
        # Handle missing values
        self.handle_missing_values()
        
        print(f"Processed dataset shape: {self.df.shape}")
        return self.df
    
    def create_target_variables(self):
        """Create interpretable target variables from survey codes."""
        # q35: Ever used electronic vapor product
        # q36: Current electronic vapor use
        
        if 'q35' in self.df.columns:
            self.df['ever_vaped'] = (self.df['q35'] == 1).astype(int)
        
        if 'q36' in self.df.columns:
            self.df['currently_vape'] = (self.df['q36'] == 1).astype(int)
            
        # Create derived variables
        if 'q35' in self.df.columns and 'q36' in self.df.columns:
            # Quit vaping: ever vaped but not currently vaping
            self.df['quit_vaping'] = ((self.df['q35'] == 1) & (self.df['q36'] != 1)).astype(int)
    
    def handle_missing_values(self):
        """Handle missing values appropriately for causal analysis."""
        # Drop columns with more than 30% missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > 0.3].index.tolist()
        self.df = self.df.drop(columns=cols_to_drop)
        
        # For remaining columns, drop rows with missing target variable
        if 'ever_vaped' in self.df.columns:
            self.df = self.df.dropna(subset=['ever_vaped'])
        
        # Fill remaining missing values with mode for categorical variables
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'int64', 'float64']:
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col] = self.df[col].fillna(mode_value[0])
    
    def categorize_variables(self):
        """Categorize variables based on domain knowledge for causal analysis."""
        
        # Define variable categories based on survey structure
        self.variable_categories = {
            'demographic': ['age', 'sex', 'grade', 'race4', 'race7', 'stheight', 'stweight', 'bmi', 'bmipct'],
            
            'mental_health': ['q26', 'q27', 'q28', 'q29', 'q30'],  # sad/hopeless, suicide ideation/planning/attempt
            
            'risk_behaviors': ['q8', 'q9', 'q10', 'q11'],  # seatbelt, drinking & driving, texting & driving
            
            'violence_safety': ['q12', 'q13', 'q14', 'q15', 'q16', 'q19', 'q20', 'q21', 'q22', 'q24', 'q25'],
            
            'substance_use': ['q32', 'q33', 'q34', 'q38', 'q39', 'q40',  # tobacco
                             'q41', 'q42', 'q43', 'q44', 'q45',  # alcohol
                             'q46', 'q47', 'q48',  # marijuana
                             'q49', 'q50', 'q51', 'q52', 'q53', 'q54', 'q55'],  # other drugs
            
            'sexual_behavior': ['q56', 'q57', 'q58', 'q59', 'q60', 'q61', 'q63'],
            
            'health_nutrition': ['q66', 'q67', 'q68', 'q69', 'q70', 'q71', 'q72', 'q73', 'q74', 'q75'],
            
            'physical_activity': ['q76', 'q77', 'q78', 'q79'],
            
            'health_services': ['q81', 'q82', 'q83', 'q85', 'q87'],
            
            'target': ['ever_vaped', 'currently_vape', 'quit_vaping']
        }
        
        # Create reverse mapping
        self.var_to_category = {}
        for category, variables in self.variable_categories.items():
            for var in variables:
                if var in self.df.columns:
                    self.var_to_category[var] = category
        
        print("Variable categorization complete:")
        for category, variables in self.variable_categories.items():
            available_vars = [v for v in variables if v in self.df.columns]
            print(f"  {category}: {len(available_vars)} variables")
    
    def create_domain_knowledge_dag(self):
        """Create a DAG based on domain knowledge about youth risk behaviors."""
        
        # Simplified causal relationships based on behavioral health literature
        causal_relationships = [
            # Demographics affect everything
            ('age', 'substance_use'),
            ('sex', 'substance_use'),
            ('grade', 'substance_use'),
            
            # Mental health pathway
            ('mental_health', 'substance_use'),
            ('mental_health', 'ever_vaped'),
            
            # Peer influence and risk behaviors
            ('risk_behaviors', 'substance_use'),
            ('risk_behaviors', 'ever_vaped'),
            
            # Violence/safety affects mental health
            ('violence_safety', 'mental_health'),
            
            # Substance use gateway effect
            ('substance_use', 'ever_vaped'),
            
            # Health behaviors
            ('health_nutrition', 'ever_vaped'),
            ('physical_activity', 'ever_vaped'),
        ]
        
        # Create networkx graph
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(causal_relationships)
        
        return self.dag
    
    def discover_causal_structure(self, method='pc', alpha=0.05):
        """Use causal discovery algorithms to learn structure from data."""
        print(f"Running causal discovery using {method} algorithm...")
        
        # Prepare data for causal discovery
        # Select key variables to avoid computational complexity
        key_vars = []
        for category in ['demographic', 'mental_health', 'substance_use', 'target']:
            if category in self.variable_categories:
                key_vars.extend([v for v in self.variable_categories[category] 
                               if v in self.df.columns])
        
        # Ensure we have target variable
        if 'ever_vaped' not in key_vars and 'ever_vaped' in self.df.columns:
            key_vars.append('ever_vaped')
        
        discovery_df = self.df[key_vars].copy()
        
        # Convert to numeric if needed
        for col in discovery_df.columns:
            if discovery_df[col].dtype == 'object':
                le = LabelEncoder()
                discovery_df[col] = le.fit_transform(discovery_df[col].astype(str))
        
        # Run PC algorithm
        if method == 'pc':
            # Use appropriate independence test for mixed data
            cg = pc(discovery_df.values, alpha=alpha, indep_test=fisherz)
            
            # Convert to networkx graph
            discovered_dag = nx.DiGraph()
            n_vars = len(key_vars)
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and cg.G.graph[i, j] == 1:
                        discovered_dag.add_edge(key_vars[i], key_vars[j])
            
            return discovered_dag, key_vars
        
        return None, key_vars
    
    def estimate_causal_effects(self, treatment_var='substance_use_score', outcome_var='ever_vaped'):
        """Estimate causal effects using DoWhy framework."""
        
        # Create composite scores for variable categories
        self.create_composite_scores()
        
        if treatment_var not in self.df.columns:
            print(f"Treatment variable {treatment_var} not found. Creating composite score...")
            treatment_var = 'substance_use_score'
        
        if outcome_var not in self.df.columns:
            print(f"Outcome variable {outcome_var} not found.")
            return None
        
        # Define confounders
        confounders = ['age', 'sex', 'grade', 'mental_health_score', 'risk_behavior_score']
        available_confounders = [c for c in confounders if c in self.df.columns]
        
        # Create causal model
        causal_graph = f"""
        digraph {{
            {treatment_var};
            {outcome_var};
            {'; '.join(available_confounders)};
            
            {' -> '.join(available_confounders + [treatment_var])};
            {' -> '.join(available_confounders + [outcome_var])};
            {treatment_var} -> {outcome_var};
        }}
        """
        
        # Prepare data
        analysis_vars = [treatment_var, outcome_var] + available_confounders
        analysis_df = self.df[analysis_vars].copy()
        
        # Create DoWhy causal model
        self.causal_model = CausalModel(
            data=analysis_df,
            treatment=treatment_var,
            outcome=outcome_var,
            graph=causal_graph
        )
        
        # Identify causal effect
        identified_estimand = self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        print("Identified estimand:")
        print(identified_estimand)
        
        # Estimate causal effect
        causal_estimate = self.causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching"
        )
        
        print("Causal Effect Estimate:")
        print(causal_estimate)
        
        # Refute the estimate
        refute_random = self.causal_model.refute_estimate(
            identified_estimand, causal_estimate,
            method_name="random_common_cause"
        )
        
        print("Refutation using random common cause:")
        print(refute_random)
        
        return causal_estimate
    
    def create_composite_scores(self):
        """Create composite scores for variable categories."""
        
        for category, variables in self.variable_categories.items():
            if category == 'target':
                continue
                
            available_vars = [v for v in variables if v in self.df.columns]
            if len(available_vars) > 1:
                # Create standardized composite score
                score_name = f"{category}_score"
                
                # Handle missing values and standardize
                category_data = self.df[available_vars].copy()
                
                # Fill missing with median
                for col in available_vars:
                    category_data[col] = category_data[col].fillna(category_data[col].median())
                
                # Standardize and sum
                scaler = StandardScaler()
                standardized = scaler.fit_transform(category_data)
                self.df[score_name] = np.mean(standardized, axis=1)
    
    def plot_causal_graph(self, graph, title="Causal Graph", figsize=(12, 8)):
        """Plot causal graph using networkx and matplotlib."""
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw graph
        nx.draw(graph, pos, 
                with_labels=True, 
                node_color='lightblue', 
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                arrowsize=20)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'output/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_variable_importance(self, target='ever_vaped'):
        """Analyze variable importance using Random Forest."""
        
        if target not in self.df.columns:
            print(f"Target variable {target} not found.")
            return None
        
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['ever_vaped', 'currently_vape', 'quit_vaping']]
        
        X = self.df[feature_cols].copy()
        y = self.df[target]
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill missing values
        X = X.fillna(X.median())
        
        # Fit Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'variable': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 important variables
        plt.figure(figsize=(10, 8))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance'])
        plt.yticks(range(len(top_20)), top_20['variable'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Variables Associated with Vaping')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('output/variable_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def run_full_analysis(self):
        """Run the complete causal analysis pipeline."""
        print("=" * 60)
        print("VAPING CAUSAL ANALYSIS - TEXAS YOUTH RISK BEHAVIOR SURVEY")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Categorize variables
        self.categorize_variables()
        
        # Step 3: Create domain knowledge DAG
        domain_dag = self.create_domain_knowledge_dag()
        
        # Step 4: Variable importance analysis
        print("\n" + "="*40)
        print("VARIABLE IMPORTANCE ANALYSIS")
        print("="*40)
        importance_df = self.analyze_variable_importance()
        print(f"\nTop 10 most important variables:")
        print(importance_df.head(10))
        
        # Step 5: Causal discovery
        print("\n" + "="*40)
        print("CAUSAL STRUCTURE DISCOVERY")
        print("="*40)
        discovered_dag, key_vars = self.discover_causal_structure()
        
        # Step 6: Causal effect estimation
        print("\n" + "="*40)
        print("CAUSAL EFFECT ESTIMATION")
        print("="*40)
        causal_estimate = self.estimate_causal_effects()
        
        # Step 7: Generate summary report
        self.generate_summary_report(importance_df, causal_estimate)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - Check output folder for visualizations")
        print("="*60)
    
    def generate_summary_report(self, importance_df=None, causal_estimate=None):
        """Generate a comprehensive summary report."""
        
        report = []
        report.append("VAPING CAUSAL ANALYSIS SUMMARY REPORT")
        report.append("="*50)
        report.append(f"Dataset: {self.data_path}")
        report.append(f"Sample size: {len(self.df)} students")
        report.append(f"Number of variables: {len(self.df.columns)}")
        
        if 'ever_vaped' in self.df.columns:
            vaping_rate = self.df['ever_vaped'].mean()
            report.append(f"Ever vaping rate: {vaping_rate:.2%}")
        
        report.append("\nVARIABLE CATEGORIES:")
        for category, variables in self.variable_categories.items():
            available_vars = [v for v in variables if v in self.df.columns]
            report.append(f"  {category}: {len(available_vars)} variables")
        
        if importance_df is not None:
            report.append("\nTOP 10 PREDICTIVE VARIABLES:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                report.append(f"  {i}. {row['variable']}: {row['importance']:.4f}")
        
        if causal_estimate is not None:
            report.append(f"\nCAUSAL EFFECT ESTIMATE:")
            report.append(f"  Effect: {causal_estimate.value:.4f}")
        
        report.append("\nRECOMMENDations:")
        report.append("  1. Focus interventions on modifiable risk factors")
        report.append("  2. Address mental health as a key pathway")
        report.append("  3. Implement comprehensive substance use prevention")
        report.append("  4. Consider peer influence in program design")
        
        # Save report
        with open('output/causal_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run analysis
    analyzer = VapingCausalAnalysis()
    analyzer.run_full_analysis()