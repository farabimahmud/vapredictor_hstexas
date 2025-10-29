import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
import dowhy
from dowhy import CausalModel
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq, fisherz, gsq
import networkx as nx
from scipy import stats
from econml.dml import LinearDML
from econml.dr import DRLearner

class RigorousCausalAnalysis:
    """
    Rigorous causal analysis for vaping behavior using proper causal inference methods.
    """
    
    def __init__(self, data_path="data/cleaned_hstexas_full.csv"):
        self.data_path = data_path
        self.df = None
        self.causal_df = None
        self.discovered_graph = None
        self.causal_models = {}
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
        
        # Select key variables for causal analysis (to manage computational complexity)
        key_variables = [
            # Demographics
            'age', 'sex', 'grade', 'race4',
            # Mental health
            'q26', 'q27', 'q28', 'q29',  # sad/hopeless, suicide ideation/plan/attempt
            # Substance use
            'q32', 'q33', 'q40',  # cigarette use, cessation
            'q41', 'q42', 'q43',  # alcohol use
            'q46', 'q47', 'q48',  # marijuana use
            'q49', 'q50', 'q51',  # other drugs
            # Risk behaviors
            'q8', 'q9', 'q10', 'q16',  # safety behaviors, fighting
            'q24', 'q25',  # bullying
            # Health behaviors
            'q76', 'q85', 'q87',  # physical activity, sleep, grades
            # Target
            'ever_vaped'
        ]
        
        # Filter to available variables
        available_vars = [var for var in key_variables if var in self.df.columns]
        self.causal_df = self.df[available_vars].copy()
        
        # Handle missing values
        print("Handling missing values for causal analysis...")
        
        # For each variable, impute missing values
        for col in self.causal_df.columns:
            if self.causal_df[col].isnull().sum() > 0:
                if col in ['age', 'sex', 'grade', 'race4']:
                    # Demographics: use mode
                    mode_val = self.causal_df[col].mode()
                    if len(mode_val) > 0:
                        self.causal_df[col] = self.causal_df[col].fillna(mode_val[0])
                else:
                    # Behavioral variables: use median or mode
                    if self.causal_df[col].dtype in ['int64', 'float64']:
                        self.causal_df[col] = self.causal_df[col].fillna(self.causal_df[col].median())
                    else:
                        mode_val = self.causal_df[col].mode()
                        if len(mode_val) > 0:
                            self.causal_df[col] = self.causal_df[col].fillna(mode_val[0])
        
        # Remove rows with target variable missing
        if 'ever_vaped' in self.causal_df.columns:
            self.causal_df = self.causal_df.dropna(subset=['ever_vaped'])
        
        print(f"Causal analysis dataset shape: {self.causal_df.shape}")
        return self.causal_df
    
    def discover_causal_structure(self, method='pc', alpha=0.01):
        """Discover causal structure using constraint-based or score-based methods."""
        print(f"\nDiscovering causal structure using {method} algorithm...")
        
        # Prepare data for causal discovery
        discovery_data = self.causal_df.copy()
        
        # Encode categorical variables for causal discovery
        label_encoders = {}
        for col in discovery_data.columns:
            if discovery_data[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                discovery_data[col] = le.fit_transform(discovery_data[col].astype(str))
                label_encoders[col] = le
        
        # Convert to numpy array
        data_matrix = discovery_data.values
        variable_names = list(discovery_data.columns)
        
        try:
            if method == 'pc':
                # PC algorithm for constraint-based discovery
                print(f"Running PC algorithm with alpha={alpha}...")
                cg = pc(data_matrix, alpha=alpha, indep_test=chisq)
                
                # Convert to networkx graph
                self.discovered_graph = nx.DiGraph()
                n_vars = len(variable_names)
                
                for i in range(n_vars):
                    for j in range(n_vars):
                        if i != j and cg.G.graph[i, j] == 1:
                            # Check for edge direction from CPDAG
                            if hasattr(cg, 'draw_pydot_graph'):
                                self.discovered_graph.add_edge(variable_names[i], variable_names[j])
                            else:
                                # If undirected, add both directions with lower weight
                                self.discovered_graph.add_edge(variable_names[i], variable_names[j], weight=0.5)
                
            elif method == 'ges':
                # GES algorithm for score-based discovery
                print("Running GES algorithm...")
                Record = ges(data_matrix)
                
                # Convert to networkx graph
                self.discovered_graph = nx.DiGraph()
                n_vars = len(variable_names)
                
                for i in range(n_vars):
                    for j in range(n_vars):
                        if Record['G'].graph[i, j] == 1:
                            self.discovered_graph.add_edge(variable_names[i], variable_names[j])
            
            print(f"Discovered graph with {self.discovered_graph.number_of_nodes()} nodes and {self.discovered_graph.number_of_edges()} edges")
            
            # Identify parents of the target variable
            if 'ever_vaped' in self.discovered_graph.nodes():
                parents = list(self.discovered_graph.predecessors('ever_vaped'))
                print(f"Discovered causal parents of 'ever_vaped': {parents}")
                self.results['causal_parents'] = parents
            
            return self.discovered_graph
            
        except Exception as e:
            print(f"Causal discovery failed: {e}")
            print("Falling back to domain knowledge graph...")
            return self._create_domain_knowledge_graph()
    
    def _create_domain_knowledge_graph(self):
        """Create a causal graph based on domain knowledge as fallback."""
        print("Creating domain knowledge-based causal graph...")
        
        self.discovered_graph = nx.DiGraph()
        
        # Add nodes
        for var in self.causal_df.columns:
            self.discovered_graph.add_node(var)
        
        # Define causal relationships based on theory
        causal_edges = [
            # Demographics affect behaviors
            ('age', 'q26'), ('age', 'q32'), ('age', 'q46'), ('age', 'ever_vaped'),
            ('sex', 'q26'), ('sex', 'q32'), ('sex', 'ever_vaped'),
            ('grade', 'q32'), ('grade', 'q46'), ('grade', 'ever_vaped'),
            
            # Mental health affects substance use
            ('q26', 'q32'), ('q26', 'q42'), ('q26', 'q46'), ('q26', 'ever_vaped'),
            ('q27', 'q32'), ('q27', 'q42'), ('q27', 'ever_vaped'),
            ('q28', 'ever_vaped'), ('q29', 'ever_vaped'),
            
            # Substance use gateway effects
            ('q32', 'ever_vaped'), ('q33', 'ever_vaped'),
            ('q42', 'ever_vaped'), ('q46', 'ever_vaped'), ('q48', 'ever_vaped'),
            
            # Risk behaviors cluster
            ('q8', 'q9'), ('q9', 'q10'), ('q16', 'ever_vaped'),
            
            # Health behaviors
            ('q76', 'ever_vaped'), ('q85', 'q26'), ('q87', 'q26')
        ]
        
        # Add edges that exist in data
        for source, target in causal_edges:
            if source in self.causal_df.columns and target in self.causal_df.columns:
                self.discovered_graph.add_edge(source, target)
        
        return self.discovered_graph
    
    def estimate_causal_effects_dowhy(self, treatment, outcome, confounders=None):
        """Estimate causal effects using DoWhy framework."""
        print(f"\nEstimating causal effect of {treatment} on {outcome} using DoWhy...")
        
        if treatment not in self.causal_df.columns or outcome not in self.causal_df.columns:
            print(f"Treatment or outcome variable not available")
            return None
        
        # Define confounders automatically if not provided
        if confounders is None:
            # Use discovered graph to find confounders
            if self.discovered_graph and treatment in self.discovered_graph.nodes() and outcome in self.discovered_graph.nodes():
                # Find common causes (confounders)
                treatment_ancestors = set(nx.ancestors(self.discovered_graph, treatment))
                outcome_ancestors = set(nx.ancestors(self.discovered_graph, outcome))
                confounders = list(treatment_ancestors.intersection(outcome_ancestors))
                
                # Add direct causes of treatment and outcome
                treatment_parents = list(self.discovered_graph.predecessors(treatment))
                outcome_parents = list(self.discovered_graph.predecessors(outcome))
                confounders.extend([p for p in treatment_parents if p != outcome])
                confounders.extend([p for p in outcome_parents if p != treatment])
                
                confounders = list(set(confounders))  # Remove duplicates
            else:
                # Default confounders based on domain knowledge
                potential_confounders = ['age', 'sex', 'grade', 'race4', 'q26', 'q27']
                confounders = [c for c in potential_confounders if c in self.causal_df.columns and c not in [treatment, outcome]]
        
        print(f"Using confounders: {confounders}")
        
        # Prepare data for DoWhy
        analysis_vars = [treatment, outcome] + confounders
        analysis_df = self.causal_df[analysis_vars].copy()
        
        # Create causal graph for DoWhy
        graph_edges = []
        
        # Add confounder edges
        for conf in confounders:
            graph_edges.append(f"{conf} -> {treatment}")
            graph_edges.append(f"{conf} -> {outcome}")
        
        # Add treatment effect
        graph_edges.append(f"{treatment} -> {outcome}")
        
        # Add discovered edges if available
        if self.discovered_graph:
            for edge in self.discovered_graph.edges():
                if edge[0] in analysis_vars and edge[1] in analysis_vars:
                    graph_str = f"{edge[0]} -> {edge[1]}"
                    if graph_str not in graph_edges:
                        graph_edges.append(graph_str)
        
        causal_graph = "digraph { " + "; ".join(graph_edges) + "; }"
        
        try:
            # Create causal model
            model = CausalModel(
                data=analysis_df,
                treatment=treatment,
                outcome=outcome,
                graph=causal_graph
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            print("Identified estimand:")
            print(identified_estimand)
            
            # Estimate causal effect using multiple methods
            methods = [
                "backdoor.propensity_score_matching",
                "backdoor.linear_regression", 
                "backdoor.propensity_score_weighting"
            ]
            
            estimates = {}
            for method in methods:
                try:
                    estimate = model.estimate_effect(identified_estimand, method_name=method)
                    estimates[method] = estimate.value
                    print(f"\n{method}: {estimate.value:.4f}")
                    
                    # Refutation tests
                    refute_result = model.refute_estimate(identified_estimand, estimate, 
                                                        method_name="random_common_cause")
                    print(f"Refutation (random common cause): {refute_result.new_effect:.4f}")
                    
                except Exception as e:
                    print(f"Method {method} failed: {e}")
                    continue
            
            self.results[f'causal_effect_{treatment}_{outcome}'] = estimates
            return estimates
            
        except Exception as e:
            print(f"DoWhy analysis failed: {e}")
            return None
    
    def estimate_causal_effects_econml(self, treatment, outcome, confounders=None):
        """Estimate causal effects using EconML double machine learning."""
        print(f"\nEstimating causal effect using EconML Double ML...")
        
        if treatment not in self.causal_df.columns or outcome not in self.causal_df.columns:
            return None
        
        # Prepare confounders
        if confounders is None:
            potential_confounders = ['age', 'sex', 'grade', 'race4', 'q26', 'q27', 'q32', 'q42']
            confounders = [c for c in potential_confounders if c in self.causal_df.columns and c not in [treatment, outcome]]
        
        # Prepare data
        X = self.causal_df[confounders].values
        T = self.causal_df[treatment].values
        Y = self.causal_df[outcome].values
        
        try:
            # Double Machine Learning
            dml = LinearDML(
                model_y=RandomForestClassifier(n_estimators=100, random_state=42),
                model_t=RandomForestClassifier(n_estimators=100, random_state=42),
                random_state=42
            )
            
            dml.fit(Y, T, X=X)
            
            # Get treatment effect
            treatment_effect = dml.effect(X)
            avg_effect = np.mean(treatment_effect)
            
            print(f"Average Treatment Effect (ATE): {avg_effect:.4f}")
            print(f"Treatment Effect Std: {np.std(treatment_effect):.4f}")
            
            # Confidence intervals
            try:
                effect_inference = dml.effect_inference(X)
                conf_int = effect_inference.conf_int()
                print(f"95% Confidence Interval: [{conf_int[0].mean():.4f}, {conf_int[1].mean():.4f}]")
            except:
                print("Could not compute confidence intervals")
            
            self.results[f'econml_effect_{treatment}_{outcome}'] = {
                'ate': avg_effect,
                'std': np.std(treatment_effect),
                'individual_effects': treatment_effect
            }
            
            return avg_effect
            
        except Exception as e:
            print(f"EconML analysis failed: {e}")
            return None
    
    def analyze_multiple_treatments(self):
        """Analyze causal effects of multiple potential treatments on vaping."""
        print("\n" + "="*60)
        print("MULTIPLE TREATMENT CAUSAL ANALYSIS")
        print("="*60)
        
        # Define potential treatments (modifiable factors)
        treatments = {
            'q26': 'Mental Health (Sad/Hopeless)',
            'q32': 'Ever Cigarette Use',
            'q42': 'Current Alcohol Use',
            'q46': 'Ever Marijuana Use',
            'q16': 'Physical Fighting',
            'q24': 'Bullying at School',
            'q76': 'Physical Activity',
            'q85': 'Sleep Duration',
            'q87': 'Academic Grades'
        }
        
        outcome = 'ever_vaped'
        causal_effects = {}
        
        for treatment_var, treatment_name in treatments.items():
            if treatment_var in self.causal_df.columns:
                print(f"\nAnalyzing treatment: {treatment_name} ({treatment_var})")
                
                # DoWhy analysis
                dowhy_effects = self.estimate_causal_effects_dowhy(treatment_var, outcome)
                
                # EconML analysis  
                econml_effect = self.estimate_causal_effects_econml(treatment_var, outcome)
                
                causal_effects[treatment_name] = {
                    'variable': treatment_var,
                    'dowhy_effects': dowhy_effects,
                    'econml_effect': econml_effect
                }
        
        self.results['multiple_treatments'] = causal_effects
        return causal_effects
    
    def plot_causal_graph(self):
        """Plot the discovered causal graph."""
        if self.discovered_graph is None:
            print("No causal graph available to plot")
            return
        
        plt.figure(figsize=(16, 12))
        
        # Create layout
        if 'ever_vaped' in self.discovered_graph.nodes():
            # Use hierarchical layout with target at bottom
            pos = nx.spring_layout(self.discovered_graph, k=3, iterations=50)
            
            # Adjust target position
            if 'ever_vaped' in pos:
                target_x = pos['ever_vaped'][0]
                pos['ever_vaped'] = (target_x, -2)  # Move target to bottom
        else:
            pos = nx.spring_layout(self.discovered_graph, k=3, iterations=50)
        
        # Define node colors
        node_colors = []
        for node in self.discovered_graph.nodes():
            if node == 'ever_vaped':
                node_colors.append('#ff4444')  # Red for target
            elif node in ['age', 'sex', 'grade', 'race4']:
                node_colors.append('#4444ff')  # Blue for demographics
            elif node in ['q26', 'q27', 'q28', 'q29']:
                node_colors.append('#44ff44')  # Green for mental health
            elif node in ['q32', 'q33', 'q40', 'q41', 'q42', 'q43', 'q46', 'q47', 'q48', 'q49', 'q50', 'q51']:
                node_colors.append('#ffaa44')  # Orange for substance use
            else:
                node_colors.append('#dddddd')  # Gray for others
        
        # Draw graph
        nx.draw(self.discovered_graph, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8)
        
        plt.title('Discovered Causal Graph for Vaping Behavior', fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444ff', markersize=15, label='Demographics'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44', markersize=15, label='Mental Health'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffaa44', markersize=15, label='Substance Use'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', markersize=15, label='Target (Vaping)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#dddddd', markersize=15, label='Other Factors')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        plt.savefig('output/discovered_causal_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evidence_based_recommendations(self):
        """Generate recommendations based on rigorous causal analysis results."""
        print("\n" + "="*60)
        print("EVIDENCE-BASED INTERVENTION RECOMMENDATIONS")
        print("="*60)
        
        if 'multiple_treatments' not in self.results:
            print("No causal analysis results available for recommendations")
            return
        
        # Rank interventions by causal effect size
        intervention_effects = []
        
        for treatment_name, results in self.results['multiple_treatments'].items():
            econml_effect = results.get('econml_effect', 0)
            if econml_effect is not None:
                intervention_effects.append((treatment_name, abs(econml_effect), econml_effect))
        
        # Sort by absolute effect size
        intervention_effects.sort(key=lambda x: x[1], reverse=True)
        
        print("INTERVENTIONS RANKED BY CAUSAL EFFECT SIZE:")
        print("(Negative effects indicate protective factors)")
        print("-" * 60)
        
        for i, (treatment, abs_effect, raw_effect) in enumerate(intervention_effects, 1):
            direction = "↑ INCREASES" if raw_effect > 0 else "↓ DECREASES"
            print(f"{i}. {treatment}: {direction} vaping risk by {abs_effect:.4f}")
        
        # Generate specific recommendations
        print(f"\nSPECIFIC EVIDENCE-BASED RECOMMENDATIONS:")
        print("-" * 60)
        
        top_protective = [item for item in intervention_effects[:5] if item[2] < 0]
        top_risk = [item for item in intervention_effects[:5] if item[2] > 0]
        
        if top_protective:
            print("PROTECTIVE FACTORS TO STRENGTHEN:")
            for treatment, _, effect in top_protective:
                if "Mental Health" in treatment:
                    print(f"• Implement comprehensive mental health support programs")
                    print(f"  - Effect size: {abs(effect):.4f} reduction in vaping risk")
                elif "Physical Activity" in treatment:
                    print(f"• Promote physical activity and sports participation")
                elif "Academic" in treatment:
                    print(f"• Support academic achievement through tutoring and mentoring")
                elif "Sleep" in treatment:
                    print(f"• Educate about healthy sleep habits")
        
        if top_risk:
            print(f"\nRISK FACTORS TO TARGET FOR REDUCTION:")
            for treatment, _, effect in top_risk:
                if "Cigarette" in treatment or "Alcohol" in treatment or "Marijuana" in treatment:
                    print(f"• Strengthen substance use prevention programs")
                    print(f"  - Effect size: {effect:.4f} increase in vaping risk")
                elif "Fighting" in treatment or "Bullying" in treatment:
                    print(f"• Implement violence prevention and social-emotional learning")
        
        # Methodology note
        print(f"\nMETHODOLOGY NOTE:")
        print("These recommendations are based on rigorous causal inference methods including:")
        print("• Causal graph discovery using constraint-based algorithms")
        print("• Double machine learning for treatment effect estimation")
        print("• Multiple identification strategies with robustness checks")
        print("• Effect sizes represent change in probability of vaping initiation")
    
    def run_complete_causal_analysis(self):
        """Run the complete rigorous causal analysis."""
        print("="*80)
        print("RIGOROUS CAUSAL ANALYSIS - VAPING BEHAVIOR")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Discover causal structure
        self.discover_causal_structure(method='pc', alpha=0.01)
        
        # Plot discovered graph
        self.plot_causal_graph()
        
        # Analyze multiple treatments
        self.analyze_multiple_treatments()
        
        # Generate evidence-based recommendations
        self.generate_evidence_based_recommendations()
        
        print("\n" + "="*80)
        print("RIGOROUS CAUSAL ANALYSIS COMPLETE")
        print("="*80)

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    analyzer = RigorousCausalAnalysis()
    analyzer.run_complete_causal_analysis()