import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
import dowhy
from dowhy import CausalModel
import networkx as nx
from itertools import combinations
import scipy.stats as stats

# Causal discovery
try:
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.utils.cit import chisq, fisherz
    from causal_learn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("causal-learn not available, using alternative methods")

class AlgorithmicCausalAnalysis:
    """
    Advanced causal analysis with algorithmic confounder selection.
    """
    
    def __init__(self, data_path="data/hstexas_full.csv"):
        self.data_path = data_path
        self.df = None
        self.causal_df = None
        self.discovered_graph = None
        self.confounder_sets = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for algorithmic causal analysis."""
        print("Loading and preparing data for algorithmic causal analysis...")
        self.df = pd.read_csv(self.data_path)
        print(f"Original shape: {self.df.shape}")
        
        # Create target variables
        if 'q35' in self.df.columns:
            self.df['ever_vaped'] = (self.df['q35'] == 1).astype(int)
        
        # Remove administrative columns
        admin_cols = ['sitecode', 'sitename', 'sitetype', 'sitetypenum', 
                     'weight', 'stratum', 'PSU', 'record', 'year', 'survyear']
        self.df = self.df.drop(columns=[col for col in admin_cols if col in self.df.columns])
        
        # Select comprehensive variable set for causal discovery
        key_variables = [
            # Demographics
            'age', 'sex', 'grade', 'race4',
            # Mental health indicators
            'q26', 'q27', 'q28', 'q29',
            # Substance use behaviors
            'q32', 'q33', 'q40', 'q41', 'q42', 'q43', 'q46', 'q47', 'q48',
            'q49', 'q50', 'q51',
            # Risk behaviors
            'q8', 'q9', 'q10', 'q16', 'q24', 'q25',
            # Health behaviors
            'q76', 'q85', 'q87',
            # Additional behavioral indicators
            'q12', 'q13', 'q14', 'q15', 'q17', 'q18', 'q19', 'q20',
            # Target
            'ever_vaped'
        ]
        
        # Filter to available variables
        available_vars = [var for var in key_variables if var in self.df.columns]
        self.causal_df = self.df[available_vars].copy()
        
        # Convert variables appropriately for causal discovery
        print("Converting variables for causal discovery...")
        
        for col in self.causal_df.columns:
            if col == 'ever_vaped':
                continue  # Already binary
            elif col in ['age', 'grade']:
                # Keep as continuous for better causal discovery
                self.causal_df[col] = self.causal_df[col].fillna(self.causal_df[col].median())
            elif col in ['sex', 'race4']:
                # Convert to binary
                self.causal_df[col] = (self.causal_df[col] == 1).astype(int)
            else:
                # Convert behavioral variables to binary
                self.causal_df[col] = (self.causal_df[col] == 1).astype(int)
        
        # Handle missing values with more sophisticated approach
        print("Handling missing values...")
        for col in self.causal_df.columns:
            if self.causal_df[col].isnull().sum() > 0:
                if col in ['age', 'grade']:
                    # Use median for continuous
                    self.causal_df[col] = self.causal_df[col].fillna(self.causal_df[col].median())
                else:
                    # Use mode for binary/categorical
                    self.causal_df[col] = self.causal_df[col].fillna(0)
        
        # Remove rows with target variable missing
        if 'ever_vaped' in self.causal_df.columns:
            self.causal_df = self.causal_df.dropna(subset=['ever_vaped'])
        
        # Sample for computational efficiency if dataset is very large
        if len(self.causal_df) > 10000:
            print(f"Sampling 10,000 observations for causal discovery...")
            self.causal_df = self.causal_df.sample(n=10000, random_state=42)
        
        print(f"Final dataset shape for causal discovery: {self.causal_df.shape}")
        return self.causal_df
    
    def discover_causal_structure(self):
        """Use PC algorithm to discover causal structure."""
        print("\n" + "="*70)
        print("ALGORITHMIC CAUSAL STRUCTURE DISCOVERY")
        print("="*70)
        
        if not CAUSAL_LEARN_AVAILABLE:
            print("causal-learn not available, using correlation-based discovery")
            return self._correlation_based_discovery()
        
        # Prepare data for PC algorithm
        data_matrix = self.causal_df.values.astype(float)
        var_names = list(self.causal_df.columns)
        
        print(f"Running PC algorithm on {data_matrix.shape[0]} samples, {data_matrix.shape[1]} variables...")
        
        # Set up background knowledge (if any)
        bk = BackgroundKnowledge()
        
        # Add some domain knowledge constraints
        if 'age' in var_names and 'ever_vaped' in var_names:
            age_idx = var_names.index('age')
            vape_idx = var_names.index('ever_vaped')
            # Age cannot be caused by vaping
            bk.add_forbidden_by_node(vape_idx, age_idx)
        
        if 'sex' in var_names and 'ever_vaped' in var_names:
            sex_idx = var_names.index('sex')
            vape_idx = var_names.index('ever_vaped')
            # Sex cannot be caused by vaping
            bk.add_forbidden_by_node(vape_idx, sex_idx)
        
        try:
            # Run PC algorithm with chi-square test for discrete variables
            cg = pc(data_matrix, 
                   alpha=0.05,  # Significance level
                   indep_test=chisq,  # Chi-square for discrete/binary variables
                   background_knowledge=bk,
                   verbose=False)
            
            # Extract graph
            self.discovered_graph = cg.G
            
            # Analyze discovered structure
            self._analyze_discovered_graph(var_names)
            
            return cg
            
        except Exception as e:
            print(f"PC algorithm failed: {e}")
            print("Falling back to correlation-based discovery...")
            return self._correlation_based_discovery()
    
    def _correlation_based_discovery(self):
        """Fallback method using correlation analysis."""
        print("Using correlation-based causal discovery...")
        
        # Calculate correlation matrix
        corr_matrix = self.causal_df.corr()
        
        # Create adjacency matrix based on significant correlations
        n_vars = len(self.causal_df.columns)
        self.discovered_graph = np.zeros((n_vars, n_vars))
        
        var_names = list(self.causal_df.columns)
        
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i != j:
                    # Use correlation threshold and sample size for significance
                    corr = abs(corr_matrix.loc[var1, var2])
                    n = len(self.causal_df)
                    # Rough significance test
                    t_stat = corr * np.sqrt((n-2)/(1-corr**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                    
                    if p_value < 0.05 and corr > 0.1:  # Significant and meaningful
                        self.discovered_graph[i, j] = 1
        
        self._analyze_discovered_graph(var_names)
        return self.discovered_graph
    
    def _analyze_discovered_graph(self, var_names):
        """Analyze the discovered causal graph."""
        print(f"\nDiscovered Graph Analysis:")
        print(f"Variables: {len(var_names)}")
        
        if hasattr(self.discovered_graph, 'shape'):
            edges = np.sum(self.discovered_graph)
            print(f"Total edges: {edges}")
        else:
            edges = len([1 for i in range(len(var_names)) for j in range(len(var_names)) 
                        if i != j and self.discovered_graph[i,j] == 1])
            print(f"Total edges: {edges}")
        
        # Find parents of target variable
        if 'ever_vaped' in var_names:
            target_idx = var_names.index('ever_vaped')
            parents = []
            
            for i, var in enumerate(var_names):
                if i != target_idx and self.discovered_graph[i, target_idx] == 1:
                    parents.append(var)
            
            print(f"\nDiscovered parents of 'ever_vaped': {parents}")
            self.results['discovered_parents'] = parents
        
        # Store variable names for later use
        self.var_names = var_names
    
    def identify_optimal_confounders(self, treatment, outcome='ever_vaped'):
        """Use algorithmic methods to identify optimal confounders."""
        print(f"\n" + "="*70)
        print(f"ALGORITHMIC CONFOUNDER IDENTIFICATION: {treatment} → {outcome}")
        print("="*70)
        
        if treatment not in self.causal_df.columns:
            print(f"Treatment variable {treatment} not found")
            return []
        
        # Method 1: Graph-based identification
        graph_confounders = self._identify_graph_confounders(treatment, outcome)
        
        # Method 2: Cross-validation based selection
        cv_confounders = self._cross_validation_confounder_selection(treatment, outcome)
        
        # Method 3: Statistical significance based
        statistical_confounders = self._statistical_confounder_selection(treatment, outcome)
        
        # Combine methods
        all_candidates = set(graph_confounders + cv_confounders + statistical_confounders)
        
        print(f"\nConfounder identification results:")
        print(f"Graph-based: {graph_confounders}")
        print(f"Cross-validation: {cv_confounders}")
        print(f"Statistical: {statistical_confounders}")
        print(f"Combined candidates: {list(all_candidates)}")
        
        # Final selection using ensemble approach
        final_confounders = self._ensemble_confounder_selection(treatment, outcome, list(all_candidates))
        
        print(f"Final selected confounders: {final_confounders}")
        
        self.confounder_sets[treatment] = {
            'graph_based': graph_confounders,
            'cv_based': cv_confounders,
            'statistical': statistical_confounders,
            'final': final_confounders
        }
        
        return final_confounders
    
    def _identify_graph_confounders(self, treatment, outcome):
        """Identify confounders using discovered causal graph."""
        if self.discovered_graph is None or not hasattr(self, 'var_names'):
            return []
        
        var_names = self.var_names
        
        if treatment not in var_names or outcome not in var_names:
            return []
        
        treatment_idx = var_names.index(treatment)
        outcome_idx = var_names.index(outcome)
        
        confounders = []
        
        # Find variables that point to both treatment and outcome
        for i, var in enumerate(var_names):
            if (i != treatment_idx and i != outcome_idx and 
                self.discovered_graph[i, treatment_idx] == 1 and 
                self.discovered_graph[i, outcome_idx] == 1):
                confounders.append(var)
        
        return confounders
    
    def _cross_validation_confounder_selection(self, treatment, outcome, max_confounders=10):
        """Use cross-validation to select optimal confounder set."""
        potential_confounders = [col for col in self.causal_df.columns 
                               if col not in [treatment, outcome]]
        
        if len(potential_confounders) == 0:
            return []
        
        # Limit search space for computational efficiency
        if len(potential_confounders) > max_confounders:
            # Pre-select based on correlation with both treatment and outcome
            scores = []
            for conf in potential_confounders:
                treat_corr = abs(self.causal_df[conf].corr(self.causal_df[treatment]))
                outcome_corr = abs(self.causal_df[conf].corr(self.causal_df[outcome]))
                scores.append((conf, treat_corr * outcome_corr))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            potential_confounders = [conf for conf, _ in scores[:max_confounders]]
        
        print(f"Testing {len(potential_confounders)} potential confounders with CV...")
        
        best_score = -np.inf
        best_confounders = []
        
        # Test different confounder set sizes
        for n_conf in range(1, min(len(potential_confounders) + 1, 8)):
            # Test combinations
            for confounders in combinations(potential_confounders, n_conf):
                score = self._evaluate_confounder_set(treatment, outcome, list(confounders))
                
                if score > best_score:
                    best_score = score
                    best_confounders = list(confounders)
        
        print(f"Best CV score: {best_score:.4f}")
        return best_confounders
    
    def _statistical_confounder_selection(self, treatment, outcome, p_threshold=0.05):
        """Select confounders based on statistical significance."""
        potential_confounders = [col for col in self.causal_df.columns 
                               if col not in [treatment, outcome]]
        
        confounders = []
        
        for conf in potential_confounders:
            # Test association with treatment
            treat_test = self._test_association(conf, treatment)
            # Test association with outcome
            outcome_test = self._test_association(conf, outcome)
            
            if treat_test < p_threshold and outcome_test < p_threshold:
                confounders.append(conf)
        
        return confounders
    
    def _test_association(self, var1, var2):
        """Test statistical association between two variables."""
        try:
            # For binary variables, use chi-square test
            if (self.causal_df[var1].nunique() <= 2 and 
                self.causal_df[var2].nunique() <= 2):
                
                crosstab = pd.crosstab(self.causal_df[var1], self.causal_df[var2])
                chi2, p_value, _, _ = stats.chi2_contingency(crosstab)
                return p_value
            
            # For continuous vs binary, use t-test
            elif (self.causal_df[var1].nunique() > 2 and 
                  self.causal_df[var2].nunique() <= 2):
                
                group0 = self.causal_df[self.causal_df[var2] == 0][var1]
                group1 = self.causal_df[self.causal_df[var2] == 1][var1]
                _, p_value = stats.ttest_ind(group0, group1)
                return p_value
            
            # For continuous variables, use correlation test
            else:
                corr, p_value = stats.pearsonr(self.causal_df[var1], self.causal_df[var2])
                return p_value
                
        except:
            return 1.0  # Return non-significant if test fails
    
    def _evaluate_confounder_set(self, treatment, outcome, confounders):
        """Evaluate a confounder set using cross-validation."""
        if len(confounders) == 0:
            return -np.inf
        
        try:
            # Prepare data
            X_vars = [treatment] + confounders
            X = self.causal_df[X_vars]
            y = self.causal_df[outcome]
            
            # Use appropriate model based on outcome type
            if y.nunique() <= 2:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = LinearRegression()
                scoring = 'neg_mean_squared_error'
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            return np.mean(cv_scores)
            
        except Exception as e:
            return -np.inf
    
    def _ensemble_confounder_selection(self, treatment, outcome, candidates):
        """Final confounder selection using ensemble approach."""
        if len(candidates) == 0:
            return []
        
        # Score each candidate
        candidate_scores = {}
        
        for candidate in candidates:
            score = 0
            
            # Add points for being identified by different methods
            methods = self.confounder_sets.get(treatment, {})
            if candidate in methods.get('graph_based', []):
                score += 3  # Higher weight for graph-based
            if candidate in methods.get('cv_based', []):
                score += 2
            if candidate in methods.get('statistical', []):
                score += 1
            
            # Add points for statistical significance
            treat_p = self._test_association(candidate, treatment)
            outcome_p = self._test_association(candidate, outcome)
            
            if treat_p < 0.01 and outcome_p < 0.01:
                score += 2
            elif treat_p < 0.05 and outcome_p < 0.05:
                score += 1
            
            candidate_scores[candidate] = score
        
        # Select top candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top candidates (max 6 to avoid overfitting)
        final_confounders = [cand for cand, score in sorted_candidates[:6] if score > 0]
        
        return final_confounders
    
    def estimate_causal_effects_with_sensitivity(self, treatment, outcome='ever_vaped'):
        """Estimate causal effects with sensitivity analysis."""
        print(f"\n" + "="*70)
        print(f"CAUSAL EFFECT ESTIMATION WITH SENSITIVITY ANALYSIS")
        print(f"Treatment: {treatment} → Outcome: {outcome}")
        print("="*70)
        
        # Get optimal confounders
        confounders = self.identify_optimal_confounders(treatment, outcome)
        
        if len(confounders) == 0:
            print("No confounders identified, using minimal adjustment set")
            confounders = ['age', 'sex'] if all(c in self.causal_df.columns for c in ['age', 'sex']) else []
        
        # Estimate effect with different confounder sets
        sensitivity_results = {}
        
        # 1. No adjustment
        effect_unadjusted = self._estimate_effect_dowhy(treatment, outcome, [])
        sensitivity_results['unadjusted'] = effect_unadjusted
        
        # 2. Minimal adjustment (demographics only)
        minimal_confounders = [c for c in ['age', 'sex', 'grade', 'race4'] 
                              if c in self.causal_df.columns and c != treatment]
        effect_minimal = self._estimate_effect_dowhy(treatment, outcome, minimal_confounders)
        sensitivity_results['minimal'] = effect_minimal
        
        # 3. Algorithmic confounders
        effect_algorithmic = self._estimate_effect_dowhy(treatment, outcome, confounders)
        sensitivity_results['algorithmic'] = effect_algorithmic
        
        # 4. Full adjustment (all available confounders)
        full_confounders = [c for c in self.causal_df.columns 
                           if c not in [treatment, outcome]][:15]  # Limit for stability
        effect_full = self._estimate_effect_dowhy(treatment, outcome, full_confounders)
        sensitivity_results['full'] = effect_full
        
        # Report results
        print(f"\nSensitivity Analysis Results:")
        print("-" * 50)
        for method, effect in sensitivity_results.items():
            if effect is not None:
                print(f"{method:15s}: {effect:8.4f}")
            else:
                print(f"{method:15s}: Failed")
        
        # Calculate sensitivity metrics
        valid_effects = [e for e in sensitivity_results.values() if e is not None]
        if len(valid_effects) > 1:
            effect_range = max(valid_effects) - min(valid_effects)
            effect_std = np.std(valid_effects)
            print(f"\nSensitivity Metrics:")
            print(f"Effect range: {effect_range:.4f}")
            print(f"Effect std:   {effect_std:.4f}")
            print(f"Robustness:   {'High' if effect_std < 0.02 else 'Medium' if effect_std < 0.05 else 'Low'}")
        
        return sensitivity_results
    
    def _estimate_effect_dowhy(self, treatment, outcome, confounders):
        """Estimate causal effect using DoWhy."""
        try:
            # Prepare data
            analysis_vars = [treatment, outcome] + confounders
            analysis_df = self.causal_df[analysis_vars].copy()
            
            # Create causal graph
            graph_str = f"digraph {{ "
            
            # Add confounder relationships
            for conf in confounders:
                graph_str += f"{conf} -> {treatment}; {conf} -> {outcome}; "
            
            # Add treatment effect
            graph_str += f"{treatment} -> {outcome}; "
            graph_str += "}"
            
            # Create causal model
            model = CausalModel(
                data=analysis_df,
                treatment=treatment,
                outcome=outcome,
                graph=graph_str
            )
            
            # Identify and estimate effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified_estimand, 
                method_name="backdoor.linear_regression"
            )
            
            return estimate.value
            
        except Exception as e:
            print(f"DoWhy estimation failed: {e}")
            return None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive algorithmic causal analysis."""
        print("="*80)
        print("ALGORITHMIC CAUSAL ANALYSIS - ADVANCED METHODS")
        print("Using Causal Discovery + Cross-Validation + Sensitivity Analysis")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Discover causal structure
        self.discover_causal_structure()
        
        # Analyze key treatments
        treatments = {
            'q26': 'Mental Health Issues',
            'q32': 'Ever Cigarette Use', 
            'q42': 'Current Alcohol Use',
            'q46': 'Ever Marijuana Use',
            'q85': 'Adequate Sleep'
        }
        
        all_results = {}
        
        for treatment_var, treatment_name in treatments.items():
            if treatment_var in self.causal_df.columns:
                print(f"\n{'='*60}")
                print(f"ANALYZING: {treatment_name} ({treatment_var})")
                print('='*60)
                
                # Estimate with sensitivity analysis
                sensitivity_results = self.estimate_causal_effects_with_sensitivity(treatment_var)
                all_results[treatment_name] = sensitivity_results
        
        # Generate comprehensive report
        self._generate_algorithmic_report(all_results)
        
        print("\n" + "="*80)
        print("ALGORITHMIC CAUSAL ANALYSIS COMPLETE")
        print("="*80)
        
        return all_results
    
    def _generate_algorithmic_report(self, all_results):
        """Generate comprehensive report of algorithmic analysis."""
        print(f"\n" + "="*80)
        print("COMPREHENSIVE ALGORITHMIC CAUSAL ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n1. CAUSAL DISCOVERY RESULTS:")
        print("-" * 40)
        if 'discovered_parents' in self.results:
            print(f"Discovered causal parents of vaping: {self.results['discovered_parents']}")
        else:
            print("No direct causal parents discovered")
        
        print(f"\n2. ROBUST CAUSAL EFFECTS (Algorithmic Selection):")
        print("-" * 60)
        
        robust_effects = []
        
        for treatment, sensitivity_results in all_results.items():
            # Use algorithmic estimate as primary
            primary_effect = sensitivity_results.get('algorithmic')
            
            # Calculate robustness
            valid_effects = [e for e in sensitivity_results.values() if e is not None]
            
            if len(valid_effects) > 1 and primary_effect is not None:
                effect_std = np.std(valid_effects)
                robustness = "High" if effect_std < 0.02 else "Medium" if effect_std < 0.05 else "Low"
                
                robust_effects.append({
                    'treatment': treatment,
                    'effect': primary_effect,
                    'robustness': robustness,
                    'std': effect_std
                })
        
        # Sort by effect magnitude
        robust_effects.sort(key=lambda x: abs(x['effect']), reverse=True)
        
        for i, result in enumerate(robust_effects, 1):
            effect = result['effect']
            direction = "increases" if effect > 0 else "decreases"
            
            print(f"{i}. {result['treatment']:25s}: {direction} vaping by {abs(effect):.4f}")
            print(f"   Robustness: {result['robustness']} (std: {result['std']:.4f})")
        
        print(f"\n3. METHODOLOGICAL ADVANTAGES:")
        print("-" * 40)
        print("✓ Algorithmic causal structure discovery")
        print("✓ Cross-validation based confounder selection")
        print("✓ Multiple method ensemble approach")
        print("✓ Comprehensive sensitivity analysis")
        print("✓ Robustness assessment for each estimate")
        
        print(f"\n4. CONFIDENCE ASSESSMENT:")
        print("-" * 40)
        high_confidence = [r for r in robust_effects if r['robustness'] == 'High']
        medium_confidence = [r for r in robust_effects if r['robustness'] == 'Medium']
        low_confidence = [r for r in robust_effects if r['robustness'] == 'Low']
        
        print(f"High confidence effects: {len(high_confidence)}")
        print(f"Medium confidence effects: {len(medium_confidence)}")
        print(f"Low confidence effects: {len(low_confidence)}")
        
        if high_confidence:
            print(f"\nMOST RELIABLE CAUSAL EFFECTS:")
            for result in high_confidence:
                effect = result['effect']
                direction = "Risk factor" if effect > 0 else "Protective factor"
                print(f"- {result['treatment']}: {direction} (effect: {effect:+.4f})")

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    analyzer = AlgorithmicCausalAnalysis()
    analyzer.run_comprehensive_analysis()