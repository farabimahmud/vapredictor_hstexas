"""
Enhanced Comprehensive Vaping Prediction Analysis
Following methodological best practices for health behavior prediction

This module implements a comprehensive approach to predicting vaping behavior
following patterns from recent literature, including:
- Frequent vaping definition (≥20 days in past 30)
- Multiple imputation strategies
- Feature categorization by domains
- Advanced interpretability analysis
- Intersectionality and interaction analysis
- Comprehensive sensitivity analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Some interpretability features will be limited.")

# Missing data handling
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    from sklearn.impute import SimpleImputer
    ITERATIVE_IMPUTER_AVAILABLE = False
    print("Warning: IterativeImputer not available. MICE imputation will be disabled.")

warnings.filterwarnings('ignore')

class ComprehensiveVapingAnalysis:
    """
    Enhanced comprehensive analysis for vaping behavior prediction
    following methodological best practices from health behavior research
    """
    
    def __init__(self, data_path: str = "../hstexas.csv", variable_names_path: str = "../variable_names.csv"):
        self.data_path = data_path
        self.variable_names_path = variable_names_path
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.df_raw = None
        self.df_processed = None
        self.variable_mapping = None
        self.feature_domains = {}
        
        # Model containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        # Analysis settings
        self.frequent_vaping_threshold = 20  # ≥20 days in past 30
        self.train_test_split_ratio = 0.6  # 60/40 split as in reference study
        self.cv_folds = 10
        self.random_state = 42
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the analysis"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_variable_mapping(self) -> Dict[str, str]:
        """Load variable name mapping from CSV file"""
        self.logger.info("Loading variable name mapping...")
        
        var_df = pd.read_csv(self.variable_names_path)
        mapping = {}
        
        for _, row in var_df.iterrows():
            var_code = row['name']
            if pd.notna(row['newname']) and row['newname'].strip():
                readable_name = row['newname'].strip()
            else:
                # Create readable name from variable code
                readable_name = self._create_readable_name(var_code)
            
            mapping[var_code] = readable_name
            
        self.variable_mapping = mapping
        self.logger.info(f"Loaded {len(mapping)} variable mappings")
        return mapping
    
    def _create_readable_name(self, var_code: str) -> str:
        """Create readable name for variables without mapping"""
        if var_code.startswith('q'):
            return f"Survey Question {var_code.upper()}"
        elif var_code.startswith('qn'):
            return f"Derived Variable {var_code.upper()}"
        else:
            return var_code.title().replace('_', ' ')
    
    def categorize_features_by_domain(self) -> Dict[str, List[str]]:
        """
        Categorize features into meaningful domains following health behavior literature
        """
        self.logger.info("Categorizing features by domain...")
        
        # Define feature domains based on content analysis
        domains = {
            'demographics': [
                'age', 'sex', 'grade', 'race4', 'race7', 'stheight', 'stweight', 
                'bmi', 'bmipct', 'qnobese', 'qnowt'
            ],
            'sexual_identity_behavior': [
                'sexid', 'sexid2', 'sexpart', 'sexpart2', 'transg', 'sextrans',
                'qtransgender', 'qntransgender', 'q56', 'q57', 'q58', 'q59', 
                'q60', 'q61', 'q62'
            ],
            'safety_violence': [
                'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 
                'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23'
            ],
            'bullying_discrimination': [
                'q24', 'q25', 'qtreatbadlyace', 'qunfairlyace', 'qunfairlydisc',
                'qntreatbadlyace', 'qnunfairlyace', 'qnunfairlydisc'
            ],
            'mental_health': [
                'q26', 'q27', 'q28', 'q29', 'q30', 'q84', 'q85', 'q86'
            ],
            'substance_use': [
                'q31', 'q32', 'q33', 'q34', 'q38', 'q39', 'q40', 'q41', 'q42', 
                'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49', 'q50', 'q51', 
                'q52', 'q53', 'q54', 'q55', 'qcurrentopioid', 'qhallucdrug',
                'qncurrentopioid', 'qnhallucdrug', 'qnillict'
            ],
            'nutrition_eating': [
                'q66', 'q67', 'q68', 'q69', 'q70', 'q71', 'q72', 'q73', 'q74', 
                'q75', 'qsportsdrink', 'qwater', 'qnsportsdrink', 'qnwater', 
                'qnwater1', 'qnwater2', 'qnwater3'
            ],
            'physical_activity': [
                'q76', 'q77', 'q78', 'q79', 'qmusclestrength', 'qnmusclestrength'
            ],
            'health_services': [
                'q81', 'q82', 'q83', 'qsunburn', 'qnsunburn'
            ],
            'social_media_technology': [
                'q80'
            ],
            'academic_performance': [
                'q87'
            ],
            'adverse_childhood_experiences': [
                'qbasicneedsace', 'qemoabuseace', 'qincarparentace', 'qintviolenceace',
                'qlivedwabuseace', 'qlivedwillace', 'qphyabuseace', 'qphyviolenceace',
                'qsexabuseace', 'qtalkadultace', 'qtalkfriendace', 'qverbalabuseace',
                'qnbasicneedsace', 'qnemoabuseace', 'qnincarparentace', 'qnintviolenceace',
                'qnlivedwabuseace', 'qnlivedwillace', 'qnphyabuseace', 'qnphyviolenceace',
                'qnsexabuseace', 'qntalkadultace', 'qntalkfriendace', 'qnverbalabuseace'
            ],
            'social_support': [
                'qclose2people', 'qparentalmonitoring', 'qnclose2people', 'qnparentalmonitoring'
            ]
        }
        
        # Add any qn variables that weren't categorized
        all_categorized = set()
        for domain_vars in domains.values():
            all_categorized.update(domain_vars)
        
        # Store the categorization
        self.feature_domains = domains
        
        # Report domain sizes
        for domain, variables in domains.items():
            self.logger.info(f"Domain '{domain}': {len(variables)} variables")
        
        return domains
    
    def load_and_explore_data(self) -> pd.DataFrame:
        """Load and perform initial exploration of the dataset"""
        self.logger.info("Loading dataset for comprehensive analysis...")
        
        # Load data
        self.df_raw = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded dataset with shape: {self.df_raw.shape}")
        
        # Load variable mapping
        self.load_variable_mapping()
        
        # Basic data exploration
        self.logger.info("Dataset Overview:")
        self.logger.info(f"- Total observations: {len(self.df_raw):,}")
        self.logger.info(f"- Total variables: {len(self.df_raw.columns):,}")
        self.logger.info(f"- Memory usage: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Check key vaping variables
        vaping_vars = ['q35', 'q36', 'qnfrevp', 'qndayevp']
        available_vaping_vars = [var for var in vaping_vars if var in self.df_raw.columns]
        self.logger.info(f"Available vaping variables: {available_vaping_vars}")
        
        # Missing data overview
        missing_summary = self.df_raw.isnull().sum()
        missing_pct = (missing_summary / len(self.df_raw) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        
        self.logger.info(f"Variables with >50% missing data: {len(high_missing)}")
        if len(high_missing) > 0:
            self.logger.info("Top variables with missing data:")
            for var, pct in high_missing.head(10).items():
                var_name = self.variable_mapping.get(var, var)
                self.logger.info(f"  {var} ({var_name}): {pct}% missing")
        
        return self.df_raw
    
    def create_target_variables(self) -> pd.DataFrame:
        """
        Create target variables for vaping behavior
        Primary: frequent vaping (≥20 days in past 30)
        Secondary: ever vaped (for comparison)
        """
        self.logger.info("Creating target variables...")
        
        df = self.df_raw.copy()
        
        # Primary target: Frequent vaping (≥20 days in past 30)
        if 'q36' in df.columns:
            # q36 appears to be current vaping - check its distribution
            q36_dist = df['q36'].value_counts().sort_index()
            self.logger.info(f"q36 (current vaping) distribution: {q36_dist.to_dict()}")
            
            # Based on YRBS coding: 1=0 days, 2=1-2 days, 3=3-5 days, 4=6-9 days, 
            # 5=10-19 days, 6=20-29 days, 7=all 30 days
            # Frequent vaping = categories 6 or 7 (≥20 days)
            df['frequent_vaping'] = ((df['q36'] >= 6) & (df['q36'] <= 7)).astype(int)
            
        elif 'qndayevp' in df.columns:
            # Alternative: use derived variable for days of vaping
            qndayevp_dist = df['qndayevp'].value_counts().sort_index()
            self.logger.info(f"qndayevp distribution: {qndayevp_dist.to_dict()}")
            
            df['frequent_vaping'] = (df['qndayevp'] >= self.frequent_vaping_threshold).astype(int)
        else:
            self.logger.warning("No suitable variable found for frequent vaping. Using ever vaped as primary target.")
            df['frequent_vaping'] = None
        
        # Secondary target: Ever vaped
        if 'q35' in df.columns:
            df['ever_vaped'] = (df['q35'] == 1).astype(int)
        elif 'qnfrevp' in df.columns:
            df['ever_vaped'] = (df['qnfrevp'] == 1).astype(int)
        else:
            raise ValueError("No suitable variable found for ever vaped")
        
        # Use appropriate primary target
        if df['frequent_vaping'].notna().any():
            frequent_rate = df['frequent_vaping'].mean()
            if frequent_rate > 0.01:  # At least 1% prevalence
                primary_target = 'frequent_vaping'
                target_description = f"≥{self.frequent_vaping_threshold} days vaping in past 30"
            else:
                self.logger.warning(f"Frequent vaping prevalence very low ({frequent_rate:.1%}). Using ever vaped as primary target.")
                primary_target = 'ever_vaped'
                target_description = "ever used e-cigarettes"
                df['frequent_vaping'] = df['ever_vaped']  # Use as fallback
        else:
            primary_target = 'ever_vaped'
            target_description = "ever used e-cigarettes"
            df['frequent_vaping'] = df['ever_vaped']  # Use as fallback
        
        # Report target distributions
        for target in ['frequent_vaping', 'ever_vaped']:
            if target in df.columns and df[target].notna().any():
                dist = df[target].value_counts()
                rate = df[target].mean()
                self.logger.info(f"{target} distribution: {dist.to_dict()}")
                self.logger.info(f"{target} prevalence: {rate:.2%}")
        
        self.logger.info(f"Primary target: {primary_target} ({target_description})")
        
        self.df_processed = df
        return df
    
    def prepare_features_for_modeling(self, target_variable: str = 'frequent_vaping') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling with comprehensive preprocessing
        """
        self.logger.info("Preparing features for modeling...")
        
        df = self.df_processed.copy()
        
        # Remove administrative and target-related columns
        exclude_cols = [
            'index', 'sitecode', 'sitename', 'sitetype', 'sitetypenum', 
            'weight', 'stratum', 'PSU', 'record', 'year', 'survyear',
            # Remove vaping-related variables to avoid leakage
            'q35', 'q36', 'q37', 'qnfrevp', 'qndayevp', 'qn35', 'qn36', 'qn37',
            'frequent_vaping', 'ever_vaped'
        ]
        
        # Remove columns that don't exist in the dataset
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        # Categorize features by domain
        self.categorize_features_by_domain()
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove variables with >80% missing data
        missing_pct = df[feature_cols].isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.8].index.tolist()
        feature_cols = [col for col in feature_cols if col not in high_missing_cols]
        
        self.logger.info(f"Removed {len(high_missing_cols)} variables with >80% missing data")
        self.logger.info(f"Final feature count: {len(feature_cols)}")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target_variable].copy()
        
        # Remove rows with missing target
        complete_cases = y.notna()
        X = X[complete_cases]
        y = y[complete_cases]
        
        self.logger.info(f"After removing missing targets: {len(X)} observations")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        self.logger.info(f"Target prevalence: {y.mean():.2%}")
        
        # Calculate missingness statistics
        overall_missing_rate = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        self.logger.info(f"Overall missingness rate: {overall_missing_rate:.1%}")
        
        return X, y
    
    def handle_missing_data(self, X: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Handle missing data using specified method
        
        Parameters:
        - method: 'simple' for median/mode imputation, 'mice' for multiple imputation
        """
        self.logger.info(f"Handling missing data using {method} imputation...")
        
        X_imputed = X.copy()
        
        if method == 'simple':
            # Simple imputation: median for numeric, mode for categorical
            for col in X_imputed.columns:
                if X_imputed[col].dtype in ['object', 'category']:
                    # Categorical - use mode
                    mode_val = X_imputed[col].mode()
                    if len(mode_val) > 0:
                        X_imputed[col] = X_imputed[col].fillna(mode_val[0])
                    else:
                        X_imputed[col] = X_imputed[col].fillna('Unknown')
                else:
                    # Numerical - use median
                    X_imputed[col] = X_imputed[col].fillna(X_imputed[col].median())
                    
        elif method == 'mice':
            # Multiple Imputation by Chained Equations
            if not ITERATIVE_IMPUTER_AVAILABLE:
                self.logger.warning("IterativeImputer not available. Falling back to simple imputation.")
                return self.handle_missing_data(X, method='simple')
            
            # Encode categorical variables first
            label_encoders = {}
            for col in X_imputed.columns:
                if X_imputed[col].dtype in ['object', 'category']:
                    le = LabelEncoder()
                    # Handle NaN values in fit
                    non_null_values = X_imputed[col].dropna()
                    if len(non_null_values) > 0:
                        le.fit(non_null_values.astype(str))
                        # Transform all values, filling NaN with -1 temporarily
                        temp_values = X_imputed[col].fillna('MISSING_VALUE').astype(str)
                        X_imputed[col] = le.transform(temp_values)
                        label_encoders[col] = le
                    else:
                        X_imputed[col] = 0
            
            # Apply MICE
            imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
            X_imputed_array = imputer.fit_transform(X_imputed)
            X_imputed = pd.DataFrame(X_imputed_array, columns=X_imputed.columns, index=X_imputed.index)
            
            # Decode categorical variables back
            for col, le in label_encoders.items():
                # Round to nearest integer for categorical variables
                X_imputed[col] = X_imputed[col].round().astype(int)
                # Clip to valid range
                X_imputed[col] = X_imputed[col].clip(0, len(le.classes_) - 1)
        
        # Final check for any remaining NaN values
        remaining_nan = X_imputed.isnull().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(f"Warning: {remaining_nan} NaN values remain, filling with 0...")
            X_imputed = X_imputed.fillna(0)
        
        self.logger.info(f"Imputation complete. Remaining NaN values: {X_imputed.isnull().sum().sum()}")
        
        return X_imputed
    
    def save_analysis_metadata(self, **kwargs):
        """Save analysis metadata and parameters"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'variable_names_path': str(self.variable_names_path),
            'frequent_vaping_threshold': self.frequent_vaping_threshold,
            'train_test_split_ratio': self.train_test_split_ratio,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'feature_domains': {k: len(v) for k, v in self.feature_domains.items()},
            **kwargs
        }
        
        metadata_path = self.output_dir / 'analysis_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Analysis metadata saved to {metadata_path}")

    def _get_feature_domain(self, feature: str) -> str:
        """Get domain for a feature"""
        for domain, features in self.feature_domains.items():
            if feature in features:
                return domain
        return 'other'


if __name__ == "__main__":
    # Example usage
    analysis = ComprehensiveVapingAnalysis()
    
    # Phase 1: Data loading and exploration
    print("="*80)
    print("COMPREHENSIVE VAPING BEHAVIOR ANALYSIS")
    print("Enhanced Methodology Following Best Practices")
    print("="*80)
    
    # Load and explore data
    df_raw = analysis.load_and_explore_data()
    
    # Create target variables
    df_processed = analysis.create_target_variables()
    
    # Prepare features
    X, y = analysis.prepare_features_for_modeling()
    
    # Handle missing data (simple imputation for now)
    X_imputed = analysis.handle_missing_data(X, method='simple')
    
    # Save initial analysis metadata
    analysis.save_analysis_metadata(
        dataset_shape=df_raw.shape,
        processed_shape=df_processed.shape,
        feature_count=len(X.columns),
        target_prevalence=y.mean()
    )
    
    print(f"\nPhase 1 Complete: Data prepared for modeling")
    print(f"- Dataset: {len(X)} observations, {len(X.columns)} features")
    print(f"- Target prevalence: {y.mean():.2%}")
    print(f"- Next: Implement advanced modeling and interpretability analysis")