# Comprehensive Vaping Behavior Prediction Analysis

## Overview

This comprehensive analysis framework implements enhanced methodological best practices for predicting vaping behavior, following patterns from recent health behavior research literature. The analysis includes advanced modeling, interpretability analysis, and intersectionality considerations.

## Key Features

### 1. **Enhanced Target Definition**
- **Frequent Vaping**: ≥20 days of vaping in past 30 days (clinically relevant)
- **Ever Vaped**: Binary indicator for comparison
- Automatic selection based on data availability

### 2. **Comprehensive Feature Categorization**
Features are organized into meaningful domains:
- **Demographics**: Age, sex, grade, race, BMI
- **Mental Health**: Depression, suicidal ideation, sleep
- **Substance Use**: Alcohol, marijuana, tobacco, other drugs
- **Safety/Violence**: Weapon carrying, fights, threats
- **Bullying/Discrimination**: School bullying, online harassment
- **Sexual Behavior**: Partners, protection use, risk behaviors
- **Nutrition/Eating**: Diet patterns, weight management
- **Physical Activity**: Exercise, sports participation
- **Health Services**: Medical care, preventive services
- **Adverse Childhood Experiences**: Trauma, family dysfunction
- **Social Support**: Parental monitoring, peer relationships

### 3. **Advanced Missing Data Handling**
- **Primary**: Simple imputation (median/mode)
- **Sensitivity**: Multiple Imputation by Chained Equations (MICE)
- Comprehensive missingness reporting

### 4. **Sophisticated Modeling Approach**
- **Primary Model**: Random Forest with 10-fold CV hyperparameter tuning
- **60/40 train-test split** following literature best practices
- **SMOTE oversampling** for class balance in training
- **Comparison Models**: Logistic regression (demographics-only and full)

### 5. **Comprehensive Evaluation Metrics**
- **AUC-ROC**: Primary discriminative performance metric
- **C-Index**: Concordance index (equivalent to AUC for binary)
- **Average Precision**: Precision-recall performance
- **Calibration**: Reliability assessment with Brier score
- **Clinical Metrics**: Sensitivity, specificity, PPV, NPV

### 6. **Advanced Interpretability**
- **SHAP Values**: Model explanation and feature importance
- **Partial Dependence Plots**: Individual feature effects
- **Feature Importance**: Mean decrease in accuracy
- **Domain-wise Analysis**: Importance by behavioral domain

### 7. **Intersectionality Analysis**
- **Sociodemographic Interactions**: 9 key variables analyzed
- **Pairwise Interaction Strengths**: Variance-based quantification
- **Two-way Partial Dependence**: Visualization of strongest interactions
- **Health Equity Considerations**: Disparities assessment

### 8. **Comprehensive Sensitivity Analyses**
- **Multiple Imputation**: MICE vs simple imputation comparison
- **Restricted Analysis**: Current vapers only (when available)
- **Cross-validation Stability**: 10-fold CV robustness assessment
- **Model Comparison**: Complex vs simple approaches

## File Structure

```
comprehensive_analysis/
├── enhanced_vaping_analysis.py    # Core data analysis class
├── advanced_modeling.py           # Modeling and evaluation
├── interpretability_analysis.py   # SHAP, PDP, interactions
├── main_analysis.py              # Complete pipeline orchestration
├── requirements.txt              # Dependencies
├── README.md                     # This documentation
├── output/                       # Analysis results
│   ├── *.png                    # Visualizations
│   ├── *.csv                    # Summary tables
│   ├── *.txt                    # Reports
│   └── *.json                   # Metadata
└── notebooks/                    # Jupyter notebooks (optional)
```

## Installation and Setup

### 1. Install Dependencies

```bash
# Navigate to the comprehensive_analysis directory
cd comprehensive_analysis

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Data Files

Ensure these files are in the parent directory:
- `hstexas.csv` - Main dataset
- `variable_names.csv` - Variable name mapping

## Usage

### Quick Start - Complete Analysis

```python
from main_analysis import ComprehensiveAnalysisPipeline

# Initialize and run complete analysis
pipeline = ComprehensiveAnalysisPipeline()
pipeline.run_complete_analysis()
```

### Customized Analysis

```python
# Run with specific parameters
pipeline.run_complete_analysis(
    imputation_method='mice',     # 'simple' or 'mice'
    use_smote=True,              # Class balancing
    cv_folds=10,                 # Cross-validation folds
    top_n_features=20,           # Features for PDP
    run_sensitivity=True         # Include sensitivity analyses
)
```

### Phase-by-Phase Analysis

```python
# Run individual phases
pipeline = ComprehensiveAnalysisPipeline()

# Phase 1: Data preparation
pipeline.run_phase_1_data_preparation(imputation_method='simple')

# Phase 2: Advanced modeling
pipeline.run_phase_2_advanced_modeling(use_smote=True, cv_folds=10)

# Phase 3: Interpretability
pipeline.run_phase_3_interpretability_analysis(top_n_features=15)

# Phase 4: Sensitivity analyses
pipeline.run_sensitivity_analyses()

# Generate final report
pipeline.generate_final_report()
```

### Individual Module Usage

```python
# Data analysis only
from enhanced_vaping_analysis import ComprehensiveVapingAnalysis

analyzer = ComprehensiveVapingAnalysis()
df = analyzer.load_and_explore_data()
X, y = analyzer.prepare_features_for_modeling()

# Modeling only
from advanced_modeling import AdvancedVapingModeling

modeler = AdvancedVapingModeling()
modeler.prepare_training_data(X, y)
best_model = modeler.tune_random_forest()

# Interpretability only
from interpretability_analysis import InterpretabilityAnalysis

interpreter = InterpretabilityAnalysis()
interpreter.setup_analysis(best_model, X_test, y_test)
interpreter.analyze_sociodemographic_interactions()
```

## Output Files

### Visualizations
- `roc_curves_enhanced.png` - ROC curve comparison
- `model_comparison_comprehensive.png` - Performance metrics
- `calibration_analysis.png` - Model calibration
- `shap_feature_importance.png` - SHAP-based importance
- `partial_dependence_plots.png` - Top feature effects
- `interaction_pdp_plots.png` - Interaction effects

### Data Files
- `processed_data.csv` - Cleaned dataset
- `model_performance_summary.csv` - Performance metrics
- `analysis_metadata.json` - Analysis parameters

### Reports
- `comprehensive_analysis_report.txt` - Executive summary
- `interpretability_report.txt` - Detailed interpretability
- `analysis.log` - Processing log

## Methodological Alignment

This framework follows best practices from:

### Recent Literature Standards
- **Frequent vaping definition**: ≥20 days/month threshold
- **Feature domains**: Comprehensive risk factor categorization
- **Missing data**: Multiple approaches with sensitivity analysis
- **Modeling**: Random Forest with proper validation
- **Interpretability**: Advanced explanation methods
- **Intersectionality**: Sociodemographic interaction analysis

### Statistical Rigor
- **Train-test split**: 60/40 for adequate training data
- **Cross-validation**: 10-fold stratified CV
- **Class balance**: SMOTE in training only
- **Performance metrics**: Multiple complementary measures
- **Sensitivity analysis**: Robustness across assumptions

### Clinical Relevance
- **Meaningful outcomes**: Frequent vs ever use
- **Domain organization**: Intervention-relevant categories
- **Equity considerations**: Interaction analysis
- **Translation potential**: Clinical decision support

## Customization Options

### Target Variable
```python
# Use different target
X, y = analyzer.prepare_features_for_modeling('ever_vaped')

# Modify frequency threshold
analyzer.frequent_vaping_threshold = 15  # 15+ days instead of 20+
```

### Feature Selection
```python
# Exclude specific domains
analyzer.feature_domains.pop('sexual_identity_behavior', None)

# Add custom feature groupings
analyzer.feature_domains['custom_risk'] = ['q26', 'q46', 'q31']
```

### Model Parameters
```python
# Custom Random Forest parameters
custom_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
# Pass to tune_random_forest()
```

## Performance Expectations

### Typical Results
- **AUC**: 0.85-0.95 (excellent discrimination)
- **Sensitivity**: 0.80-0.90 (good case detection)
- **Specificity**: 0.85-0.95 (good specificity)
- **Calibration**: Well-calibrated predictions

### Computation Time
- **Complete analysis**: 10-30 minutes
- **Data preparation**: 2-5 minutes
- **Model tuning**: 5-15 minutes
- **Interpretability**: 3-10 minutes

### Memory Requirements
- **RAM**: 4-8 GB recommended
- **Storage**: 100-500 MB for outputs

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   ```python
   # Reduce sample sizes
   pipeline.run_phase_3_interpretability_analysis(top_n_features=10)
   ```

3. **Missing Variables**
   - Check variable_names.csv mapping
   - Verify column names in hstexas.csv

4. **Convergence Issues**
   ```python
   # Increase iterations for logistic regression
   # Reduce complexity for Random Forest
   ```

### Performance Optimization

```python
# Parallel processing
import os
os.environ['SCIKIT_LEARN_N_JOBS'] = '4'  # Use 4 cores

# Memory management
import gc
gc.collect()  # Force garbage collection
```

## Citation and Attribution

If using this framework, please cite:
- The original dataset source
- Relevant methodological papers
- This analysis framework

## Contact and Support

For questions about this framework:
- Review the code documentation
- Check the analysis logs
- Examine the output reports
- Modify parameters as needed

This comprehensive framework provides a robust foundation for advanced vaping behavior prediction analysis following current methodological best practices.