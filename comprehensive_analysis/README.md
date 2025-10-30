# Comprehensive Vaping Behavior Prediction Analysis

This repository contains a comprehensive machine learning framework for predicting vaping behavior in adolescents using the Texas Youth Risk Behavior Survey (YRBS) data. The analysis follows enhanced methodological patterns from health behavior research literature and implements multiple advanced modeling approaches.

## 📊 Overview

The framework implements a complete pipeline including:
- **Data preprocessing** with domain-specific feature engineering
- **Four advanced machine learning models**: Random Forest, XGBoost, AdaBoost, and Logistic Regression
- **Comprehensive model evaluation** with ROC analysis and feature importance
- **Advanced interpretability analysis** with SHAP values and partial dependence plots
- **Enhanced reporting** with readable variable names and clinical interpretations

## 🎯 Key Results

- **Best Model**: XGBoost Classifier (AUC: 0.9697, 94.86% accuracy)
- **Dataset**: 32,324 observations with 157 features across 13 behavioral domains
- **Target**: Ever vaped e-cigarettes (8.56% prevalence)
- **Top Predictors**: Cigarette smoking behaviors, mental health factors, risk-taking behaviors

## 🛠️ Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 8GB (16GB recommended for full analysis)
- **Storage**: ~2GB free space for data and outputs

### Required Data Files
Ensure these files are present in the parent directory (`../`):
- `hstexas.csv` - Texas YRBS dataset
- `variable_names.csv` - Variable name mapping file

## 🚀 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd vapredictor_hstexas/comprehensive_analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost, shap; print('All dependencies installed successfully!')"
```

## 📋 Step-by-Step Instructions to Regenerate All Reports and Graphs

### Option 1: Complete Analysis Pipeline (Recommended)

Run the full comprehensive analysis pipeline:

```bash
python main_analysis.py
```

**What this generates:**
- Complete data preprocessing and feature engineering
- All four models with hyperparameter tuning
- Comprehensive model evaluation and comparison
- Advanced interpretability analysis with SHAP
- All reports and visualizations

**Estimated runtime**: 15-30 minutes (depending on system)

**Output files created:**
- `output/comprehensive_analysis_report.txt` - Main analysis report
- `output/model_comparison_comprehensive.png` - Model performance comparison
- `output/roc_curves_enhanced.png` - ROC curve comparison
- `output/shap_feature_importance.png` - SHAP feature importance
- `output/partial_dependence_plots.png` - Partial dependence analysis
- `output/calibration_analysis.png` - Model calibration curves
- `output/interaction_pdp_plots.png` - Interaction analysis
- `output/interpretability_report.txt` - Detailed interpretability analysis
- `output/model_performance_summary.csv` - Performance metrics table

### Option 2: Quick Model Comparison (Fast Alternative)

For a faster analysis focusing on model comparison:

```bash
python quick_test_models.py
```

**What this generates:**
- All four models with default/minimal tuning
- Model performance comparison
- Feature importance analysis with readable names
- ROC curve visualization

**Estimated runtime**: 2-5 minutes

**Output files created:**
- `output/quick_model_comparison.csv` - Performance metrics
- `output/quick_model_comparison_roc.png` - ROC curves
- Feature importance displayed in console

### Option 3: Enhanced Feature Importance Report

Generate detailed feature importance analysis with variable names:

```bash
python feature_importance_report.py
```

**What this generates:**
- Cross-model feature importance comparison
- Readable variable names and descriptions
- Domain-level importance analysis
- Clinical interpretation of findings

**Output files created:**
- `output/enhanced_feature_importance_report.txt` - Detailed feature analysis

### Option 4: Individual Analysis Components

Run specific components of the analysis:

#### A. Data Analysis Only
```bash
python -c "
from enhanced_vaping_analysis import ComprehensiveVapingAnalysis
analyzer = ComprehensiveVapingAnalysis('../hstexas.csv', '../variable_names.csv')
analyzer.load_and_explore_data()
analyzer.create_target_variables()
X, y = analyzer.prepare_features_for_modeling('ever_vaped')
X_final = analyzer.handle_missing_data(X)
print('Data preprocessing complete!')
"
```

#### B. Advanced Modeling Only
```bash
python -c "
from advanced_modeling import AdvancedVapingModeling
modeling = AdvancedVapingModeling('output')
# Add your modeling code here
print('Modeling complete!')
"
```

#### C. Interpretability Analysis Only
```bash
python -c "
from interpretability_analysis import InterpretabilityAnalysis
interpreter = InterpretabilityAnalysis('output')
# Add your interpretability code here
print('Interpretability analysis complete!')
"
```

## 📊 Understanding the Output Files

### Main Reports
1. **`comprehensive_analysis_report.txt`**
   - Executive summary with key findings
   - Model comparison results
   - Feature importance analysis
   - Clinical and policy implications

2. **`enhanced_feature_importance_report.txt`**
   - Detailed feature importance with variable names
   - Cross-model comparison
   - Domain-level analysis
   - Clinical interpretation

### Visualizations
1. **`model_comparison_comprehensive.png`**
   - Performance metrics comparison (AUC, Accuracy, Precision, Recall)
   - Bar charts for easy comparison

2. **`quick_model_comparison_roc.png`** / **`roc_curves_enhanced.png`**
   - ROC curves for all models
   - AUC values for each model

3. **`shap_feature_importance.png`**
   - SHAP feature importance summary
   - Top predictors with impact direction

4. **`partial_dependence_plots.png`**
   - Partial dependence plots for top features
   - Shows relationship between features and predictions

### Data Files
1. **`quick_model_comparison.csv`** / **`model_performance_summary.csv`**
   - Detailed performance metrics for all models
   - Includes AUC, accuracy, precision, recall, F1-score

2. **`processed_data.csv`**
   - Final preprocessed dataset used for modeling
   - Can be used for external analysis

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# If you get import errors, ensure all dependencies are installed:
pip install -r requirements.txt

# For specific packages:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap
```

#### 2. Memory Issues
```bash
# If you encounter memory issues, try the quick analysis:
python quick_test_models.py

# Or reduce the dataset size in the code
```

#### 3. Missing Data Files
```bash
# Ensure data files are in the correct location:
ls ../hstexas.csv ../variable_names.csv

# If files are missing, check the parent directory structure
```

#### 4. XGBoost Installation Issues
```bash
# On macOS with Apple Silicon:
conda install -c conda-forge xgboost

# Or using pip:
pip install xgboost --no-binary xgboost
```

#### 5. Long Runtime
```bash
# For faster execution, use fewer cross-validation folds:
# Edit the script to use cv_folds=3 instead of cv_folds=10

# Or run the quick version:
python quick_test_models.py
```

## 📈 Expected Outputs and Results

### Model Performance (Expected Results)
- **XGBoost**: AUC ~0.970, Accuracy ~94.9%
- **Random Forest**: AUC ~0.969, Accuracy ~94.7%
- **AdaBoost**: AUC ~0.958, Accuracy ~93.8%
- **Logistic Regression**: AUC ~0.957, Accuracy ~89.3%

### Top Predictive Features
1. Ever smoked cigarettes (qn85)
2. Current cigarette use frequency (q32)
3. Age first tried cigarettes (q85)
4. Binge drinking episodes (qn49)
5. Ever used hard drugs (q75)

### Key Findings
- Substance use variables dominate predictions
- Gateway substance hypothesis strongly supported
- Mental health factors show significant importance
- All models achieve excellent discrimination (AUC > 0.95)

## 🔬 Customization Options

### Modifying Analysis Parameters

#### Change Target Variable
```python
# In enhanced_vaping_analysis.py, modify:
X, y = analyzer.prepare_features_for_modeling('frequent_vaping')  # Instead of 'ever_vaped'
```

#### Adjust Model Parameters
```python
# In quick_test_models.py or advanced_modeling.py, modify:
rf_model = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=20,      # Increase depth
    random_state=42
)
```

#### Change Cross-Validation Folds
```python
# Modify cv_folds parameter:
modeling.tune_random_forest(cv_folds=5)  # Instead of 10
```

## 📝 File Structure

```
comprehensive_analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main_analysis.py                   # Complete analysis pipeline
├── quick_test_models.py              # Quick model comparison
├── feature_importance_report.py      # Enhanced feature analysis
├── enhanced_vaping_analysis.py       # Data preprocessing
├── advanced_modeling.py              # Machine learning models
├── interpretability_analysis.py      # SHAP and interpretability
├── output/                           # Generated reports and plots
│   ├── comprehensive_analysis_report.txt
│   ├── enhanced_feature_importance_report.txt
│   ├── *.png                        # Visualization files
│   └── *.csv                        # Results tables
└── __pycache__/                      # Python cache files
```

## 🤝 Contributing

To contribute to this analysis:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -am 'Add some feature'`
5. **Push to the branch**: `git push origin feature/your-feature`
6. **Submit a pull request**

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Verify all dependencies** are installed correctly
3. **Ensure data files** are in the correct location
4. **Check Python version** compatibility (3.8+)

## 📄 License

This project is part of academic research. Please cite appropriately if used in publications.

## 🏆 Acknowledgments

- Texas Youth Risk Behavior Survey (YRBS) for providing the dataset
- Enhanced methodological patterns from health behavior research literature
- Open source machine learning community for tools and libraries

---

**Last Updated**: October 30, 2025  
**Version**: 2.0 (Enhanced with XGBoost and AdaBoost)  
**Contact**: [Your contact information]