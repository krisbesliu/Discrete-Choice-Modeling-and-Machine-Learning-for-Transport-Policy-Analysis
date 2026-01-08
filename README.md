# Discrete-Choice-Modeling-and-Machine-Learning-for-Transport-Policy-Analysis
An empirical comparison of classical discrete choice models and machine learning methods applied to urban transport mode choice. The project focuses on predictive accuracy, substitution patterns, elasticity estimation, and counterfactual policy simulations using real-world travel data.

This project implements and compares multiple machine learning and econometric models for predicting transport mode choices (Public Transit, Drive, Cycle, Walk) using trip-level data.

## Project Overview

The analysis includes three complementary modeling approaches:
1. **Neural Networks (NN)** - Deep learning approach for mode choice prediction
2. **Multinomial Logit / Nested Logit (MNL/NL)** - Traditional econometric discrete choice models
3. **Tree-Based Ensemble Methods** - Random Forest, LightGBM, and XGBoost models

Each approach is implemented in a separate Jupyter notebook with comprehensive model training, hyperparameter tuning, evaluation, and interpretability analysis.

---

## Project Structure

```
Advanced Data Analysis Code/
├── Data/
│   └── dataset.csv                          # Transport mode choice dataset
├── Random Forest/
│   └── Transport_RandomForest_FE.ipynb      # Tree-based models notebook
├── Neural Networks/
│   └── NN_Notebook.ipynb                    # Neural network models notebook
├── Econometric Models/
│   └── MNL_NL_Notebook.ipynb                # Discrete choice models notebook
└── README.md                                # This file
```

### Data Requirements

**Input File**: `Data/dataset.csv`
- Contains trip-level records with the following information:
  - **Target Variable**: `travel_mode` (PT, Drive, Cycle, Walk)
  - **Temporal Features**: `travel_month`, `travel_date`, `day_of_week`, `start_time_linear`
  - **Distance Features**: `distance`
  - **Travel Time**: `dur_walking`, `dur_cycling`, `dur_pt_total`, `dur_driving`
  - **Cost Features**: `cost_transit`, `cost_driving_total`
  - **Traffic Features**: `driving_traffic_percent`
  - **PT Features**: `dur_pt_rail`, `dur_pt_bus`, `dur_pt_access`, `dur_pt_int_walking`, `dur_pt_int_waiting`, `pt_n_interchanges`
  - **Individual Characteristics**: `car_ownership`, `female`, `driving_license`, `age`
  - **Household Variables**: `household_id`, `person_n`
  - **Purpose**: `purpose` (categorical)

---

## Required Libraries & Dependencies

### Core Data Science Stack
```
pandas >= 2.2.3
numpy >= 2.1.3
matplotlib >= 3.10.7
seaborn >= 0.13.2
scikit-learn >= 1.6.1
```

### Tree-Based Models (Tree_Based_Notebook)
```
lightgbm >= 4.6.0
xgboost >= 3.1.2
```

### Neural Networks (NN_Notebook)
```
tensorflow >= 2.20.0
keras >= 3.11.3
```

### Econometric Models (MNL_NL_Notebook)
```
scipy >= 1.15.3
biogeme >= 3.2.13
```
**Biogeme Package Details**:
- **Name**: biogeme
- **Version**: 3.2.13
- **Summary**: Estimation and application of discrete choice models
- **Home-page**: http://biogeme.epfl.ch

### Interpretability & Analysis
```
shap >= 0.50.0
scikit-optimize >= 0.10.2
```

### Utilities
```
openpyxl >= 3.1.5  # Excel export
warnings (built-in)
```

### Installation

Install all dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost tensorflow keras scipy scikit-optimize shap openpyxl "biogeme==3.2.13"
```

Or create a virtual environment with the included requirements.txt:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Notebook Descriptions & Execution Guide

### 1. Tree-Based Models Notebook (`Tree_Based_Notebook.ipynb`)

**Purpose**: Compare ensemble tree-based models for transport mode choice prediction

**Key Components**:
- Data Loading, Feature Engineering, and Preparation
- Model Training and Evaluation
- Scenario Analysis & Counterfactuals
- Partial Dependence Analysis
- Comparative Model Analysis
- SHAP Interpretability Analysis

**Execution Order**:
1. Run Section 1-4 sequentially (data loading and preparation)
2. Run Section 5 to train models (do not re-run the bayesian hyperparameter tuning as results are saved)
3. Run Sections 6-9 for analysis and interpretation


### 2. Neural Networks Notebook (`NN_Notebook.ipynb`)

**Purpose**: Apply deep learning to transport mode choice prediction

**Key Components**:
- Data loading and preprocessing
- Neural network architecture design
- Layer configuration (dense, dropout, batch normalization)
- Training with validation and early stopping
- Hyperparameter tuning (learning rate, batch size, epochs)
- Model evaluation (accuracy, loss curves, confusion matrices)
- Feature importance analysis
- Sensitivity analysis

**Execution Order**:
1. Load and preprocess data (normalize/standardize features)
2. Define and compile neural network architecture
3. Train model with cross-validation (do not re-run NAS as results are saved)
4. Evaluate performance on test set
5. Generate visualizations and analysis

---

### 3. Multinomial Logit / Nested Logit Notebook (`MNL_NL_Notebook.ipynb`)

**Purpose**: Apply traditional econometric discrete choice models for interpretation and policy analysis

**Key Components**:
- Data preparation for discrete choice analysis
- Nested Logit (NL) specification testing
- Multinomial Logit (MNL) estimation
- Parameter estimation and significance testing
- Price and time elasticities
- Policy scenario simulations
- Market share predictions

**Execution Order**:
1. Load and format data for choice models
2. Estimate MNL model
3. Test for IIA (Independence of Irrelevant Alternatives)
4. Estimate NL model if IIA violated
5. Calculate elasticities
6. Run policy scenarios
7. Generate summary tables and visualizations

---

## Project Workflow Diagram

```
1. Data Loading & Exploration
         ↓
2. Feature Engineering
         ↓
3. Data Preparation & Encoding
         ↓
4. Train/Test Split (80/20)
         ├─────────────────┬──────────────────┐
         ↓                 ↓                  ↓
   Tree-Based Models  Neural Networks    Econometric Models
   (Random Forest,    (Multi-layer       (MNL, Nested Logit)
    LightGBM,         Perceptron)
    XGBoost)
         ├─────────────────┼──────────────────┤
         ↓                 ↓                  ↓
   Model Evaluation      Model           Elasticity &
   - Metrics            Training        Policy Analysis
   - Feature Imp.        - Validation
   - SHAP Analysis       - Tuning
         ├─────────────────┼──────────────────┤
         └─────────────────┴──────────────────┘
                        ↓
            5. Comparative Analysis
            6. Scenario Analysis
            7. Report Generation
```

## Author Notes

- These notebooks implement best practices in machine learning and econometric modeling
- All models use 80/20 train/test split with random_state=0 for reproducibility
- Hyperparameter tuning uses cross-validation (3-10 folds) to prevent overfitting
- Results are validated using multiple metrics (accuracy, F1, NLL) to ensure robustness

---

