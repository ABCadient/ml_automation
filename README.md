<<<<<<< HEAD
# ML Automation App

A comprehensive Streamlit-based machine learning automation tool that provides end-to-end ML workflow automation with advanced features for model comparison, explanation, and management.

## Features

### 🚀 Core Features
- **CSV Data Processing**: Upload and process CSV files with automatic data cleaning
- **Multi-Model Training**: Support for 6+ ML algorithms with automatic comparison
- **Interactive Visualizations**: Rich EDA and model analysis capabilities
- **Explainable AI**: Multiple explanation methods (SHAP, LIME, Feature Importance, PDP)

### 🆕 Enhanced Features (New)

#### 1. **Model Persistence**
- Save trained models with all components (scaler, feature names, target column)
- Load saved models for reuse
- Automatic model management with file organization

#### 2. **Hyperparameter Tuning**
- Automated GridSearchCV for optimal parameter selection
- Support for all major algorithms (Random Forest, XGBoost, LightGBM, etc.)
- Configurable parameter grids for each model type

#### 3. **Automated Feature Selection**
- **Recursive Feature Elimination (RFE)**: Eliminates features recursively
- **SelectKBest**: Selects top K features based on statistical tests
- Configurable number of features to select
- Automatic feature importance ranking

#### 4. **Model Type Differentiation**
- **Automatic Problem Detection**: Determines if your problem is classification or regression
- **Manual Override**: Option to manually specify problem type
- **Appropriate Model Selection**: Shows relevant models based on problem type
- **Different Metrics**: Classification metrics (Accuracy, Precision, Recall, F1) vs Regression metrics (R², MSE, RMSE, MAE)

## Supported Algorithms

### Classification Models
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- LightGBM
- Extra Trees
- SVM

### Regression Models
- Linear Regression
- Random Forest
- XGBoost
- Gradient Boosting
- LightGBM
- Extra Trees
- SVR

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Usage

### 1. Data Upload
- Upload your CSV file using the sidebar
- The app will automatically analyze your data

### 2. Problem Type Selection
- Choose between "Auto-detect", "Classification", or "Regression"
- Auto-detect will analyze your target variable to determine the problem type

### 3. Model Selection
- Select the models you want to train
- Available models change based on problem type

### 4. Advanced Options
- **Feature Selection**: Enable automated feature selection with RFE or SelectKBest
- **Hyperparameter Tuning**: Enable GridSearchCV for optimal parameters
- **Number of Features**: Specify how many features to select

### 5. Training and Results
- Click "Train Models" to start the training process
- View performance metrics and comparisons
- Download results as CSV

### 6. Model Management
- Save your best models for future use
- Load previously saved models
- View model metadata (problem type, target column, features)

## App Structure

### Tabs
1. **Data Analysis**: Comprehensive EDA with visualizations
2. **Model Training**: Train and compare multiple models
3. **Model Explanation**: Detailed model analysis with SHAP, LIME, etc.
4. **Model Management**: Save and load models

### Key Components
- `MLComparison` class: Core orchestrator for model training
- `utils.py`: Data analysis and model explanation functions
- Session state management for persistent data
- Automatic problem type detection
- Feature engineering pipeline

## Advanced Features

### Model Persistence
```python
# Save model
model_path = ml_comp.save_model(model, scaler, feature_names, model_name, target_col)

# Load model
model, scaler, feature_names, target_col, problem_type = ml_comp.load_model(model_name)
```

### Feature Selection
```python
# Automated feature selection
X_selected, selected_features = ml_comp.automated_feature_selection(X, y, method='rfe', n_features=20)
```

### Hyperparameter Tuning
```python
# GridSearchCV for optimal parameters
tuned_model, best_params = ml_comp.hyperparameter_tuning(model, X_train, y_train, model_name)
```

## Data Requirements

### Input Format
- CSV file with headers
- Numeric and categorical features supported
- Missing values handled automatically

### Target Variable
- **Classification**: Binary or multi-class (automatically detected)
- **Regression**: Continuous numeric values

## Performance Metrics

### Classification
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC AUC (for binary classification)

### Regression
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Dependencies

- **Streamlit**: Web interface
- **Scikit-learn**: ML algorithms and utilities
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Plotly**: Interactive plots
- **SHAP**: Model explanations
- **LIME**: Local explanations
- **PDPbox**: Partial dependence plots
- **Joblib**: Model persistence

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML algorithms from [Scikit-learn](https://scikit-learn.org/)
- Model explanations with [SHAP](https://github.com/slundberg/shap) and [LIME](https://github.com/marcotcr/lime) 
=======
# ml_automation
This app automated the ML related tasks and steps.
>>>>>>> caf8d25a67c729935dcff5d9f5e2cf80dced3482
