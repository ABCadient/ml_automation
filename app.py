import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import joblib
import os
from utils import create_data_analysis_tab, create_best_model_tab

# Set page configuration with competition branding
st.set_page_config(
    page_title="AI-Powered ML Automation Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Competition branding and styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .competition-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Competition header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered Machine Learning Automation Platform</h1>
    <p>Democratizing AI/ML for Everyone - No Code Required</p>
</div>
""", unsafe_allow_html=True)

# Competition badge
st.markdown("""
<div class="competition-badge">
    🏆 Mission AI Possible Competition Entry 🏆
</div>
""", unsafe_allow_html=True)

# Demo datasets for competition
DEMO_DATASETS = {
    "Employee Tenure Prediction": {
        "description": "Predict employee retention based on various features",
        "url": "https://raw.githubusercontent.com/datasets/employee-attrition/main/data.csv",
        "type": "classification"
    },
    "House Price Prediction": {
        "description": "Predict house prices using regression",
        "url": "https://raw.githubusercontent.com/datasets/house-prices/main/data.csv", 
        "type": "regression"
    },
    "Customer Churn Prediction": {
        "description": "Predict customer churn for telecom company",
        "url": "https://raw.githubusercontent.com/datasets/customer-churn/main/data.csv",
        "type": "classification"
    }
}

def load_demo_data(dataset_name):
    """Load demo dataset or create synthetic data for competition"""
    if dataset_name == "Employee Tenure Prediction":
        # Create synthetic employee data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'Age': np.random.normal(35, 10, n_samples),
            'Salary': np.random.normal(60000, 20000, n_samples),
            'YearsExperience': np.random.normal(8, 5, n_samples),
            'EducationLevel': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
            'WorkHistoryDays': np.random.exponential(365, n_samples),
            'EducationCount': np.random.poisson(2, n_samples),
            'WorkHistoryCount': np.random.poisson(3, n_samples),
            'ReferenceCount': np.random.poisson(2, n_samples),
            'Distance': np.random.exponential(20, n_samples),
            'AvailabilityAfterDays': np.random.exponential(30, n_samples),
            'WorkPreference': np.random.choice(['Remote', 'Hybrid', 'Office'], n_samples),
            'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples),
            'SourceCategory': np.random.choice(['Job Board', 'Referral', 'Recruiter', 'Social Media'], n_samples),
            'JobCategory': np.random.choice(['Technical', 'Non-Technical', 'Management'], n_samples),
            'ReasonForLeaving': np.random.choice(['Career Growth', 'Better Pay', 'Relocation', 'Retirement'], n_samples),
            'IsCurrentlyEmployed': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Tenure_orig': np.random.exponential(365, n_samples)
        })
        
        # Create target variable (long tenure = 1, short tenure = 0)
        data['BinTenure'] = (data['Tenure_orig'] > 90).astype(int)
        
    elif dataset_name == "House Price Prediction":
        # Create synthetic house price data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'SquareFeet': np.random.normal(2000, 500, n_samples),
            'Bedrooms': np.random.poisson(3, n_samples),
            'Bathrooms': np.random.poisson(2, n_samples),
            'YearBuilt': np.random.randint(1980, 2020, n_samples),
            'LotSize': np.random.exponential(5000, n_samples),
            'GarageSpaces': np.random.poisson(2, n_samples),
            'DistanceToCity': np.random.exponential(10, n_samples),
            'SchoolRating': np.random.uniform(1, 10, n_samples),
            'CrimeRate': np.random.exponential(0.1, n_samples),
            'Price': np.random.normal(300000, 100000, n_samples)
        })
        
        # Adjust price based on features
        data['Price'] = (data['SquareFeet'] * 100 + 
                        data['Bedrooms'] * 20000 + 
                        data['Bathrooms'] * 15000 +
                        (2024 - data['YearBuilt']) * -1000 +
                        data['SchoolRating'] * 5000 +
                        np.random.normal(0, 20000, n_samples))
        
    elif dataset_name == "Customer Churn Prediction":
        # Create synthetic customer churn data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'Age': np.random.normal(45, 15, n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples),
            'TotalCharges': np.random.exponential(2000, n_samples),
            'ContractLength': np.random.choice([1, 12, 24], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber', 'None'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic', 'Mailed', 'Bank Transfer', 'Credit Card'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
    
    return data

class MLComparison:
    def __init__(self, selected_models, problem_type='auto'):
        # Define models for both classification and regression
        classification_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            'XGBoost': xgb.XGBClassifier(scale_pos_weight=2/3,
                                        max_depth=6,
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100,
                                          learning_rate=0.1,
                                          max_depth=6,
                                          random_state=42,
                                          class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100,
                                              max_depth=6,
                                              random_state=42,
                                              class_weight='balanced'),
            'SVM': SVC(probability=True)
        }
        
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(max_depth=6,
                                       learning_rate=0.1,
                                       n_estimators=100,
                                       random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100,
                                         learning_rate=0.1,
                                         max_depth=6,
                                         random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100,
                                             max_depth=6,
                                             random_state=42),
            'SVR': SVR()
        }
        
        self.problem_type = problem_type
        self.classification_models = classification_models
        self.regression_models = regression_models
        
        # Select appropriate models based on problem type
        if problem_type == 'classification':
            self.models = {name: model for name, model in classification_models.items() 
                          if name in selected_models}
        elif problem_type == 'regression':
            self.models = {name: model for name, model in regression_models.items() 
                          if name in selected_models}
        else:
            # Auto-detect based on target variable
            self.models = {name: model for name, model in classification_models.items() 
                          if name in selected_models}

        self.results = {}
        self.feature_selector = None
        self.best_features = None

    def detect_problem_type(self, target_col, data):
        """Detect if the problem is classification or regression"""
        if self.problem_type != 'auto':
            return self.problem_type
            
        target_values = data[target_col].dropna()
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target_values):
            # Check if it's binary classification (0/1 or similar)
            unique_values = target_values.nunique()
            if unique_values == 2:
                return 'classification'
            elif unique_values <= 10:
                # Could be multi-class classification or regression
                # Check if values are mostly integers
                if target_values.dtype in ['int64', 'int32']:
                    return 'classification'
                else:
                    return 'regression'
            else:
                return 'regression'
        else:
            return 'classification'

    def automated_feature_selection(self, X, y, method='rfe', n_features=None):
        """Perform automated feature selection"""
        if n_features is None:
            n_features = min(20, X.shape[1])  # Default to 20 features or all if less
        
        if method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42) if self.problem_type == 'classification' else RandomForestRegressor(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator=estimator, n_features_to_select=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.best_features = X.columns[self.feature_selector.support_].tolist()
            
        elif method == 'kbest':
            # SelectKBest
            if self.problem_type == 'classification':
                self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.best_features = X.columns[self.feature_selector.get_support()].tolist()
            
        return X_selected, self.best_features

    def hyperparameter_tuning(self, model, X_train, y_train, model_name):
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'Extra Trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3, 
                scoring='accuracy' if self.problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            return model, {}

    def save_model(self, model, scaler, feature_names, model_name, target_col):
        """Save model and related components"""
        # Create models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Save model
        model_path = f'saved_models/{model_name}_model.pkl'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = f'saved_models/{model_name}_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        features_path = f'saved_models/{model_name}_features.pkl'
        joblib.dump(feature_names, features_path)
        
        # Save target column name
        target_path = f'saved_models/{model_name}_target.pkl'
        joblib.dump(target_col, target_path)
        
        # Save problem type
        problem_path = f'saved_models/{model_name}_problem_type.pkl'
        joblib.dump(self.problem_type, problem_path)
        
        return model_path

    def load_model(self, model_name):
        """Load saved model and components"""
        try:
            model_path = f'saved_models/{model_name}_model.pkl'
            scaler_path = f'saved_models/{model_name}_scaler.pkl'
            features_path = f'saved_models/{model_name}_features.pkl'
            target_path = f'saved_models/{model_name}_target.pkl'
            problem_path = f'saved_models/{model_name}_problem_type.pkl'
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_names = joblib.load(features_path)
            target_col = joblib.load(target_path)
            problem_type = joblib.load(problem_path)
            
            return model, scaler, feature_names, target_col, problem_type
        except FileNotFoundError:
            return None, None, None, None, None

    def get_saved_models(self):
        """Get list of saved models"""
        if not os.path.exists('saved_models'):
            return []
        
        saved_models = []
        for file in os.listdir('saved_models'):
            if file.endswith('_model.pkl'):
                model_name = file.replace('_model.pkl', '')
                saved_models.append(model_name)
        return saved_models

    def clean_state(self, state_value):
        """Extract state code from the full path format"""
        if pd.isna(state_value):
            return 'Missing'
        if 'States.' in str(state_value):
            return str(state_value).split('States.')[-1]
        return str(state_value)
    
    def data_processing(self, df, threshold_days):
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Print initial NaN counts
        print("\nInitial NaN counts:")
        print(df.isna().sum())
        
        # Clean state values
        df['State'] = df['State'].apply(self.clean_state)
        if 'WorkHistoryState' in df.columns:
            df['WorkHistoryState'] = df['WorkHistoryState'].apply(self.clean_state)
        
        # Identify numeric and categorical columns
        numeric_columns = ['WorkHistoryDays', 'EducationCount', 'WorkHistoryCount', 
                        'ReferenceCount', 'Distance', 'AvailabilityAfterDays']
        
        categorical_columns = ['WorkPreference', 'State', 'SourceCategory',
                            'JobCategory', 'ReasonForLeaving']
        
        # Handle numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Fill NaN with median for numeric columns
                median_value = df[col].median()
                if pd.isna(median_value):  # If median is also NaN
                    median_value = 0
                df[col] = df[col].fillna(median_value)
        
        # Handle categorical columns
        for col in categorical_columns:
            if col in df.columns:
                # Fill NaN with 'Missing' for categorical columns
                df[col] = df[col].fillna('Missing')
        
        # Create label encoders for categorical columns
        encoders = {}
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        
        # Feature engineering
        # Binary flag for missing work history
        df['has_work_history'] = (df['WorkHistoryDays'] > 0).astype(int)
        
        # Ratio of education to work history (avoiding division by zero)
        df['edu_work_ratio'] = df['EducationCount'] / (df['WorkHistoryCount'] + 1)
        
        # Distance categories (with proper NaN handling)
        df['distance_category'] = pd.qcut(
            df['Distance'].fillna(df['Distance'].median()), 
            q=3, 
            labels=[0, 1, 2], 
            duplicates='drop'
        ).astype(float)  # Convert to float to handle NaN
        
        # Binary flags
        df['is_currently_employed'] = df['IsCurrentlyEmployed'].fillna(0).astype(int)
        if 'IsFormerEmployee' in df.columns:
            df['IsFormerEmployee'] = df['IsFormerEmployee'].fillna(0).astype(int)

        # Add threshold days calculation if provided
        if threshold_days is not None:
            # df['long_tenure'] = (df['WorkHistoryDays'] >= threshold_days).astype(int)
            df.loc[df['Tenure_orig']<=90, 'BinTenure'] = 0
            df.loc[df['Tenure_orig']>90, 'BinTenure'] = 1
            df.dropna(subset=['BinTenure'], inplace=True)
        
        # Drop unnecessary columns
        columns_to_drop = ['WorkHistoryState', 'Tenure_orig']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Print final NaN counts
        print("\nFinal NaN counts:")
        print(df.isna().sum())
        
        # Print final columns
        print("\nFinal columns:")
        print(df.columns.tolist())
        
        return df, encoders

        
    def prepare_data(self, data: pd.DataFrame, target_col: str, test_size: float=0.2, threshold_days=None, 
                    enable_feature_selection=False, feature_selection_method='rfe', n_features=None):
        """Prepare data for modeling"""
        prepared_data, encoders = self.data_processing(data, threshold_days)
        
        # Detect problem type
        self.problem_type = self.detect_problem_type(target_col, prepared_data)
        
        # Update models based on detected problem type
        if self.problem_type == 'regression':
            self.models = {name: model for name, model in self.regression_models.items() 
                          if name in self.models.keys()}
        
        # Final check for any remaining NaN values
        if prepared_data.isna().any().any():
            print("\nWarning: There are still NaN values in the following columns:")
            print(prepared_data.columns[prepared_data.isna().any()].tolist())
            # Fill any remaining NaN values with 0
            prepared_data = prepared_data.fillna(0)

        # Separate features and target
        X = prepared_data.drop(columns=[target_col], axis=1, errors='ignore')
        y = prepared_data[target_col]
        
        # Apply feature selection if enabled
        if enable_feature_selection:
            X_selected, selected_features = self.automated_feature_selection(X, y, feature_selection_method, n_features)
            X = pd.DataFrame(X_selected, columns=selected_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, enable_tuning=False):
        """Train models and compute metrics"""
        for name, model in self.models.items():
            # Perform hyperparameter tuning if enabled
            if enable_tuning:
                tuned_model, best_params = self.hyperparameter_tuning(model, X_train, y_train, name)
                model = tuned_model
                print(f"Best parameters for {name}: {best_params}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if self.problem_type == 'classification':
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Get probabilities if available
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                except:
                    y_prob = None
                    fpr, tpr, roc_auc = None, None, None
                
                conf_matrix = confusion_matrix(y_test, y_pred)
            else:
                # Regression metrics
                metrics = {
                    'R² Score': r2_score(y_test, y_pred),
                    'Mean Squared Error': mean_squared_error(y_test, y_pred),
                    'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'Mean Absolute Error': np.mean(np.abs(y_test - y_pred))
                }
                fpr, tpr, roc_auc = None, None, None
                conf_matrix = None
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='accuracy' if self.problem_type == 'classification' else 'r2')
            
            # Feature importance
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                else:
                    importance = abs(model.coef_[0])
            except:
                importance = None
                
            self.results[name] = {
                'metrics': metrics,
                'cv_scores': cv_scores,
                'importance': importance,
                'confusion_matrix': conf_matrix,
                'roc_curve': (fpr, tpr) if fpr is not None else None,
                'roc_auc': roc_auc,
                'model': model  # Store the trained model
            }
        
        return self.results

def main():
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'metrics_df' not in st.session_state:
        st.session_state.metrics_df = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = None
    if 'ml_comp' not in st.session_state:
        st.session_state.ml_comp = None

    # Competition overview section
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Competition Value Proposition</h3>
        <p><strong>Problem:</strong> AI/ML is complex and requires extensive coding knowledge</p>
        <p><strong>Solution:</strong> Democratized ML platform that makes AI accessible to everyone</p>
        <p><strong>Impact:</strong> Enables non-technical users to leverage AI for business insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Key features showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI-Powered</h4>
            <p>Automated problem detection and model selection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 No-Code</h4>
            <p>Drag-and-drop interface for ML workflows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔍 Explainable AI</h4>
            <p>SHAP, LIME, and feature importance analysis</p>
        </div>
        """, unsafe_allow_html=True)

    # Reset function to clear all session state
    def reset_session_state():
        st.session_state.training_completed = False
        st.session_state.results = None
        st.session_state.feature_names = None
        st.session_state.models = None
        st.session_state.metrics_df = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.problem_type = None
        st.session_state.ml_comp = None
    
    # Data input section
    st.sidebar.header("📁 Data Input")
    
    # Demo mode for competition
    demo_mode = st.sidebar.checkbox("🎯 Demo Mode (Competition Ready)", value=True, 
                                   help="Use pre-built datasets to showcase the platform")
    
    if demo_mode:
        st.sidebar.subheader("Demo Datasets")
        demo_dataset = st.sidebar.selectbox(
            "Choose Demo Dataset",
            list(DEMO_DATASETS.keys()),
            help="Select a dataset to demonstrate the platform capabilities"
        )
        
        if st.sidebar.button("🚀 Load Demo Dataset"):
            with st.spinner("Loading demo dataset..."):
                data = load_demo_data(demo_dataset)
                st.success(f"✅ Loaded {demo_dataset} dataset with {len(data)} records!")
                st.session_state.demo_data = data
                st.session_state.demo_dataset_name = demo_dataset
    else:
        # File upload
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.demo_data = data
            st.session_state.demo_dataset_name = "Custom Dataset"
    
    # Check if data is loaded
    if 'demo_data' not in st.session_state:
        st.info("👆 Please load a dataset using the sidebar to get started!")
        return
    
    data = st.session_state.demo_data

    # Enhanced tabs with competition focus
    tabs = st.tabs([
        "🎯 Quick Demo",
        "📊 Data Analysis", 
        "🤖 Model Training",
        "🔍 Model Explanation",
        "💾 Model Management",
        "🏆 Competition Info"
    ])
    
    with tabs[0]:
        st.header("🚀 Quick Demo - See AI in Action!")
        
        if st.session_state.demo_dataset_name:
            st.markdown(f"""
            <div class="feature-card">
                <h3>Current Dataset: {st.session_state.demo_dataset_name}</h3>
                <p><strong>Records:</strong> {len(data):,} | <strong>Features:</strong> {len(data.columns)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-demo functionality
        if st.button("🎯 Run Auto-Demo", type="primary"):
            with st.spinner("Running automated demo..."):
                # Auto-select target and features
                if st.session_state.demo_dataset_name == "Employee Tenure Prediction":
                    target_col = "BinTenure"
                    feature_cols = [col for col in data.columns if col not in [target_col, "Tenure_orig"]]
                elif st.session_state.demo_dataset_name == "House Price Prediction":
                    target_col = "Price"
                    feature_cols = [col for col in data.columns if col != target_col]
                elif st.session_state.demo_dataset_name == "Customer Churn Prediction":
                    target_col = "Churn"
                    feature_cols = [col for col in data.columns if col != target_col]
                else:
                    target_col = data.columns[-1]
                    feature_cols = data.columns[:-1]
                
                # Auto-detect problem type
                problem_type = "Auto-detect"
                
                # Select top models
                selected_models = ['Random Forest', 'XGBoost', 'LightGBM']
                
                # Initialize and train
                ml_comp = MLComparison(selected_models, "auto")
                st.session_state.ml_comp = ml_comp
                
                # Prepare data with feature selection
                X_train, X_test, y_train, y_test, feature_names = ml_comp.prepare_data(
                    data[feature_cols + [target_col]], target_col,
                    enable_feature_selection=True,
                    feature_selection_method='rfe',
                    n_features=min(10, len(feature_cols))
                )
                
                # Train with hyperparameter tuning
                results = ml_comp.train_and_evaluate(X_train, X_test, y_train, y_test, enable_tuning=True)
                
                st.session_state.results = results
                st.session_state.feature_names = feature_names
                st.session_state.problem_type = ml_comp.problem_type
                st.session_state.training_completed = True
                
                st.success("✅ Auto-demo completed! Check the Model Training tab for results.")
        
        # Show dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(data.head(), use_container_width=True)
        
        # Show basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Features", len(data.columns))
        with col2:
            st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with tabs[1]:
        create_data_analysis_tab(data)
        
    with tabs[1]:
        st.write("Data Preview:", data.head())

        # Problem type selection
        st.sidebar.header("🎯 Problem Type")
        problem_type = st.sidebar.selectbox(
            "Select Problem Type",
            ["Auto-detect", "Classification", "Regression"],
            help="Auto-detect will determine the problem type based on your target variable"
        )
        
        # Convert to MLComparison format
        problem_type_map = {
            "Auto-detect": "auto",
            "Classification": "classification", 
            "Regression": "regression"
        }

        threshold_days = st.number_input(
            "Enter threshold days for tenure classification",
            min_value=1,
            value=90,  # Default to 1 year
            help="Number of Days above which an employee is considered to have long tenure"
        )

        # Model selection checkboxes
        st.sidebar.header("🤖 Select Models")
        if problem_type == "Regression":
            model_options = {
                'Linear Regression': st.sidebar.checkbox('Linear Regression', value=True),
                'Random Forest': st.sidebar.checkbox('Random Forest', value=True),
                'Gradient Boosting': st.sidebar.checkbox('Gradient Boosting', value=True),
                'XGBoost': st.sidebar.checkbox('XGBoost', value=True),
                'LightGBM': st.sidebar.checkbox('LightGBM', value=True),
                'Extra Trees': st.sidebar.checkbox('Extra Trees', value=True),
                'SVR': st.sidebar.checkbox('SVR', value=True)
            }
        else:
            model_options = {
                'Logistic Regression': st.sidebar.checkbox('Logistic Regression', value=True),
                'Random Forest': st.sidebar.checkbox('Random Forest', value=True),
                'Gradient Boosting': st.sidebar.checkbox('Gradient Boosting', value=True),
                'XGBoost': st.sidebar.checkbox('XGBoost', value=True),
                'LightGBM': st.sidebar.checkbox('LightGBM', value=True),
                'Extra Trees': st.sidebar.checkbox('Extra Trees', value=True),
                'SVM': st.sidebar.checkbox('SVM', value=True)
            }
        
        # Filter selected models
        selected_models = [name for name, selected in model_options.items() if selected]
        
        if not selected_models:
            st.warning("Please select at least one model to proceed.")
            return
        
        # Select target variable
        target_col = st.sidebar.selectbox(
            "Select Target Variable",
            options=data.columns
        )
        
        # Select features
        feature_cols = st.sidebar.multiselect(
            "Select Features",
            options=[col for col in data.columns if col != target_col],
            default=[col for col in data.columns if col != target_col]
        )

        # Advanced options
        st.sidebar.header("⚙️ Advanced Options")
        enable_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=False)
        if enable_feature_selection:
            feature_selection_method = st.sidebar.selectbox(
                "Feature Selection Method",
                ["rfe", "kbest"],
                help="RFE: Recursive Feature Elimination, KBest: Select K Best features"
            )
            n_features = st.sidebar.number_input(
                "Number of Features to Select",
                min_value=1,
                max_value=len(feature_cols),
                value=min(20, len(feature_cols)),
                help="Number of features to select"
            )
        else:
            feature_selection_method = 'rfe'
            n_features = None

        enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)

        # Initialize ML comparison
        ml_comp = MLComparison(selected_models, problem_type_map[problem_type])
        st.session_state.ml_comp = ml_comp
        
        # Train models button
        if st.sidebar.button("🚀 Train Models", type="primary") or st.session_state.training_completed:
            if not st.session_state.training_completed:
                with st.spinner('Training models...'):
                    # Prepare data
                    X_train, X_test, y_train, y_test, feature_names = ml_comp.prepare_data(
                        data[feature_cols + [target_col]], target_col, threshold_days=threshold_days,
                        enable_feature_selection=enable_feature_selection,
                        feature_selection_method=feature_selection_method,
                        n_features=n_features
                    )

                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_names
                    st.session_state.models = ml_comp.models
                    st.session_state.problem_type = ml_comp.problem_type
                    
                    # Train and evaluate
                    results = ml_comp.train_and_evaluate(X_train, X_test, y_train, y_test, enable_tuning)
                    st.session_state.results = results
                    st.session_state.training_completed = True
                    
            # Display results
            st.header("📊 Model Performance Comparison")
            
            # Show problem type
            st.info(f"Problem Type: {st.session_state.problem_type.title()}")
            
            # Metrics comparison
            metrics_df = pd.DataFrame({
                model_name: result['metrics'] 
                for model_name, result in st.session_state.results.items()
            }).T
            
            st.write("Performance Metrics:")
            st.dataframe(metrics_df)
            
            # Plot metrics comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.plot(kind='bar', ax=ax)
            plt.title("Model Performance Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Cross-validation results
            st.header("Cross-validation Results")
            for model_name, result in st.session_state.results.items():
                st.write(f"{model_name} CV Scores:")
                cv_df = pd.DataFrame(result['cv_scores'], columns=['Score'])
                st.write(f"Mean: {cv_df['Score'].mean():.3f} (±{cv_df['Score'].std()*2:.3f})")
            
            # Feature importance
            st.header("Feature Importance")
            for model_name, result in st.session_state.results.items():
                if result['importance'] is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': result['importance']
                    }).sort_values('Importance', ascending=False)
                    
                    sns.barplot(data=importance_df, x='Importance', y='Feature')
                    plt.title(f"{model_name} - Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Add download button for results
            csv = metrics_df.to_csv()
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )

            # Add a button to reset/retrain
            if st.sidebar.button("Reset and Retrain", key="reset_retrain_button"):
                reset_session_state()
                st.experimental_rerun()

    with tabs[2]:
        if 'results' in st.session_state:
            create_best_model_tab(
                st.session_state.results,
                st.session_state.models,
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test,
                st.session_state.feature_names
            )
        else:
            st.warning("Please train models first in the Model Training tab.")

    with tabs[3]:
        st.header("💾 Model Management")
        
        # Save models section
        st.subheader("Save Models")
        if st.session_state.training_completed and st.session_state.results:
            model_to_save = st.selectbox(
                "Select Model to Save",
                list(st.session_state.results.keys())
            )
            
            if st.button("Save Selected Model"):
                result = st.session_state.results[model_to_save]
                model_path = st.session_state.ml_comp.save_model(
                    result['model'],
                    StandardScaler().fit(st.session_state.X_train),
                    st.session_state.feature_names,
                    model_to_save,
                    target_col
                )
                st.success(f"Model saved successfully to {model_path}")
        
        # Load models section
        st.subheader("Load Saved Models")
        saved_models = st.session_state.ml_comp.get_saved_models() if st.session_state.ml_comp else []
        
        if saved_models:
            model_to_load = st.selectbox("Select Model to Load", saved_models)
            
            if st.button("Load Selected Model"):
                model, scaler, feature_names, target_col, problem_type = st.session_state.ml_comp.load_model(model_to_load)
                if model is not None:
                    st.success(f"Model {model_to_load} loaded successfully!")
                    st.write(f"Problem Type: {problem_type}")
                    st.write(f"Target Column: {target_col}")
                    st.write(f"Features: {len(feature_names)} features")
                else:
                    st.error("Failed to load model")
        else:
            st.info("No saved models found")

    with tabs[4]:
        st.header("🏆 Mission AI Possible Competition")
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Our Mission</h3>
            <p>To democratize AI/ML by making it accessible to everyone, regardless of technical background.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🚀 Innovation Highlights</h4>
                <ul>
                    <li>No-code ML platform</li>
                    <li>Automated problem detection</li>
                    <li>Explainable AI integration</li>
                    <li>Real-time model comparison</li>
                    <li>Feature selection automation</li>
                    <li>Hyperparameter tuning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>💡 Business Impact</h4>
                <ul>
                    <li>Reduces ML adoption barriers</li>
                    <li>Enables data-driven decisions</li>
                    <li>Accelerates AI implementation</li>
                    <li>Lowers technical requirements</li>
                    <li>Increases AI accessibility</li>
                    <li>Democratizes data science</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 Technical Architecture</h3>
            <p><strong>Frontend:</strong> Streamlit for rapid prototyping and beautiful UI</p>
            <p><strong>Backend:</strong> Scikit-learn, XGBoost, LightGBM for ML algorithms</p>
            <p><strong>Explainability:</strong> SHAP, LIME for model interpretation</p>
            <p><strong>Deployment:</strong> Streamlit Cloud for easy sharing and collaboration</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
