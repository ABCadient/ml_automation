import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
from lime import lime_tabular
from pdpbox import pdp
import matplotlib.pyplot as plt

def create_data_analysis_tab(data):
    """Create data analysis tab with visualizations and statistics"""
    st.header("Data Analysis")
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Number of Records: {data.shape[0]}")
    with col2:
        st.info(f"Number of Features: {data.shape[1]}")
    with col3:
        st.info(f"Missing Values: {data.isnull().sum().sum()}")
    
    # Data Statistics
    if st.checkbox("Show Basic Statistics"):
        st.subheader("Basic Statistics")
        st.write(data.describe())
    
    # Missing Values Analysis
    if st.checkbox("Show Missing Values Analysis"):
        st.subheader("Missing Values Analysis")
        missing_vals = data.isnull().sum()
        if missing_vals.any():
            missing_df = pd.DataFrame({
                'Feature': missing_vals.index,
                'Missing Count': missing_vals.values,
                'Percentage': (missing_vals.values / len(data)) * 100
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            st.write(missing_df)
            
            # Plot missing values
            fig = px.bar(
                missing_df,
                x='Feature',
                y='Percentage',
                title='Percentage of Missing Values by Feature'
            )
            st.plotly_chart(fig)
        else:
            st.success("No missing values found in the dataset!")
    
    # Feature Distribution Analysis
    st.subheader("Feature Distribution Analysis")
    feature = st.selectbox("Select Feature for Distribution", data.columns)
    
    # Create container for better spacing
    distribution_container = st.container()
    
    # Use columns with proper width allocation
    with distribution_container:
        col1, col2 = st.columns([2, 1])  # Allocate more space for plot
        
        with col1:
            # Histogram with proper sizing
            fig = px.histogram(
                data,
                x=feature,
                title=f'Distribution of {feature}',
                marginal='box',  # adds a box plot above the histogram
                height=400      # Fixed height
            )
            # Update layout for better fit
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Basic statistics of selected feature
            st.write("Feature Statistics:")
            stats = data[feature].describe()
            # Format statistics for better display
            stats_df = pd.DataFrame(stats).round(2)
            st.dataframe(stats_df, height=400)  # Fixed height to match plot
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        # Correlation Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=600
        )
        st.plotly_chart(fig)
        
        # Show highest correlations
        st.subheader("Highest Correlations")
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find top correlations
        high_corr = pd.DataFrame(
            upper.unstack(),
            columns=['correlation']
        ).reset_index()
        high_corr = high_corr[high_corr['correlation'].notna()]
        high_corr = high_corr.sort_values('correlation', key=abs, ascending=False)
        st.write(high_corr.head(10))
    
    # Feature Relationships
    st.subheader("Feature Relationships")
    if len(numeric_cols) > 1:
        x_feat = st.selectbox("Select X Feature", numeric_cols)
        y_feat = st.selectbox("Select Y Feature", 
                            [col for col in numeric_cols if col != x_feat])
        
        fig = px.scatter(
            data,
            x=x_feat,
            y=y_feat,
            title=f'Relationship between {x_feat} and {y_feat}',
            trendline="ols"  # adds trend line
        )
        st.plotly_chart(fig)


def create_best_model_tab(results, model_dict, X_train, X_test, y_train, y_test, feature_names):
    """Create best model analysis tab with detailed explanations"""
    st.header("Best Model Analysis")
    
    # Check if results is None or empty
    if not results:
        st.warning("No model results available. Please train models first.")
        return
    
    try:
        # Find best model based on accuracy
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] 
            for model_name, result in results.items()
            if result and 'metrics' in result  # Add validation
        }).T
        
        if metrics_df.empty:
            st.warning("No valid metrics found in results.")
            return
    
        best_model_name = metrics_df['Accuracy'].idxmax()
        if best_model_name not in model_dict:
            st.error(f"Model {best_model_name} not found in model dictionary.")
            return
        best_model = model_dict[best_model_name]
    except Exception as e:
        st.error(f"Error analyzing model results: {str(e)}")
        return
        
    # Display best model details
    st.subheader("Best Performing Model")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Best Model: {best_model_name}")
        st.info(f"Accuracy: {metrics_df.loc[best_model_name, 'Accuracy']:.4f}")
    
    with col2:
        st.info(f"F1 Score: {metrics_df.loc[best_model_name, 'F1 Score']:.4f}")
        st.info(f"Precision: {metrics_df.loc[best_model_name, 'Precision']:.4f}")
    
    # Model Explanation Methods
    explanation_method = st.selectbox(
        "Select Explanation Method",
        ["SHAP", "LIME", "Feature Importance", "Partial Dependence Plots"]
    )
    
    if explanation_method == "SHAP":
        st.subheader("SHAP Analysis")
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(best_model) if hasattr(best_model, 'predict_proba') else shap.LinearExplainer(best_model, X_train)
            shap_values = explainer.shap_values(X_test)
            
            # Plot SHAP summary
            st.write("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            st.pyplot(fig)
            
            # SHAP Dependence Plots
            st.write("SHAP Dependence Plots")
            feature = st.selectbox("Select feature for dependence plot", feature_names)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feature_names.get_loc(feature), shap_values, X_test, 
                               feature_names=feature_names, show=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Unable to generate SHAP analysis: {str(e)}")
    
    elif explanation_method == "LIME":
        st.subheader("LIME Analysis")
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=['Short Tenure', 'Long Tenure'],
                mode='classification'
            )
            
            # Select instance to explain
            instance_idx = st.slider("Select instance to explain", 0, len(X_test)-1, 0)
            exp = explainer.explain_instance(
                X_test[instance_idx], 
                best_model.predict_proba,
                num_features=10
            )
            
            # Plot LIME explanation
            st.write("LIME Feature Importance")
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Unable to generate LIME analysis: {str(e)}")
    
    elif explanation_method == "Feature Importance":
        st.subheader("Feature Importance Analysis")
        try:
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            else:
                importances = np.abs(best_model.coef_[0])
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'{best_model_name} Feature Importance'
            )
            st.plotly_chart(fig)
            
            # Feature importance table
            st.write("Feature Importance Table")
            st.dataframe(importance_df)
            
        except Exception as e:
            st.error(f"Unable to generate feature importance analysis: {str(e)}")
    
    elif explanation_method == "Partial Dependence Plots":
        st.subheader("Partial Dependence Analysis")
        try:
            # Select feature for PDP
            feature = st.selectbox("Select feature for partial dependence plot", feature_names)
            
            # Create PDP plot
            pdp_isolate = pdp.pdp_isolate(
                model=best_model,
                dataset=X_train,
                model_features=feature_names,
                feature=feature
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pdp.pdp_plot(pdp_isolate, feature, plot_lines=True, frac_to_plot=0.5)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Unable to generate partial dependence plot: {str(e)}")
    
    # Model Performance Details
    st.subheader("Detailed Performance Metrics")
    
    # Confusion Matrix
    fig = px.imshow(
        results[best_model_name]['confusion_matrix'],
        labels=dict(x="Predicted", y="Actual"),
        x=['Short Tenure', 'Long Tenure'],
        y=['Short Tenure', 'Long Tenure'],
        title="Confusion Matrix",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig)
    
    # ROC Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[best_model_name]['roc_curve'][0],
        y=results[best_model_name]['roc_curve'][1],
        name=f"ROC (AUC = {results[best_model_name]['roc_auc']:.3f})"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash='dash'),
        name='Random'
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    st.plotly_chart(fig)
