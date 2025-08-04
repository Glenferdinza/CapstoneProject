import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# Configure page
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Load models and data
@st.cache_data
def load_data_and_models():
    try:
        # Load fraud detection models
        kmeans_model = joblib.load("optimized_clustering_cc.h5")
        pca_model = joblib.load("optimized_PCA_clustering_cc.h5")
        scaler = joblib.load("optimized_scaler_cc.h5")
        fraud_model = joblib.load("fraud_classifier_best.h5")
        df = pd.read_csv('optimized_data_clustering_cc.csv')
        original_df = pd.read_csv("CC GENERAL.csv")
        
        return {
            'kmeans_model': kmeans_model,
            'pca_model': pca_model,
            'scaler': scaler,
            'fraud_model': fraud_model,
            'df': df,
            'original_df': original_df,
            'status': 'fraud_detection_loaded'
        }
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return {'status': 'error', 'error': str(e)}

# Define cluster labels for fraud detection
cluster_labels = {
    0: "Normal Customers",
    1: "Potential Fraud"
}

# Load data
data_dict = load_data_and_models()

if data_dict['status'] == 'fraud_detection_loaded':
    df = data_dict['df']
    fraud_model = data_dict['fraud_model']
    kmeans_model = data_dict['kmeans_model']
    scaler = data_dict['scaler']
    original_df = data_dict['original_df']
    
    # Main title
    st.title("Advanced Fraud Detection Dashboard")
    st.markdown("---")
    
    # Sidebar metrics
    st.sidebar.header("Model Performance")
    if df is not None:
        total_customers = len(df)
        num_features = len([col for col in df.columns if col not in ['CUST_ID', 'cluster', 'pseudo_label', 'pca1', 'pca2']])
        num_clusters = df['cluster'].nunique() if 'cluster' in df.columns else 2
        
        st.sidebar.metric("Total Customers", f"{total_customers:,}")
        st.sidebar.metric("Features Used", num_features)
        st.sidebar.metric("Model Accuracy", "98.19%")
        st.sidebar.metric("Clusters Found", str(num_clusters))
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Fraud Detection", "Cluster Analysis", "Model Performance"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                fraud_count = df['pseudo_label'].sum() if 'pseudo_label' in df.columns else 0
                st.metric("Potential Fraud Cases", fraud_count)
            with col3:
                fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            
            st.subheader("Data Sample")
            st.dataframe(df.head())
    
    with tab2:
        st.subheader("Fraud Detection Analysis")
        
        if df is not None and 'pseudo_label' in df.columns:
            # Fraud distribution
            fraud_dist = df['pseudo_label'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fraud Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = ['Normal', 'Potential Fraud']
                colors = ['lightblue', 'red']
                plt.pie(fraud_dist.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title("Distribution of Fraud Cases")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Fraud by Cluster")
                if 'cluster' in df.columns:
                    cluster_fraud = df.groupby('cluster')['pseudo_label'].agg(['count', 'sum', 'mean']).round(3)
                    cluster_fraud.columns = ['Total', 'Fraud_Cases', 'Fraud_Rate']
                    cluster_fraud['Fraud_Rate'] = cluster_fraud['Fraud_Rate'] * 100
                    st.dataframe(cluster_fraud)
    
    with tab3:
        st.subheader("Cluster Analysis")
        
        if df is not None and 'cluster' in df.columns:
            # Cluster distribution
            cluster_dist = df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                cluster_dist.plot(kind='bar', ax=ax, color='skyblue')
                plt.title("Number of Customers per Cluster")
                plt.xlabel("Cluster")
                plt.ylabel("Count")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                st.subheader("PCA Visualization")
                try:
                    numeric_features = df.select_dtypes(include=[np.number]).columns
                    numeric_features = [col for col in numeric_features if col not in ['CUST_ID', 'cluster', 'pseudo_label']]
                    
                    if len(numeric_features) > 1:
                        X_for_pca = df[numeric_features].fillna(0)
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X_for_pca)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                            c=df['cluster'], cmap='viridis', alpha=0.7)
                        plt.title("Customer Clusters in PCA Space")
                        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                        plt.colorbar(scatter, label='Cluster')
                        st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate PCA: {e}")
    
    with tab4:
        st.subheader("Model Performance")
        
        if df is not None and fraud_model is not None:
            try:
                # Prepare features for prediction
                feature_cols = [col for col in df.columns if col not in ['CUST_ID', 'cluster', 'pseudo_label', 'pca1', 'pca2']]
                X = df[feature_cols].fillna(0)
                
                if 'pseudo_label' in df.columns:
                    y_true = df['pseudo_label'].values
                    
                    # Make predictions
                    y_pred = fraud_model.predict(X)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.3f}")
                    col2.metric("Precision", f"{precision:.3f}")
                    col3.metric("Recall", f"{recall:.3f}")
                    col4.metric("F1-Score", f"{f1:.3f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                    plt.title("Confusion Matrix")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error evaluating model: {e}")
    
    # File upload section
    st.sidebar.header("Upload New Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
            
            if st.sidebar.button("Predict Fraud"):
                try:
                    # Preprocess and predict
                    feature_cols = [col for col in df.columns if col not in ['CUST_ID', 'cluster', 'pseudo_label', 'pca1', 'pca2']]
                    available_cols = [col for col in feature_cols if col in new_data.columns]
                    
                    if available_cols:
                        X_new = new_data[available_cols].fillna(0)
                        predictions = fraud_model.predict(X_new)
                        
                        new_data['Fraud_Prediction'] = predictions
                        new_data['Risk_Level'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                        
                        st.subheader("Fraud Predictions")
                        st.dataframe(new_data[['CUST_ID', 'Fraud_Prediction', 'Risk_Level'] if 'CUST_ID' in new_data.columns else ['Fraud_Prediction', 'Risk_Level']])
                        
                        fraud_count_new = sum(predictions)
                        st.metric("Potential Fraud Cases", fraud_count_new)
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    # About section
    st.sidebar.header("ℹAbout")
    st.sidebar.write("""
    This dashboard uses advanced machine learning to detect potential fraud in credit card transactions.
    
    **Features:**
    - Multi-algorithm clustering
    - Fraud detection with 98.19% accuracy
    - Real-time predictions
    - Interactive visualizations
    
    **Models:** KMeans, Random Forest, PCA
    """)

else:
    st.error("Failed to load models. Please ensure model files exist.")
    st.info("Run the optimized_clustering_v2.py script first to generate the required models.")
