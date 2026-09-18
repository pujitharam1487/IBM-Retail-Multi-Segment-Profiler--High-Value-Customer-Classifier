"""
Retail Multi-Segment Profiler & High-Value Customer Classifier
Interactive Streamlit Web Application.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Configure Streamlit page settings
st.set_page_config(
    page_title="Retail Multi-Segment Profiler & High-Value Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .persona-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.data_loader import initialize_datasets, load_mall_customers_df, generate_retail_intelligence_df
from src.preprocessing import RetailPreprocessor
from src.clustering import CustomerSegmenter
from src.classification import HighValueClassifier
from src.recommendation import RetailRecommendationEngine
from src.pipeline import RetailIntelligencePipeline


@st.cache_resource(show_spinner="Loading datasets and models...")
def load_app_resources():
    """
    Initializes or loads cached dataset and trained ML pipeline.
    """
    data_dir = Path("data")
    models_dir = Path("models")
    
    # Initialize datasets
    mall_df, intel_df = initialize_datasets(str(data_dir))
    
    # Train or load pipeline
    if (models_dir / "preprocessor.joblib").exists() and (models_dir / "classifier.joblib").exists():
        try:
            pipeline = RetailIntelligencePipeline.load(str(models_dir))
        except Exception:
            pipeline = RetailIntelligencePipeline()
            pipeline.fit(intel_df)
            pipeline.save(str(models_dir))
    else:
        pipeline = RetailIntelligencePipeline()
        pipeline.fit(intel_df)
        pipeline.save(str(models_dir))
        
    # Generate scored dataset
    scored_df = pipeline.score_batch(intel_df)
    
    return mall_df, intel_df, pipeline, scored_df


# Load resources
mall_df, intel_df, pipeline, scored_df = load_app_resources()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1555421689-491a97ff2040?w=600&auto=format&fit=crop&q=80", width='stretch')
    st.markdown("### 🛍️ Retail Intelligence Studio")
    st.markdown("**Multi-Segment Persona Profiler & High-Value Classifier**")
    st.markdown("---")
    
    st.markdown("#### ⚙️ System Controls")
    dataset_view = st.radio(
        "Active Dataset View:",
        ["Retail Intelligence (Enriched)", "Classic Mall Customers (200 Records)"],
        index=0
    )
    
    current_df = intel_df if "Enriched" in dataset_view else mall_df
    
    st.markdown("---")
    st.markdown("#### 📐 Architecture Overview")
    st.markdown("""
    - **Stage 1 (Unsupervised)**:  
      `K-Means Clustering` with Elbow (WCSS) & Silhouette validation for persona discovery.
    - **Stage 2 (Supervised)**:  
      `XGBoost Classifier` with Stratified 5-Fold CV for High-Value customer prediction.
    - **Stage 3 (Decisioning)**:  
      Personalized marketing recommendation & retention strategy engine.
    """)
    st.markdown("---")
    
    if st.button("🔄 Retrain ML Pipeline", width='stretch'):
        with st.spinner("Retraining K-Means & XGBoost models..."):
            pipeline = RetailIntelligencePipeline()
            pipeline.fit(intel_df)
            pipeline.save("models")
            st.cache_resource.clear()
            st.success("Pipeline retrained and cached!")
            st.rerun()

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">Retail Multi-Segment Profiler & High-Value Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Two-Stage Machine Learning Architecture: K-Means Persona Discovery + XGBoost High-Value Customer Prediction & Recommendation Engine</div>', unsafe_allow_html=True)

# Metric Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{len(scored_df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    hv_count = (scored_df["High_Value_Prediction"] == "High-Value").sum()
    hv_pct = (hv_count / len(scored_df)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High-Value Share</div>
        <div class="metric-value">{hv_pct:.1f}% <span style="font-size: 1rem; color: #10B981;">({hv_count})</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_income = scored_df["Annual_Income_k"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Annual Income</div>
        <div class="metric-value">${avg_income:.1f}k</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_spend = scored_df["Spending_Score"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Spending Score</div>
        <div class="metric-value">{avg_spend:.1f}<span style="font-size: 1rem; color: #64748B;">/100</span></div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    n_personas = scored_df["Persona_Name"].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Discovered Personas</div>
        <div class="metric-value">{n_personas} Personas</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# MAIN TABS
# ==========================================
tabs = st.tabs([
    "📊 1. Executive EDA & Overview",
    "🎯 2. K-Means Customer Personas",
    "🚀 3. XGBoost Classifier Benchmark",
    "🔮 4. Real-time Customer Simulator",
    "📋 5. Scored Database & Export",
    "💡 6. Marketing Strategy Playbook"
])

# ==========================================
# TAB 1: EXECUTIVE EDA
# ==========================================
with tabs[0]:
    st.subheader("Exploratory Data Analysis & Statistical Overview")
    st.markdown("Analyze customer demographics, spending behaviors, income distribution, and correlation dynamics.")
    
    eda_col1, eda_col2 = st.columns([1, 1])
    
    with eda_col1:
        # Scatter: Income vs Spending Score
        fig_scatter = px.scatter(
            current_df,
            x="Annual_Income_k" if "Annual_Income_k" in current_df.columns else "Annual Income (k$)",
            y="Spending_Score" if "Spending_Score" in current_df.columns else "Spending Score (1-100)",
            color="Gender",
            size="Age",
            hover_data=["Age"],
            title="Annual Income vs Spending Score (Sized by Age)",
            color_discrete_map={"Male": "#3B82F6", "Female": "#EC4899"},
            template="plotly_white"
        )
        fig_scatter.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_scatter, width='stretch')
        
    with eda_col2:
        # Age vs Spending Score Box Plot
        age_col_name = "Age"
        spend_col_name = "Spending_Score" if "Spending_Score" in current_df.columns else "Spending Score (1-100)"
        
        fig_hist = px.histogram(
            current_df,
            x=spend_col_name,
            color="Gender",
            marginal="box",
            nbins=30,
            title="Spending Score Distribution by Gender",
            color_discrete_map={"Male": "#3B82F6", "Female": "#EC4899"},
            template="plotly_white"
        )
        fig_hist.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_hist, width='stretch')
        
    st.markdown("---")
    
    eda_row2_1, eda_row2_2 = st.columns([1, 1])
    with eda_row2_1:
        # Correlation Heatmap
        numeric_df = current_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().round(2)
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Numerical Feature Correlation Matrix",
            template="plotly_white"
        )
        fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_corr, width='stretch')
        
    with eda_row2_2:
        # Age Distribution by Age Groups
        df_age_group = current_df.copy()
        df_age_group["Age_Cohort"] = pd.cut(
            df_age_group["Age"],
            bins=[0, 25, 40, 55, 100],
            labels=["<25 Young Adults", "26-40 Career Builders", "41-55 Mature Families", "55+ Seniors"]
        )
        age_group_counts = df_age_group["Age_Cohort"].value_counts().reset_index()
        age_group_counts.columns = ["Cohort", "Count"]
        
        fig_donut = px.pie(
            age_group_counts,
            names="Cohort",
            values="Count",
            hole=0.45,
            title="Customer Base Age Cohort Breakdown",
            color_discrete_sequence=px.colors.qualitative.Prism,
            template="plotly_white"
        )
        fig_donut.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_donut, width='stretch')

# ==========================================
# TAB 2: K-MEANS PERSONA CLUSTERING
# ==========================================
with tabs[1]:
    st.subheader("Unsupervised Customer Persona Segmentation (K-Means)")
    st.markdown("Evaluate clustering optimality using the **Elbow Method (Inertia WCSS)** and **Silhouette Score Analysis**, and explore distinct multi-dimensional customer personas.")
    
    # 1. Elbow & Silhouette Curves
    eval_df = pipeline.segmenter.evaluation_results.get("k_evaluation", None)
    if eval_df is None:
        X_scaled, _ = pipeline.preprocessor.prepare_clustering_data(intel_df)
        eval_df = pipeline.segmenter.evaluate_k_range(X_scaled, k_min=2, k_max=8)
        
    k_col1, k_col2 = st.columns(2)
    with k_col1:
        fig_elbow = px.line(
            eval_df,
            x="K",
            y="Inertia_WCSS",
            markers=True,
            title="Elbow Method: Within-Cluster Sum of Squares (Inertia)",
            labels={"Inertia_WCSS": "Inertia (WCSS)", "K": "Number of Clusters (K)"},
            template="plotly_white"
        )
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#EF4444", annotation_text="Chosen K=5")
        fig_elbow.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_elbow, width='stretch')
        
    with k_col2:
        fig_sil = px.line(
            eval_df,
            x="K",
            y="Silhouette_Score",
            markers=True,
            title="Silhouette Analysis across K Values",
            labels={"Silhouette_Score": "Silhouette Score", "K": "Number of Clusters (K)"},
            template="plotly_white"
        )
        fig_sil.add_vline(x=5, line_dash="dash", line_color="#10B981", annotation_text="Optimal Balance K=5")
        fig_sil.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_sil, width='stretch')
        
    st.markdown("---")
    
    # 2. 2D and 3D Cluster Visualizations
    st.markdown("#### 🗺️ Multi-Dimensional Cluster Visualizations")
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        fig_2d = px.scatter(
            scored_df,
            x="Annual_Income_k",
            y="Spending_Score",
            color="Persona_Name",
            symbol="Persona_Name",
            size="Age",
            hover_name="CustomerID",
            hover_data=["Age", "Annual_Income_k", "Spending_Score", "High_Value_Prediction"],
            title="2D Cluster Map: Annual Income vs Spending Score",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_2d.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_2d, width='stretch')
        
    with viz_col2:
        fig_3d = px.scatter_3d(
            scored_df,
            x="Age",
            y="Annual_Income_k",
            z="Spending_Score",
            color="Persona_Name",
            opacity=0.85,
            size_max=8,
            title="3D Persona Space (Age × Income × Spending)",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_3d.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_3d, width='stretch')
        
    st.markdown("---")
    
    # 3. Discovered Persona Profiles Table & Cards
    st.markdown("#### 👥 Discovered Persona Profiles & Business Interpretation")
    profiles_df = pipeline.segmenter.get_profiles_dataframe()
    
    for _, row in profiles_df.iterrows():
        st.markdown(f"""
        <div class="persona-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #1E293B;">Cluster {row['cluster_id']}: {row['persona_name']}</h4>
                <span style="background: #EFF6FF; color: #1D4ED8; font-weight: 600; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem;">
                    {row['tier']} ({row['percentage']}% of Base)
                </span>
            </div>
            <p style="color: #475569; margin: 8px 0 12px 0; font-size: 0.95rem;">{row['description']}</p>
            <div style="display: flex; gap: 24px; font-size: 0.9rem; color: #334155;">
                <div><strong>Avg Age:</strong> {row['mean_age']} yrs</div>
                <div><strong>Avg Annual Income:</strong> ${row['mean_income']}k</div>
                <div><strong>Avg Spending Score:</strong> {row['mean_spending_score']}/100</div>
                <div><strong>Customer Count:</strong> {row['count']} shoppers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: SUPERVISED CLASSIFIER BENCHMARK
# ==========================================
with tabs[2]:
    st.subheader("Supervised High-Value Customer Classifier Benchmark")
    st.markdown("Compare machine learning models on **Stratified 5-Fold Cross-Validation** to prevent overfitting, examine test set ROC & PR curves, Confusion Matrix, and inspect **SHAP / Feature Importances**.")
    
    # Model comparison table
    preprocessor = pipeline.preprocessor
    X_clf, y_clf, _ = preprocessor.prepare_classification_data(intel_df)
    
    if "cv_comparison" not in pipeline.classifier.benchmark_results:
        cv_df = pipeline.classifier.benchmark_models_cv(X_clf, y_clf, n_splits=5)
    else:
        cv_df = pipeline.classifier.benchmark_results["cv_comparison"]
        
    st.markdown("#### 🏆 Stratified 5-Fold Cross-Validation Benchmark")
    st.dataframe(
        cv_df.style.format({
            "Accuracy (CV Mean)": "{:.3f}",
            "Accuracy (Std)": "±{:.3f}",
            "Precision (CV Mean)": "{:.3f}",
            "Recall (CV Mean)": "{:.3f}",
            "F1-Score (CV Mean)": "{:.3f}",
            "ROC-AUC (CV Mean)": "{:.3f}"
        }).background_gradient(subset=["ROC-AUC (CV Mean)", "F1-Score (CV Mean)"], cmap="Blues"),
        width='stretch'
    )
    
    st.markdown("---")
    
    # Model Test Evaluation Curves
    clf_row1_1, clf_row1_2 = st.columns([1, 1])
    test_metrics = pipeline.classifier.test_metrics.get("XGBoost Classifier", {})
    
    with clf_row1_1:
        if test_metrics:
            y_test = np.array(test_metrics["y_test"])
            y_prob = np.array(test_metrics["y_prob"])
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC = {test_metrics['roc_auc']:.3f})", line=dict(color="#3B82F6", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(color="#94A3B8", dash="dash")))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC Curve)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_roc, width='stretch')
            
    with clf_row1_2:
        if test_metrics:
            cm = np.array(test_metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Standard (0)", "High-Value (1)"],
                y=["Standard (0)", "High-Value (1)"],
                color_continuous_scale="Blues",
                title=f"Holdout Confusion Matrix (Accuracy = {test_metrics['accuracy']:.3f})",
                template="plotly_white"
            )
            fig_cm.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_cm, width='stretch')
            
    st.markdown("---")
    
    # Feature Importances & SHAP
    st.markdown("#### 🧠 Model Explainability & Feature Importances (XGBoost)")
    feat_imp_df = pipeline.classifier.get_feature_importances()
    
    fig_imp = px.bar(
        feat_imp_df.head(8),
        x="Relative_Pct",
        y="Feature",
        orientation="h",
        title="Top Predictive Behavioral & Demographic Features (% Contribution)",
        labels={"Relative_Pct": "Importance (%)", "Feature": "Feature Name"},
        color="Relative_Pct",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_imp, width='stretch')

# ==========================================
# TAB 4: REAL-TIME CUSTOMER SIMULATOR
# ==========================================
with tabs[3]:
    st.subheader("Interactive Customer Intelligence Simulator")
    st.markdown("Simulate a live customer profile to instantly classify high-value likelihood, assign persona clusters, and generate automated marketing strategies.")
    
    sim_col_in, sim_col_out = st.columns([1.1, 1.2])
    
    with sim_col_in:
        st.markdown("##### 📝 Input Customer Profile")
        with st.container(border=True):
            sim_id = st.text_input("Customer Identifier", value="CUST_LIVE_SIM_001")
            
            c_g, c_a = st.columns(2)
            with c_g:
                sim_gender = st.selectbox("Gender", ["Female", "Male"])
            with c_a:
                sim_age = st.slider("Age", min_value=18, max_value=80, value=32)
                
            c_inc, c_spd = st.columns(2)
            with c_inc:
                sim_income = st.slider("Annual Income ($k)", min_value=15, max_value=160, value=85)
            with c_spd:
                sim_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=78)
                
            st.markdown("###### 🛒 Historical Purchasing & Engagement")
            c_freq, c_aov = st.columns(2)
            with c_freq:
                sim_freq = st.slider("Orders / Year", min_value=1, max_value=52, value=28)
            with c_aov:
                sim_aov = st.slider("Avg Order Value ($)", min_value=20.0, max_value=450.0, value=210.0)
                
            c_rec, c_app = st.columns(2)
            with c_rec:
                sim_recency = st.slider("Days Since Last Order", min_value=1, max_value=365, value=15)
            with c_app:
                sim_app = st.slider("App Sessions / Month", min_value=0, max_value=40, value=22)
                
            c_ret, c_disc = st.columns(2)
            with c_ret:
                sim_return = st.slider("Return Rate (%)", min_value=0.0, max_value=30.0, value=5.0)
            with c_disc:
                sim_disc = st.slider("Discount Usage (%)", min_value=0.0, max_value=90.0, value=15.0)
                
            sim_customer_dict = {
                "CustomerID": sim_id,
                "Gender": sim_gender,
                "Age": sim_age,
                "Annual_Income_k": sim_income,
                "Spending_Score": sim_score,
                "Purchase_Frequency_Year": sim_freq,
                "Avg_Order_Value": sim_aov,
                "Recency_Days": sim_recency,
                "Return_Rate_Pct": sim_return,
                "Discount_Usage_Pct": sim_disc,
                "Online_Order_Ratio": 0.65,
                "App_Sessions_Month": sim_app,
                "Years_As_Customer": 3.5
            }

    with sim_col_out:
        st.markdown("##### 🎯 Intelligence & Scoring Output")
        sim_res = pipeline.score_single_customer(sim_customer_dict)
        rec = sim_res["recommendation"]
        
        # High-Value Probability Gauge
        prob_val = sim_res["high_value_probability"]
        is_hv = sim_res["high_value_prediction"] == "High-Value"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_val * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "High-Value Shopper Propensity (%)", 'font': {'size': 18}},
            delta={'reference': 50.0, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#10B981" if is_hv else "#3B82F6"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 40], 'color': '#F1F5F9'},
                    {'range': [40, 70], 'color': '#E0F2FE'},
                    {'range': [70, 100], 'color': '#DCFCE7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 75.0
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, width='stretch')
        
        # Output Profile Card
        status_color = "#10B981" if is_hv else "#64748B"
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 10px; padding: 18px 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">{sim_res['persona_name']}</span>
                <span style="background: {status_color}; color: white; font-weight: 700; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem;">
                    {sim_res['high_value_prediction']} ({prob_val*100:.1f}%)
                </span>
            </div>
            <p style="color: #475569; font-size: 0.9rem; margin-top: 6px;">{sim_res['persona_description']}</p>
            <hr style="border-top: 1px solid #E2E8F0; margin: 12px 0;">
            <h5 style="margin: 0 0 6px 0; color: #1E293B;">🎯 Tailored Strategy: {rec['strategy']} {rec['badge']}</h5>
            <p style="color: #334155; font-size: 0.9rem; margin-bottom: 8px;"><strong>Primary Channel:</strong> {rec['channel']}</p>
            <p style="color: #334155; font-size: 0.9rem; margin-bottom: 8px;"><strong>Discount Policy:</strong> {rec['promo_policy']}</p>
            <strong style="font-size: 0.9rem; color: #1E293B;">Actionable Campaign Tactics:</strong>
            <ul style="color: #475569; font-size: 0.88rem; margin-top: 4px; padding-left: 20px;">
                {''.join([f"<li>{t}</li>" for t in rec['tactics']])}
            </ul>
            <div style="background: #EFF6FF; border-left: 4px solid #3B82F6; padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; color: #1E40AF; margin-top: 10px;">
                <strong>Expected ROI Impact:</strong> {rec['expected_roi']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 5: SCORED DATABASE & EXPORT
# ==========================================
with tabs[4]:
    st.subheader("Customer Intelligence Database & Batch Exporter")
    st.markdown("Filter, search, and export the scored customer base with cluster personas and high-value probabilities.")
    
    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        sel_persona = st.multiselect(
            "Filter by Persona:",
            options=scored_df["Persona_Name"].unique(),
            default=scored_df["Persona_Name"].unique()
        )
    with f_col2:
        sel_hv = st.multiselect(
            "Filter Prediction:",
            options=["High-Value", "Standard"],
            default=["High-Value", "Standard"]
        )
    with f_col3:
        min_prob = st.slider("Min High-Value Prob:", 0.0, 1.0, 0.0, 0.05)
    with f_col4:
        search_query = st.text_input("Search Customer ID:", "")
        
    # Apply filters
    filtered_df = scored_df[
        (scored_df["Persona_Name"].isin(sel_persona)) &
        (scored_df["High_Value_Prediction"].isin(sel_hv)) &
        (scored_df["High_Value_Probability"] >= min_prob)
    ]
    if search_query:
        filtered_df = filtered_df[filtered_df["CustomerID"].astype(str).str.contains(search_query, case=False)]
        
    st.markdown(f"**Showing {len(filtered_df):,} of {len(scored_df):,} Customers**")
    
    # Display table
    display_cols = [
        "CustomerID", "Gender", "Age", "Annual_Income_k", "Spending_Score",
        "Persona_Name", "Tier", "High_Value_Prediction", "High_Value_Probability",
        "Marketing_Strategy", "Recommended_Channel"
    ]
    st.dataframe(
        filtered_df[[c for c in display_cols if c in filtered_df.columns]],
        width='stretch',
        height=420
    )
    
    # Download Button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Scored Customers CSV",
        data=csv_data,
        file_name="retail_scored_customers.csv",
        mime="text/csv",
        width='content'
    )

# ==========================================
# TAB 6: MARKETING STRATEGY PLAYBOOK
# ==========================================
with tabs[5]:
    st.subheader("Retail Marketing Strategy Playbook & ROI Simulator")
    st.markdown("Actionable tactical matrix designed for CMOs, Retail Growth Marketers, and CRM teams to maximize Customer Lifetime Value (CLV).")
    
    playbook_cards = [
        {
            "title": "💎 Affluent VIP Spenders (Tier 1)",
            "persona": "High Income | High Spending",
            "goal": "Maximizing Customer Lifetime Value & Zero-Discount Luxury Retention",
            "color": "#8B5CF6",
            "tactics": [
                "Private invitation-only seasonal trunk shows and early product access.",
                "Dedicated 1-on-1 concierge desk with bespoke packaging and styling.",
                "Zero price discounting to protect brand equity and luxury margins.",
                "Exclusive VIP loyalty perks: complimentary priority shipping & anniversary gifts."
            ],
            "channel": "Personal Concierge / Private Invitation / VIP SMS",
            "roi": "+22% CLV retention, 95%+ renewal rates."
        },
        {
            "title": "🎯 Affluent Conservative Savers (Tier 2)",
            "persona": "High Income | Low Spending",
            "goal": "Converting Latent Spending Power through High-Value Upselling",
            "color": "#3B82F6",
            "tactics": [
                "Bundle premium hero products with high-threshold gifts (e.g. Free gift on orders >$300).",
                "Emphasize durability, lifetime warranty, artisan craftsmanship, and quality assurance.",
                "Targeted lookbooks highlighting flagship luxury lines.",
                "Exclusive trial upgrades to higher tier loyalty perks."
            ],
            "channel": "Curated Email Lookbooks / Category Highlights",
            "roi": "+35% Average Order Value (AOV) expansion."
        },
        {
            "title": "🔥 Young Enthusiasts & Trendsetters (Tier 3)",
            "persona": "Moderate Income | High Spending",
            "goal": "Driving Social Virality, Gamification & Fast Mobile Checkout",
            "color": "#EC4899",
            "tactics": [
                "24-hour flash mobile drops with app countdowns and push alerts.",
                "Gamified loyalty points for product reviews, TikTok/Instagram tags, and daily app logins.",
                "Dual-sided referral bonuses ($15 for both inviter and friend).",
                "Frictionless checkout integrations (Apple Pay, Buy Now Pay Later)."
            ],
            "channel": "Mobile App Push / Social Drops / SMS",
            "roi": "+40% repeat transaction frequency."
        },
        {
            "title": "🏷️ Frugal Budget Conscious (Tier 4)",
            "persona": "Low Income | Low Spending",
            "goal": "Driving Off-Season Stock Turnover & Volume Conversions",
            "color": "#F59E0B",
            "tactics": [
                "Clearance notifications, seasonal outlet promos, and multi-buy savings.",
                "Price-match guarantees on essential categories.",
                "Cart abandonment nudges with incremental 10% coupon codes.",
                "Threshold-based free shipping (e.g., 'Add $8 to qualify for free shipping')."
            ],
            "channel": "Weekly Deal Email / Discount SMS Digest",
            "roi": "+15% conversion on clearance inventory."
        }
    ]
    
    p_cols = st.columns(2)
    for i, p in enumerate(playbook_cards):
        with p_cols[i % 2]:
            st.markdown(f"""
            <div style="background: white; border: 1px solid #E2E8F0; border-top: 4px solid {p['color']}; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <h4 style="margin: 0 0 4px 0; color: #1E293B;">{p['title']}</h4>
                <p style="color: #64748B; font-size: 0.85rem; font-weight: 600; margin-bottom: 10px;">{p['persona']}</p>
                <div style="background: #F8FAFC; border-radius: 6px; padding: 8px 12px; font-size: 0.88rem; color: #334155; margin-bottom: 12px;">
                    <strong>Core Objective:</strong> {p['goal']}
                </div>
                <strong style="font-size: 0.88rem; color: #1E293B;">Key Strategic Plays:</strong>
                <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; margin-top: 6px;">
                    {''.join([f"<li>{t}</li>" for t in p['tactics']])}
                </ul>
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 10px; margin-top: 12px;">
                    <span><strong>Channel:</strong> {p['channel']}</span>
                    <span style="color: #10B981; font-weight: 600;">{p['roi']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
