"""
Unit and Integration Test Suite for Retail Multi-Segment Intelligence System.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import initialize_datasets, load_mall_customers_df, generate_retail_intelligence_df
from src.preprocessing import RetailPreprocessor
from src.clustering import CustomerSegmenter
from src.classification import HighValueClassifier
from src.recommendation import RetailRecommendationEngine
from src.pipeline import RetailIntelligencePipeline


@pytest.fixture
def sample_data():
    mall_df = load_mall_customers_df()
    intel_df = generate_retail_intelligence_df(n_samples=100, random_seed=42)
    return mall_df, intel_df


def test_data_loader(sample_data):
    mall_df, intel_df = sample_data
    assert len(mall_df) == 200
    assert "Spending Score (1-100)" in mall_df.columns
    assert len(intel_df) == 100
    assert "High_Value_Customer" in intel_df.columns
    assert intel_df["High_Value_Customer"].isin([0, 1]).all()


def test_preprocessor(sample_data):
    _, intel_df = sample_data
    preprocessor = RetailPreprocessor()
    
    # 1. Clean dataset
    cleaned = preprocessor.clean_dataset(intel_df)
    assert not cleaned.isnull().any().any()
    
    # 2. Feature engineering
    engineered = preprocessor.engineer_features(cleaned)
    assert "Income_to_Spend_Ratio" in engineered.columns
    assert "Engagement_Index" in engineered.columns
    
    # 3. Clustering data prep
    X_scaled, feat_names = preprocessor.prepare_clustering_data(intel_df)
    assert X_scaled.shape[0] == len(intel_df)
    assert X_scaled.shape[1] == 3
    
    # 4. Classification data prep
    X_clf, y_clf, clf_feats = preprocessor.prepare_classification_data(intel_df)
    assert len(X_clf) == len(y_clf)
    assert "High_Value_Customer" not in X_clf.columns


def test_clustering(sample_data):
    _, intel_df = sample_data
    preprocessor = RetailPreprocessor()
    X_scaled, feat_names = preprocessor.prepare_clustering_data(intel_df)
    
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(X_scaled, intel_df, feat_names)
    
    profiles_df = segmenter.get_profiles_dataframe()
    assert len(profiles_df) == 4
    assert "persona_name" in profiles_df.columns
    
    preds = segmenter.predict(X_scaled)
    assert len(preds) == len(intel_df)
    assert set(preds).issubset({0, 1, 2, 3})


def test_classification(sample_data):
    _, intel_df = sample_data
    preprocessor = RetailPreprocessor()
    X_clf, y_clf, _ = preprocessor.prepare_classification_data(intel_df)
    
    clf = HighValueClassifier(random_state=42)
    eval_dict = clf.train_and_evaluate(X_clf, y_clf, test_size=0.25)
    
    assert "XGBoost Classifier" in eval_dict["test_scores"]
    xgb_score = eval_dict["test_scores"]["XGBoost Classifier"]
    assert 0.0 <= xgb_score["roc_auc"] <= 1.0
    assert 0.0 <= xgb_score["f1"] <= 1.0
    
    # Check feature importance
    feat_df = clf.get_feature_importances()
    assert len(feat_df) == X_clf.shape[1]
    assert feat_df["Importance"].sum() > 0


def test_recommendation_engine():
    engine = RetailRecommendationEngine()
    
    rec_vip = engine.get_recommendation(
        persona_name="Affluent VIP Spenders",
        is_high_value=1,
        high_value_prob=0.95,
        annual_income=90,
        spending_score=85
    )
    assert "VIP" in rec_vip["strategy"] or "Elite" in rec_vip["badge"]
    assert len(rec_vip["tactics"]) >= 3
    
    rec_budget = engine.get_recommendation(
        persona_name="Frugal Budget Conscious",
        is_high_value=0,
        high_value_prob=0.15,
        annual_income=25,
        spending_score=20
    )
    assert "Value" in rec_budget["strategy"] or "Seekers" in rec_budget["badge"]


def test_end_to_end_pipeline(sample_data, tmp_path):
    _, intel_df = sample_data
    
    pipeline = RetailIntelligencePipeline()
    pipeline.fit(intel_df)
    
    # 1. Single Customer Scoring
    sample_cust = {
        "CustomerID": "CUST_TEST",
        "Gender": "Male",
        "Age": 28,
        "Annual_Income_k": 85,
        "Spending_Score": 90,
        "Purchase_Frequency_Year": 30,
        "Avg_Order_Value": 250.0,
        "Recency_Days": 14,
        "Return_Rate_Pct": 3.0,
        "Discount_Usage_Pct": 10.0,
        "Online_Order_Ratio": 0.70,
        "App_Sessions_Month": 20,
        "Years_As_Customer": 3.0
    }
    
    result = pipeline.score_single_customer(sample_cust)
    assert result["customer_id"] == "CUST_TEST"
    assert "persona_name" in result
    assert result["high_value_prediction"] in ["High-Value", "Standard"]
    assert 0.0 <= result["high_value_probability"] <= 1.0
    assert "strategy" in result["recommendation"]
    
    # 2. Batch Scoring
    scored_df = pipeline.score_batch(intel_df)
    assert "Persona_Name" in scored_df.columns
    assert "High_Value_Prediction" in scored_df.columns
    assert "Marketing_Strategy" in scored_df.columns
    assert len(scored_df) == len(intel_df)
    
    # 3. Save and Load
    model_dir = tmp_path / "models"
    pipeline.save(str(model_dir))
    loaded_pipeline = RetailIntelligencePipeline.load(str(model_dir))
    loaded_res = loaded_pipeline.score_single_customer(sample_cust)
    assert loaded_res["persona_name"] == result["persona_name"]
