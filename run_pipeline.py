"""
Master Pipeline Runner Script.
Executes the full machine learning workflow:
1. Data initialization & validation
2. K-Means clustering evaluation & persona profiling
3. Supervised classification model benchmarking & XGBoost training
4. SHAP & Feature importance extraction
5. Full customer base scoring & marketing strategy assignment
6. Model artifact persistence to models/ and JSON reports to reports/
"""

import sys
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure utf-8 output encoding for windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import initialize_datasets
from src.preprocessing import RetailPreprocessor
from src.clustering import CustomerSegmenter
from src.classification import HighValueClassifier
from src.pipeline import RetailIntelligencePipeline


def main():
    print("=" * 80)
    print(" [*] RETAIL MULTI-SEGMENT PROFILER & HIGH-VALUE CUSTOMER CLASSIFIER")
    print("=" * 80)
    
    # 1. Initialize data directories and datasets
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/6] Ingesting & Validating Datasets...")
    mall_df, intel_df = initialize_datasets("data")
    print(f"  [+] Mall Customers Dataset: {mall_df.shape[0]} records, {mall_df.shape[1]} features.")
    print(f"  [+] Retail Intelligence Dataset: {intel_df.shape[0]} records, {intel_df.shape[1]} features.")
    print(f"  [+] High-Value Target Distribution: {dict(intel_df['High_Value_Customer'].value_counts())}")
    
    # 2. Preprocessing & Clustering
    print("\n[Step 2/6] Running Unsupervised Persona Discovery (K-Means)...")
    preprocessor = RetailPreprocessor(scaling_method="standard")
    X_cluster_scaled, cluster_feat_names = preprocessor.prepare_clustering_data(intel_df)
    
    segmenter = CustomerSegmenter(n_clusters=5, random_state=42)
    k_eval_df = segmenter.evaluate_k_range(X_cluster_scaled, k_min=2, k_max=8)
    print("\n  --- K-Means Evaluation (Elbow & Silhouette) ---")
    print(k_eval_df[["K", "Inertia_WCSS", "Silhouette_Score", "Davies_Bouldin_Index"]].to_string(index=False))
    
    segmenter.fit(X_cluster_scaled, intel_df, cluster_feat_names)
    profiles_df = segmenter.get_profiles_dataframe()
    print("\n  --- Discovered Customer Personas ---")
    print(profiles_df[["cluster_id", "persona_name", "tier", "percentage", "mean_income", "mean_spending_score"]].to_string(index=False))
    
    # 3. Supervised Classification Benchmarking
    print("\n[Step 3/6] Benchmarking Supervised Models (Stratified 5-Fold CV)...")
    X_clf, y_clf, clf_feat_names = preprocessor.prepare_classification_data(intel_df, target_col="High_Value_Customer")
    
    classifier = HighValueClassifier(random_state=42)
    cv_comparison = classifier.benchmark_models_cv(X_clf, y_clf, n_splits=5)
    print("\n  --- Stratified 5-Fold Cross Validation Results ---")
    print(cv_comparison.to_string(index=False))
    
    # 4. Train Best XGBoost Model & Test Set Evaluation
    print("\n[Step 4/6] Training & Evaluating XGBoost Classifier on Holdout Set...")
    eval_results = classifier.train_and_evaluate(X_clf, y_clf, test_size=0.20)
    xgb_test_metrics = eval_results["test_scores"]["XGBoost Classifier"]
    
    print(f"  [+] XGBoost Test Accuracy:  {xgb_test_metrics['accuracy']:.4f}")
    print(f"  [+] XGBoost Test Precision: {xgb_test_metrics['precision']:.4f}")
    print(f"  [+] XGBoost Test Recall:    {xgb_test_metrics['recall']:.4f}")
    print(f"  [+] XGBoost Test F1-Score:  {xgb_test_metrics['f1']:.4f}")
    print(f"  [+] XGBoost Test ROC-AUC:   {xgb_test_metrics['roc_auc']:.4f}")
    print(f"  [+] Confusion Matrix:       {xgb_test_metrics['confusion_matrix']}")
    
    # Feature Importances
    feat_imp = classifier.get_feature_importances()
    print("\n  --- Top Predictive Features (XGBoost) ---")
    print(feat_imp.head(6).to_string(index=False))
    
    # 5. Build Unified Pipeline & Score Customer Base
    print("\n[Step 5/6] Building Unified Scoring Pipeline & Scoring Customer Base...")
    pipeline = RetailIntelligencePipeline(
        preprocessor=preprocessor,
        segmenter=segmenter,
        classifier=classifier
    )
    
    scored_df = pipeline.score_batch(intel_df)
    scored_csv_path = Path("data/scored_customers.csv")
    scored_df.to_csv(scored_csv_path, index=False)
    print(f"  [+] Successfully scored {len(scored_df)} customers -> Saved to {scored_csv_path}")
    
    # Test single customer scoring
    sample_cust = {
        "CustomerID": "CUST_TEST_VIP",
        "Gender": "Female",
        "Age": 34,
        "Annual_Income_k": 95,
        "Spending_Score": 88,
        "Purchase_Frequency_Year": 32,
        "Avg_Order_Value": 280.0,
        "Recency_Days": 12,
        "Return_Rate_Pct": 4.5,
        "Discount_Usage_Pct": 15.0,
        "Online_Order_Ratio": 0.65,
        "App_Sessions_Month": 24,
        "Years_As_Customer": 4.0
    }
    sample_result = pipeline.score_single_customer(sample_cust)
    print("\n  --- Sample Real-Time Customer Scoring Simulation ---")
    print(f"  Customer ID:       {sample_result['customer_id']}")
    print(f"  Discovered Persona:{sample_result['persona_name']} ({sample_result['tier']})")
    print(f"  High-Value Pred:   {sample_result['high_value_prediction']} (Prob: {sample_result['high_value_probability']})")
    print(f"  Strategy:          {sample_result['recommendation']['strategy']}")
    print(f"  Badge:             {sample_result['recommendation']['badge']}")
    print(f"  Channel:           {sample_result['recommendation']['channel']}")
    
    # 6. Model Persistence & Reporting
    print("\n[Step 6/6] Persisting Model Artifacts & Generating Reports...")
    pipeline.save("models")
    print("  [+] Saved model artifacts to models/ directory.")
    
    report_dict = {
        "dataset_summary": {
            "total_records": len(intel_df),
            "high_value_count": int(intel_df["High_Value_Customer"].sum()),
            "high_value_ratio": float(intel_df["High_Value_Customer"].mean())
        },
        "clustering_metrics": {
            "optimal_k": 5,
            "silhouette_score": float(segmenter.evaluation_results.get("final_silhouette", 0.0)),
            "davies_bouldin_index": float(segmenter.evaluation_results.get("final_davies_bouldin", 0.0)),
            "inertia": float(segmenter.evaluation_results.get("final_inertia", 0.0))
        },
        "classifier_cv_benchmark": cv_comparison.to_dict(orient="records"),
        "best_model_test_metrics": xgb_test_metrics,
        "feature_importances": feat_imp.to_dict(orient="records"),
        "persona_profiles": profiles_df.to_dict(orient="records")
    }
    
    with open("reports/pipeline_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print("  [+] Saved pipeline report to reports/pipeline_report.json")
    
    print("\n" + "=" * 80)
    print(" [OK] ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
