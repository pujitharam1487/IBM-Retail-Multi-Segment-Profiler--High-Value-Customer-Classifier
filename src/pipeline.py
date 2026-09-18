"""
End-to-End Retail Multi-Segment Intelligence Pipeline.
Orchestrates data preprocessing, K-Means clustering, XGBoost classification,
and recommendation generation for real-time and batch scoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import RetailPreprocessor
from src.clustering import CustomerSegmenter
from src.classification import HighValueClassifier
from src.recommendation import RetailRecommendationEngine


class RetailIntelligencePipeline:
    """
    Unified end-to-end inference and scoring pipeline.
    """

    def __init__(
        self,
        preprocessor: Optional[RetailPreprocessor] = None,
        segmenter: Optional[CustomerSegmenter] = None,
        classifier: Optional[HighValueClassifier] = None
    ):
        self.preprocessor = preprocessor or RetailPreprocessor()
        self.segmenter = segmenter or CustomerSegmenter(n_clusters=5)
        self.classifier = classifier or HighValueClassifier()
        self.recommendation_engine = RetailRecommendationEngine()

    def fit(self, df: pd.DataFrame, target_col: str = "High_Value_Customer") -> "RetailIntelligencePipeline":
        """
        Fits both the K-Means clustering model and the Supervised Classification model.
        """
        # 1. Prepare clustering data
        X_cluster_scaled, cluster_feat_names = self.preprocessor.prepare_clustering_data(df)
        self.segmenter.fit(X_cluster_scaled, df, cluster_feat_names)
        
        # 2. Prepare classification data
        X_clf, y_clf, clf_feat_names = self.preprocessor.prepare_classification_data(df, target_col=target_col)
        self.classifier.train_and_evaluate(X_clf, y_clf)
        
        return self

    def score_single_customer(self, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scores a single customer record and returns Persona, High-Value probability, and Recommendation.
        """
        df_single = pd.DataFrame([customer_dict])
        
        # 1. Clean & Feature Engineering
        df_clean = self.preprocessor.clean_dataset(df_single)
        df_feat = self.preprocessor.engineer_features(df_clean)
        
        # 2. Clustering Persona
        age_col = [c for c in df_clean.columns if "age" in c.lower() and "group" not in c.lower()][0]
        income_col = [c for c in df_clean.columns if "income" in c.lower()][0]
        spend_col = [c for c in df_clean.columns if "spend" in c.lower() and "score" in c.lower()][0]
        
        cluster_inputs = np.array([[df_clean[age_col].iloc[0], df_clean[income_col].iloc[0], df_clean[spend_col].iloc[0]]])
        cluster_inputs_scaled = self.preprocessor.cluster_scaler.transform(cluster_inputs)
        cluster_id = int(self.segmenter.predict(cluster_inputs_scaled)[0])
        persona_info = self.segmenter.get_persona_for_cluster(cluster_id)
        
        # 3. Supervised Classification
        # Align features with classifier training set
        clf_inputs = {}
        for feat in self.classifier.feature_names:
            if feat == "Gender_Male":
                gender = str(customer_dict.get("Gender", "Male")).lower()
                clf_inputs["Gender_Male"] = 1 if gender == "male" else 0
            elif feat in df_feat.columns:
                clf_inputs[feat] = df_feat[feat].iloc[0]
            elif feat in customer_dict:
                clf_inputs[feat] = customer_dict[feat]
            else:
                clf_inputs[feat] = 0.0
                
        df_clf_input = pd.DataFrame([clf_inputs])
        hv_pred, hv_prob = self.classifier.predict_customer(df_clf_input)
        
        # 4. Recommendation Engine
        rec_package = self.recommendation_engine.get_recommendation(
            persona_name=persona_info["persona_name"],
            is_high_value=hv_pred,
            high_value_prob=hv_prob,
            annual_income=float(df_clean[income_col].iloc[0]),
            spending_score=float(df_clean[spend_col].iloc[0])
        )
        
        return {
            "customer_id": customer_dict.get("CustomerID", "CUST_NEW"),
            "cluster_id": cluster_id,
            "persona_name": persona_info["persona_name"],
            "persona_tag": persona_info["persona_tag"],
            "tier": persona_info["tier"],
            "persona_description": persona_info["description"],
            "high_value_prediction": "High-Value" if hv_pred == 1 else "Standard",
            "high_value_probability": round(hv_prob, 3),
            "recommendation": rec_package
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores an entire DataFrame of customers and returns an enriched DataFrame.
        """
        df_out = df.copy()
        df_clean = self.preprocessor.clean_dataset(df_out)
        
        # 1. Cluster Scoring
        X_cluster_scaled, _ = self.preprocessor.prepare_clustering_data(df_out)
        cluster_labels = self.segmenter.predict(X_cluster_scaled)
        
        persona_names = []
        persona_tags = []
        tiers = []
        for c_id in cluster_labels:
            p_info = self.segmenter.get_persona_for_cluster(int(c_id))
            persona_names.append(p_info["persona_name"])
            persona_tags.append(p_info["persona_tag"])
            tiers.append(p_info["tier"])
            
        df_out["Cluster_ID"] = cluster_labels
        df_out["Persona_Name"] = persona_names
        df_out["Persona_Tag"] = persona_tags
        df_out["Tier"] = tiers
        
        # 2. Classification Scoring
        X_clf, _, _ = self.preprocessor.prepare_classification_data(df_out)
        # Reindex to ensure feature columns match exactly
        X_clf_aligned = X_clf.reindex(columns=self.classifier.feature_names, fill_value=0)
        
        if hasattr(self.classifier.best_model, "predict_proba"):
            probs = self.classifier.best_model.predict_proba(X_clf_aligned)[:, 1]
        else:
            probs = self.classifier.best_model.predict(X_clf_aligned)
            
        preds = (probs >= 0.5).astype(int)
        
        df_out["High_Value_Prediction"] = ["High-Value" if p == 1 else "Standard" for p in preds]
        df_out["High_Value_Probability"] = np.round(probs, 3)
        
        # 3. Recommendation strategies
        strategies = []
        channels = []
        income_col = [c for c in df_clean.columns if "income" in c.lower()][0]
        spend_col = [c for c in df_clean.columns if "spend" in c.lower() and "score" in c.lower()][0]
        
        for idx in range(len(df_out)):
            rec = self.recommendation_engine.get_recommendation(
                persona_name=persona_names[idx],
                is_high_value=int(preds[idx]),
                high_value_prob=float(probs[idx]),
                annual_income=float(df_clean[income_col].iloc[idx]),
                spending_score=float(df_clean[spend_col].iloc[idx])
            )
            strategies.append(rec["strategy"])
            channels.append(rec["channel"])
            
        df_out["Marketing_Strategy"] = strategies
        df_out["Recommended_Channel"] = channels
        
        return df_out

    def save(self, directory: str = "models"):
        """
        Saves all trained pipeline components.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, dir_path / "preprocessor.joblib")
        joblib.dump(self.segmenter, dir_path / "segmenter.joblib")
        joblib.dump(self.classifier, dir_path / "classifier.joblib")

    @classmethod
    def load(cls, directory: str = "models") -> "RetailIntelligencePipeline":
        """
        Loads all pipeline components from a directory.
        """
        dir_path = Path(directory)
        preprocessor = joblib.load(dir_path / "preprocessor.joblib")
        segmenter = joblib.load(dir_path / "segmenter.joblib")
        classifier = joblib.load(dir_path / "classifier.joblib")
        return cls(preprocessor=preprocessor, segmenter=segmenter, classifier=classifier)
