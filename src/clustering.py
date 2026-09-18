"""
Unsupervised Learning Module - K-Means Customer Persona Segmentation.
Implements K-Means clustering, evaluation via Elbow Method & Silhouette Scores,
and dynamic business persona profiling.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib


class CustomerSegmenter:
    """
    K-Means Customer Segmentation Engine with Persona Profiling.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.feature_names: List[str] = []
        self.cluster_profiles: Dict[int, Dict[str, Any]] = {}
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate_k_range(
        self,
        X_scaled: np.ndarray,
        k_min: int = 2,
        k_max: int = 10
    ) -> pd.DataFrame:
        """
        Evaluates a range of K values using Inertia (Elbow Method), Silhouette Score,
        Davies-Bouldin Index, and Calinski-Harabasz Score.
        """
        results = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X_scaled)
            
            inertia = km.inertia_
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            
            results.append({
                "K": k,
                "Inertia_WCSS": inertia,
                "Silhouette_Score": sil_score,
                "Davies_Bouldin_Index": db_score,
                "Calinski_Harabasz_Score": ch_score
            })
            
        eval_df = pd.DataFrame(results)
        self.evaluation_results["k_evaluation"] = eval_df
        return eval_df

    def fit(
        self,
        X_scaled: np.ndarray,
        df_original: pd.DataFrame,
        feature_names: List[str]
    ) -> "CustomerSegmenter":
        """
        Fits the K-Means clustering model and generates statistical persona profiles.
        """
        self.feature_names = feature_names
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=15,
            random_state=self.random_state
        )
        labels = self.model.fit_predict(X_scaled)
        
        # Calculate cluster metrics
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        self.evaluation_results["final_silhouette"] = sil
        self.evaluation_results["final_davies_bouldin"] = db
        self.evaluation_results["final_inertia"] = self.model.inertia_
        
        # Persona profiling
        self._build_persona_profiles(df_original, labels)
        return self

    def _build_persona_profiles(self, df_original: pd.DataFrame, labels: np.ndarray):
        """
        Analyzes characteristics of each cluster and assigns descriptive business persona labels.
        """
        df_profile = df_original.copy()
        df_profile["Cluster"] = labels
        
        income_col = [c for c in df_profile.columns if "income" in c.lower()][0]
        spend_col = [c for c in df_profile.columns if "spend" in c.lower() and "score" in c.lower()][0]
        age_col = [c for c in df_profile.columns if "age" in c.lower() and "group" not in c.lower()][0]
        
        overall_mean_income = df_profile[income_col].mean()
        overall_mean_spend = df_profile[spend_col].mean()
        overall_mean_age = df_profile[age_col].mean()
        
        self.cluster_profiles = {}
        
        for c_id in range(self.n_clusters):
            cluster_subset = df_profile[df_profile["Cluster"] == c_id]
            c_count = len(cluster_subset)
            c_pct = (c_count / len(df_profile)) * 100
            
            c_income = cluster_subset[income_col].mean()
            c_spend = cluster_subset[spend_col].mean()
            c_age = cluster_subset[age_col].mean()
            
            # Dynamic business persona label assignment
            if c_income >= overall_mean_income and c_spend >= overall_mean_spend:
                persona_name = "Affluent VIP Spenders"
                persona_tag = "Premium / High-Value"
                description = "High income with high spending propensity. Prime revenue drivers with high luxury demand."
                tier = "Tier 1 - Highest Value"
            elif c_income >= overall_mean_income and c_spend < overall_mean_spend:
                persona_name = "Affluent Conservative Savers"
                persona_tag = "Target / High-Potential"
                description = "High income but conservative spending. Prime opportunity for upselling with high-value propositions."
                tier = "Tier 2 - High Potential"
            elif c_income < overall_mean_income and c_spend >= overall_mean_spend:
                persona_name = "Young Enthusiasts & Trendsetters"
                persona_tag = "Potential / High-Engagement"
                description = "Moderate/lower income with strong spending appetite. Trend-conscious and responsive to brand buzz."
                tier = "Tier 3 - High Engagement"
            elif c_income < overall_mean_income and c_spend < overall_mean_spend:
                persona_name = "Frugal Budget Conscious"
                persona_tag = "Budget Shoppers"
                description = "Careful spenders with lower income. Seek discounts, promotions, and essential value items."
                tier = "Tier 4 - Cost Sensitive"
            else:
                persona_name = "Balanced Mainstream Shoppers"
                persona_tag = "Average / Regular Shoppers"
                description = "Moderate income and average spending habits. Stable customer base requiring consistent engagement."
                tier = "Tier 3 - Stable Base"
                
            self.cluster_profiles[c_id] = {
                "cluster_id": c_id,
                "persona_name": persona_name,
                "persona_tag": persona_tag,
                "tier": tier,
                "description": description,
                "count": c_count,
                "percentage": round(c_pct, 1),
                "mean_age": round(c_age, 1),
                "mean_income": round(c_income, 1),
                "mean_spending_score": round(c_spend, 1),
            }

    def get_profiles_dataframe(self) -> pd.DataFrame:
        """
        Returns persona profiles in a structured pandas DataFrame.
        """
        return pd.DataFrame(list(self.cluster_profiles.values()))

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Predicts cluster IDs for scaled input features.
        """
        if self.model is None:
            raise ValueError("CustomerSegmenter model has not been fitted yet.")
        return self.model.predict(X_scaled)

    def get_persona_for_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """
        Returns persona metadata for a specific cluster ID.
        """
        return self.cluster_profiles.get(cluster_id, {
            "cluster_id": cluster_id,
            "persona_name": f"Cluster {cluster_id}",
            "persona_tag": "Standard Segment",
            "tier": "Standard",
            "description": "Customer cluster segment."
        })

    def save(self, filepath: str):
        """
        Saves the fitted segmenter and its persona profiles.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerSegmenter":
        """
        Loads a saved segmenter instance.
        """
        return joblib.load(filepath)
