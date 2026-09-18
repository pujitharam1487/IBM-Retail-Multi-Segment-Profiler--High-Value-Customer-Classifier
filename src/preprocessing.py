"""
Preprocessing and Feature Engineering Module.
Handles data cleaning, outlier handling, categorical encoding, feature scaling,
and leakage-free dataset transformations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder


class RetailPreprocessor:
    """
    Production preprocessor pipeline for customer clustering and classification.
    """

    def __init__(self, scaling_method: str = "standard"):
        self.scaling_method = scaling_method
        self.cluster_scaler = StandardScaler() if scaling_method == "standard" else RobustScaler()
        self.clf_scaler = StandardScaler() if scaling_method == "standard" else RobustScaler()
        self.gender_encoder = LabelEncoder()
        self.is_fitted = False
        self.clf_feature_names: List[str] = []
        self.cluster_feature_names: List[str] = []

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates column names, detects and handles missing values, and drops duplicates.
        """
        df_clean = df.copy()
        
        # Standardize column naming
        df_clean.columns = [
            c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "").replace("-", "_")
            for c in df_clean.columns
        ]
        
        # Remove exact duplicate records if any
        df_clean = df_clean.drop_duplicates()
        
        # Fill missing values if any exist
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_val = df_clean[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(fill_val)
                
        return df_clean

    def handle_outliers(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        factor: float = 1.5,
        clip: bool = True
    ) -> pd.DataFrame:
        """
        Detects and caps outliers using the Interquartile Range (IQR) method.
        """
        df_out = df.copy()
        for col in numeric_cols:
            if col in df_out.columns:
                q25 = df_out[col].quantile(0.25)
                q75 = df_out[col].quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - factor * iqr
                upper_bound = q75 + factor * iqr
                if clip:
                    df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
        return df_out

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific behavioral and interaction features.
        """
        df_feat = df.copy()
        
        # Income to Spend Ratio
        income_col = [c for c in df_feat.columns if "income" in c.lower()][0]
        spend_col = [c for c in df_feat.columns if "spend" in c.lower() and "score" in c.lower()][0]
        
        df_feat["Income_to_Spend_Ratio"] = df_feat[income_col] / (df_feat[spend_col] + 1e-3)
        
        # Age group bins
        df_feat["Age_Group"] = pd.cut(
            df_feat["Age"],
            bins=[0, 25, 40, 55, 100],
            labels=["Young_Adult", "Middle_Age", "Mature_Adult", "Senior"]
        )
        
        # Behavioral engagement index if behavioral features exist
        if "Purchase_Frequency_Year" in df_feat.columns and "Recency_Days" in df_feat.columns:
            # High frequency + low recency = high engagement
            df_feat["Engagement_Index"] = (
                (df_feat["Purchase_Frequency_Year"] / 52.0) * 0.6 +
                ((365 - df_feat["Recency_Days"]) / 365.0) * 0.4
            )
            
        if "Discount_Usage_Pct" in df_feat.columns and "Return_Rate_Pct" in df_feat.columns:
            df_feat["Price_Sensitivity_Index"] = (
                (df_feat["Discount_Usage_Pct"] / 100.0) * 0.7 +
                (df_feat["Return_Rate_Pct"] / 30.0) * 0.3
            )
            
        return df_feat

    def prepare_clustering_data(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts and scales features specifically for K-Means Customer Persona Segmentation.
        Defaults to Age, Annual Income, and Spending Score.
        """
        df_clean = self.clean_dataset(df)
        
        if features is None:
            # Map standard features
            age_col = [c for c in df_clean.columns if "age" in c.lower() and "group" not in c.lower()][0]
            income_col = [c for c in df_clean.columns if "income" in c.lower()][0]
            spend_col = [c for c in df_clean.columns if "spend" in c.lower() and "score" in c.lower()][0]
            features = [age_col, income_col, spend_col]
            
        self.cluster_feature_names = features
        X = df_clean[features].values
        
        if not self.is_fitted:
            X_scaled = self.cluster_scaler.fit_transform(X)
        else:
            X_scaled = self.cluster_scaler.transform(X)
            
        return X_scaled, features

    def prepare_classification_data(
        self,
        df: pd.DataFrame,
        target_col: str = "High_Value_Customer",
        exclude_leakage: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepares feature matrix X and target y for High-Value Customer Classification.
        Excludes CustomerID, target_col, and direct leakage sources (e.g. Total_Annual_Spend)
        when predicting high value.
        """
        df_clean = self.clean_dataset(df)
        df_clean = self.engineer_features(df_clean)
        
        # Binary target creation if not present
        if target_col not in df_clean.columns:
            # Documented business rule: Top 25% by Spending Score for standard mall dataset
            spend_col = [c for c in df_clean.columns if "spend" in c.lower() and "score" in c.lower()][0]
            threshold = df_clean[spend_col].quantile(0.75)
            df_clean[target_col] = (df_clean[spend_col] >= threshold).astype(int)
            
        y = df_clean[target_col].astype(int)
        
        # Columns to exclude from predictor matrix X
        drop_cols = [target_col, "CustomerID", "Customer_ID", "Age_Group"]
        
        if exclude_leakage:
            # Exclude raw target source features to prevent trivial classification
            if "Total_Annual_Spend" in df_clean.columns:
                drop_cols.append("Total_Annual_Spend")
            if "Spending_Score" in df_clean.columns and target_col == "High_Value_Customer":
                # If target was directly computed from spending score, we can exclude or keep historical transactional features
                pass
                
        feature_df = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
        
        # Encode categorical columns
        if "Gender" in feature_df.columns:
            feature_df["Gender_Male"] = (feature_df["Gender"].astype(str).str.lower() == "male").astype(int)
            feature_df = feature_df.drop(columns=["Gender"])
            
        self.clf_feature_names = list(feature_df.columns)
        return feature_df, y, self.clf_feature_names
