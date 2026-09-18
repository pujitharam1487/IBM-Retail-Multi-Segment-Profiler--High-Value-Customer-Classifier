from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import joblib

# Safe import for SHAP in case DLL/numba is restricted by Windows Application Control
try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class HighValueClassifier:
    """
    Supervised Machine Learning System for High-Value Customer Prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state),
            "XGBoost Classifier": XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=random_state
            )
        }
        self.best_model_name: str = "XGBoost Classifier"
        self.best_model: Optional[Any] = None
        self.benchmark_results: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.test_metrics: Dict[str, Any] = {}
        self.shap_explainer = None
        self.shap_values = None

    def benchmark_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Runs Stratified K-Fold Cross-Validation across all candidate models and returns a comparison table.
        """
        self.feature_names = list(X.columns)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }
        
        results = []
        for name, model in self.models.items():
            cv_out = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
            results.append({
                "Model": name,
                "Accuracy (CV Mean)": cv_out["test_accuracy"].mean(),
                "Accuracy (Std)": cv_out["test_accuracy"].std(),
                "Precision (CV Mean)": cv_out["test_precision"].mean(),
                "Recall (CV Mean)": cv_out["test_recall"].mean(),
                "F1-Score (CV Mean)": cv_out["test_f1"].mean(),
                "ROC-AUC (CV Mean)": cv_out["test_roc_auc"].mean(),
            })
            
        res_df = pd.DataFrame(results).sort_values(by="ROC-AUC (CV Mean)", ascending=False).reset_index(drop=True)
        self.benchmark_results["cv_comparison"] = res_df
        return res_df

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20
    ) -> Dict[str, Any]:
        """
        Performs train-test split, trains all models, fits SHAP explainability on XGBoost,
        and computes comprehensive evaluation metrics.
        """
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        test_scores = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            test_scores[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "y_test": y_test.tolist(),
                "y_prob": y_prob.tolist(),
                "y_pred": y_pred.tolist()
            }
            
        self.best_model = self.models[self.best_model_name]
        self.test_metrics = test_scores
        
        # Fit SHAP Explainer on XGBoost if available
        if HAS_SHAP and shap is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(self.best_model)
                self.shap_values = self.shap_explainer(X_test)
            except Exception:
                try:
                    self.shap_explainer = shap.Explainer(self.best_model, X_train)
                    self.shap_values = self.shap_explainer(X_test)
                except Exception:
                    self.shap_values = None
        else:
            self.shap_values = None
                
        return {
            "test_scores": test_scores,
            "best_model_name": self.best_model_name,
            "feature_names": self.feature_names,
            "X_test": X_test,
            "y_test": y_test
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Extracts feature importance scores from the best model (XGBoost).
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
            
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        feat_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        feat_df["Relative_Pct"] = (feat_df["Importance"] / feat_df["Importance"].sum()) * 100
        return feat_df

    def predict_customer(self, customer_features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predicts High-Value status (0 or 1) and High-Value probability for a single customer feature vector.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Ensure column ordering matches training set
        X_cust = customer_features[self.feature_names]
        pred = int(self.best_model.predict(X_cust)[0])
        prob = float(self.best_model.predict_proba(X_cust)[0][1])
        return pred, prob

    def save(self, filepath: str):
        """
        Persists the trained classifier and metrics.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HighValueClassifier":
        """
        Loads a saved classifier instance.
        """
        return joblib.load(filepath)
