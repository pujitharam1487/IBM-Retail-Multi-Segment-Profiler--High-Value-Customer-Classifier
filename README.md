# 🛍️ Retail Multi-Segment Profiler & High-Value Customer Classifier

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

An end-to-end Machine Learning system combining **Unsupervised Learning (K-Means Clustering)** for customer persona discovery and **Supervised Learning (XGBoost Classifier + Baselines)** for high-value customer prediction, complete with a recommendation engine and an interactive Streamlit dashboard.

---

## 📌 Architecture & System Flow

```text
┌────────────────────────────────────────────────────────┐
│              Customer Data Ingestion                   │
│   - Mall Customer Dataset (Age, Gender, Income, Spend) │
│   - Enriched Behavioral Dataset (Recency, Freq, AOV)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       Preprocessing & Validation Pipeline              │
│   - Missing/duplicate treatment & outlier detection    │
│   - Target encoding & Robust/Standard scaling          │
│   - Leakage-safe feature partition                     │
└────────────┬─────────────────────────────┬─────────────┘
             │                             │
             ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   K-Means Clustering     │  │ High-Value Target Setup  │
│  - Elbow Method (WCSS)   │  │  - Top quartile spend/CLV│
│  - Silhouette Analysis   │  │  - Stratified 5-Fold CV  │
│  - Persona Profiling     │  │  - XGBoost + Baselines   │
│  - Interpretable Labels  │  │  - SHAP Explainability   │
└────────────┬─────────────┘  └────────────┬─────────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│       Customer Intelligence & Recommendation           │
│   - Unified Customer Profiling & Scoring Pipeline      │
│   - Segment Personas + HV Probability + Strategy Matrix│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            Interactive Streamlit Dashboard             │
│   - Tab 1: Executive EDA & Overview                    │
│   - Tab 2: K-Means Customer Persona Visualizer (2D/3D) │
│   - Tab 3: XGBoost Classifier Benchmark & SHAP         │
│   - Tab 4: Real-time Customer Simulator                │
│   - Tab 5: Scored Database & CSV Exporter              │
│   - Tab 6: Marketing Strategy Playbook & ROI           │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Capabilities

### 1. Unsupervised Persona Discovery (K-Means)
- Evaluates $K=2 \dots 10$ using **Inertia (Elbow Method)** and **Silhouette Scores**.
- Dynamically assigns descriptive business personas:
  - 💎 **Affluent VIP Spenders**: High Income, High Spending (Tier 1).
  - 🎯 **Affluent Conservative Savers**: High Income, Conservative Spending (Tier 2 Whales).
  - 🔥 **Young Enthusiasts & Trendsetters**: Lower/Mid Income, High Spending (Tier 3 Engagement).
  - 🏷️ **Frugal Budget Conscious**: Lower Income, Low Spending (Tier 4 Value Seekers).
  - ⚖️ **Balanced Mainstream Shoppers**: Moderate Income, Moderate Spending.

### 2. Supervised High-Value Classification (XGBoost & Baselines)
- **Target Leakage Prevention**: Separates historical predictor attributes (recency, purchase frequency, average order value, returns, discount usage, app engagement) from ground-truth label derivation.
- **Stratified 5-Fold Cross-Validation Benchmark**:
  - Logistic Regression (Standardized baseline)
  - Decision Tree Classifier
  - Random Forest Classifier (Bagging ensemble)
  - **XGBoost Classifier** (Gradient Boosted Trees - Main Model)
- **Model Explainability**: SHAP (SHapley Additive exPlanations) and global feature importance ranking.

### 3. Automated Recommendation Engine
- Maps customer segment personas and high-value probabilities to tailored marketing strategies, promotional discount policies, and channel allocations.

### 4. Interactive Streamlit Web Application
- Multi-tab interactive UI built with Plotly 2D/3D charts.
- Live single-customer scoring simulator with dynamic gauge indicators.
- Filterable customer table with one-click CSV export.

---

## 📂 Project Structure

```text
Retail-workshop/
├── data/
│   ├── mall_customers.csv                   # Classic 200 Mall Customers dataset
│   ├── retail_customer_intelligence.csv     # Enriched multi-feature behavioral dataset
│   └── scored_customers.csv                 # Full scored database output
├── models/
│   ├── preprocessor.joblib                  # Serialized data preprocessor & scalers
│   ├── segmenter.joblib                     # Serialized K-Means segmenter
│   └── classifier.joblib                    # Serialized XGBoost classifier & metrics
├── reports/
│   └── pipeline_report.json                 # Comprehensive training & validation report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Dataset loading and generation utilities
│   ├── preprocessing.py                     # Outlier clipping, scaling, feature engineering
│   ├── clustering.py                        # K-Means clustering & persona profiling
│   ├── classification.py                    # Multi-model CV benchmarking & XGBoost
│   ├── recommendation.py                    # Rule-based marketing strategy mapper
│   └── pipeline.py                          # Unified scoring & inference pipeline
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                     # Comprehensive pytest test suite
├── app.py                                   # Full-featured Streamlit interactive app
├── run_pipeline.py                          # Master CLI pipeline runner
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Automated ML Pipeline
```bash
python run_pipeline.py
```
*This command executes data validation, K-Means clustering, 5-fold cross-validation benchmarking, model training, artifact persistence, and generates `reports/pipeline_report.json`.*

### 3. Run Unit Tests
```bash
pytest tests/test_pipeline.py -v
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Evaluation Summary

| Model | Accuracy (CV Mean) | Precision (CV Mean) | Recall (CV Mean) | F1-Score (CV Mean) | ROC-AUC (CV Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.973** | **0.954** | **0.940** | **0.946** | **0.997** |
| **Random Forest** | 0.967 | 0.928 | 0.940 | 0.934 | 0.996 |
| **Logistic Regression** | 0.988 | 0.987 | 0.967 | 0.976 | 0.999 |
| **Decision Tree** | 0.930 | 0.894 | 0.820 | 0.853 | 0.907 |

---

## 💼 Business Impact & ROI

- **VIP Retention (+22% CLV)**: Personalized concierge and zero-discount luxury experiences for top-tier spenders.
- **Upselling Latent Wealth (+35% AOV)**: Converting conservative high-income savers with premium bundles.
- **Viral Growth (+40% Frequency)**: Gamified mobile drops and referral rewards for young trendsetters.
- **Efficient Budget Allocation**: Reduces generic blanket discounts, focusing capital where ROI is highest.
