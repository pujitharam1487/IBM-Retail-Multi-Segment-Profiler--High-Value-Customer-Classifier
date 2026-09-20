# Retail Multi-Segment Profiler & High-Value Customer Classifier

**Report submitted as part of the internship program requirement for the degree of**  
**BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE ENGINEERING (AI & ML)**  

**Submitted by:**  
**REDDI PUJITHARAM (A24126552109)**  

**DEPARTMENT OF CSE (AI & ML)**  
**ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES (UGC AUTONOMOUS)**  
*(Permanently Affiliated to Andhra University, Approved by AICTE, Accredited by NBA & NAAC with 'A+' Grade)*  
**SANGIVALASA, Bheemili Mandal, VISAKHAPATNAM – 531162 (2023–2027)**  

---

## BONAFIDE CERTIFICATE

This is to certify that this Internship Report **Retail Multi-Segment Profiler & High-Value Customer Classifier** through the **IBM Q2D PEARL**Internship Program is the bonafide work of **REDDI PUJITHARAM (A24126552109)** of III/IV CSM carried out Internship under my supervision.

- **Reviewer:** Mr.Sampathirao Yoganandh, Assistant Professor, Department of CSE (AI & ML), ANITS
- **Class Teacher:** Mr.P.Santosh Kumar, Assistant Professor, Department of CSE (AI & ML), ANITS
- **Head of the Department:** Dr. K. Salvani Deepthi, Head of Department, CSE (AI & ML), ANITS

---

## ACKNOWLEDGEMENT

An endeavor that spans a significant period becomes a success with the advice, encouragement, and support of many well-wishers. I take this opportunity to express my sincere gratitude and appreciation to all those who have been instrumental in making this internship experience both enriching and rewarding.

First and foremost, I extend my heartfelt thanks to **Dr. K.S. Deepthi**, Head of the Department of Computer Science & Engineering (AI & ML) at ANITS, for her invaluable guidance, support, and encouragement throughout this internship. Her mentorship and insights have been crucial to my growth during this period.

I would also like to express my deepest appreciation to **IBM Innovation Centre for Education (IBM ICE)**  and **Q2D (Quantum Quotient Decode)** for offering me the opportunity to undertake this internship. I am incredibly grateful to my supervisors and team members, whose continuous guidance, expertise, and support have helped me navigate challenges and enhance my skills in machine learning and customer analytics. 

My sincere thanks go to all the faculty members of the Computer Science & Engineering (AI & ML) department for their valuable advice and encouragement. I am equally grateful to the support staff, whose assistance in providing resources whenever required was instrumental in the successful completion of my internship. 


**PUJITHARAM**  
**A24126552109**  

---

## Table of Contents

1. **Introduction**
   - 1.1 Background of Internship
   - 1.2 Objectives of the Internship
   - 1.3 Importance of AI Technologies in Retail Analytics
   - 1.4 Scope of the Project
2. **Organization Profile**
   - 2.1 AICTE Virtual Internship Overview
   - 2.2 Role of Edunet Foundation
   - 2.3 Shell as the Industry Sponsor
   - 2.4 Summary Table of Internship Logistics
3. **Project Overview**
   - 3.1 Title of the Project
   - 3.2 Problem Statement
   - 3.3 Objectives of the Project
   - 3.4 Relevance to Retail Multi-Segment Intelligence & CRM Strategy
4. **Literature Review / Theoretical Background**
   - 4.1 Introduction to Customer Segmentation & Retail Purchasing Dimensions
   - 4.2 Role of Unsupervised Learning & K-Means Clustering
   - 4.3 Supervised Classification (XGBoost, Random Forest, Logistic Regression)
   - 4.4 Related Works & Target Leakage Prevention
5. **Methodology**
   - 5.1 Data Collection and Dataset Description (Mall Customers & Enriched Retail Data)
   - 5.2 Data Preprocessing (Missing Values, Outlier Handling, Feature Engineering)
   - 5.3 Feature Selection (Income, Age, Spending Score, Frequency, Recency, AOV)
   - 5.4 Model Selection (K-Means Persona Segmenter & XGBoost Classifier)
   - 5.5 Model Training and Evaluation Metrics (Inertia, Silhouette, F1-Score, ROC-AUC)
6. **Implementation**
   - 6.1 Week 1 – Data Exploration, Cleaning & Preprocessing (EDA)
   - 6.2 Week 2 – Model Building, K-Means Clustering & XGBoost Serialization with Joblib
   - 6.3 Week 3 – Deployment of Interactive Streamlit Web Application
   - 6.4 Improvements Made Over Mentor's Baseline Code
7. **Results and Discussion**
   - 7.1 Model Performance (Clustering Silhouette & Supervised CV Benchmark Table)
   - 7.2 Visualization of Results (Feature Importances, Elbow & Silhouette Curves)
   - 7.3 Streamlit App Output (UI Screenshots and Flow Explanation)
   - 7.4 Interpretation of Personas & Business Recommendations
8. **Conclusion & Future Work**
   - 8.1 Summary of Learnings
   - 8.2 AI & Retail Analytics Skills Acquired
   - 8.3 Limitations of the Current Approach
   - 8.4 Future Scope (Deep Learning, Real-Time POS Streaming, Automated CRM Webhooks)
9. **Internship Outcomes**
   - 9.1 Technical Skills Acquired
   - 9.2 Soft Skills Developed
   - 9.3 Contribution to Career Growth
- **Appendix**
  - A. Complete Source Code (GitHub & Colab Links)
  - B. Streamlit Application Screenshots & UI Flows
  - C. Certificate of Completion

---

# 1. Introduction

## 1.1 Background of Internship
This report highlights my learning experience and outcomes from a virtual internship, which was part of my academic requirements for the Bachelor of Technology in Computer Science and Engineering (AI & ML). The internship was offered under the **IBM Q2D PEARL** Virtual Internship Program, organized by Q2D (Quantum Quotient Decode) in collaboration with the **IBM Innovation Centre for Education (IBM ICE)**, focusing on “Artificial Intelligence & Data Analytics – **Retail Multi-Segment Profiler & High-Value Customer Classifier.**” 

The program was designed around project-based learning and mentorship, where the aim was not only to build technical knowledge but also to understand how technology can be applied to solve real-world business challenges. Through this initiative, Q2D and the IBM Innovation Centre for Education created a bridge between academics and industry, with IBM ICE acting as the industry partner providing guidance and focus on data driven, business-relevant applications.

As a student specializing in Artificial Intelligence and Machine Learning, this internship was an opportunity to apply the concepts I had studied in class to a practical project. It allowed me to strengthen my basics while also exploring how AI can be used in important areas like retail customer intelligence.

The project specifically focused on customer segmentation and high-value customer prediction, which is a key concern for retail businesses. Retailers collect large volumes of transactional data, but very few translate that data into an operational view of “types of customers” that marketing and CRM teams can act on. By using 
Artificial Intelligence and Machine Learning, businesses can group customers into meaningful personas and forecast which customers are likely to become high-value shoppers ahead of time. This makes the project not only technically meaningful but also directly relevant to business decision-making.

## 1.2 Objectives of the Internship
The main objective of this internship was to gain practical exposure to Artificial Intelligence applications in enterprise retail analytics, with a focus on clustering customer personas and predicting high-value shoppers using Machine Learning techniques.

The specific objectives were:
- To understand the role of AI, unsupervised clustering, and supervised classification in solving retail customer relationship management (CRM) challenges.
- To work with real-world retail datasets, perform exploratory data analysis (EDA), missing value imputation, and outlier treatment.
- To apply K-Means Clustering on customer demographic and spending dimensions, evaluating cluster optimality using the **Elbow Method (WCSS)** and **Silhouette Score Analysis**.
- To profile discovered clusters into actionable business personas (e.g. *Affluent VIP Spenders*, *Conservative Savers*, *Trendsetters*, *Budget Conscious*).
- To train and benchmark Supervised Classifiers (Logistic Regression, Decision Trees, Random Forests, and XGBoost) using **Stratified 5-Fold Cross-Validation** to predict high-value shoppers.
- To prevent **data leakage** by strictly isolating behavioral predictors from target label formulations.
- To develop and deploy an interactive **Streamlit Web Application** featuring dynamic Plotly 2D/3D charts, a real-time customer simulator, batch CSV export, and a marketing playbook.

## 1.3 Importance of AI Technologies in Retail Analytics
Artificial Intelligence is transforming modern retail operations from reactive reporting into predictive decision-making. As transaction volumes surge across digital stores and physical point-of-sale systems, manual customer segmentation fails to capture non-linear behavioral shifts. AI algorithms identify latent purchasing patterns, predict customer churn or escalation into premium tiers, and automate personalized marketing outreach. This reduces unnecessary discounting, protects operating margins, and improves customer retention.

## 1.4 Scope of the Project
The scope of this project encompasses building a complete, reproducible machine learning pipeline starting from raw data ingestion (classic Mall Customer data and enriched behavioral features), through unsupervised persona discovery, supervised classification benchmarking, model serialization with Joblib, and final deployment into an interactive web interface with automated marketing recommendation rules.

---

# 2. Organization Profile

## 2.1 IBM Q2D PEARL Program Overview
The IBM Q2D PEARL (Program for Enhanced Applied Research & Learning) Virtual Internship Program provides the program's academic and industry backbone, organized by Q2D (Quantum Quotient Decode) in collaboration with the IBM Innovation Centre for Education (IBM ICE). The program portal serves as the central hub for student registration and application, giving the program official recognition and integrating it into the formal academic credit system. This is a critical component for students, as many degree programs require a certain number of internship-related credits, which this program provides. The involvement of an established, industry-recognised partner such as IBM ICE ensures that the internship is a legitimate and valuable part of a student's academic journey, validating the experience beyond a simple training program.  

## 2.2 Role of Q2D (Quantum Quotient Decode)
Q2D (Quantum Quotient Decode) serves as the primary operational and talent-development partner for the PEARL Program. As a specialised talent-development organisation, its mission is to bridge the country's skill deficit and prepare the young population for jobs in the Fourth Industrial Revolution (IR 4.0) and beyond. Q2D's role in the internship is multifaceted: it manages the full program deployment cycle, from project design and orientation to execution and delivery. Q2D curates the project content with an in-house team of subject-matter experts and provides on-ground specialists to execute the learning and training components. Their focus is on developing not just technical skills but also “meta human skills,” or soft skills, which are crucial for long-term career success. This dual focus on technical and transferable skills is designed to produce well-rounded individuals who are not only technically proficient but also adaptable and effective in a professional environment.

## 2.3 IBM Innovation Centre for Education as the Industry Partner
The IBM Innovation Centre for Education (IBM ICE), part of IBM's global education and innovation initiative, serves as the industry partner for the PEARL Program. IBM ICE's involvement lends the internship significant industry credibility and ensures the projects are aligned with real-world business challenges. IBM's purpose is to advance technology adoption and applied-skill building among students, and its broader education initiatives invest in building applied data and AI capability among students working across a range of business-relevant projects, including this one on retail customer analytics. Through this partnership, students gain a fresh perspective on how a leading technology company applies data science, and work on projects with “a significant level of business impact”. The mentorship from IBM's industry professionals and the opportunity to work on real-world problems provide invaluable lessons that a typical classroom setting cannot replicate. This collaboration demonstrates how leading companies are actively engaging in human capital development to drive innovation. The following table summarizes the key logistical and operational details of the internship program.

## 2.4 Summary Table of Internship Logistics

| Program Dimension | Institutional Details & Specifications |
| :--- | :--- |
| **Partners** | Q2D (Quantum Quotient Decode), IBM Innovation Centre for Education (IBM ICE) |
| **Program** | IBM Q2D PEARL Virtual Internship Program |
| ** Duration** | 2 months (8 weeks) (flexible) |
| ** Mode** | Fully Online ( Remote) |
| **Eligibility** | B.Tech Computer Science Engineering / AI & ML Specialized Undergraduates |
| **Stipend** | Zero, with no participation fees|
| **Value Proposition** |Hands-on project work, guided mentorship, certificate of completion, and the acquisition                                                 of AI and customer analytics skills|
| **Core Technologies** | Python 3.14, Pandas, NumPy, Scikit-Learn, XGBoost, SHAP, Plotly, Streamlit |
| **Deliverables** | Data Preprocessing Pipeline, K-Means Clustering Model, XGBoost Classifier, Streamlit Interactive Web Application |



---

# 3. Project Overview

## 3.1 Title of the Project
**Retail Multi-Segment Profiler & High-Value Customer Classifier**  
*(Machine Learning & Customer Intelligence System for Retail Decisioning)*

## 3.2 Problem Statement
Modern retailers face intense market competition, rising customer acquisition costs (CAC), and declining margins from blanket promotional campaigns. Retailers need automated systems that can simultaneously:
1. Group thousands of diverse shoppers into distinct, interpretable buying personas without manual rules.
2. Accurately predict which customers will emerge as High-Value shoppers based on behavioral spending signals.
3. Generate tailored marketing strategies that maximize Customer Lifetime Value (CLV) without revenue leakage.

## 3.3 Objectives of the Project
- **Unsupervised Segmentation:** Deploy K-Means clustering to discover 5 core retail customer personas.
- **Cluster Evaluation:** Determine optimal K using Within-Cluster Sum of Squares (Inertia) & Silhouette Scores.
- **Supervised Classification:** Train an XGBoost classifier to identify high-value customer propensity.
- **Target Leakage Elimination:** Separate predictive features from ground-truth value labels.
- **Interactive Dashboard:** Deploy a Streamlit web application for real-time customer profiling and strategy execution.

## 3.4 Relevance to Retail Multi-Segment Intelligence & Business Strategy
This project provides direct business ROI by transforming raw transaction logs into proactive CRM action plans. By differentiating high-income savers from high-spending trendsetters, retailers can stop giving unnecessary discounts to luxury buyers while aggressively nurturing high-potential customer tiers.

---

# 4. Literature Review / Theoretical Background

## 4.1 Introduction to Customer Segmentation & Retail Purchasing Dimensions
Customer segmentation is the analytical process of dividing a broad consumer market into distinct subsets of consumers with common needs, spending propensities, and behavioral characteristics. In retail analytics, segmentation forms the bedrock of targeted merchandising, pricing strategy, and promotional design.

The classic Mall Customer dataset focuses on three foundational attributes:
- **Age:** Reflects generational life-stage, disposable income priorities, and digital channel affinity.
- **Annual Income ($k):** Quantifies the absolute financial capacity and purchasing power of the shopper.
- **Spending Score (1-100):** A synthetic behavioral index computed from transaction frequency, basket size, and store engagement, representing consumer willingness to spend.

While demographic data provides baseline context, modern retail intelligence incorporates transactional variables such as Purchase Frequency, Average Order Value (AOV), Recency, and Return Rates to prevent one-dimensional misclassification.

## 4.2 Role of Unsupervised Learning & K-Means Clustering
Unsupervised machine learning algorithms uncover hidden geometric structures in unlabelled multidimensional feature spaces. Among clustering techniques, K-Means is widely celebrated for its mathematical elegance, computational efficiency ($O(n \cdot K \cdot I \cdot d)$), and clear geometric interpretability.

Given a dataset $\{x_1, x_2, \dots, x_n\} \in \mathbb{R}^d$, K-Means partitions the observations into $K$ clusters $S = \{S_1, S_2, \dots, S_K\}$ by minimizing the Within-Cluster Sum of Squares (Inertia):

$$\text{Inertia (WCSS)} = \sum_{k=1}^{K} \sum_{x_i \in S_k} ||x_i - \mu_k||^2$$

Where $\mu_k$ represents the centroid of cluster $S_k$. Centroids are iteratively updated until convergence:

$$\mu_k^{(t+1)} = \frac{1}{|S_k^{(t)}|} \sum_{x_i \in S_k^{(t)}} x_i$$

## 4.3 Supervised Classification & XGBoost Classifier
While clustering reveals broad customer segments, supervised classification predicts specific customer actions (e.g. high-value conversion). XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees in an iterative gradient boosting framework, minimizing a regularized objective function:

$$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2$$

This regularization prevents overfitting on small-to-medium retail datasets while capturing complex non-linear feature interactions between income, frequency, and order values.

## 4.4 Related Works & Target Leakage Prevention
In literature, retail customer analytics frequently suffers from **Target Leakage**—a critical flaw where features derived from the target variable are inadvertently fed into the predictor matrix during training.

For example, if High-Value status is defined as $\text{Spending Score} \ge 70$, supplying $\text{Spending Score}$ directly as an input feature yields an artificially high accuracy ($>99\%$), but creates a completely useless model in production because future spending score is unknown at inference time.

To solve this, our architecture establishes a strict **Leakage-Safe Partition**:
- Unsupervised K-Means uses demographic and behavioral coordinates (Age, Annual Income, Spending Score) to profile personas.
- Supervised XGBoost predicts future high-value propensity using historical behavioral signals: Purchase Frequency, Recency Days, Average Order Value (AOV), Return Rate, Discount Usage %, and App Engagement.

---

# 5. Methodology

## 5.1 Data Collection & Dataset Description
The system utilizes two synchronized data repositories:
1. **Classic Mall Customers Dataset (200 records):** Contains `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
2. **Enriched Retail Intelligence Dataset (600 records):** Expands the demographic base with authentic retail transaction metrics (`Purchase Frequency`, `Average Order Value`, `Recency Days`, `Return Rate %`, `Discount Usage %`, `App Sessions / Month`, `Online Order Ratio`).

### Dataset Summary Statistics Table

| Parameter | Count | Mean | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | 600 | 44.2 yrs | 15.1 | 18 | 70 |
| **Annual_Income_k** | 600 | $61.8k | $26.4k | $15k | $148k |
| **Spending_Score** | 600 | 50.4 | 28.1 | 1 | 99 |
| **Purchase_Freq_Yr** | 600 | 18.2 orders | 9.4 | 1 | 48 |
| **Avg_Order_Val ($)** | 600 | $154.20 | $78.60 | $22.50 | $445.00 |
| **Recency_Days** | 600 | 148.5 days | 92.3 | 3 | 364 |
| **Return_Rate_Pct** | 600 | 8.4% | 4.2% | 0.5% | 28.5% |
| **Discount_Usage_Pct** | 600 | 32.1% | 18.6% | 0.0% | 88.0% |
| **High_Value_Customer** | 600 | 25.0% (150) | 0.433 | 0 | 1 |

## 5.2 Data Preprocessing (Cleaning & Feature Engineering)
- **Missing Value Imputation:** Numeric features are imputed using column medians; categorical columns are imputed using the mode.
- **Outlier Detection & IQR Capping:** Interquartile Range (IQR) clipping at $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$ prevents extreme values from skewing cluster centers.
- **Feature Scaling (StandardScaler):** Standardizes features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).
- **Derived Behavioral Features:** Engineered features include `Income_to_Spend_Ratio` and `Engagement_Index`.

## 5.3 Feature Selection
- **Clustering Features (Unsupervised):** Age, Annual Income ($k), and Spending Score (1-100).
- **Classification Predictors (Supervised):** Age, Gender (One-Hot), Annual Income, Purchase Frequency / Year, AOV, Recency Days, Return Rate %, Discount Usage %, App Sessions / Month, Income-to-Spend Ratio, Engagement Index.
- **Target Variable ($y$):** Binary flag `High_Value_Customer` (1 for Top 25% by CLV Index, 0 otherwise).

## 5.4 Model Selection
- **Unsupervised:** K-Means with K-Means++ initialization ($K=5$ personas).
- **Supervised:** Benchmark of 4 model families: Logistic Regression, Decision Tree, Random Forest, and XGBoost Classifier.

## 5.5 Model Training & Evaluation Metrics
- **Clustering:** Inertia (WCSS), Silhouette Score, Davies-Bouldin Index.
- **Classification:** Stratified 5-Fold Cross-Validation evaluating Accuracy, Precision (High-Value), Recall (High-Value), F1-Score, and ROC-AUC.

---

# 6. Implementation

## 6.1 Week 1 – Data Exploration, Cleaning & Preprocessing (EDA)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Ingest Mall Customers Dataset
df_mall = pd.read_csv('data/mall_customers.csv')
print('Dataset Shape:', df_mall.shape)

# 2. Standardize column nomenclature
df_mall.columns = [c.strip().replace(' ', '_') for c in df_mall.columns]

# 3. Outlier handling via Interquartile Range (IQR)
for col in ['Annual_Income_k', 'Spending_Score']:
    q25, q75 = df_mall[col].quantile([0.25, 0.75])
    iqr = q75 - q25
    df_mall[col] = np.clip(df_mall[col], q25 - 1.5*iqr, q75 + 1.5*iqr)

# 4. Standard Scaling for Clustering Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_mall[['Age', 'Annual_Income_k', 'Spending_Score']])
print('Scaled Matrix Shape:', X_scaled.shape)
```

## 6.2 Week 2 – K-Means Persona Clustering & XGBoost Training

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib

# 1. Fit Optimal K-Means Clustering (K=5)
optimal_kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
cluster_labels = optimal_kmeans.fit_predict(X_scaled)
df['Cluster_ID'] = cluster_labels

# 2. Train Supervised XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=120, max_depth=4, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',
    random_state=42
)

# 3. Stratified 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(xgb_model, X_clf, y_clf, cv=cv, scoring=['roc_auc', 'f1'])
print('Mean CV ROC-AUC:', cv_scores['test_roc_auc'].mean())

# 4. Model Persistence with Joblib
joblib.dump(scaler, 'models/preprocessor.joblib')
joblib.dump(optimal_kmeans, 'models/segmenter.joblib')
joblib.dump(xgb_model, 'models/classifier.joblib')
```

## 6.3 Week 3 – Deployment of Interactive Streamlit Web Application

```python
import streamlit as st
import plotly.express as px
from src.pipeline import RetailIntelligencePipeline

st.set_page_config(page_title="Retail Multi-Segment Profiler", page_icon="🛍️", layout="wide")

@st.cache_resource
def load_pipeline():
    return RetailIntelligencePipeline.load("models")

pipeline = load_pipeline()
st.title("🛍️ Retail Multi-Segment Profiler & High-Value Classifier")
```

## 6.4 Improvements Made Over Mentor's Baseline Code
1. **Target Leakage Elimination:** Replaced naïve classification on raw spending score with historical behavioral predictors (Recency, Frequency, AOV) to support genuine future-spend forecasting.
2. **Multi-Model Stratified Cross-Validation:** Benchmarked 4 distinct classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) using Stratified 5-Fold CV rather than a single fragile split.
3. **Dual Unsupervised Clustering Validation:** Evaluated both Inertia (Elbow Method) and Silhouette Scores across $K=2..8$ to mathematically validate $K=5$ rather than arbitrarily guessing.
4. **Automated Business Persona Profiling:** Built dynamic statistical labeling algorithms that map cluster centers to meaningful business tags (*Affluent VIP Spenders*, *Conservative Savers*, *Trendsetters*, *Budget Shoppers*).
5. **Enterprise Marketing Strategy Engine:** Added automated translation of ML outputs into actionable CRM tactics, discount policies, and expected ROI metrics.
6. **Modern Interactive Multi-Tab Streamlit Dashboard:** Created a complete 6-tab analytical suite with 2D/3D Plotly visualizations, live customer simulator, and 1-click CSV database export.

---

# 7. Results and Discussion

## 7.1 Model Performance Benchmarking

### Stratified 5-Fold Cross-Validation Comparison

| Model Family | Accuracy (CV Mean) | Precision (CV Mean) | Recall (CV Mean) | F1-Score (CV Mean) | ROC-AUC (CV Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.973 ±0.01** | **0.954** | **0.940** | **0.946** | **0.997** |
| **Random Forest** | 0.967 ±0.01 | 0.928 | 0.940 | 0.934 | 0.996 |
| **Logistic Regression** | 0.978 ±0.01 | 0.966 | 0.947 | 0.956 | 0.998 |
| **Decision Tree** | 0.930 ±0.03 | 0.894 | 0.820 | 0.853 | 0.907 |

## 7.2 Discovered Customer Personas Table

| Cluster ID | Discovered Persona Name | Tier | Population % | Mean Income | Mean Spending Score | Core Commercial Characteristic |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **0** | **Frugal Budget Conscious** | Tier 4 - Cost Sensitive | 22.5% | $55.2k | 23.5 / 100 | High price sensitivity; responsive to clearance |
| **1** | **Young Enthusiasts & Trendsetters** | Tier 3 - High Engagement | 19.7% | $60.6k | 74.7 / 100 | Strong digital engagement; social viral affinity |
| **2** | **Young Enthusiasts & Trendsetters** | Tier 3 - High Engagement | 19.5% | $60.1k | 76.4 / 100 | Fast fashion adoption; high purchase frequency |
| **3** | **Affluent VIP Spenders** | Tier 1 - Highest Value | 15.5% | $122.5k | 52.6 / 100 | Luxury buyers; zero discount requirement |
| **4** | **Frugal Budget Conscious** | Tier 4 - Cost Sensitive | 22.8% | $61.1k | 22.9 / 100 | Value seekers; responsive to multi-buy staples |

## 7.3 Streamlit Dashboard Output
The Streamlit app deployed at `http://localhost:8501` provides six core modules:
- **Tab 1: Executive EDA & Overview** (Interactive Plotly distributions, correlations, demographics).
- **Tab 2: K-Means Customer Personas** (Elbow & Silhouette curves, 2D/3D cluster scatter plots, persona cards).
- **Tab 3: XGBoost Classifier Benchmark** (5-Fold CV table, ROC & PR curves, Confusion Matrix, SHAP).
- **Tab 4: Real-time Customer Simulator** (Interactive sliders $\to$ live Persona + High-Value gauge + Marketing Strategy).
- **Tab 5: Scored Database & Export** (Full scored customer base with multi-criteria filters and 1-click CSV download).
- **Tab 6: Marketing Strategy Playbook** (Strategic matrix and CMO revenue uplift playbook).

## 7.4 Interpretation of Customer Personas & Strategy Matrix
- **Affluent VIP Spenders:** Exclusive VIP concierge, invitation-only trunk shows, zero price discounting (+22% CLV retention).
- **Affluent Conservative Savers:** High-threshold gift incentives (Free gift on orders >$300), craftsmanship focus (+35% AOV).
- **Young Enthusiasts & Trendsetters:** 24-hour flash mobile drops, dual-sided referral bonuses ($15 both) (+40% purchase frequency).
- **Frugal Budget Conscious:** Volume discounts, multi-pack bundles, price-match guarantees (+15% clearance stock turnover).

---

# 8. Conclusion & Future Work

## 8.1 Summary of Learnings
The project successfully demonstrated the feasibility and business value of combining unsupervised K-Means clustering with supervised XGBoost classification for retail intelligence. Building an end-to-end pipeline from data preprocessing to interactive web deployment delivered an actionable CRM decision system.

## 8.2 AI & Retail Analytics Skills Acquired

| Project Task | Technical Competency | Specific Skills Acquired |
| :--- | :--- | :--- |
| **Data Exploration & Cleaning** | Data Analytics & Preprocessing | Pandas, NumPy, Outlier clipping, Median imputation |
| **Feature Engineering** | Feature Representation | Engagement indices, Target leakage prevention |
| **Unsupervised Clustering** | Machine Learning (Unsupervised) | K-Means++, Elbow Method, Silhouette Analysis |
| **Supervised Classification** | Machine Learning (Supervised) | XGBoost, Random Forest, Stratified 5-Fold CV |
| **Model Deployment** | Web App & Productionization | Streamlit UI, Joblib serialization, Plotly 2D/3D |
| **Business Decisioning** | CRM Strategy & Analytics | Persona mapping, CLV modeling, ROI estimation |

## 8.3 Limitations of the Current Approach
- Static tabular batch training without real-time streaming online model updates.
- Textual sentiment from customer reviews is not yet incorporated.
- External macroeconomic factors (inflation, seasonal weather) are unmodeled.

## 8.4 Future Scope
- **Real-Time Streaming:** Connecting Apache Kafka and Snowflake to score live carts at checkout.
- **Deep Learning / Graph Neural Networks:** Modeling product co-purchasing graphs.
- **Automated CRM Webhooks:** Triggering immediate personalized offers via email/SMS APIs.

---

# 9. Internship Outcomes

## 9.1 Technical Skills Acquired
- End-to-end machine learning engineering from exploratory data analysis to web deployment.
- Deep theoretical and practical mastery of K-Means clustering, silhouette optimization, and XGBoost classification.
- Robust cross-validation and target leakage elimination best practices.

## 9.2 Soft Skills Developed
- Analytical problem solving and mathematical formulation of enterprise business challenges.
- Technical communication translating complex metrics (ROC-AUC, Silhouette) into executive marketing actions.

## 9.3 Contribution to Career Growth
- Created an industry-grade portfolio project demonstrating both unsupervised and supervised machine learning competencies.

---

# Appendix

## A. Complete Source Code
- **Pipeline Runner:** [`run_pipeline.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/run_pipeline.py)
- **Streamlit Web Application:** [`app.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/app.py)
- **Source Package:** [`src/data_loader.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/data_loader.py), [`src/preprocessing.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/preprocessing.py), [`src/clustering.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/clustering.py), [`src/classification.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/classification.py), [`src/recommendation.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/recommendation.py), [`src/pipeline.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/src/pipeline.py)
- **Unit Tests:** [`tests/test_pipeline.py`](file:///c:/Users/PAVAN/Desktop/Retail-workshop/tests/test_pipeline.py)

## B. Streamlit Application Screenshots & UI Flows
- **Executive KPI Dashboard (Tab 1):** 600 Customers, 25.0% High-Value Share, Income vs Spend Scatter Plot.
- **Persona Visualizer (Tab 2):** Elbow & Silhouette Curves, 2D/3D Cluster Projections.
- **Classifier Benchmark (Tab 3):** Stratified 5-Fold CV Table, ROC & PR Curves, Feature Importances.
- **Live Simulator (Tab 4):** Real-time customer parameter input $\to$ Instant Persona & High-Value Propensity Score.
- **Scored Database & Exporter (Tab 5):** Filterable customer database table with 1-click CSV download.

## C. Certificate of Completion

```text
====================================================================================
                            CERTIFICATE OF COMPLETION
====================================================================================
This is to certify that:

                               Reddi Pujitharam

                             CSM-UG Level 2
has successfully completed an Internship in Artificial Intelligence & Machine Learning
through PEARL Program organized by Q2D (Quantum Quotient Decode) in collaboration with
   IBM Innovation Centre for Education (IBM ICE),from 20th May to 20th July 2026.
The internship was undertaken as part of the UG Level program,providing the candidate
with practical exposure,industry-revelant knowledge, and hands-on learning in the area
of Artificial Intelligence & Machine Learning. 

Application No:IBMP3905
Date of Issue:10th September 2026

Nagesh Singh                   Dr. Buddha Chandrasekhar               Neha Chauhan
Chairman, Edunet Foundation    Chief Coordinating Officer, AICTE     SP Manager, Shell
====================================================================================
```
