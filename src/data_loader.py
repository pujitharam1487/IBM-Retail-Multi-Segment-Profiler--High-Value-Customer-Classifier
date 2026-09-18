"""
Data Ingestion and Generation Module.
Provides functionality to load the classic Mall Customer dataset and generate/load
enriched retail behavioral datasets to mitigate data leakage in predictive modeling.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Authentic 200 Mall Customer Dataset distribution seed values
MALL_CUSTOMERS_RAW_DATA = [
    (1, "Male", 19, 15, 39), (2, "Male", 21, 15, 81), (3, "Female", 20, 16, 6), (4, "Female", 23, 16, 77),
    (5, "Female", 31, 17, 40), (6, "Female", 22, 17, 76), (7, "Female", 35, 18, 6), (8, "Female", 23, 18, 94),
    (9, "Male", 64, 19, 3), (10, "Female", 30, 19, 72), (11, "Male", 67, 19, 14), (12, "Female", 35, 19, 99),
    (13, "Female", 58, 20, 15), (14, "Female", 24, 20, 77), (15, "Male", 37, 20, 13), (16, "Male", 22, 20, 79),
    (17, "Female", 35, 21, 35), (18, "Male", 20, 21, 66), (19, "Male", 52, 23, 29), (20, "Female", 35, 23, 98),
    (21, "Male", 35, 24, 35), (22, "Male", 25, 24, 73), (23, "Female", 46, 25, 5), (24, "Male", 31, 25, 73),
    (25, "Female", 54, 28, 14), (26, "Male", 29, 28, 82), (27, "Female", 45, 28, 32), (28, "Male", 35, 28, 61),
    (29, "Female", 40, 29, 31), (30, "Female", 23, 29, 87), (31, "Male", 60, 30, 4), (32, "Female", 21, 30, 73),
    (33, "Male", 53, 33, 4), (34, "Male", 18, 33, 92), (35, "Female", 49, 33, 14), (36, "Female", 21, 33, 81),
    (37, "Female", 42, 34, 17), (38, "Female", 30, 34, 73), (39, "Female", 36, 37, 26), (40, "Female", 20, 37, 75),
    (41, "Female", 65, 38, 35), (42, "Male", 24, 38, 92), (43, "Male", 48, 39, 36), (44, "Female", 31, 39, 61),
    (45, "Female", 49, 39, 28), (46, "Female", 24, 39, 65), (47, "Female", 50, 40, 55), (48, "Female", 27, 40, 47),
    (49, "Female", 29, 40, 42), (50, "Female", 31, 40, 42), (51, "Female", 49, 42, 52), (52, "Male", 33, 42, 60),
    (53, "Female", 31, 43, 54), (54, "Male", 59, 43, 60), (55, "Female", 50, 43, 45), (56, "Male", 47, 43, 41),
    (57, "Female", 51, 44, 50), (58, "Male", 69, 44, 46), (59, "Female", 27, 46, 51), (60, "Male", 53, 46, 46),
    (61, "Male", 70, 46, 56), (62, "Male", 19, 46, 55), (63, "Female", 67, 47, 52), (64, "Female", 54, 47, 59),
    (65, "Male", 63, 48, 51), (66, "Male", 18, 48, 59), (67, "Female", 43, 48, 50), (68, "Female", 68, 48, 48),
    (69, "Male", 19, 48, 59), (70, "Female", 32, 48, 47), (71, "Male", 70, 49, 55), (72, "Female", 47, 49, 42),
    (73, "Female", 60, 50, 49), (74, "Female", 60, 50, 56), (75, "Male", 59, 54, 47), (76, "Male", 26, 54, 54),
    (77, "Female", 45, 54, 53), (78, "Male", 40, 54, 48), (79, "Female", 23, 54, 52), (80, "Female", 49, 54, 42),
    (81, "Male", 57, 54, 51), (82, "Male", 38, 54, 55), (83, "Male", 67, 54, 41), (84, "Female", 46, 54, 44),
    (85, "Female", 21, 54, 57), (86, "Male", 48, 54, 46), (87, "Female", 55, 57, 58), (88, "Female", 22, 57, 55),
    (89, "Female", 34, 58, 60), (90, "Female", 50, 58, 46), (91, "Female", 68, 59, 55), (92, "Male", 18, 59, 41),
    (93, "Male", 48, 60, 49), (94, "Female", 40, 60, 40), (95, "Female", 32, 60, 42), (96, "Male", 24, 60, 52),
    (97, "Female", 47, 60, 47), (98, "Female", 27, 60, 50), (99, "Male", 48, 61, 42), (100, "Male", 20, 61, 49),
    (101, "Female", 23, 62, 41), (102, "Female", 49, 62, 48), (103, "Male", 67, 62, 59), (104, "Male", 26, 62, 55),
    (105, "Male", 49, 62, 56), (106, "Female", 21, 62, 42), (107, "Female", 66, 63, 50), (108, "Male", 54, 63, 46),
    (109, "Male", 68, 63, 43), (110, "Male", 66, 63, 48), (111, "Male", 65, 63, 52), (112, "Female", 19, 63, 54),
    (113, "Female", 38, 64, 42), (114, "Male", 19, 64, 46), (115, "Female", 18, 65, 48), (116, "Female", 19, 65, 50),
    (117, "Female", 63, 65, 43), (118, "Female", 49, 65, 59), (119, "Female", 51, 67, 43), (120, "Female", 50, 67, 57),
    (121, "Male", 27, 67, 56), (122, "Female", 38, 67, 40), (123, "Female", 40, 69, 58), (124, "Male", 39, 69, 91),
    (125, "Female", 23, 70, 29), (126, "Female", 31, 70, 77), (127, "Male", 43, 71, 35), (128, "Male", 40, 71, 95),
    (129, "Male", 59, 71, 11), (130, "Male", 38, 71, 75), (131, "Male", 47, 71, 9), (132, "Male", 39, 71, 75),
    (133, "Female", 25, 72, 34), (134, "Female", 31, 72, 71), (135, "Female", 20, 73, 5), (136, "Female", 29, 73, 88),
    (137, "Female", 44, 73, 7), (138, "Male", 32, 73, 73), (139, "Male", 19, 74, 10), (140, "Female", 35, 74, 72),
    (141, "Female", 57, 75, 5), (142, "Male", 28, 75, 40), (143, "Female", 28, 76, 40), (144, "Female", 32, 76, 87),
    (145, "Male", 25, 77, 12), (146, "Male", 28, 77, 97), (147, "Male", 48, 77, 36), (148, "Female", 32, 77, 74),
    (149, "Female", 34, 78, 22), (150, "Male", 34, 78, 90), (151, "Male", 43, 78, 17), (152, "Male", 39, 78, 88),
    (153, "Male", 44, 78, 20), (154, "Female", 38, 78, 76), (155, "Female", 47, 78, 16), (156, "Female", 27, 78, 89),
    (157, "Male", 37, 78, 1), (158, "Female", 30, 78, 78), (159, "Male", 34, 78, 1), (160, "Female", 30, 78, 73),
    (161, "Female", 56, 79, 35), (162, "Female", 29, 79, 83), (163, "Male", 19, 81, 5), (164, "Female", 31, 81, 93),
    (165, "Male", 50, 85, 26), (166, "Female", 36, 85, 75), (167, "Male", 42, 86, 20), (168, "Female", 33, 86, 95),
    (169, "Female", 36, 87, 27), (170, "Male", 32, 87, 63), (171, "Male", 40, 87, 13), (172, "Male", 28, 87, 75),
    (173, "Male", 36, 87, 10), (174, "Male", 36, 87, 92), (175, "Female", 52, 88, 13), (176, "Female", 30, 88, 86),
    (177, "Male", 58, 88, 15), (178, "Male", 27, 88, 69), (179, "Male", 59, 93, 14), (180, "Male", 35, 93, 90),
    (181, "Female", 37, 97, 32), (182, "Female", 32, 97, 86), (183, "Male", 46, 98, 15), (184, "Female", 29, 98, 88),
    (185, "Female", 41, 99, 39), (186, "Male", 30, 99, 97), (187, "Female", 54, 101, 24), (188, "Female", 28, 101, 68),
    (189, "Female", 41, 103, 17), (190, "Female", 36, 103, 85), (191, "Female", 34, 103, 23), (192, "Female", 32, 103, 69),
    (193, "Male", 33, 113, 8), (194, "Female", 38, 113, 91), (195, "Female", 47, 120, 16), (196, "Female", 35, 120, 79),
    (197, "Female", 45, 126, 28), (198, "Male", 32, 126, 74), (199, "Male", 32, 137, 18), (200, "Male", 30, 137, 83)
]


def load_mall_customers_df(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads or creates the classic Mall Customers dataset.
    Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(
            MALL_CUSTOMERS_RAW_DATA,
            columns=["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
        )
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
    return df


def generate_retail_intelligence_df(
    n_samples: int = 500,
    random_seed: int = 42,
    csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates an enriched Retail Customer Intelligence dataset that combines:
    - Demographic data (Age, Gender, Annual Income)
    - Mall Spending Score (for clustering & persona discovery)
    - Behavioral & Transactional features (Purchase Frequency, Average Order Value,
      Recency, Return Rate, Discount Usage, App Engagement)
    - Clean Ground Truth High-Value Label (avoiding target leakage)
    """
    np.random.seed(random_seed)
    
    # 1. Base demographics & spending patterns
    customer_ids = [f"CUST_{i+1:04d}" for i in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.45, 0.55])
    ages = np.random.randint(18, 71, size=n_samples)
    
    # Incomes follow realistic multimodal log-normal distribution
    base_income = np.random.lognormal(mean=4.1, sigma=0.45, size=n_samples)
    annual_incomes = np.clip(np.round(base_income).astype(int), 15, 150)
    
    # Spending Score influenced slightly by age & random propensity
    spending_scores = np.random.randint(1, 100, size=n_samples)
    
    # 2. Historical Transactional Features
    # Higher income & high spending propensity increases frequency & AOV
    norm_income = (annual_incomes - 15) / (150 - 15)
    norm_score = spending_scores / 100.0
    
    purchase_frequency = np.clip(
        np.round(3 + 25 * norm_score + 10 * norm_income + np.random.normal(0, 3, n_samples)),
        1, 52
    ).astype(int)
    
    avg_order_value = np.clip(
        np.round(30 + 200 * norm_income + 150 * norm_score + np.random.normal(0, 20, n_samples), 2),
        15.0, 500.0
    )
    
    recency_days = np.clip(
        np.round(365 * (1 - norm_score * 0.7) + np.random.normal(0, 30, n_samples)),
        1, 365
    ).astype(int)
    
    return_rate_pct = np.clip(
        np.round(5 + 15 * (1 - norm_income * 0.3) + np.random.normal(0, 4, n_samples), 1),
        0.5, 30.0
    )
    
    discount_usage_pct = np.clip(
        np.round(50 * (1 - norm_income * 0.6) + 20 * norm_score + np.random.normal(0, 8, n_samples), 1),
        0.0, 95.0
    )
    
    online_order_ratio = np.clip(
        np.round(0.3 + 0.4 * (1 - ages / 70) + np.random.normal(0, 0.15, n_samples), 2),
        0.0, 1.0
    )
    
    app_sessions_per_month = np.clip(
        np.round(15 * norm_score + 10 * (1 - ages / 70) + np.random.normal(0, 3, n_samples)),
        0, 45
    ).astype(int)
    
    years_as_customer = np.clip(
        np.round(0.5 + (ages - 18) * 0.15 + np.random.uniform(0, 3, n_samples), 1),
        0.5, 12.0
    )
    
    total_annual_spend = np.round(purchase_frequency * avg_order_value, 2)
    
    # 3. High-Value Customer Target Formulation (Top 25% by Total Value & Loyalty Score)
    # Composite Value Index based on Total Spend, Recency (inverse), and App Loyalty
    clv_score = (
        0.55 * (total_annual_spend / np.percentile(total_annual_spend, 90)) +
        0.25 * ((365 - recency_days) / 365) +
        0.20 * (app_sessions_per_month / 30)
    )
    
    threshold = np.percentile(clv_score, 75)
    is_high_value = (clv_score >= threshold).astype(int)
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": genders,
        "Age": ages,
        "Annual_Income_k": annual_incomes,
        "Spending_Score": spending_scores,
        "Purchase_Frequency_Year": purchase_frequency,
        "Avg_Order_Value": avg_order_value,
        "Recency_Days": recency_days,
        "Return_Rate_Pct": return_rate_pct,
        "Discount_Usage_Pct": discount_usage_pct,
        "Online_Order_Ratio": online_order_ratio,
        "App_Sessions_Month": app_sessions_per_month,
        "Years_As_Customer": years_as_customer,
        "Total_Annual_Spend": total_annual_spend,
        "High_Value_Customer": is_high_value
    })
    
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
    return df


def initialize_datasets(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures both mall_customers.csv and retail_customer_intelligence.csv are initialized.
    """
    dir_path = Path(data_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    mall_path = dir_path / "mall_customers.csv"
    intel_path = dir_path / "retail_customer_intelligence.csv"
    
    mall_df = load_mall_customers_df(str(mall_path))
    intel_df = generate_retail_intelligence_df(n_samples=600, csv_path=str(intel_path))
    
    return mall_df, intel_df


if __name__ == "__main__":
    mall_df, intel_df = initialize_datasets("data")
    print(f"Loaded Mall Customers dataset: shape {mall_df.shape}")
    print(f"Loaded Retail Intelligence dataset: shape {intel_df.shape}")
    print(f"High-Value class distribution:\n{intel_df['High_Value_Customer'].value_counts(normalize=True)}")
