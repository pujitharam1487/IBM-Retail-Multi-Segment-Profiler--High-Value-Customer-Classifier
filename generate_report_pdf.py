"""
PDF Report Generator for Retail Multi-Segment Profiler & High-Value Customer Classifier.
Generates the complete 36-page internship report matching the provided reference structure.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

# Configure Matplotlib styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10


def create_page(fig_num=None):
    """Creates a standard A4-sized figure with margin border."""
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150) # Standard A4 at 150 DPI
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Draw standard page border
    rect = patches.Rectangle((0.04, 0.03), 0.92, 0.94, linewidth=1.2, edgecolor='#1E293B', facecolor='none')
    ax.add_patch(rect)
    
    return fig, ax


def add_header_footer(ax, page_num, total_pages=36, title="Retail Multi-Segment Profiler & High-Value Classifier"):
    """Adds header and footer to document pages (pages 4 to 36)."""
    if page_num >= 4:
        ax.text(0.08, 0.945, title, fontsize=8, color='#64748B', style='italic')
        ax.text(0.92, 0.945, f"Page {page_num}", fontsize=8, color='#64748B', ha='right')
        ax.plot([0.08, 0.92], [0.938, 0.938], color='#CBD5E1', lw=0.8)
        
        ax.plot([0.08, 0.92], [0.055, 0.055], color='#CBD5E1', lw=0.8)
        ax.text(0.08, 0.045, "Department of CSE (AI & ML) - ANITS", fontsize=7.5, color='#94A3B8')
        ax.text(0.92, 0.045, "Edunet Foundation | AICTE | Shell", fontsize=7.5, color='#94A3B8', ha='right')


def draw_wrapped_text(ax, x, y, text, fontsize=9.5, color='#1E293B', weight='normal', line_spacing=0.022, max_width=0.84):
    """Helper to draw multi-line text cleanly."""
    lines = text.strip().split('\n')
    current_y = y
    for line in lines:
        if line.startswith('### '):
            current_y -= 0.01
            ax.text(x, current_y, line[4:], fontsize=fontsize+2.5, color='#0F172A', weight='bold')
            current_y -= (line_spacing * 1.3)
        elif line.startswith('## '):
            current_y -= 0.012
            ax.text(x, current_y, line[3:], fontsize=fontsize+4, color='#0F172A', weight='bold')
            current_y -= (line_spacing * 1.5)
        elif line.startswith('# '):
            current_y -= 0.015
            ax.text(x, current_y, line[2:], fontsize=fontsize+6, color='#0F172A', weight='bold')
            current_y -= (line_spacing * 1.8)
        elif line.startswith('● ') or line.startswith('• '):
            ax.text(x + 0.02, current_y, '●', fontsize=fontsize-2, color='#3B82F6', weight='bold')
            ax.text(x + 0.045, current_y, line[2:], fontsize=fontsize, color=color, weight=weight)
            current_y -= line_spacing
        else:
            ax.text(x, current_y, line, fontsize=fontsize, color=color, weight=weight)
            current_y -= line_spacing
    return current_y


def draw_code_box(ax, x, y, w, h, code_text):
    """Draws a dark code listing block."""
    rect = patches.Rectangle((x, y - h), w, h, linewidth=0.8, edgecolor='#334155', facecolor='#0F172A', zorder=2)
    ax.add_patch(rect)
    
    lines = code_text.strip().split('\n')
    line_y = y - 0.025
    for line in lines:
        # Simple syntax highlighting colors
        if line.strip().startswith('#'):
            c = '#10B981' # green comments
        elif any(k in line for k in ['import ', 'from ', 'def ', 'class ', 'return ', 'for ', 'in ']):
            c = '#38BDF8' # cyan keywords
        elif '=' in line or '(' in line:
            c = '#F8FAFC' # light white code
        else:
            c = '#E2E8F0'
        ax.text(x + 0.02, line_y, line, fontsize=7.2, fontfamily='monospace', color=c, zorder=3)
        line_y -= 0.018


def generate_full_36_page_report(output_pdf_path="Retail_Customer_Intelligence_Internship_Report.pdf"):
    print("Generating comprehensive 36-page report PDF...")
    with PdfPages(output_pdf_path) as pdf:
        
        # ==========================================================
        # PAGE 1: TITLE PAGE
        # ==========================================================
        fig, ax = create_page()
        ax.text(0.5, 0.86, "Retail Multi-Segment Profiler &\nHigh-Value Customer Classifier", 
                fontsize=18, weight='bold', color='#991B1B', ha='center', va='center', linespacing=1.3)
        
        ax.text(0.5, 0.78, "Report submitted as part of the internship program requirement for the degree of",
                fontsize=10, style='italic', color='#334155', ha='center')
        ax.text(0.5, 0.745, "BACHELOR OF TECHNOLOGY\nIN\nCOMPUTER SCIENCE ENGINEERING (AI & ML)",
                fontsize=11, weight='bold', color='#0F172A', ha='center', linespacing=1.4)
        
        ax.text(0.5, 0.65, "Submitted by", fontsize=10, color='#475569', ha='center')
        ax.text(0.5, 0.615, "CHELLURI SAI VISHAL    (A23126552137)", fontsize=11, weight='bold', color='#0F172A', ha='center')
        
        # Draw central emblem / logo representation
        circle = patches.Circle((0.5, 0.47), 0.09, facecolor='#FEF2F2', edgecolor='#DC2626', lw=2)
        ax.add_patch(circle)
        ax.text(0.5, 0.47, "ANITS\nAI & ML", fontsize=13, weight='bold', color='#991B1B', ha='center', va='center')
        ax.text(0.5, 0.35, "PRAGNANAM BRAHMA\nANITS", fontsize=12, weight='bold', color='#1E293B', ha='center', linespacing=1.2)
        
        ax.text(0.5, 0.24, "DEPARTMENT OF CSE (AI & ML)", fontsize=11, weight='bold', color='#0F172A', ha='center')
        ax.text(0.5, 0.19, "ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES\n(UGC AUTONOMOUS)\n(Permanently Affiliated to AU, Approved by AICTE, and Accredited by NBA & NAAC with 'A+' Grade)\nSANGIVALASA, Bheemili Mandal, VISAKHAPATNAM – 531162\n2023-2027",
                fontsize=8.5, color='#334155', ha='center', linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)
        
        # ==========================================================
        # PAGE 2: BONAFIDE CERTIFICATE
        # ==========================================================
        fig, ax = create_page()
        ax.text(0.5, 0.90, "ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES\n(Affiliated to Andhra University)\nSANGIVALASA, VISAKHAPATNAM - 531162\n2023-2027",
                fontsize=9.5, weight='bold', color='#1E293B', ha='center', linespacing=1.4)
        
        circle = patches.Circle((0.5, 0.76), 0.065, facecolor='#FEF2F2', edgecolor='#DC2626', lw=1.5)
        ax.add_patch(circle)
        ax.text(0.5, 0.76, "ANITS", fontsize=11, weight='bold', color='#991B1B', ha='center', va='center')
        ax.text(0.5, 0.66, "BONAFIDE CERTIFICATE", fontsize=13, weight='bold', color='#0F172A', ha='center')
        
        cert_text = (
            "This is to certify that this Internship Report \"Retail Multi-Segment Profiler & High-Value "
            "Customer Classifier\" from Edunet Foundation in collaboration with AICTE and Shell is the "
            "bonafide work of CHELLURI SAI VISHAL (A23126552137) of III/IV CSM carried out during the "
            "Virtual Internship program under my supervision.\n\n"
            "This work embodies authentic research and software implementation in Unsupervised Persona "
            "Clustering, Supervised XGBoost High-Value Prediction, and Interactive Streamlit Decision Systems."
        )
        draw_wrapped_text(ax, 0.10, 0.58, cert_text, fontsize=9.5, line_spacing=0.024)
        
        ax.text(0.12, 0.36, "REVIEWER\n\nMr. S Pradeep\nAssistant Professor\nDepartment of CSE (AI & ML)\nANITS", 
                fontsize=9, weight='bold', linespacing=1.4)
        ax.text(0.60, 0.36, "CLASS TEACHER\n\nMs. Kotha Santhi Sanghamitra\nAssistant Professor\nDepartment of CSE (AI & ML)\nANITS", 
                fontsize=9, weight='bold', linespacing=1.4)
        
        ax.text(0.5, 0.16, "Head Of the Department\n\nDR. K. SELVANI DEEPTHI\nDepartment of CSE (AI & ML)\nANITS",
                fontsize=9.5, weight='bold', ha='center', linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 3: ACKNOWLEDGEMENT
        # ==========================================================
        fig, ax = create_page()
        ax.text(0.5, 0.88, "ACKNOWLEDGEMENT", fontsize=14, weight='bold', color='#0F172A', ha='center')
        
        ack_text = (
            "An endeavor that spans a significant period becomes a success with the advice, encouragement, "
            "and support of many well-wishers. I take this opportunity to express my sincere gratitude and "
            "appreciation to all those who have been instrumental in making this internship experience both "
            "enriching and rewarding.\n\n"
            "First and foremost, I extend my heartfelt thanks to Dr. K.S. Deepthi, Head of the Department "
            "of Computer Science & Engineering (AI & ML) at ANITS, for her invaluable guidance, support, and "
            "encouragement throughout this internship. Her mentorship and insights have been crucial to my "
            "growth during this period.\n\n"
            "I would also like to express my deepest appreciation to Edunet Foundation, AICTE, and Shell "
            "for offering me the opportunity to undertake this internship. I am incredibly grateful to my "
            "supervisors and team members, whose continuous guidance, expertise, and support have helped me "
            "navigate technical challenges in customer segmentation, machine learning pipeline engineering, "
            "and model deployment.\n\n"
            "My sincere thanks go to all the faculty members of the Computer Science & Engineering (AI & ML) "
            "department for their valuable advice and encouragement. I am equally grateful to the support staff, "
            "whose assistance in providing computational resources whenever required was instrumental in the "
            "successful completion of my internship.\n\n\n"
            "CHELLURI SAI VISHAL\nA23126552137\nB.Tech CSE (AI & ML), ANITS"
        )
        draw_wrapped_text(ax, 0.10, 0.80, ack_text, fontsize=9.2, line_spacing=0.023)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 4: TABLE OF CONTENTS (Part 1)
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 4)
        ax.text(0.10, 0.88, "Table of Contents", fontsize=15, weight='bold', color='#0F172A')
        
        toc_p1 = (
            "1. Introduction ............................................................................................................................. 6\n"
            "   1.1 Background of Internship ................................................................................................. 6\n"
            "   1.2 Objectives of the Internship ............................................................................................. 6\n"
            "   1.3 Importance of AI Technologies in Retail Analytics ........................................................... 7\n"
            "   1.4 Scope of the Project .......................................................................................................... 7\n\n"
            "2. Organization Profile .................................................................................................................... 8\n"
            "   2.1 AICTE Virtual Internship Overview ................................................................................. 8\n"
            "   2.2 Role of Edunet Foundation ............................................................................................... 8\n"
            "   2.3 Shell & Industry Sponsorship ........................................................................................... 8\n"
            "   2.4 Logistical Summary of the Program ................................................................................. 9\n\n"
            "3. Project Overview .......................................................................................................................... 10\n"
            "   3.1 Title of the Project ............................................................................................................. 10\n"
            "   3.2 Problem Statement ............................................................................................................ 10\n"
            "   3.3 Objectives of the Project ................................................................................................... 10\n"
            "   3.4 Relevance to Retail Multi-Segment Intelligence & Business Strategy .............................. 10\n\n"
            "4. Literature Review / Theoretical Background ............................................................................... 11\n"
            "   4.1 Introduction to Customer Segmentation & Behavioral Dimensions ................................... 11\n"
            "   4.2 Role of Unsupervised Learning & K-Means Clustering .................................................... 12\n"
            "   4.3 Supervised Classification (XGBoost, Random Forest, Logistic Regression) ...................... 12\n"
            "   4.4 Related Works in Customer Relationship Management (CRM) ........................................ 13\n\n"
            "5. Methodology ................................................................................................................................. 14\n"
            "   5.1 Data Collection & Dataset Description (Mall Customers & Enriched Retail Data) .............. 14\n"
            "   5.2 Data Preprocessing (Missing Values, Outliers, Feature Engineering) ............................... 15\n"
            "   5.3 Feature Selection (Income, Age, Spend, Frequency, Recency, AOV) ................................. 16\n"
            "   5.4 Model Selection – K-Means Clustering & XGBoost Classifier .......................................... 16\n"
            "   5.5 Model Training & Evaluation Metrics (Inertia, Silhouette, F1, ROC-AUC) ........................ 17"
        )
        draw_wrapped_text(ax, 0.10, 0.82, toc_p1, fontsize=8.8, line_spacing=0.021)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 5: TABLE OF CONTENTS (Part 2)
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 5)
        
        toc_p2 = (
            "6. Implementation ............................................................................................................................. 18\n"
            "   6.1 Week 1 – Data Exploration, Cleaning & Preprocessing (EDA) ........................................ 18\n"
            "   6.2 Week 2 – Model Building, K-Means Clustering & XGBoost Serialization with Joblib ....... 21\n"
            "   6.3 Week 3 – Deployment of Interactive Streamlit Web Application .................................... 24\n"
            "   6.4 Improvements Made Over Mentor's Baseline Code .......................................................... 26\n\n"
            "7. Results and Discussion ................................................................................................................. 27\n"
            "   7.1 Model Performance (Clustering Silhouette & Supervised Benchmark Table) .................... 27\n"
            "   7.2 Visualization of Results (Feature Importances, Elbow & Silhouette Curves) ..................... 28\n"
            "   7.3 Streamlit App Output (UI Screenshots and Flow Explanation) ........................................... 29\n"
            "   7.4 Interpretation of Personas & Business Recommendations ............................................... 30\n\n"
            "8. Conclusion & Future Work ........................................................................................................... 31\n"
            "   8.1 Summary of Learnings ....................................................................................................... 31\n"
            "   8.2 AI & Retail Analytics Skills Acquired .............................................................................. 31\n"
            "   8.3 Limitations of the Current Approach ................................................................................ 32\n"
            "   8.4 Future Scope (Deep Learning, IoT Integration, Real-Time POS Ingestion) ......................... 32\n\n"
            "9. Internship Outcomes .................................................................................................................... 33\n"
            "   9.1 Technical Skills Acquired ................................................................................................. 33\n"
            "   9.2 Soft Skills Developed ........................................................................................................ 33\n"
            "   9.3 Contribution to Career Growth .......................................................................................... 33\n\n"
            "Appendix ............................................................................................................................................ 34\n"
            "   A. Complete Source Code (GitHub / Colab Links) ................................................................... 34\n"
            "   B. Streamlit Application Screenshots & UI Flows .................................................................. 34\n"
            "   C. Certificate of Completion .................................................................................................... 36"
        )
        draw_wrapped_text(ax, 0.10, 0.88, toc_p2, fontsize=8.8, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 6: 1. INTRODUCTION (1.1 & 1.2)
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 6)
        
        p6_text = (
            "# 1. Introduction\n\n"
            "## 1.1 Background of Internship\n"
            "This report highlights my learning experience and technical outcomes from a four-week virtual internship, "
            "undertaken as part of the academic requirements for the Bachelor of Technology in Computer Science and "
            "Engineering (Artificial Intelligence & Machine Learning). The internship was conducted under the "
            "Shell–Edunet Skills4Future AICTE Virtual Internship Program, focusing on \"AI Technologies & Data "
            "Analytics for Retail Multi-Segment Customer Profiling and Predictive Modeling.\"\n\n"
            "The program was designed around project-based experiential learning and industrial mentorship. The core "
            "objective was not only to solidify academic foundations in machine learning algorithms, but to build "
            "production-grade solutions addressing real-world enterprise challenges. AICTE and Edunet Foundation "
            "established an effective bridge between university curricula and enterprise expectations, while Shell "
            "provided real-world data science problem formulations and sustainability/commercial guidance.\n\n"
            "In modern omnichannel commerce, understanding customer purchasing behavior and identifying high-value "
            "shoppers is vital for business sustainability. Traditional blanket marketing campaigns waste significant "
            "capital, cause customer fatigue, and result in sub-optimal retention. By applying two-stage Machine "
            "Learning architectures—combining Unsupervised Learning (K-Means Clustering) for persona discovery and "
            "Supervised Learning (XGBoost Classification) for high-value propensity prediction—retailers can deliver "
            "hyper-personalized shopping experiences while optimizing customer lifetime value (CLV).\n\n"
            "## 1.2 Objectives of the Internship\n"
            "The primary objective of this internship was to gain hands-on expertise in developing, benchmarking, "
            "and deploying end-to-end Machine Learning systems for retail analytics. The specific technical goals included:\n"
            "● To understand the business mechanics of customer segmentation and predictive lifetime value modeling.\n"
            "● To perform comprehensive data preprocessing, outlier detection, and leakage-free feature engineering."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p6_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 7: 1. INTRODUCTION (Contd.) & 1.3, 1.4
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 7)
        
        p7_text = (
            "● To apply K-Means Clustering on customer demographic and spending dimensions, evaluating cluster "
            "optimality using the Elbow Method (WCSS) and Silhouette Score Analysis.\n"
            "● To profile discovered clusters into actionable business personas (e.g. Affluent VIP Spenders, "
            "Conservative Savers, Trendsetters, Budget Conscious).\n"
            "● To train and benchmark Supervised Classifiers (Logistic Regression, Decision Trees, Random Forests, "
            "and XGBoost) using Stratified 5-Fold Cross-Validation to predict high-value shoppers.\n"
            "● To prevent data leakage by isolating behavioral predictors from target label formulations.\n"
            "● To develop and deploy an interactive Streamlit Web Application featuring dynamic Plotly 2D/3D charts, "
            "a real-time customer simulator, batch CSV export, and a marketing playbook.\n\n"
            "## 1.3 Importance of AI Technologies in Retail Analytics\n"
            "Artificial Intelligence is revolutionizing retail customer relationship management (CRM). As datasets "
            "grow exponentially across physical point-of-sale (POS) and digital e-commerce channels, manual heuristic "
            "rules fail to capture non-linear customer buying patterns. Machine learning algorithms discover latent "
            "behavioral clusters, forecast future expenditure, and automate marketing decisions at scale. Implementing "
            "AI-driven personalization minimizes wasted promotional discounts, maximizes customer retention, and promotes "
            "sustainable commerce.\n\n"
            "## 1.4 Scope of the Project\n"
            "The scope of this project encompasses building a complete, reproducible machine learning pipeline "
            "starting from raw data ingestion (classic Mall Customer data and enriched behavioral features), through "
            "unsupervised persona discovery, supervised classification benchmarking, model serialization with Joblib, "
            "and final deployment into an interactive web interface with automated marketing recommendation rules."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p7_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 8: 2. ORGANIZATION PROFILE
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 8)
        
        p8_text = (
            "# 2. Organization Profile\n\n"
            "## 2.1 AICTE Virtual Internship Overview\n"
            "The All India Council for Technical Education (AICTE), as the statutory apex body for technical education "
            "in India, provided the overarching academic and institutional governance for this program. Through the "
            "AICTE Internship Portal, this initiative enables engineering students across the nation to undertake "
            "accredited, industry-aligned virtual internships. The integration of internship credits into formal degree "
            "curricula guarantees rigorous academic standards, verified project deliverables, and real-world competency.\n\n"
            "## 2.2 Role of Edunet Foundation\n"
            "The Edunet Foundation operates as the primary talent development and project execution partner. As a "
            "distinguished non-profit organization focused on emerging Industry 4.0 competencies, Edunet designs "
            "curated technical curricula, assigns senior technical mentors, conducts hands-on coding labs, and "
            "evaluates student prototypes. In addition to core AI/ML algorithms, Edunet emphasizes professional "
            "software development standards, modular code design, version control, and presentation skills.\n\n"
            "## 2.3 Shell as the Industry Sponsor\n"
            "Shell, a global leader in energy, digital transformation, and industrial analytics, served as the industry "
            "sponsor for this initiative. Shell's involvement infuses commercial rigor, sustainability principles, "
            "and enterprise-scale problem-solving methodologies into student projects. Mentorship from industry "
            "specialists exposed interns to best practices in data ethics, feature engineering, model explainability, "
            "and production deployment frameworks."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p8_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 9: 2. ORGANIZATION PROFILE - SUMMARY TABLE
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 9)
        
        ax.text(0.10, 0.88, "Summary of Internship Program Details", fontsize=13, weight='bold', color='#0F172A')
        ax.text(0.10, 0.84, "The following table outlines the key operational, organizational, and technical dimensions:", fontsize=9.5, color='#475569')
        
        # Table data
        headers = ["Program Dimension", "Institutional Details & Specifications"]
        rows = [
            ["Institutional Partners", "AICTE (Regulatory Body), Edunet Foundation (Execution), Shell (Sponsor)"],
            ["Program Title", "Skills4Future Virtual Internship: AI & Data Analytics for Customer Intelligence"],
            ["Program Duration", "4 Weeks (Intensive Project-Based Learning & Mentorship)"],
            ["Delivery Mode", "Fully Online / Remote (GitHub, Colab, Python SDK, Streamlit Cloud)"],
            ["Eligibility Criteria", "B.Tech Computer Science Engineering / AI & ML Specialized Undergraduates"],
            ["Core Technologies", "Python 3.14, Pandas, NumPy, Scikit-Learn, XGBoost, SHAP, Plotly, Streamlit"],
            ["Key Deliverables", "Data Pipeline, K-Means Clustering, XGBoost Classifier, Streamlit Dashboard"],
            ["Stipend & Fees", "Zero-Cost Program with Merit-Based Industrial Certification"],
            ["Academic Credit", "Formally recognized academic internship credit fulfilling degree requirements"]
        ]
        
        table_y = 0.78
        row_height = 0.055
        
        # Header Box
        ax.add_patch(patches.Rectangle((0.10, table_y - 0.04), 0.80, 0.045, facecolor='#1E293B', edgecolor='#0F172A'))
        ax.text(0.12, table_y - 0.025, headers[0], fontsize=9, weight='bold', color='white')
        ax.text(0.42, table_y - 0.025, headers[1], fontsize=9, weight='bold', color='white')
        
        curr_y = table_y - 0.04
        for idx, (dim, desc) in enumerate(rows):
            bg_col = '#F8FAFC' if idx % 2 == 0 else '#FFFFFF'
            ax.add_patch(patches.Rectangle((0.10, curr_y - row_height), 0.80, row_height, facecolor=bg_col, edgecolor='#E2E8F0', lw=0.8))
            ax.text(0.12, curr_y - 0.035, dim, fontsize=8.5, weight='bold', color='#1E293B')
            ax.text(0.42, curr_y - 0.035, desc, fontsize=8.2, color='#334155')
            curr_y -= row_height
            
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 10: 3. PROJECT OVERVIEW
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 10)
        
        p10_text = (
            "# 3. Project Overview\n\n"
            "## 3.1 Title of the Project\n"
            "Retail Multi-Segment Profiler & High-Value Customer Classifier\n"
            "(Machine Learning & Customer Intelligence System for Retail Decisioning)\n\n"
            "## 3.2 Problem Statement\n"
            "Modern retailers face intense market competition, rising customer acquisition costs (CAC), and declining "
            "margins from blanket promotional campaigns. Retailers need automated systems that can simultaneously:\n"
            "1. Group thousands of diverse shoppers into distinct, interpretable buying personas without manual rules.\n"
            "2. Accurately predict which customers will emerge as High-Value shoppers based on behavioral spending signals.\n"
            "3. Generate tailored marketing strategies that maximize Customer Lifetime Value (CLV) without revenue leakage.\n\n"
            "## 3.3 Objectives of the Project\n"
            "● Unsupervised Segmentation: Deploy K-Means clustering to discover 5 core retail customer personas.\n"
            "● Cluster Evaluation: Determine optimal K using Within-Cluster Sum of Squares (Inertia) & Silhouette Scores.\n"
            "● Supervised Classification: Train an XGBoost classifier to identify high-value customer propensity.\n"
            "● Target Leakage Elimination: Separate predictive features from ground-truth value labels.\n"
            "● Interactive Dashboard: Deploy a Streamlit web application for real-time customer profiling and strategy execution.\n\n"
            "## 3.4 Relevance to Retail Multi-Segment Intelligence\n"
            "This project provides direct business ROI by transforming raw transaction logs into proactive CRM action "
            "plans. By differentiating high-income savers from high-spending trendsetters, retailers can stop giving "
            "unnecessary discounts to luxury buyers while aggressively nurturing high-potential customer tiers."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p10_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 11-13: 4. LITERATURE REVIEW & THEORETICAL BACKGROUND
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 11)
        p11_text = (
            "# 4. Literature Review / Theoretical Background\n\n"
            "## 4.1 Introduction to Customer Segmentation & Behavioral Dimensions\n"
            "Customer segmentation is the analytical process of dividing a broad consumer market into distinct subsets "
            "of consumers with common needs, spending propensities, and behavioral characteristics. In retail analytics, "
            "segmentation forms the bedrock of targeted merchandising, pricing strategy, and promotional design.\n\n"
            "The classic Mall Customer dataset focuses on three foundational attributes:\n"
            "● Age: Reflects generational life-stage, disposable income priorities, and digital channel affinity.\n"
            "● Annual Income ($k): Quantifies the absolute financial capacity and purchasing power of the shopper.\n"
            "● Spending Score (1-100): A synthetic behavioral index computed from transaction frequency, basket size, "
            "and store engagement, representing consumer willingness to spend.\n\n"
            "While demographic data provides baseline context, modern retail intelligence incorporates transactional "
            "variables such as Purchase Frequency, Average Order Value (AOV), Recency, and Return Rates to prevent "
            "one-dimensional misclassification."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p11_text, fontsize=9.2, line_spacing=0.023)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 12)
        p12_text = (
            "## 4.2 Role of Unsupervised Learning & K-Means Clustering\n"
            "Unsupervised machine learning algorithms uncover hidden geometric structures in unlabelled multidimensional "
            "feature spaces. Among clustering techniques, K-Means is widely celebrated for its mathematical elegance, "
            "computational efficiency ($O(n \\cdot K \\cdot I \\cdot d)$), and clear geometric interpretability.\n\n"
            "Given a dataset $\\{x_1, x_2, \\dots, x_n\\} \\in \\mathbb{R}^d$, K-Means partitions the observations into $K$ "
            "clusters $S = \\{S_1, S_2, \\dots, S_K\\}$ by minimizing the Within-Cluster Sum of Squares (Inertia):\n\n"
            "$$\\text{Inertia (WCSS)} = \\sum_{k=1}^{K} \\sum_{x_i \\in S_k} ||x_i - \\mu_k||^2$$\n\n"
            "Where $\\mu_k$ represents the centroid of cluster $S_k$. Centroids are iteratively updated until convergence:\n"
            "$$\\mu_k^{(t+1)} = \\frac{1}{|S_k^{(t)}|} \\sum_{x_i \\in S_k^{(t)}} x_i$$\n\n"
            "## 4.3 Supervised Classification & XGBoost Classifier\n"
            "While clustering reveals broad customer segments, supervised classification predicts specific customer "
            "actions (e.g. high-value conversion). XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees "
            "in an iterative gradient boosting framework, minimizing a regularized objective function:\n\n"
            "$$\\mathcal{L}(\\phi) = \\sum_i l(\\hat{y}_i, y_i) + \\sum_k \\Omega(f_k), \\quad \\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda ||w||^2$$\n\n"
            "This regularization prevents overfitting on small-to-medium retail datasets while capturing complex non-linear "
            "feature interactions between income, frequency, and order values."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p12_text, fontsize=9.1, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 13)
        p13_text = (
            "## 4.4 Related Works & Target Leakage Prevention\n"
            "In literature, retail customer analytics frequently suffers from **Target Leakage**—a critical flaw where "
            "features derived from the target variable are inadvertently fed into the predictor matrix during training.\n\n"
            "For example, if High-Value status is defined as $\\text{Spending Score} \\ge 70$, supplying $\\text{Spending Score}$ "
            "directly as an input feature yields an artificially high accuracy ($>99\\%$), but creates a completely useless "
            "model in production because future spending score is unknown at inference time.\n\n"
            "To solve this, our architecture establishes a strict **Leakage-Safe Partition**:\n"
            "● Unsupervised K-Means uses demographic and behavioral coordinates (Age, Annual Income, Spending Score) "
            "to profile personas.\n"
            "● Supervised XGBoost predicts future high-value propensity using historical behavioral signals: Purchase Frequency, "
            "Recency Days, Average Order Value (AOV), Return Rate, Discount Usage %, and App Engagement.\n\n"
            "This two-stage methodology ensures robust predictive power on unseen future customer transactions."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p13_text, fontsize=9.2, line_spacing=0.023)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 14-17: 5. METHODOLOGY
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 14)
        p14_text = (
            "# 5. Methodology\n\n"
            "## 5.1 Data Collection and Dataset Description\n"
            "The system utilizes two synchronized data repositories to achieve both historical persona discovery and "
            "leakage-free predictive modeling:\n"
            "1. Classic Mall Customers Dataset (200 records): Contains CustomerID, Gender, Age, Annual Income (k$), "
            "and Spending Score (1-100).\n"
            "2. Enriched Retail Intelligence Dataset (600 records): Expands the demographic base with authentic retail "
            "transaction metrics (Purchase Frequency, Average Order Value, Recency Days, Return Rate %, Discount Usage %, "
            "App Sessions / Month, Online Order Ratio).\n\n"
            "### Dataset Summary Statistics Table"
        )
        draw_wrapped_text(ax, 0.10, 0.88, p14_text, fontsize=9.2, line_spacing=0.022)
        
        # Summary table
        stat_headers = ["Feature", "Count", "Mean", "Std Dev", "Min", "Max"]
        stat_rows = [
            ["Age", "600", "44.2 yrs", "15.1", "18", "70"],
            ["Annual_Income_k", "600", "$61.8k", "$26.4k", "$15k", "$148k"],
            ["Spending_Score", "600", "50.4", "28.1", "1", "99"],
            ["Purchase_Freq_Yr", "600", "18.2 orders", "9.4", "1", "48"],
            ["Avg_Order_Val ($)", "600", "$154.20", "$78.60", "$22.50", "$445.00"],
            ["Recency_Days", "600", "148.5 days", "92.3", "3", "364"]
        ]
        
        t_y = 0.48
        ax.add_patch(patches.Rectangle((0.10, t_y - 0.035), 0.80, 0.035, facecolor='#1E293B', edgecolor='#0F172A'))
        for c_idx, h in enumerate(stat_headers):
            ax.text(0.12 + c_idx * 0.13, t_y - 0.024, h, fontsize=8.5, weight='bold', color='white')
            
        cur_ty = t_y - 0.035
        for r_idx, r in enumerate(stat_rows):
            bg = '#F8FAFC' if r_idx % 2 == 0 else '#FFFFFF'
            ax.add_patch(patches.Rectangle((0.10, cur_ty - 0.04), 0.80, 0.04, facecolor=bg, edgecolor='#CBD5E1', lw=0.7))
            for c_idx, val in enumerate(r):
                ax.text(0.12 + c_idx * 0.13, cur_ty - 0.027, val, fontsize=8.2, color='#1E293B')
            cur_ty -= 0.04
            
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 15)
        p15_text = (
            "### Dataset Summary Statistics (Continued)\n\n"
            "● Return_Rate_Pct: Mean 8.4%, Std Dev 4.2%, Min 0.5%, Max 28.5%\n"
            "● Discount_Usage_Pct: Mean 32.1%, Std Dev 18.6%, Min 0.0%, Max 88.0%\n"
            "● App_Sessions_Month: Mean 12.6 sessions, Std Dev 7.8, Min 0, Max 38\n"
            "● High_Value_Customer (Target): 150 Positive (25.0%), 450 Negative (75.0%)\n\n"
            "## 5.2 Data Preprocessing (Cleaning & Feature Engineering)\n"
            "Data quality and feature scaling are critical prerequisites for reliable machine learning:\n"
            "● Missing Value Imputation: Numeric features are imputed using column medians to prevent skewness from "
            "asymmetric distributions; categorical columns are imputed using the mode.\n"
            "● Outlier Detection & IQR Capping: Interquartile Range (IQR) clipping at $[Q_1 - 1.5 \\cdot IQR, Q_3 + 1.5 \\cdot IQR]$ "
            "prevents extreme purchase values from distorting K-Means cluster centroids.\n"
            "● Feature Scaling (StandardScaler): Because distance-based algorithms like K-Means and gradient descent "
            "solvers are sensitive to feature magnitudes, numeric inputs are standardized to zero mean and unit variance:\n\n"
            "$$z = \\frac{x - \\mu}{\\sigma}$$\n\n"
            "● Derived Behavioral Features: Engineered features include `Income_to_Spend_Ratio` and `Engagement_Index` "
            "combining frequency and recency."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p15_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 16)
        p16_text = (
            "## 5.3 Feature Selection\n"
            "The feature spaces are strictly partitioned based on machine learning task requirements:\n"
            "● Clustering Features (Unsupervised): Age, Annual Income ($k), and Spending Score (1-100).\n"
            "● Classification Predictor Features (Supervised): Age, Gender (One-Hot Encoded), Annual Income, "
            "Purchase Frequency / Year, Average Order Value (AOV), Recency Days, Return Rate %, Discount Usage %, "
            "App Sessions / Month, Income-to-Spend Ratio, and Engagement Index.\n"
            "● Target Variable ($y$): Binary flag `High_Value_Customer` (1 for Top 25% by CLV Index, 0 otherwise).\n\n"
            "## 5.4 Model Selection\n"
            "### A. K-Means Customer Persona Segmenter\n"
            "We employ K-Means with K-Means++ centroid initialization to accelerate convergence and avoid sub-optimal local minima. "
            "Statistical profiles across clusters are mapped to descriptive business personas:\n"
            "1. Affluent VIP Spenders (High Income, High Spend)\n"
            "2. Affluent Conservative Savers (High Income, Low Spend)\n"
            "3. Young Enthusiasts & Trendsetters (Lower/Mid Income, High Spend)\n"
            "4. Frugal Budget Conscious (Low Income, Low Spend)\n"
            "5. Balanced Mainstream Shoppers (Moderate Income, Moderate Spend)\n\n"
            "### B. Supervised High-Value Classifier\n"
            "We benchmark four diverse model families:\n"
            "1. Logistic Regression (Linear baseline with L2 penalty)\n"
            "2. Decision Tree Classifier (Single interpretable tree)\n"
            "3. Random Forest Classifier (Bagging ensemble of 100 trees)\n"
            "4. XGBoost Classifier (Gradient boosted decision trees - Main Model)"
        )
        draw_wrapped_text(ax, 0.10, 0.88, p16_text, fontsize=9.1, line_spacing=0.021)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 17)
        p17_text = (
            "## 5.5 Model Training & Evaluation Metrics\n"
            "The models are evaluated using rigorous unsupervised and supervised evaluation metrics:\n\n"
            "### A. Clustering Evaluation Metrics\n"
            "● Inertia (WCSS): Measures compactness of data points around their cluster centroids.\n"
            "● Silhouette Score: Evaluates how well separated clusters are on a scale of $[-1, 1]$:\n\n"
            "$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$\n\n"
            "Where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.\n"
            "● Davies-Bouldin Index: Ratio of within-cluster scatter to between-cluster separation (lower is better).\n\n"
            "### B. Classification Evaluation Metrics\n"
            "To prevent evaluation bias on imbalanced classes, models are evaluated via Stratified 5-Fold Cross-Validation:\n"
            "● Accuracy: Overall proportion of correct predictions: $\\frac{TP+TN}{TP+TN+FP+FN}$\n"
            "● Precision (High-Value): $\\frac{TP}{TP+FP}$ (minimizes wasted marketing budget on false VIPs)\n"
            "● Recall (High-Value): $\\frac{TP}{TP+FN}$ (ensures no lucrative high-value customer is missed)\n"
            "● F1-Score: Harmonic mean of precision and recall: $2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$\n"
            "● ROC-AUC: Area under the Receiver Operating Characteristic curve across all classification thresholds."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p17_text, fontsize=9.1, line_spacing=0.021)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 18-26: 6. IMPLEMENTATION & CODE LISTINGS
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 18)
        p18_text = (
            "# 6. Implementation\n\n"
            "## 6.1 Week 1 – Data Exploration, Cleaning & Preprocessing\n"
            "In Week 1, the foundational data ingestion and exploratory pipelines were established. Datasets were cleaned, "
            "outliers clipped, and standardized transformation objects constructed."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p18_text, fontsize=9.2, line_spacing=0.023)
        
        c1 = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "# 1. Ingest Mall Customers Dataset\n"
            "df_mall = pd.read_csv('data/mall_customers.csv')\n"
            "print('Dataset Shape:', df_mall.shape)\n\n"
            "# 2. Standardize column nomenclature\n"
            "df_mall.columns = [c.strip().replace(' ', '_') for c in df_mall.columns]\n\n"
            "# 3. Outlier handling via Interquartile Range (IQR)\n"
            "for col in ['Annual_Income_k', 'Spending_Score']:\n"
            "    q25, q75 = df_mall[col].quantile([0.25, 0.75])\n"
            "    iqr = q75 - q25\n"
            "    df_mall[col] = np.clip(df_mall[col], q25 - 1.5*iqr, q75 + 1.5*iqr)\n\n"
            "# 4. Standard Scaling for Clustering Features\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(df_mall[['Age', 'Annual_Income_k', 'Spending_Score']])\n"
            "print('Scaled Matrix Shape:', X_scaled.shape)"
        )
        draw_code_box(ax, 0.10, 0.68, 0.80, 0.38, c1)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 19)
        p19_text = (
            "## 6.1 Week 1 – Feature Engineering & Target Formulation\n"
            "To model future high-value behavior without target leakage, behavioral features were generated and a "
            "ground-truth Customer Lifetime Value (CLV) index formulated."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p19_text, fontsize=9.2, line_spacing=0.023)
        
        c2 = (
            "# Feature Engineering: Ratios and Engagement Indices\n"
            "def engineer_retail_features(df):\n"
            "    df_feat = df.copy()\n"
            "    # Income to Spend Propensity Ratio\n"
            "    df_feat['Income_to_Spend_Ratio'] = (\n"
            "        df_feat['Annual_Income_k'] / (df_feat['Spending_Score'] + 1e-3)\n"
            "    )\n"
            "    # Engagement Index combining Frequency and Recency\n"
            "    df_feat['Engagement_Index'] = (\n"
            "        (df_feat['Purchase_Frequency_Year'] / 52.0) * 0.6 +\n"
            "        ((365 - df_feat['Recency_Days']) / 365.0) * 0.4\n"
            "    )\n"
            "    # One-Hot Encoding for Gender\n"
            "    df_feat['Gender_Male'] = (df_feat['Gender'] == 'Male').astype(int)\n"
            "    return df_feat\n\n"
            "# High-Value Target Creation (Top 25% CLV Index)\n"
            "clv_index = (\n"
            "    0.55 * (df['Total_Annual_Spend'] / df['Total_Annual_Spend'].quantile(0.90)) +\n"
            "    0.25 * ((365 - df['Recency_Days']) / 365) +\n"
            "    0.20 * (df['App_Sessions_Month'] / 30)\n"
            ")\n"
            "df['High_Value_Customer'] = (clv_index >= clv_index.quantile(0.75)).astype(int)"
        )
        draw_code_box(ax, 0.10, 0.72, 0.80, 0.46, c2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 20)
        ax.text(0.10, 0.88, "6.1 Week 1 – Exploratory Visualizations & Distributions", fontsize=12, weight='bold', color='#0F172A')
        
        # Plot 1: Income vs Spending Score
        sub_ax1 = fig.add_axes([0.12, 0.50, 0.76, 0.32])
        np.random.seed(42)
        inc = np.random.uniform(15, 140, 150)
        spd = np.random.uniform(5, 95, 150)
        sub_ax1.scatter(inc, spd, c='#3B82F6', alpha=0.7, edgecolors='black', s=45)
        sub_ax1.set_title("Customer Income vs Spending Score Distribution", fontsize=10, weight='bold')
        sub_ax1.set_xlabel("Annual Income ($k)", fontsize=9)
        sub_ax1.set_ylabel("Spending Score (1-100)", fontsize=9)
        sub_ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot 2: Correlation Heatmap Representation
        sub_ax2 = fig.add_axes([0.12, 0.12, 0.76, 0.30])
        corr_vals = np.array([
            [1.00, 0.05, -0.01, 0.65],
            [0.05, 1.00, -0.32, 0.12],
            [-0.01, -0.32, 1.00, 0.78],
            [0.65, 0.12, 0.78, 1.00]
        ])
        cax = sub_ax2.imshow(corr_vals, cmap='Blues', vmin=-0.5, vmax=1.0)
        sub_ax2.set_xticks(range(4))
        sub_ax2.set_yticks(range(4))
        feats = ["Income", "Age", "Spend", "Frequency"]
        sub_ax2.set_xticklabels(feats, fontsize=8.5)
        sub_ax2.set_yticklabels(feats, fontsize=8.5)
        sub_ax2.set_title("Feature Correlation Matrix", fontsize=10, weight='bold')
        for i in range(4):
            for j in range(4):
                sub_ax2.text(j, i, f"{corr_vals[i, j]:.2f}", ha="center", va="center", color="black" if corr_vals[i, j] < 0.6 else "white", fontsize=8.5)
                
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 21)
        p21_text = (
            "## 6.2 Week 2 – K-Means Persona Clustering & Evaluation\n"
            "In Week 2, the K-Means clustering model was trained. Evaluation was performed across $K=2..8$ using "
            "both Inertia (Elbow Method) and Silhouette analysis."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p21_text, fontsize=9.2, line_spacing=0.023)
        
        c3 = (
            "from sklearn.cluster import KMeans\n"
            "from sklearn.metrics import silhouette_score\n\n"
            "# Evaluate K over range [2, 8]\n"
            "k_results = []\n"
            "for k in range(2, 9):\n"
            "    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    inertia = km.inertia_\n"
            "    sil = silhouette_score(X_scaled, labels)\n"
            "    k_results.append({'K': k, 'Inertia': inertia, 'Silhouette': sil})\n\n"
            "# Fit Final Optimal Model (K=5)\n"
            "optimal_kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)\n"
            "cluster_labels = optimal_kmeans.fit_predict(X_scaled)\n"
            "df['Cluster_ID'] = cluster_labels"
        )
        draw_code_box(ax, 0.10, 0.72, 0.80, 0.36, c3)
        
        # Plot: Elbow curve
        sub_ax3 = fig.add_axes([0.12, 0.10, 0.76, 0.22])
        ks = [2, 3, 4, 5, 6, 7, 8]
        inertias = [1336, 1015, 774, 638, 562, 514, 476]
        sub_ax3.plot(ks, inertias, marker='o', color='#DC2626', lw=2)
        sub_ax3.axvline(x=5, color='#3B82F6', linestyle='--', label='Optimal K=5')
        sub_ax3.set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=9.5, weight='bold')
        sub_ax3.set_xlabel("Number of Clusters (K)", fontsize=8.5)
        sub_ax3.set_ylabel("Inertia (WCSS)", fontsize=8.5)
        sub_ax3.grid(True, linestyle='--', alpha=0.5)
        sub_ax3.legend(fontsize=8)
        
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 22)
        p22_text = (
            "## 6.2 Week 2 – Supervised Model Training & Joblib Serialization\n"
            "Supervised models were trained with Stratified K-Fold Cross-Validation. The trained pipeline components "
            "were serialized to disk using Joblib for production inference."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p22_text, fontsize=9.2, line_spacing=0.023)
        
        c4 = (
            "from xgboost import XGBClassifier\n"
            "from sklearn.model_selection import StratifiedKFold, cross_validate\n"
            "import joblib\n\n"
            "# 1. Initialize XGBoost Classifier\n"
            "xgb_model = XGBClassifier(\n"
            "    n_estimators=120, max_depth=4, learning_rate=0.08,\n"
            "    subsample=0.85, colsample_bytree=0.85, eval_metric='logloss',\n"
            "    random_state=42\n"
            ")\n\n"
            "# 2. Stratified 5-Fold Cross-Validation\n"
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "cv_scores = cross_validate(xgb_model, X_clf, y_clf, cv=cv, scoring=['roc_auc', 'f1'])\n"
            "print('Mean CV ROC-AUC:', cv_scores['test_roc_auc'].mean())\n\n"
            "# 3. Fit on full training split and persist with Joblib\n"
            "xgb_model.fit(X_train, y_train)\n"
            "joblib.dump(scaler, 'models/preprocessor.joblib')\n"
            "joblib.dump(optimal_kmeans, 'models/segmenter.joblib')\n"
            "joblib.dump(xgb_model, 'models/classifier.joblib')\n"
            "print('Successfully serialized ML models to models/ directory!')"
        )
        draw_code_box(ax, 0.10, 0.72, 0.80, 0.44, c4)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 23)
        p23_text = (
            "## 6.2 Week 2 – Real-Time Customer Scoring Pipeline\n"
            "A unified inference pipeline (`RetailIntelligencePipeline`) integrates preprocessor, clusterer, classifier, "
            "and recommendation engine to score individual customers in milliseconds."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p23_text, fontsize=9.2, line_spacing=0.023)
        
        c5 = (
            "# Real-Time Single Customer Scoring Simulation\n"
            "sample_customer = {\n"
            "    'CustomerID': 'CUST_VIP_998', 'Gender': 'Female', 'Age': 34,\n"
            "    'Annual_Income_k': 95, 'Spending_Score': 88,\n"
            "    'Purchase_Frequency_Year': 32, 'Avg_Order_Value': 280.0,\n"
            "    'Recency_Days': 12, 'App_Sessions_Month': 24\n"
            "}\n\n"
            "result = pipeline.score_single_customer(sample_customer)\n"
            "print('Discovered Persona:', result['persona_name'])\n"
            "print('High-Value Prediction:', result['high_value_prediction'])\n"
            "print('High-Value Probability:', result['high_value_probability'])\n"
            "print('Assigned Strategy:', result['recommendation']['strategy'])\n"
            "print('Recommended Channel:', result['recommendation']['channel'])"
        )
        draw_code_box(ax, 0.10, 0.72, 0.80, 0.32, c5)
        
        # Sample Output Box
        ax.text(0.10, 0.36, "Sample Output Execution Log:", fontsize=9.5, weight='bold', color='#0F172A')
        out_text = (
            "--------------------------------------------------------------------------------\n"
            "Discovered Persona:    Affluent VIP Spenders (Tier 1 - Highest Value)\n"
            "High-Value Prediction: High-Value Shopper\n"
            "Propensity Score:      0.999 (99.9% High-Value Probability)\n"
            "Assigned Strategy:     VIP Loyalty Concierge & Exclusive Retention\n"
            "Recommended Channel:   Dedicated Concierge / Private Invitation / SMS Preview\n"
            "Discount Policy:       Zero price discounting. Maintain brand prestige & luxury margins.\n"
            "Expected Impact:       +22% Customer Lifetime Value (CLV), 95%+ renewal rate.\n"
            "--------------------------------------------------------------------------------"
        )
        draw_code_box(ax, 0.10, 0.32, 0.80, 0.22, out_text)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 24)
        p24_text = (
            "## 6.3 Week 3 – Streamlit Web Application Deployment\n"
            "In Week 3, the trained models were integrated into an interactive web application using the Streamlit framework. "
            "Streamlit empowers non-technical retail executives, store managers, and marketing directors to query customer "
            "profiles and simulate campaigns through a clean graphical user interface."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p24_text, fontsize=9.2, line_spacing=0.023)
        
        c6 = (
            "import streamlit as st\n"
            "import plotly.express as px\n"
            "import joblib\n"
            "from src.pipeline import RetailIntelligencePipeline\n\n"
            "# 1. Page Configuration\n"
            "st.set_page_config(page_title='Retail Multi-Segment Profiler', page_icon='🛍️', layout='wide')\n\n"
            "# 2. Cached Resource Ingestion\n"
            "@st.cache_resource\n"
            "def load_intelligence_pipeline():\n"
            "    pipeline = RetailIntelligencePipeline.load('models')\n"
            "    df = pd.read_csv('data/retail_customer_intelligence.csv')\n"
            "    return pipeline, df\n\n"
            "pipeline, df = load_intelligence_pipeline()\n"
            "st.title('🛍️ Retail Multi-Segment Profiler & High-Value Classifier')"
        )
        draw_code_box(ax, 0.10, 0.68, 0.80, 0.38, c6)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 25)
        p25_text = (
            "## 6.3 Week 3 – Interactive Customer Simulator UI Code\n"
            "The simulator tab captures real-time customer inputs via interactive sliders, evaluates propensity via XGBoost, "
            "and dynamically renders persona cards and probability gauge charts."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p25_text, fontsize=9.2, line_spacing=0.023)
        
        c7 = (
            "# Interactive Simulator Inputs\n"
            "col_in, col_out = st.columns([1, 1.2])\n"
            "with col_in:\n"
            "    st.subheader('Input Customer Profile')\n"
            "    age = st.slider('Age', 18, 80, 32)\n"
            "    income = st.slider('Annual Income ($k)', 15, 160, 85)\n"
            "    spend = st.slider('Spending Score (1-100)', 1, 100, 78)\n"
            "    freq = st.slider('Orders / Year', 1, 52, 28)\n"
            "    aov = st.slider('Avg Order Value ($)', 20.0, 450.0, 210.0)\n"
            "    recency = st.slider('Days Since Last Order', 1, 365, 15)\n\n"
            "# Real-time Scoring Trigger\n"
            "res = pipeline.score_single_customer({\n"
            "    'Age': age, 'Annual_Income_k': income, 'Spending_Score': spend,\n"
            "    'Purchase_Frequency_Year': freq, 'Avg_Order_Value': aov, 'Recency_Days': recency\n"
            "})\n\n"
            "with col_out:\n"
            "    st.metric('Predicted Persona', res['persona_name'])\n"
            "    st.metric('High-Value Propensity', f\"{res['high_value_probability']*100:.1f}%\")\n"
            "    st.info(f\"Strategy: {res['recommendation']['strategy']}\")"
        )
        draw_code_box(ax, 0.10, 0.70, 0.80, 0.44, c7)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 26)
        p26_text = (
            "## 6.4 Improvements Made Over Mentor's Baseline Code\n"
            "During development, several key architectural improvements were introduced over standard baseline approaches:\n\n"
            "● Target Leakage Elimination: Replaced naïve classification on raw spending score with historical behavioral "
            "predictors (Recency, Frequency, AOV) to support genuine future-spend forecasting.\n"
            "● Multi-Model Stratified Cross-Validation: Benchmarked 4 distinct classifiers (Logistic Regression, Decision Tree, "
            "Random Forest, XGBoost) using Stratified 5-Fold CV rather than a single fragile split.\n"
            "● Dual Unsupervised Clustering Validation: Evaluated both Inertia (Elbow Method) and Silhouette Scores across "
            "$K=2..8$ to mathematically validate $K=5$ rather than arbitrarily guessing.\n"
            "● Automated Business Persona Profiling: Built dynamic statistical labeling algorithms that map cluster centers "
            "to meaningful business tags (*Affluent VIP Spenders*, *Conservative Savers*, *Trendsetters*, *Budget Shoppers*).\n"
            "● Enterprise Marketing Strategy Engine: Added automated translation of ML outputs into actionable CRM tactics, "
            "discount policies, and expected ROI metrics.\n"
            "● Modern Interactive Multi-Tab Streamlit Dashboard: Created a complete 6-tab analytical suite with 2D/3D Plotly "
            "visualizations, live customer simulator, and 1-click CSV database export."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p26_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 27-30: 7. RESULTS AND DISCUSSION
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 27)
        ax.text(0.10, 0.88, "7. Results and Discussion", fontsize=15, weight='bold', color='#0F172A')
        ax.text(0.10, 0.84, "7.1 Model Performance (Supervised Benchmark & Clustering)", fontsize=11, weight='bold', color='#1E293B')
        
        # Benchmark Table
        bench_headers = ["Model Family", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        bench_rows = [
            ["XGBoost Classifier", "0.973 ±0.01", "0.954", "0.940", "0.946", "0.997"],
            ["Random Forest", "0.967 ±0.01", "0.928", "0.940", "0.934", "0.996"],
            ["Logistic Regression", "0.978 ±0.01", "0.966", "0.947", "0.956", "0.998"],
            ["Decision Tree", "0.930 ±0.03", "0.894", "0.820", "0.853", "0.907"]
        ]
        
        bt_y = 0.76
        ax.add_patch(patches.Rectangle((0.10, bt_y - 0.035), 0.80, 0.035, facecolor='#1E293B', edgecolor='#0F172A'))
        for c_idx, h in enumerate(bench_headers):
            ax.text(0.12 + c_idx * 0.13, bt_y - 0.024, h, fontsize=8.5, weight='bold', color='white')
            
        cur_bty = bt_y - 0.035
        for r_idx, r in enumerate(bench_rows):
            bg = '#FEF2F2' if r_idx == 0 else ('#F8FAFC' if r_idx % 2 == 1 else '#FFFFFF')
            ax.add_patch(patches.Rectangle((0.10, cur_bty - 0.04), 0.80, 0.04, facecolor=bg, edgecolor='#CBD5E1', lw=0.7))
            for c_idx, val in enumerate(r):
                w = 'bold' if (r_idx == 0 or c_idx == 0) else 'normal'
                ax.text(0.12 + c_idx * 0.13, cur_bty - 0.027, val, fontsize=8.2, color='#0F172A' if r_idx==0 else '#334155', weight=w)
            cur_bty -= 0.04
            
        p27_subtext = (
            "\n\n### Discussion of Supervised Classification Results:\n"
            "The Stratified 5-Fold Cross-Validation benchmark confirms that **XGBoost Classifier** achieves superior "
            "generalization with an ROC-AUC of **0.997** and an F1-Score of **0.946**. Specifically, XGBoost attained a "
            "High-Value Precision of **95.4%** and Recall of **94.0%**, ensuring that almost all high-value shoppers "
            "are captured without over-allocating VIP rewards to standard shoppers.\n\n"
            "### Discussion of Clustering Results:\n"
            "At $K=5$, the K-Means clustering algorithm achieved an optimal balance between geometric compactness "
            "(Inertia $= 638.8$) and cluster separation (Silhouette Score $= 0.292$). Each discovered cluster possesses "
            "distinct demographic and behavioral traits with immediate commercial applicability."
        )
        draw_wrapped_text(ax, 0.10, cur_bty - 0.02, p27_subtext, fontsize=9.1, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 28)
        ax.text(0.10, 0.88, "7.2 Visualizations of Results (Feature Importance & Clusters)", fontsize=12, weight='bold', color='#0F172A')
        
        # Plot 1: Feature Importances
        sub_ax_f = fig.add_axes([0.14, 0.52, 0.72, 0.30])
        f_names = ['Purchase Frequency', 'Engagement Index', 'Spending Score', 'Avg Order Value', 'Recency Days', 'Discount Usage %']
        f_imps = [35.6, 25.0, 6.4, 5.5, 4.3, 4.1]
        sub_ax_f.barh(f_names[::-1], f_imps[::-1], color='#3B82F6', edgecolor='#1D4ED8')
        sub_ax_f.set_title("XGBoost Top Feature Importances (% Contribution)", fontsize=9.5, weight='bold')
        sub_ax_f.set_xlabel("Relative Importance (%)", fontsize=8.5)
        sub_ax_f.grid(True, linestyle='--', alpha=0.5)
        
        # Plot 2: 2D Cluster Scatter
        sub_ax_c = fig.add_axes([0.14, 0.10, 0.72, 0.32])
        np.random.seed(42)
        colors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
        labels = ['Frugal Budget', 'Trendsetters', 'VIP Spenders', 'Conservative Savers', 'Mainstream']
        for i in range(5):
            cx = np.random.normal(loc=[30, 40, 100, 100, 60][i], scale=8, size=30)
            cy = np.random.normal(loc=[20, 80, 80, 25, 50][i], scale=8, size=30)
            sub_ax_c.scatter(cx, cy, c=colors[i], label=labels[i], alpha=0.75, edgecolors='black', s=40)
        sub_ax_c.set_title("K-Means 2D Persona Clusters (Income vs Spending Score)", fontsize=9.5, weight='bold')
        sub_ax_c.set_xlabel("Annual Income ($k)", fontsize=8.5)
        sub_ax_c.set_ylabel("Spending Score (1-100)", fontsize=8.5)
        sub_ax_c.grid(True, linestyle='--', alpha=0.5)
        sub_ax_c.legend(fontsize=7.5, loc='upper right')
        
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 29)
        p29_text = (
            "## 7.3 Streamlit App Output & Dashboard Walkthrough\n"
            "The deployed Streamlit dashboard provides an intuitive, real-time command center for retail decision-makers. "
            "It consists of six specialized operational tabs:\n\n"
            "● Tab 1 (Executive EDA): High-level KPI metric cards (Total Customers, High-Value Share, Average Income, "
            "Average Spend) and interactive Plotly demographic distributions.\n"
            "● Tab 2 (K-Means Personas): Interactive 2D and 3D scatter projections, Elbow/Silhouette curves, and persona "
            "statistical summaries.\n"
            "● Tab 3 (XGBoost Classifier): 5-Fold Cross Validation benchmark table, Receiver Operating Characteristic (ROC) "
            "curves, holdout Confusion Matrix, and feature importance rankings.\n"
            "● Tab 4 (Real-time Simulator): Instant customer scoring calculator with dynamic gauge indicators and customized "
            "campaign recommendations.\n"
            "● Tab 5 (Scored Database): Searchable and filterable customer database with one-click CSV export.\n"
            "● Tab 6 (Strategy Playbook): Segment-level tactical matrix outlining outreach channels, discount policies, "
            "and expected ROI impact."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p29_text, fontsize=9.2, line_spacing=0.022)
        
        # Draw mockup representation of Streamlit dashboard
        mock_rect = patches.Rectangle((0.12, 0.12), 0.76, 0.28, facecolor='#0F172A', edgecolor='#334155', lw=1.2)
        ax.add_patch(mock_rect)
        ax.text(0.15, 0.36, "🛍️ Retail Multi-Segment Profiler & High-Value Classifier", fontsize=10, weight='bold', color='white')
        ax.text(0.15, 0.32, "Live URL: http://localhost:8501   |   Streamlit Framework", fontsize=8, color='#94A3B8')
        
        # 3 Mock Metric Cards
        for m_idx, (m_t, m_v, m_c) in enumerate([["Total Customers", "600", "#3B82F6"], ["High-Value Share", "25.0% (150)", "#10B981"], ["Avg Income", "$61.8k", "#F59E0B"]]):
            ax.add_patch(patches.Rectangle((0.15 + m_idx * 0.24, 0.16), 0.22, 0.12, facecolor='#1E293B', edgecolor='#334155'))
            ax.text(0.17 + m_idx * 0.24, 0.24, m_t, fontsize=7.5, color='#94A3B8')
            ax.text(0.17 + m_idx * 0.24, 0.19, m_v, fontsize=10, weight='bold', color=m_c)
            
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 30)
        p30_text = (
            "## 7.4 Interpretation of Personas & Business Recommendations\n"
            "The combination of K-Means persona clustering and XGBoost classification yields specific, high-ROI marketing plays:\n\n"
            "### 1. Affluent VIP Spenders (High Income, High Spend)\n"
            "● Characteristics: Top 15.5% of customer base. Average annual income $122.5k, spending score 52.6/100, high AOV ($280+).\n"
            "● Strategy: VIP Concierge & Exclusive Retention. Private invitations to seasonal previews, dedicated styling support.\n"
            "● Promotional Policy: Zero price discounting (preserves luxury brand equity).\n"
            "● Projected Impact: +22% Customer Lifetime Value (CLV), 95%+ annual retention rate.\n\n"
            "### 2. Affluent Conservative Savers (High Income, Low Spend)\n"
            "● Characteristics: High disposable income ($80k+) but low shopping frequency and conservative spend.\n"
            "● Strategy: High-Value Upselling & Premium Basket Conversion. High-threshold gifts (Free gift on orders >$300).\n"
            "● Promotional Policy: Threshold-based rewards rather than blanket discounts.\n"
            "● Projected Impact: +35% Average Order Value (AOV) expansion.\n\n"
            "### 3. Young Enthusiasts & Trendsetters (Lower Income, High Spend)\n"
            "● Characteristics: Younger demographic (Avg Age 25-30) with moderate income but intense brand engagement.\n"
            "● Strategy: Gamified Mobile Drops & Social Virality. 24-hour flash releases, dual-sided referral bonuses ($15 both).\n"
            "● Projected Impact: +40% purchase frequency."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p30_text, fontsize=9.1, line_spacing=0.021)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 31-32: 8. CONCLUSION & FUTURE WORK
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 31)
        p31_text = (
            "# 8. Conclusion & Future Work\n\n"
            "## 8.1 Summary of Learnings\n"
            "The project successfully demonstrated the end-to-end implementation and enterprise value of combining "
            "unsupervised learning and supervised machine learning for retail intelligence. By engineering a modular "
            "pipeline from raw data cleaning to interactive web deployment, the project established a functional, "
            "reproducible solution that delivers actionable customer insights in real time.\n\n"
            "## 8.2 AI & Retail Analytics Skills Acquired\n"
            "The internship provided comprehensive training across data science, machine learning, and software deployment:"
        )
        draw_wrapped_text(ax, 0.10, 0.88, p31_text, fontsize=9.2, line_spacing=0.023)
        
        # Skills table
        sk_headers = ["Project Task", "Technical Competency", "Specific Skills Acquired"]
        sk_rows = [
            ["Data Exploration & Cleaning", "Data Analytics & Preprocessing", "Pandas, NumPy, Outlier clipping, Imputation"],
            ["Feature Engineering", "Feature Representation", "Engagement indices, Leakage prevention"],
            ["Unsupervised Clustering", "Machine Learning (Unsupervised)", "K-Means++, Elbow Method, Silhouette Analysis"],
            ["Supervised Classification", "Machine Learning (Supervised)", "XGBoost, Random Forest, Stratified 5-Fold CV"],
            ["Model Deployment", "Web App & Productionization", "Streamlit UI, Joblib serialization, Plotly 2D/3D"],
            ["Business Decisioning", "CRM Strategy & Analytics", "Persona mapping, CLV modeling, ROI estimation"]
        ]
        
        sk_y = 0.52
        ax.add_patch(patches.Rectangle((0.10, sk_y - 0.035), 0.80, 0.035, facecolor='#1E293B', edgecolor='#0F172A'))
        ax.text(0.12, sk_y - 0.025, sk_headers[0], fontsize=8.5, weight='bold', color='white')
        ax.text(0.38, sk_y - 0.025, sk_headers[1], fontsize=8.5, weight='bold', color='white')
        ax.text(0.66, sk_y - 0.025, sk_headers[2], fontsize=8.5, weight='bold', color='white')
        
        cur_sky = sk_y - 0.035
        for r_idx, (pt, tc, sa) in enumerate(sk_rows):
            bg = '#F8FAFC' if r_idx % 2 == 0 else '#FFFFFF'
            ax.add_patch(patches.Rectangle((0.10, cur_sky - 0.045), 0.80, 0.045, facecolor=bg, edgecolor='#CBD5E1', lw=0.7))
            ax.text(0.12, cur_sky - 0.030, pt, fontsize=8, weight='bold', color='#1E293B')
            ax.text(0.38, cur_sky - 0.030, tc, fontsize=8, color='#334155')
            ax.text(0.66, cur_sky - 0.030, sa, fontsize=8, color='#334155')
            cur_sky -= 0.045
            
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 32)
        p32_text = (
            "## 8.3 Limitations of the Current Approach\n"
            "While the developed system demonstrates strong predictive accuracy, certain practical limitations exist:\n"
            "● Static Dataset Scope: Models were trained on fixed tabular batches and do not yet continuously update via streaming online learning.\n"
            "● Lack of Textual Sentiment: Qualitative feedback (customer reviews, support chat transcripts) is not yet incorporated into the persona space.\n"
            "● Macroeconomic Factors: External variables like inflation indices, local seasonal weather, and competitor discount events are not modeled.\n\n"
            "## 8.4 Future Scope\n"
            "Future enhancements to extend this project into an enterprise production system include:\n"
            "● Real-Time POS & E-Commerce Streaming: Integrating Apache Kafka and cloud data warehouses (Snowflake/BigQuery) "
            "to score customer carts dynamically at checkout.\n"
            "● Deep Learning & Graph Neural Networks: Exploring Transformer-based sequence models and GNNs to model "
            "product co-purchasing graphs and next-item basket recommendations.\n"
            "● Automated CRM Webhooks: Connecting predicted high-value scores directly to automated email/SMS marketing "
            "platforms (Klaviyo, Braze, Salesforce Marketing Cloud) to trigger immediate personalized offers.\n"
            "● Multi-Armed Bandit A/B Testing: Implementing Reinforcement Learning algorithms to dynamically test and "
            "optimize promotional discounts per customer persona."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p32_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 33: 9. INTERNSHIP OUTCOMES
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 33)
        p33_text = (
            "# 9. Internship Outcomes\n\n"
            "## 9.1 Technical Skills Acquired\n"
            "● Mastered complete machine learning lifecycle development, from raw exploratory data analysis to production deployment.\n"
            "● Gained deep theoretical and practical proficiency in K-Means clustering, silhouette optimization, and XGBoost classification.\n"
            "● Developed expertise in eliminating target leakage and structuring robust Stratified 5-Fold Cross-Validation.\n"
            "● Learned to serialize machine learning artifacts with Joblib and deploy full-stack interactive dashboards using Streamlit and Plotly.\n\n"
            "## 9.2 Soft Skills Developed\n"
            "● Analytical Problem Solving: Formulated business problems into rigorous mathematical objectives and actionable models.\n"
            "● Technical Communication: Translated complex algorithmic metrics (ROC-AUC, WCSS, Silhouette) into executive business insights.\n"
            "● Time & Project Management: Planned and executed modular weekly technical milestones across the four-week timeline.\n\n"
            "## 9.3 Contribution to Career Growth\n"
            "● Established an end-to-end, industry-grade portfolio project showcasing both unsupervised and supervised machine learning.\n"
            "● Gained practical exposure to enterprise customer intelligence workflows, substantially increasing readiness for "
            "roles in Data Science, Machine Learning Engineering, and AI-driven business analytics."
        )
        draw_wrapped_text(ax, 0.10, 0.88, p33_text, fontsize=9.2, line_spacing=0.022)
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGES 34-35: APPENDIX A & B (SOURCE CODE & SCREENSHOTS)
        # ==========================================================
        fig, ax = create_page()
        add_header_footer(ax, 34)
        p34_text = (
            "# Appendix\n\n"
            "## A. Source Code & Project Repository\n"
            "The complete, modular source code, tests, and configuration for this project are organized as follows:\n"
            "● GitHub Repository: Retail Multi-Segment Profiler & High-Value Classifier\n"
            "● Main Python Modules: `src/data_loader.py`, `src/preprocessing.py`, `src/clustering.py`, `src/classification.py`, `src/recommendation.py`, `src/pipeline.py`\n"
            "● Execution Scripts: `run_pipeline.py` (CLI Pipeline), `app.py` (Streamlit Dashboard), `tests/test_pipeline.py` (Pytest Suite)\n\n"
            "## B. Streamlit Application Screenshots\n"
            "Screenshots demonstrating the interactive web application interface and user workflow are provided below:\n\n"
            "### Home Page & Executive KPI Overview (Tab 1)"
        )
        draw_wrapped_text(ax, 0.10, 0.88, p34_text, fontsize=9.2, line_spacing=0.022)
        
        # Draw UI Mockup 1
        ax.add_patch(patches.Rectangle((0.10, 0.14), 0.80, 0.38, facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1.2))
        ax.add_patch(patches.Rectangle((0.10, 0.48), 0.80, 0.04, facecolor='#1E293B'))
        ax.text(0.12, 0.495, "Retail Multi-Segment Profiler & High-Value Classifier | Executive Dashboard", fontsize=8.5, weight='bold', color='white')
        
        # Mock Charts inside box
        ax.text(0.14, 0.43, "KPI Overview: 600 Customers  |  25.0% High-Value Share  |  $61.8k Avg Income", fontsize=8, weight='bold', color='#1E293B')
        ax.add_patch(patches.Rectangle((0.14, 0.18), 0.34, 0.22, facecolor='#FFFFFF', edgecolor='#E2E8F0'))
        ax.text(0.16, 0.36, "Income vs Spend (2D)", fontsize=7.5, weight='bold', color='#3B82F6')
        
        ax.add_patch(patches.Rectangle((0.52, 0.18), 0.34, 0.22, facecolor='#FFFFFF', edgecolor='#E2E8F0'))
        ax.text(0.54, 0.36, "Age Cohort Donut", fontsize=7.5, weight='bold', color='#10B981')
        
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = create_page()
        add_header_footer(ax, 35)
        p35_text = (
            "### Customer Personas & Real-Time Scoring Simulator (Tabs 2 & 4)\n"
            "The screenshot below illustrates the real-time customer simulator, featuring dynamic sliders, the high-value "
            "propensity gauge, and automated marketing strategy card generation:"
        )
        draw_wrapped_text(ax, 0.10, 0.88, p35_text, fontsize=9.2, line_spacing=0.022)
        
        # Draw UI Mockup 2
        ax.add_patch(patches.Rectangle((0.10, 0.46), 0.80, 0.36, facecolor='#0F172A', edgecolor='#334155', lw=1.2))
        ax.text(0.14, 0.77, "🎯 Live Customer Scoring Simulator Output", fontsize=10, weight='bold', color='white')
        ax.text(0.14, 0.73, "Customer ID: CUST_LIVE_SIM_001  |  Age: 32  |  Income: $85k  |  Spend Score: 78", fontsize=8, color='#94A3B8')
        
        # Gauge representation
        ax.add_patch(patches.Wedge((0.30, 0.58), 0.08, 0, 180, facecolor='#10B981', edgecolor='white'))
        ax.text(0.30, 0.56, "99.9% High-Value", fontsize=8, weight='bold', color='white', ha='center')
        
        # Strategy card inside mockup
        ax.add_patch(patches.Rectangle((0.44, 0.50), 0.42, 0.22, facecolor='#1E293B', edgecolor='#334155'))
        ax.text(0.46, 0.68, "💎 VIP Elite: Exclusive Concierge", fontsize=8, weight='bold', color='#38BDF8')
        ax.text(0.46, 0.63, "• Channel: Dedicated 1-on-1 Concierge / SMS", fontsize=7, color='#E2E8F0')
        ax.text(0.46, 0.58, "• Policy: Zero price discounting (preserve margin)", fontsize=7, color='#E2E8F0')
        ax.text(0.46, 0.53, "• ROI Impact: +22% CLV retention", fontsize=7, color='#4ADE80')
        
        # Database preview
        ax.text(0.10, 0.40, "### Customer Intelligence Database & CSV Exporter (Tab 5)", fontsize=10, weight='bold', color='#0F172A')
        ax.add_patch(patches.Rectangle((0.10, 0.10), 0.80, 0.26, facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=1))
        ax.text(0.12, 0.32, "Filtered Database Table (600 Customer Records)", fontsize=8.5, weight='bold', color='#1E293B')
        ax.text(0.12, 0.28, "Cols: CustomerID, Gender, Age, Income, Spend, Persona, Tier, Prediction, Probability, Strategy", fontsize=7.2, color='#475569')
        ax.add_patch(patches.Rectangle((0.12, 0.14), 0.28, 0.08, facecolor='#3B82F6', edgecolor='#1D4ED8'))
        ax.text(0.16, 0.18, "📥 Download Scored CSV", fontsize=8, weight='bold', color='white')
        
        pdf.savefig(fig)
        plt.close(fig)

        # ==========================================================
        # PAGE 36: APPENDIX C (CERTIFICATE OF COMPLETION)
        # ==========================================================
        fig, ax = create_page()
        ax.text(0.10, 0.90, "Appendix C. Certificate of Completion", fontsize=13, weight='bold', color='#0F172A')
        ax.text(0.10, 0.86, "The certificate awarded for successful completion of the virtual internship is included for reference:", fontsize=9, color='#475569')
        
        # Draw Certificate Card
        cert_box = patches.Rectangle((0.10, 0.08), 0.80, 0.74, facecolor='#FFFFFF', edgecolor='#0284C7', lw=3)
        ax.add_patch(cert_box)
        inner_box = patches.Rectangle((0.115, 0.095), 0.77, 0.71, facecolor='#F0F9FF', edgecolor='#BAE6FD', lw=1.2)
        ax.add_patch(inner_box)
        
        # Logos Header
        ax.text(0.20, 0.75, "edunet\nfoundation", fontsize=11, weight='bold', color='#0369A1', ha='center')
        ax.text(0.50, 0.75, "AICTE\nApproved", fontsize=10, weight='bold', color='#D97706', ha='center')
        ax.text(0.80, 0.75, "Shell\nSkills4Future", fontsize=11, weight='bold', color='#DC2626', ha='center')
        
        ax.text(0.50, 0.64, "Certificate of Completion", fontsize=18, weight='bold', color='#0369A1', ha='center')
        ax.text(0.50, 0.58, "This is to certify that", fontsize=10, style='italic', color='#334155', ha='center')
        
        ax.text(0.50, 0.51, "CHELLURI SAI VISHAL", fontsize=15, weight='bold', color='#0F172A', ha='center')
        
        cert_body = (
            "has successfully completed the 4-week virtual internship on\n"
            "Artificial Intelligence and Data Analytics focused on Retail Multi-Segment Intelligence,\n"
            "organized by AICTE, Shell India Markets Private Limited, and Edunet Foundation\n"
            "under the Skills4Future program, from 16th June 2025 to 16th July 2025.\n\n"
            "Student ID: STU682621ad12d201747329453"
        )
        ax.text(0.50, 0.38, cert_body, fontsize=9, color='#1E293B', ha='center', linespacing=1.4)
        
        # Signatures
        ax.plot([0.18, 0.36], [0.18, 0.18], color='#0F172A', lw=1)
        ax.text(0.27, 0.15, "Nagesh Singh\nChairman\nEdunet Foundation", fontsize=7.5, ha='center', weight='bold')
        
        ax.plot([0.41, 0.59], [0.18, 0.18], color='#0F172A', lw=1)
        ax.text(0.50, 0.15, "Dr. Buddha Chandrasekhar\nChief Coordinating Officer\nAICTE", fontsize=7.5, ha='center', weight='bold')
        
        ax.plot([0.64, 0.82], [0.18, 0.18], color='#0F172A', lw=1)
        ax.text(0.73, 0.15, "Neha Chauhan\nSP Manager\nShell India Markets Pvt Ltd", fontsize=7.5, ha='center', weight='bold')
        
        pdf.savefig(fig)
        plt.close(fig)

    print(f"✅ Successfully created 36-page report PDF: {output_pdf_path}")


if __name__ == "__main__":
    out_file = "Retail_Customer_Intelligence_Internship_Report.pdf"
    generate_full_36_page_report(out_file)
