# Expedia Hotel Recommendation System
### End-to-End Machine Learning Platform | Personalisation · Behavioural Analytics · Business Intelligence

---

## Overview

A full-stack hotel recommendation system built on 41,000+ real Expedia booking transactions. This project goes beyond model accuracy — it translates machine learning outputs into actionable product strategy, marketing segmentation, and statistically validated deployment decisions.

**Live Dashboard:** [Expedia Intelligence Suite](https://expedia-intelligence-suite.onrender.com) *(live link — may take 30s to wake up on free tier)*

---

## Business Problem

A user visits Expedia, searches for a hotel, and we want to surface the most relevant hotel cluster for them to book. Better recommendations drive higher conversion. This is exactly the problem Expedia's real ML team solves at scale.

**Metric:** MAP@5 (Mean Average Precision at 5) — did the correct hotel cluster appear in the user's top 5 recommendations?

---

## Results

| Model | MAP@5 | Type |
|---|---|---|
| Logistic Regression | 0.1209 | Baseline |
| Random Forest | 0.3936 | Content-Based |
| LightGBM | 0.3249 | Content-Based |
| SVD Collaborative Filtering | 0.4850 | Collaborative |
| XGBoost | 0.5323 | Content-Based |
| **Hybrid (XGBoost + SVD)** | **0.6369** | **Best** |

**A/B Test Result:** Hybrid model delivers 26x uplift over random baseline. Validated at 99.99% statistical confidence (p < 0.0001, Cohen's d = 2.15).

---

## Project Architecture
```
Data (500K rows)
       ↓
Data Cleaning + Feature Engineering
       ↓
Exploratory Data Analysis (4 charts)
       ↓
Preprocessing (RobustScaler)
       ↓
┌─────────────────────┬──────────────────────┐
│  Content-Based      │  Collaborative       │
│  LR · RF · LGBM     │  SVD Matrix          │
│  XGBoost            │  Factorisation       │
└─────────────────────┴──────────────────────┘
              ↓ Hybrid (70/30)
       MAP@5 = 0.6369
              ↓
    SHAP Interpretability
              ↓
    A/B Test Simulation
              ↓
    RFM Customer Segmentation
              ↓
    Plotly Dash Dashboard
```

---

## Technical Stack

| Category | Tools |
|---|---|
| Languages | Python 3.12 |
| ML Models | XGBoost, LightGBM, scikit-learn, SciPy SVD |
| Data Processing | pandas, NumPy, RobustScaler |
| Interpretability | SHAP |
| Statistical Testing | scipy.stats (t-test, Cohen's d, confidence intervals) |
| Visualisation | Plotly Dash, Matplotlib, Seaborn |
| Deployment | Render.com |

---

## Key Findings

**What drives a hotel recommendation?**
- Geography (hotel continent + market) accounts for 50% of total model impact
- Destination searched matters 3x more than any user attribute
- Device type (mobile vs desktop) has the lowest predictive impact of all features

**Four traveller personas identified via RFM analysis:**

| Persona | Users | Avg Stay | Lead Time | Strategy |
|---|---|---|---|---|
| Spontaneous Explorer | 8,084 | 2.1 nights | 23 days | Flash deals, urgency messaging |
| Luxury Long-Stay | 1,304 | 6.4 nights | 54 days | Premium curated newsletters |
| Frequent Business | 417 | 2.3 nights | 28 days | Weekly availability alerts |
| Careful Planner | 1,212 | 3.1 nights | 147 days | Early-bird campaigns |

---

## Dataset

[Expedia Hotel Recommendations](https://www.kaggle.com/competitions/expedia-hotel-recommendations) — Kaggle Competition Dataset

- Training: 500,000 rows loaded (37M total available)
- Filtered to actual bookings: 41,054 rows
- Test set: 2,528,243 rows
- Target: hotel_cluster (0-99, 100 classes)

---

## Project Structure
```
hotel-recommendation-system/
├── Hotel_Recommendation_System.ipynb
├── README.md
├── requirements.txt
└── images/
    ├── model_comparison.png
    ├── shap_importance.png
    ├── ab_test_results.png
    └── rfm_segments.png
```

---

## How to Run
```bash
git clone https://github.com/Darakhshannazir/hotel-recommendation-system.git
pip install -r requirements.txt
jupyter notebook Hotel_Recommendation_System.ipynb
```

---

## About

Built by **Darakhshan Nazir** — Data Scientist II at Afiniti, Fulbright Scholar, MS Business Analytics (Bentley University, GPA 3.88).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-darakhshan--nazir-blue)](https://www.linkedin.com/in/darakhshan-nazir/)
[![GitHub](https://img.shields.io/badge/GitHub-Darakhshannazir-black)](https://github.com/Darakhshannazir)

---

*This project demonstrates end-to-end data science — from raw data to production-ready ML systems with business intelligence.*
