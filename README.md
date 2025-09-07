# SmartDevTime – Predicting Project Effort/Duration

This repository contains three experiments exploring **machine learning models for predicting software project effort and duration**.  
The goal is to assess feasibility of deployment and identify next steps for building a reliable estimation tool.

---

## 📂 Projects Overview

| Project | Dataset | Size | Notes |
|---------|---------|------|-------|
| **Project 1** | Real dataset (Function Point–based) | ~80 rows (after cleaning ~77) | Small, real-world data with team/manager experience and FP metrics. |
| **Project 2** | Hybrid (80 real + simulated expansion to ~300) | ~300 rows | Combines real observations with synthetic extension for stability. |
| **Side Project** | GitHub metadata (~37k repos) | ~37,000 rows | Public GitHub stats (stars, forks, PRs, watchers, etc.). Indirect proxy for duration. |

---

## ⚙️ Methods

Across projects, the following methods were applied:

- **Feature Engineering**
  - Log-transform of `Effort` to stabilize variance.
  - Handling of categorical features (one-hot).
  - Scaling where necessary.

- **Models Tested**
  - Linear Regression / Ridge / Lasso / ElasticNet
  - Decision Trees / Random Forests
  - XGBoost
  - Cross-validation strategies (5-Fold, 5x2 CV)

- **Evaluation Metrics**
  - R² (coefficient of determination)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

## 📊 Results Summary

### Project 1 (Real, ~80 rows)

- **Best Model**: Linear Regression  
- **Metrics**:  
  - R² ≈ 0.44  
  - MAE ≈ 2024  
  - RMSE ≈ 2831  

*Interpretation:*  
Real data but too small. Some predictive power exists, but uncertainty is high.

---

### Project 2 (Hybrid, ~300 rows)

- **Best Results** (Random Forest, 5x2 CV on Effort):  
  - R² ≈ 0.52 ± 0.16  
  - MAE ≈ 2074 ± 253  
  - RMSE ≈ 2637 ± 310  

- **Linear / ElasticNet (on Log Effort)**:  
  - R² ≈ 0.42–0.45  
  - MAE (back-transformed) ≈ 2160–2170  

*Interpretation:*  
Slightly better stability with cross-validation and log transform. Synthetic data helps training but carries risk of distribution shift.

---

### Side Project (GitHub, ~37k repos)

- **Linear Regression (5-Fold)**:  
  - R² ≈ 0.06  
  - MAE ≈ 450 days  
  - RMSE ≈ 551 days  

- **XGBoost with cumulative features (stars, forks, PRs, etc.)**:  
  - R² ≈ 0.993 (suspiciously high due to data leakage)  

*Interpretation:*  
GitHub metadata is unsuitable for pre-project prediction. Cumulative features leak post-hoc information, inflating results artificially.

---

## ✅ Conclusions

- **Realistic Deployment Today:**  
  Not yet recommended for external users expecting accurate single-point estimates.  
  Models can be deployed **internally as an MVP**, with:
  - Prediction intervals (e.g., P10/P50/P90).
  - Transparency about high uncertainty.

- **Project 1 & 2:**  
  Provide baseline performance (R² ≈ 0.4–0.5, MAE ≈ 2000+). Useful for proof-of-concept but not for robust external deployment.

- **Side Project:**  
  Not directly useful for effort estimation due to leakage and irrelevance of post-start metrics.

---

## 🚀 Next Steps

1. **Collect More Real Data**
   - Aim for 300–500+ real projects with consistent labels (Effort in person-hours/days).
   - Ensure all features are *known before project start* (e.g., team experience, estimated size, domain, tech stack).

2. **Refine Feature Set**
   - Remove post-hoc features (stars, forks, commits, PRs).
   - Focus on **pre-project descriptors**.

3. **Improve Validation**
   - Use **time-aware CV** if chronological ordering matters.
   - Report confidence intervals and calibration.

4. **Prototype Deployment**
   - Package pipeline with `scikit-learn` (`ColumnTransformer`, `RidgeCV` or `RandomForest`).
   - Expose via FastAPI endpoint: `/predict` → returns `{p10, p50, p90}` estimates.
   - Add monitoring (MAE, SMAPE, input drift).

5. **Iterative Retraining**
   - Continuously log new project data.
   - Retrain monthly or quarterly as dataset grows.

---

## 📌 Key Takeaways

- Current models capture ~40–50% of variance (moderate).  
- Predictive uncertainty is large (MAE ~2000 effort units).  
- **Data quality and size** are the main bottlenecks.  
- For reliable deployment, **more real, pre-project data** is essential.  

---

## 📝 Acknowledgements

This work was developed in three stages, combining real-world software project datasets, simulated data augmentation, and exploratory GitHub metadata.  
The experiments provide a foundation for future **data-driven project time estimation systems**.





