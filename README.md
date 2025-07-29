Title:
Predict Time to Complete IT Projects using ML

Description:
We are looking for a Junior to Mid-level Machine Learning Developer to build a predictive model that estimates how long IT projects will take to complete.

You’ll work with a dataset containing information about past projects (type, team size, experience level, complexity, etc.), and your task will be to preprocess the data, build a regression model, evaluate it, and deliver predictions and insights in a clear report.

Expected deliverables:

Cleaned dataset

Jupyter Notebook with well-commented code

Final model (pickle or joblib)

Visualizations showing data relationships and model performance

Final report (Markdown and README)

Tech stack preferred: Python, Scikit-learn, Pandas, Matplotlib/Seaborn
Bonus if you use advanced models (e.g. XGBoost) or demonstrate good model interpretability (SHAP/LIME).

Time: ~30 hours
Budget: $200–250

Country: India 🇮🇳
Language: English (Intermediate)




## This is for small and real dataset fp.csv


# Effort Prediction in Software Projects using Machine Learning

## 🧩 Problem Statement

The goal of this project is to predict the **development effort** (time/cost) of software projects using basic input features such as team and project characteristics.

---

## 🗂 Dataset Summary

* Source: Provided CSV file with 81 rows and 13 columns
* Cleaned data: 77 valid rows after removing -1 values in experience fields
* Features include: team/manager experience, code metrics, project language, etc.

---

## 🔧 Steps Followed

### 1. Data Inspection & Cleaning

* Used `.info()`, `.describe()`, and `.isnull()` to inspect structure
* Removed outliers and invalid values (`-1`) in experience columns

### 2. Feature Engineering

* Removed ID, Project, and YearEnd columns
* One-hot encoding for `Language`
* Added `TeamExp × ManagerExp` as interaction term

### 3. Model Training & Evaluation

We tested the following models:

| Model                      | MAE      | RMSE     | R²       |
| -------------------------- | -------- | -------- | -------- |
| Linear (selected features) | **2024** | **2831** | **0.44** |
| Linear (full + engineered) | 2230     | 2924     | 0.40     |
| Lasso Regression           | 2231     | 2925     | 0.40     |
| Ridge Regression           | 2259     | 2967     | 0.38     |
| Decision Tree              | 2779     | 3601     | 0.09     |
| XGBoost                    | 2511     | 3506     | 0.14     |
| Random Forest              | 2443     | 3325     | 0.23     |

---

## 📌 Conclusion

* **Simpler Linear Regression** with key features performed best
* Adding interaction & one-hot encoding did not improve significantly
* Complex models (XGBoost, RandomForest) underperformed due to small dataset size

---

## 🚀 Recommendations for Further Work

* Use more data (100+ samples)
* Advanced feature combinations (polynomial, temporal grouping)
* Scale features before Lasso/Ridge
* Test with ensemble stacking or regularized PCA

---

> Project developed step-by-step using "Prompt-e-Ghadam" method under guidance of Amir with Matilda (ChatGPT-4)
