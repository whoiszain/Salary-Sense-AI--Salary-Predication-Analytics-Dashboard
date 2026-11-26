<div align="center">

# 💼 SalarySense AI

### Salary Prediction & Analytics Dashboard

Predict a yearly salary from age, experience, gender, education and job title, with an **80% prediction range**, a **K-Means salary band**, data insights, model evaluation and saved prediction history.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

</div>

---

## 📌 Overview

**SalarySense AI** is an end-to-end machine learning project: it cleans a real-world salary dataset, compares four models, tunes a Random Forest, and serves it through a **FastAPI** backend with a clean, single-page **dashboard**.

It is **Python-only**. The dashboard is one self-contained HTML file served by FastAPI, so there is **no Node.js, npm or build step** to deal with.

> Built as the *Machine Learning (AI-3202), Lab 14 – Complex Computing Activity* project.

## ✨ Features

- **Salary prediction** (USD per year) from five inputs: age, years of experience, gender, education level and job title
- **80% prediction range** (lower and upper bound) built from the model's residuals, so you see uncertainty and not just one number
- **K-Means salary band** (Low / Medium / High / Top pay) for every prediction
- **Data insights tab**: salary distribution, experience vs salary, salary by education, top job titles and a "what cleaning changed" table
- **Model tab**: metrics, model comparison, feature importance, actual vs predicted, and a fairness check by gender
- **Prediction history** stored in SQLite (view and clear from the UI)
- **Input validation**: unknown categories, impossible ages and unrealistic experience (e.g. started working at age 7) are rejected with clear errors
- **Automated API tests** with `pytest`

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Single-page dashboard in plain HTML, CSS and JavaScript (`app/static/index.html`), served by FastAPI. Five tabs: Predict, Data insights, Model, History, How it works |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic v2](https://docs.pydantic.dev/) |
| **Machine Learning** | scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomForestRegressor`, `LinearRegression`, `DecisionTreeRegressor`, `KMeans`, `GridSearchCV`), joblib |
| **Data** | pandas, NumPy, Matplotlib |
| **Database** | SQLite (prediction history) |
| **Testing** | pytest, httpx / FastAPI `TestClient` |

## 🧠 How It Works

```
Browser form ──▶ POST /api/predict ──▶ FastAPI (app/main.py)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
   scikit-learn Pipeline                              Validation (Pydantic + rules)
   Imputer → Scaler + OneHotEncoder → Random Forest
                    │
                    ├─▶ salary
                    ├─▶ 80% range  = salary × (1 + q10) … salary × (1 + q90)
                    ├─▶ salary band = nearest K-Means centre
                    └─▶ saved to SQLite history ──▶ result shown in the browser
```

### ML pipeline (`train.py`)

1. **Cleaning and preprocessing**: removes rows with missing salary, fixes inconsistent labels (e.g. `Bachelor's Degree` → `Bachelor's`, `phD` → `PhD`), drops duplicates, imputes missing features, groups rare job titles (fewer than 5 records) into `Other`, one-hot encodes categories, scales numbers, and splits 80/20
2. **Baselines**: Linear Regression and a depth-8 Decision Tree
3. **Random Forest** with default settings, then tuned with **GridSearchCV**
4. **Evaluation**: MAE, RMSE, R² on the test set plus 5-fold cross-validation
5. **Prediction range**: the 10th and 90th percentile of relative residuals give an 80% interval
6. **K-Means (k = 4)** on experience and salary, with elbow and silhouette analysis, to create the salary bands
7. **Fairness check**: mean prediction error by gender
8. Saves the model, metadata, figures and a filled-in report draft

## 📊 Results

| Model | MAE | RMSE | R² (test) | R² (5-fold CV) |
|---|---:|---:|---:|---:|
| Linear Regression | $16,573 | $22,432 | 0.814 | 0.841 ± 0.014 |
| Decision Tree (depth 8) | $15,595 | $23,039 | 0.804 | 0.828 ± 0.023 |
| Random Forest (default) | $10,214 | $15,968 | 0.906 | 0.900 ± 0.017 |
| **Random Forest (tuned, deployed)** | **$10,914** | **$16,589** | **0.898** | **0.899 ± 0.009** |

The tuned Random Forest is deployed because it has the most stable cross-validation score (lowest variance across folds).

**Fairness check (mean prediction error by gender, test set)**

| Gender | Samples | Mean actual | Mean predicted | Mean error |
|---|---:|---:|---:|---:|
| Female | 161 | $106,503 | $106,815 | −1.34% |
| Male | 193 | $120,249 | $118,332 | +1.00% |

<details>
<summary><b>📈 More figures</b></summary>

| | |
|---|---|
| ![Model comparison](outputs/figures/06_model_comparison.png) | ![Actual vs predicted](outputs/figures/07_actual_vs_predicted.png) |
| ![Feature importance](outputs/figures/09_feature_importance.png) | ![K-Means clusters](outputs/figures/11_kmeans_clusters.png) |

</details>

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)). On Windows, tick *"Add Python to PATH"* during installation.

### Option 1: One-click (Windows)

Double-click **`run.bat`**. It creates a virtual environment, installs the packages, trains the model (first run only, about 2–5 minutes) and starts the server. Then open **http://localhost:8000**.

### Option 2: Manual setup

```bash
# 1. Clone the repository
git clone https://github.com/syedumerfarooq24-ops/SalarySense-AI-v2.git
cd SalarySense-AI-v2

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Train the model (creates models/, outputs/figures/ and docs/REPORT_DRAFT.md)
python train.py

# 5. Start the app
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

> The model must be trained (step 4) before the API can make predictions. If it is missing, the API returns `503 Model not trained yet`.

### Run the tests

```bash
python -m pytest -q
```

## 🔌 API Reference

Interactive docs are available at **http://localhost:8000/docs** (Swagger UI).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Is the model loaded? |
| `GET` | `/api/options` | Valid genders, education levels, job titles and age/experience ranges |
| `POST` | `/api/predict` | Predict salary, 80% range and salary band (saved to history) |
| `GET` | `/api/history?limit=20` | Latest predictions |
| `DELETE` | `/api/history` | Clear history |
| `GET` | `/api/analytics` | Data insights for the charts |
| `GET` | `/api/model` | Metrics, model comparison, feature importance, fairness, K-Means bands |

**Example request**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "experience": 5, "gender": "Male", "education": "Bachelor'"'"'s", "job_title": "Software Engineer"}'
```

**Example response** (values depend on your trained model)

```json
{
  "id": "a1b2c3d4e5",
  "predicted_salary": 95000,
  "lower": 82000,
  "upper": 108000,
  "band": "Medium pay",
  "interval_level": 0.8,
  "model_r2": 0.898,
  "model_mae": 10914,
  "currency": "USD"
}
```

## 📁 Project Structure

```
SalarySense-AI/
├── app/
│   ├── main.py              # FastAPI app: API routes, validation, SQLite history
│   └── static/
│       └── index.html       # Dashboard (Predict, Insights, Model, History, How it works)
├── data/
│   └── Salary_Data.csv      # Dataset
├── docs/
│   └── REPORT_DRAFT.md      # Report draft generated by train.py
├── outputs/
│   ├── figures/             # Charts produced during training
│   ├── model_comparison.csv
│   └── fairness_by_gender.csv
├── tests/
│   └── test_app.py          # API tests
├── train.py                 # Full ML pipeline
├── report_writer.py         # Fills the report draft with real numbers
├── requirements.txt
└── run.bat                  # One-click start (Windows)
```

## 🗂️ Dataset

Kaggle-origin **"Salary Data"** with the columns *Age, Gender, Education Level, Job Title, Years of Experience, Salary*. It has **6,704 raw rows**, which become **1,783 unique, clean rows** after removing missing values, label inconsistencies and duplicates. Salaries are in **USD**.

## ⚠️ Limitations

- The dataset is small after de-duplication, and rare job titles are grouped into `Other`.
- There is no country or company column, so predictions are **not** specific to any country or employer (including Pakistan).
- Age and experience are strongly correlated, which makes their individual effects hard to separate.
- Gender is a model input. The fairness table above is a basic check, not proof that the model is free of bias.
- Predictions are estimates for learning and demonstration, **not** a basis for real salary or hiring decisions.

## 🔭 Future Improvements

- Add country and industry features with a larger dataset
- Try gradient boosting (XGBoost / LightGBM) and compare against the Random Forest
- Add explainability for each prediction (e.g. SHAP)
- Dockerize the app and deploy it online
- Add CSV upload for batch predictions

## 👤 Author

**Syed Umer Ahmed** — BS Artificial Intelligence student, Karachi, Pakistan

[![GitHub](https://img.shields.io/badge/GitHub-syedumerfarooq24--ops-181717?logo=github)](https://github.com/syedumerfarooq24-ops)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed%20Umer%20Ahmed-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syed-umer-ahmed-b47226406/)

---

<div align="center">

If you found this project useful, consider giving it a ⭐

</div>
