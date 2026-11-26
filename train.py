"""
SalarySense AI - Training pipeline  (Lab 14 / CCA)
===================================================
Ye ek script poora ML kaam karti hai:

    python train.py

Steps and the labs they cover
  1. Data cleaning + preprocessing  ........ Lab 1
  2. Linear Regression baseline ............ Lab 2
  3. Decision Tree (single tree) ........... Lab 4
  4. Random Forest + GridSearchCV .......... Lab 5 + CLO-2 (tuning, cross-validation)
  5. 80% prediction range (residuals) ...... CLO-2 (evaluation)
  6. K-Means salary bands .................. Lab 10

Outputs
  models/salary_model.joblib   trained scikit-learn Pipeline (used by the API)
  models/metadata.json         metrics, options, charts data (used by the dashboard)
  outputs/figures/*.png        figures for the report
  outputs/model_comparison.csv model comparison table
  docs/REPORT_DRAFT.md         report draft filled with YOUR numbers
"""
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # figures save hotay hain, koi window nahi khulti
import matplotlib.pyplot as plt  # noqa: E402
import sklearn  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (mean_absolute_error, mean_squared_error,  # noqa: E402
                             r2_score, silhouette_score)
from sklearn.model_selection import (GridSearchCV, cross_val_predict,  # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "Salary_Data.csv"
MODELS = BASE / "models"
FIGS = BASE / "outputs" / "figures"
DOCS = BASE / "docs"
for d in (MODELS, FIGS, DOCS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
NUMERIC = ["age", "experience"]
CATEGORICAL = ["gender", "education", "job_title"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "salary"
MIN_SALARY = 10_000          # below this = data-entry error (e.g. 350)
MIN_TITLE_COUNT = 5          # rarer job titles are grouped into "Other"
BAND_NAMES = ["Low pay", "Medium pay", "High pay", "Top pay"]


# =========================================================================
# 1. DATA CLEANING  (Lab 1)
# =========================================================================
def clean_data(path=DATA_FILE):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = ["age", "gender", "education", "job_title", "experience", "salary"]
    rep = {"rows_raw": int(len(raw)),
           "missing_per_column": {c: int(v) for c, v in raw.isnull().sum().items()},
           "duplicates": int(raw.duplicated().sum())}

    df = raw.copy()
    # (a) rows without salary cannot be used for training
    n = len(df)
    df = df.dropna(subset=[TARGET])
    rep["dropped_missing_salary"] = n - len(df)

    # (b) inconsistent labels: "Bachelor's" vs "Bachelor's Degree", "phD" vs "PhD"
    def edu(s):
        if pd.isna(s):
            return np.nan
        s = str(s).lower()
        if "phd" in s:
            return "PhD"
        if "master" in s:
            return "Master's"
        if "bachelor" in s:
            return "Bachelor's"
        return "High School"
    df["education"] = df["education"].map(edu)
    rep["education_labels_before"] = int(raw["education"].nunique())
    rep["education_labels_after"] = int(df["education"].nunique())

    # (c) exact duplicate rows -> keep one (otherwise same row lands in train AND test)
    n = len(df)
    df = df.drop_duplicates()
    rep["dropped_duplicates"] = n - len(df)

    # (d) impossible salaries (e.g. 350) are data-entry errors
    n = len(df)
    df = df[df[TARGET] >= MIN_SALARY]
    rep["dropped_salary_below_10000"] = n - len(df)

    # (e) group rare job titles (193 titles is too sparse for 1800 rows)
    counts = df["job_title"].value_counts()
    rep["job_titles_before"] = int(df["job_title"].nunique())
    df["job_title"] = df["job_title"].where(df["job_title"].map(counts) >= MIN_TITLE_COUNT, "Other")
    df.loc[df["job_title"].isna(), "job_title"] = np.nan  # keep real missing as missing
    rep["job_titles_after"] = int(df["job_title"].nunique())

    rep["rows_clean"] = int(len(df))
    rep["missing_features_left_for_imputer"] = {c: int(df[c].isnull().sum()) for c in FEATURES}
    return df.reset_index(drop=True), rep


def as_model_frame(df):
    """Categorical columns as plain object dtype (safe for all pandas/sklearn versions)."""
    out = df[FEATURES].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out


def make_preprocessor():
    """Lab 1: impute missing -> scale numbers -> one-hot encode text."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def metrics(y_true, y_pred):
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 0),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 0),
            "R2": round(float(r2_score(y_true, y_pred)), 4)}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=130)
    plt.close()


# =========================================================================
# 2. EDA FIGURES + dashboard aggregates
# =========================================================================
def eda(df):
    plt.figure(figsize=(7, 4)); plt.hist(df[TARGET], bins=30, color="#0f8b6d")
    plt.title("Salary distribution"); plt.xlabel("Salary (USD)"); plt.ylabel("Employees")
    savefig("01_salary_distribution.png")

    plt.figure(figsize=(7, 4.5)); plt.scatter(df["experience"], df[TARGET], s=8, alpha=.4, color="#14213d")
    plt.title("Experience vs salary"); plt.xlabel("Years of experience"); plt.ylabel("Salary (USD)")
    savefig("02_experience_vs_salary.png")

    order = ["High School", "Bachelor's", "Master's", "PhD"]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot([df[df.education == e][TARGET] for e in order])
    plt.xticks(range(1, len(order) + 1), order)   # works on every matplotlib version
    plt.title("Salary by education"); plt.ylabel("Salary (USD)")
    savefig("03_salary_by_education.png")

    top = df.groupby("job_title")[TARGET].agg(["mean", "count"]).query("count >= 10").sort_values("mean").tail(10)
    plt.figure(figsize=(8, 4.5)); plt.barh(top.index, top["mean"], color="#0f8b6d")
    plt.title("Highest paying job titles (>=10 employees)"); plt.xlabel("Average salary (USD)")
    savefig("04_top_job_titles.png")

    corr = df[["age", "experience", TARGET]].corr()
    plt.figure(figsize=(5, 4)); plt.imshow(corr, cmap="Greens", vmin=0, vmax=1)
    plt.xticks(range(3), corr.columns); plt.yticks(range(3), corr.columns)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Correlation (numeric columns)")
    savefig("05_correlation.png")


def analytics_aggregates(df):
    """Small JSON-friendly summaries that the dashboard draws as charts."""
    hist, edges = np.histogram(df[TARGET], bins=12)
    bins = [{"label": f"{int(edges[i] / 1000)}-{int(edges[i + 1] / 1000)}k", "count": int(hist[i])} for i in range(12)]

    exp_buckets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 15), (16, 20), (21, 40)]
    by_exp = []
    for lo, hi in exp_buckets:
        s = df[(df.experience >= lo) & (df.experience <= hi)][TARGET]
        if len(s):
            by_exp.append({"label": f"{lo}-{hi} yrs", "value": round(float(s.mean())), "n": int(len(s))})

    order = ["High School", "Bachelor's", "Master's", "PhD"]
    by_edu = [{"label": e, "value": round(float(df[df.education == e][TARGET].mean())),
               "n": int((df.education == e).sum())} for e in order]

    g = df.groupby("job_title")[TARGET].agg(["mean", "count"]).query("count >= 10").sort_values("mean", ascending=False).head(10)
    top_titles = [{"label": i, "value": round(float(r["mean"])), "n": int(r["count"])} for i, r in g.iterrows()]

    by_gender = [{"label": gname, "value": round(float(s.mean())), "n": int(len(s))}
                 for gname, s in df.groupby("gender")[TARGET] if len(s) >= 30]

    return {"kpi": {"rows": int(len(df)), "mean_salary": round(float(df[TARGET].mean())),
                    "median_salary": round(float(df[TARGET].median())),
                    "min_salary": round(float(df[TARGET].min())), "max_salary": round(float(df[TARGET].max())),
                    "job_titles": int(df.job_title.nunique())},
            "salary_histogram": bins, "salary_by_experience": by_exp, "salary_by_education": by_edu,
            "top_job_titles": top_titles, "salary_by_gender": by_gender}


# =========================================================================
# 3. K-MEANS SALARY BANDS  (Lab 10)
# =========================================================================
def kmeans_bands(df):
    X = df[["experience", TARGET]].to_numpy(float)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    ks = list(range(2, 9))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        inertia.append(float(km.inertia_)); sil.append(float(silhouette_score(Z, km.labels_)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
    ax[1].plot(ks, sil, "o-", color="green"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    savefig("10_kmeans_elbow_silhouette.png")

    K = 4
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
    centers = scaler.inverse_transform(km.cluster_centers_)
    order = np.argsort(centers[:, 1])                       # lowest salary first
    rank = {int(c): i for i, c in enumerate(order)}         # cluster id -> band index
    labels = np.array([rank[int(c)] for c in km.labels_])

    bands = []
    for i in range(K):
        s = df[labels == i]
        bands.append({"name": BAND_NAMES[i], "count": int(len(s)),
                      "mean_salary": round(float(s[TARGET].mean())), "min_salary": round(float(s[TARGET].min())),
                      "max_salary": round(float(s[TARGET].max())), "mean_experience": round(float(s.experience.mean()), 1)})
    plt.figure(figsize=(7.5, 5))
    for i in range(K):
        s = df[labels == i]
        plt.scatter(s.experience, s[TARGET], s=9, alpha=.5, label=BAND_NAMES[i])
    plt.legend(); plt.xlabel("Years of experience"); plt.ylabel("Salary (USD)"); plt.title("K-Means salary bands (k=4)")
    savefig("11_kmeans_clusters.png")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(350, len(df)), replace=False)
    sample = [{"x": float(df.experience.iloc[i]), "y": float(df[TARGET].iloc[i]), "band": int(labels[i])} for i in idx]
    return {"k": K, "names": BAND_NAMES, "centers": centers[order].round(2).tolist(),   # [experience, salary]
            "mean": scaler.mean_.round(4).tolist(), "scale": scaler.scale_.round(4).tolist(),
            "bands": bands, "elbow": {"k": ks, "inertia": [round(v, 2) for v in inertia],
                                      "silhouette": [round(v, 4) for v in sil]},
            "sample": sample}


# =========================================================================
# 4. MAIN
# =========================================================================
def main():
    print("[1/6] Cleaning data (Lab 1)")
    df, cleaning = clean_data()
    print("      ", cleaning)
    eda(df)

    X, y = as_model_frame(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"      train={len(X_train)}  test={len(X_test)}")

    print("[2/6] Comparing models (Lab 2, 4, 5)")
    candidates = {
        "Linear Regression (Lab 2)": LinearRegression(),
        "Decision Tree, depth 8 (Lab 4)": DecisionTreeRegressor(max_depth=8, random_state=SEED),
        "Random Forest, default (Lab 5)": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    rows, fitted = [], {}
    for name, reg in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", reg)]).fit(X_train, y_train)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        row = {"model": name, **metrics(y_test, pipe.predict(X_test)),
               "cv_r2_mean": round(float(cv.mean()), 4), "cv_r2_std": round(float(cv.std()), 4)}
        rows.append(row); fitted[name] = pipe
        print("      ", row)

    print("[3/6] Tuning Random Forest with GridSearchCV (CLO-2)")
    grid = {"model__n_estimators": [200, 400], "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 1.0]}
    base = Pipeline([("prep", make_preprocessor()),
                     ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    gs = GridSearchCV(base, grid, cv=3, scoring="r2", n_jobs=1).fit(X_train, y_train)
    best = gs.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    cv = cross_val_score(best, X_train, y_train, cv=5, scoring="r2")
    final = {"model": "Random Forest, TUNED (GridSearchCV)", **metrics(y_test, best.predict(X_test)),
             "cv_r2_mean": round(float(cv.mean()), 4), "cv_r2_std": round(float(cv.std()), 4)}
    rows.append(final)
    print("       best params:", best_params); print("      ", final)
    pd.DataFrame(rows).to_csv(BASE / "outputs" / "model_comparison.csv", index=False)

    pred = best.predict(X_test)
    plt.figure(figsize=(8, 3.8)); names = [r["model"] for r in rows]
    plt.barh(names, [r["R2"] for r in rows], color="#0f8b6d"); plt.xlim(0, 1); plt.xlabel("R2 on test set")
    for i, r in enumerate(rows):
        plt.text(r["R2"] + .01, i, f'{r["R2"]:.3f}', va="center")
    plt.title("Model comparison"); savefig("06_model_comparison.png")

    plt.figure(figsize=(5.5, 5)); plt.scatter(y_test, pred, s=9, alpha=.5, color="#14213d")
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lim, lim, "r--"); plt.xlabel("Actual salary"); plt.ylabel("Predicted salary")
    plt.title("Actual vs predicted (test set)"); savefig("07_actual_vs_predicted.png")

    plt.figure(figsize=(6.5, 4)); plt.scatter(pred, y_test - pred, s=9, alpha=.5, color="#0f8b6d")
    plt.axhline(0, color="r"); plt.xlabel("Predicted salary"); plt.ylabel("Residual (actual - predicted)")
    plt.title("Residual plot"); savefig("08_residuals.png")

    print("[4/6] Feature importance + 80% prediction range")
    pi = permutation_importance(best, X_test, y_test, n_repeats=15, random_state=SEED, scoring="r2")
    raw_imp = np.clip(pi.importances_mean, 0, None)
    imp = {c: round(float(v / raw_imp.sum()), 4) for c, v in zip(FEATURES, raw_imp)}
    imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    plt.figure(figsize=(6.5, 3.8)); plt.barh(list(imp)[::-1], list(imp.values())[::-1], color="#0f8b6d")
    plt.xlabel("Share of importance"); plt.title("Feature importance (permutation, test set)")
    savefig("09_feature_importance.png")

    # Range: relative error (actual-pred)/pred on out-of-fold TRAIN predictions -> 10th/90th percentile.
    # Then we CHECK on the untouched test set how often reality really falls inside the range.
    oof = cross_val_predict(best, X_train, y_train, cv=5)
    rel = (y_train.to_numpy() - oof) / oof
    q10, q90 = (float(v) for v in np.percentile(rel, [10, 90]))
    low, high = pred * (1 + q10), pred * (1 + q90)
    coverage = float(np.mean((y_test.to_numpy() >= low) & (y_test.to_numpy() <= high)))
    interval = {"level": 0.8, "rel_q10": round(q10, 4), "rel_q90": round(q90, 4),
                "test_coverage": round(coverage, 3)}
    print("      ", interval)

    print("[5/6] Fairness check by gender (ethics)")
    t = pd.DataFrame({"gender": X_test["gender"].to_numpy(), "actual": y_test.to_numpy(), "pred": pred})
    t["err_pct"] = (t.actual - t.pred) / t.pred * 100
    fair = []
    for g, s in t.groupby("gender"):
        if len(s) >= 15:
            fair.append({"gender": g, "n": int(len(s)), "mean_actual": round(float(s.actual.mean())),
                         "mean_pred": round(float(s.pred.mean())), "mean_error_pct": round(float(s.err_pct.mean()), 2)})
    pd.DataFrame(fair).to_csv(BASE / "outputs" / "fairness_by_gender.csv", index=False)

    print("[6/6] K-Means salary bands (Lab 10)")
    km = kmeans_bands(df)
    for b in km["bands"]:
        print("      ", b)

    # ---- save everything ------------------------------------------------
    joblib.dump(best, MODELS / "salary_model.joblib")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(250, len(y_test)), replace=False)
    scatter = [{"actual": float(y_test.iloc[i]), "pred": float(pred[i])} for i in idx]
    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "sklearn_version": sklearn.__version__,
        "dataset": {"source": "Kaggle-origin 'Salary Data' (Age, Gender, Education Level, Job Title, Years of Experience, Salary), GitHub mirror",
                    "currency": "USD", "cleaning": cleaning},
        "options": {"genders": df.gender.value_counts().index.tolist(),   # most common first
                    "educations": ["High School", "Bachelor's", "Master's", "PhD"],
                    "job_titles": sorted(t for t in df.job_title.dropna().unique() if t != "Other") + ["Other"],
                    "age": [int(df.age.min()), int(df.age.max())], "experience": [0, int(df.experience.max())]},
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "test_size": 0.2, "seed": SEED},
        "comparison": rows, "best_params": best_params, "final": final, "interval": interval,
        "importance": imp, "fairness": fair, "kmeans": km, "analytics": analytics_aggregates(df), "scatter": scatter,
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    from report_writer import write_report
    write_report(meta, DOCS / "REPORT_DRAFT.md")
    print("\nDONE. Model:", MODELS / "salary_model.joblib")
    print("      Figures:", FIGS)
    print("      Report draft:", DOCS / "REPORT_DRAFT.md")


if __name__ == "__main__":
    main()
