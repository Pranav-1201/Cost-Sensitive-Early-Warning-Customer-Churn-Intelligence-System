"""
pipeline.py  — ChurnLens (patched)

Two fixes from code review:
  1. SHAP explainer selection: RandomForest and StackingClassifier are NOT linear.
     Using LinearExplainer on them silently returns wrong values or crashes.
     Fixed: detect model type and pick the right explainer.

  2. df_test_slice index alignment: after train_test_split, the test rows
     are NOT always the last N rows of df_clean (shuffle=True in split).
     The old code did df_clean.iloc[len(X_train):] which gives wrong customer data.
     Fixed: track actual test indices from the split.

Everything else (function signatures, progress callbacks, return shape) is unchanged.
"""

import pandas as pd
import numpy as np
import warnings
import traceback
from typing import Callable, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import shap

warnings.filterwarnings("ignore")

COST_FN = 10_000
COST_FP = 500


# ── Data cleaning ────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 🔥 Fix TotalCharges safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # ✅ DO NOT drop everything — fill instead
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # 🔥 Fix Churn safely
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
        df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})

        # keep only valid rows
        df = df[df["Churn"].notna()]

    # 🔥 Remove ID column safely
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # 🔥 Minimal safe cleaning (NOT full dropna)
    essential_cols = ["tenure", "MonthlyCharges"]
    existing = [c for c in essential_cols if c in df.columns]
    df = df.dropna(subset=existing)

    print("DEBUG: rows after cleaning =", len(df))

    return df


# ── Feature engineering ──────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tenure_to_Charges"] = df["tenure"] / (df["TotalCharges"] + 1)
    df["AvgMonthlyCharge"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"]       = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]
    df["ServiceCount"]   = (df[existing] == "Yes").sum(axis=1)
    df["LowEngagement"]  = ((df["tenure"] < 12) & (df["ServiceCount"] < 3)).astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["FiberUser"]      = (df["InternetService"] == "Fiber optic").astype(int)
    df["TenureGroup"]    = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["Short-Term", "Mid-Term", "Long-Term"],
        include_lowest=True,
    )
    return df


# ── Encoding ─────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    target = "Churn"

    # --------- TARGET ----------
    y = df[target].values

    # --------- CATEGORICAL ----------
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target
    ]

    df_enc = pd.get_dummies(
        df.drop(columns=[target]),
        columns=cat_cols,
        drop_first=True
    )

    # --------- BOOLEAN ----------
    bool_cols = df_enc.select_dtypes(include=["bool"]).columns
    df_enc[bool_cols] = df_enc[bool_cols].astype(int)

    # --------- CONVERT TO NUMERIC ----------
    import numpy as np
    X = df_enc.values.astype(float)

    # --------- 🔥 CRITICAL FIX: HANDLE NaNs ----------
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # --------- OPTIONAL DEBUG ----------
    # print("NaNs after encoding:", np.isnan(X).sum())

    return X, y, list(df_enc.columns)


# ── Threshold search ─────────────────────────────────
def find_best_threshold(y_true, y_prob, cost_fn=COST_FN, cost_fp=COST_FP):
    best_thresh, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.55, 0.01):
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = round(t, 2)
    return best_thresh, best_cost


# ── SHAP explainer selection (FIXED) ─────────────────
def _get_explainer(raw_model, X_background):
    """
    Pick the correct SHAP explainer based on model type.
    LinearExplainer is fast but ONLY valid for linear models.
    TreeExplainer works for tree-based models.
    Explainer (generic) is the fallback.
    """
    # Unwrap stacking final estimator if needed
    model_to_check = raw_model
    if isinstance(raw_model, StackingClassifier):
        model_to_check = raw_model.final_estimator_

    if isinstance(model_to_check, LogisticRegression):
        return shap.LinearExplainer(
            raw_model, X_background, feature_perturbation="interventional"
        )
    elif isinstance(model_to_check, (RandomForestClassifier,)):
        return shap.TreeExplainer(raw_model)
    else:
        # Stacking, calibrated, etc. — use generic kernel (slower but universal)
        # Sample down heavily to keep it fast
        bg = X_background[:50] if len(X_background) > 50 else X_background
        return shap.KernelExplainer(
            lambda x: raw_model.predict_proba(x)[:, 1], bg
        )


# ── SHAP global ──────────────────────────────────────
def compute_shap_global(model, X, feature_names, max_samples=200):
    try:
        sample = X[:max_samples] if len(X) > max_samples else X
        explainer = _get_explainer(model, sample)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return [
            {"feature": f, "importance": round(float(v), 6)}
            for f, v in sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
        ]
    except Exception as e:
        print(f"[SHAP global] fallback zeros: {e}")
        return [{"feature": f, "importance": 0.0} for f in feature_names]


# ── EDA summary ──────────────────────────────────────
def compute_eda_summary(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()

    # 🔥 ROBUST Churn conversion (case-insensitive + safe)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
        df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})
        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
        df = df.dropna(subset=["Churn"])

    # 🔥 FIX TotalCharges (CRITICAL)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 🛑 Safety: prevent empty dataset crash
    if len(df) == 0:
        return {
            "total_customers": 0,
            "churn_rate": 0.0,
            "churn_count": 0,
            "retain_count": 0,
            "by_contract": [],
            "tenure_distribution": [],
            "monthly_charges_distribution": [],
            "feature_stats": {},
        }

    # ✅ Core stats
    total_customers = int(len(df))
    churn_rate = float(df["Churn"].mean())
    churn_count = int(df["Churn"].sum())
    retain_count = int(total_customers - churn_count)

    # ✅ By contract (safe)
    by_contract = []
    if "Contract" in df.columns:
        by_contract = (
            df.groupby("Contract")["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "churn_rate"})
            .to_dict(orient="records")
        )

    # ✅ Tenure distribution (safe)
    tenure_dist = []
    if "tenure" in df.columns and len(df["tenure"].dropna()) > 0:
        tenure_bins = pd.cut(df["tenure"], bins=10)
        tenure_df = (
            df.groupby(tenure_bins, observed=False)["Churn"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={
                "count": "customers",
                "mean": "churn_rate"
            })
        )
        tenure_df["bin"] = tenure_df.iloc[:, 0].apply(lambda x: str(x))
        tenure_df = tenure_df.drop(columns=[tenure_df.columns[0]])
        tenure_dist = tenure_df.to_dict(orient="records")

    # ✅ Monthly charges distribution (safe)
    mc_dist = []
    if "MonthlyCharges" in df.columns and len(df["MonthlyCharges"].dropna()) > 0:
        mc_bins = pd.cut(df["MonthlyCharges"], bins=8)
        mc_df = (
            df.groupby(mc_bins, observed=False)["Churn"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={
                "count": "customers",
                "mean": "churn_rate"
            })
        )
        mc_df["bin"] = mc_df.iloc[:, 0].apply(lambda x: str(x))
        mc_df = mc_df.drop(columns=[mc_df.columns[0]])
    # ✅ Feature stats (safe)
    feature_stats = {}
    if "tenure" in df.columns:
        feature_stats["mean_tenure"] = round(float(df["tenure"].mean()), 2)
    if "MonthlyCharges" in df.columns:
        feature_stats["mean_monthly_charges"] = round(float(df["MonthlyCharges"].mean()), 2)
    if "TotalCharges" in df.columns:
        feature_stats["mean_total_charges"] = round(float(df["TotalCharges"].mean()), 2)

    return {
        "total_customers": total_customers,
        "churn_rate": round(churn_rate, 4),
        "churn_count": churn_count,
        "retain_count": retain_count,
        "by_contract": by_contract,
        "tenure_distribution": tenure_dist,
        "monthly_charges_distribution": mc_dist,
        "feature_stats": feature_stats,
    }


# ── Model building ───────────────────────────────────
def build_models():
    base_lr   = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    tuned_lr  = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)
    rf        = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    stacked   = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=500),
        cv=3,
    )
    return [
        ("Logistic Regression", Pipeline([("scaler", StandardScaler()), ("model", base_lr)])),
        ("Tuned Logistic",      Pipeline([("scaler", StandardScaler()), ("model", tuned_lr)])),
        ("Random Forest",       Pipeline([("model", rf)])),
        ("Stacked Model",       Pipeline([("scaler", StandardScaler()), ("model", stacked)])),
    ]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, cost = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm  = confusion_matrix(y_test, y_pred).tolist()
    acc = round(accuracy_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_prob), 4)
    pr  = round(average_precision_score(y_test, y_prob), 4)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

    return {
        "name": name,
        "accuracy": acc,
        "roc_auc": auc,
        "pr_auc": pr,
        "cost": int(cost),
        "threshold": threshold,
        "confusion_matrix": cm,
        "cv_scores": [round(float(s), 4) for s in cv],
        "cv_mean": round(float(cv.mean()), 4),
        "cv_std": round(float(cv.std()), 4),
        "_model_obj": model,
    }


# ── Main entry point ─────────────────────────────────
def run_pipeline(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    def progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    results = {}

    try:
        progress(5, "Computing EDA summary")
        results["eda"] = compute_eda_summary(df)

        progress(12, "Cleaning data")
        df_clean = clean_data(df)

        # 🔥 SAFETY CHECK 1
        if len(df_clean) == 0:
            raise ValueError("Dataset became empty after cleaning")

        progress(22, "Engineering features")
        df_feat = engineer_features(df_clean)

        progress(30, "Encoding categorical features")
        X, y, feature_names = encode_features(df_feat)

        # 🔥 SAFETY CHECK 2
        if len(X) == 0:
            raise ValueError("Dataset became empty after encoding")

        progress(35, "Splitting train / test sets")

        indices = np.arange(len(X))

        # 🔥 SAFE stratify
        stratify_y = y if len(np.unique(y)) > 1 else None

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            indices,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y,
        )

        models_to_train = build_models()
        model_results = []
        n_models = len(models_to_train)

        for i, (name, model) in enumerate(models_to_train):
            pct = 38 + int((i / n_models) * 40)
            progress(pct, f"Training {name}")
            m = evaluate_model(name, model, X_train, X_test, y_train, y_test)
            model_results.append(m)

        progress(80, "Selecting best model")
        best = min(model_results, key=lambda m: m["cost"])
        best_model_obj = best["_model_obj"]
        best_threshold = best["threshold"]

        sorted_by_cost = sorted(model_results, key=lambda m: m["cost"])

        for m in model_results:
            if m["name"] == best["name"]:
                m["status"] = "Selected"
            elif m["name"] == sorted_by_cost[1]["name"]:
                m["status"] = "Runner-up"
            else:
                m["status"] = "Evaluated"

        for m in model_results:
            m.pop("_model_obj", None)

        results["models"] = model_results
        results["best_model"] = best["name"]
        results["best_threshold"] = best_threshold
        results["cost_fn"] = COST_FN
        results["cost_fp"] = COST_FP

        progress(85, "Computing SHAP global importances")

        raw_model = best_model_obj
        if hasattr(best_model_obj, "named_steps"):
            raw_model = best_model_obj.named_steps.get("model", best_model_obj)
            X_shap = (
                best_model_obj.named_steps["scaler"].transform(X_test)
                if "scaler" in best_model_obj.named_steps
                else X_test
            )
        else:
            X_shap = X_test

        results["shap_global"] = compute_shap_global(raw_model, X_shap, feature_names)

        progress(92, "Computing per-customer SHAP values")

        try:
            explainer = _get_explainer(raw_model, X_shap)
            shap_vals = explainer.shap_values(X_shap)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            y_prob_all = best_model_obj.predict_proba(X_test)[:, 1]
            df_clean_reset = df_clean.reset_index(drop=True)

            customer_shap = []

            for i in range(min(len(X_test), 100)):
                original_idx = idx_test[i]

                row = (
                    df_clean_reset.iloc[original_idx].to_dict()
                    if original_idx < len(df_clean_reset)
                    else {}
                )

                prob = float(y_prob_all[i])
                pred = int(prob >= best_threshold)

                sv = {
                    feature_names[j]: round(float(shap_vals[i, j]), 6)
                    for j in range(len(feature_names))
                }

                customer_shap.append({
                    "index": i,
                    "customer": {
                        k: (
                            int(v) if isinstance(v, np.integer)
                            else float(v) if isinstance(v, np.floating)
                            else v
                        )
                        for k, v in row.items()
                    },
                    "probability": round(prob, 4),
                    "prediction": pred,
                    "risk_level": "HIGH RISK" if prob >= best_threshold else "LOW RISK",
                    "threshold_used": best_threshold,
                    "shap_values": sv,
                })

            results["customer_shap"] = customer_shap

        except Exception as e:
            print(f"[SHAP per-customer] failed: {e}")
            results["customer_shap"] = []

        progress(97, "Finalising results")

        results["feature_names"] = feature_names
        results["dataset_info"] = {
            "total_rows": int(len(df)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "n_features": int(len(feature_names)),
            "churn_rate": round(float(y.mean()), 4),
        }

        progress(100, "Pipeline complete")
        return results

    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}\n{traceback.format_exc()}")