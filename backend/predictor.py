import pandas as pd
import numpy as np

# ── Feature engineering ─────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ✅ ALWAYS convert to numeric (CRITICAL FIX)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)

    df["Tenure_to_Charges"] = df["tenure"] / (df["MonthlyCharges"] + 1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Short-Term", "Mid-Term", "Long-Term"]
    )

    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["HighSpender"] = (df["MonthlyCharges"] > 80).astype(int)

    df["ServiceCount"] = (
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )

    df["LowEngagement"] = (df["ServiceCount"] <= 2).astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["FiberUser"] = (df["InternetService"] == "Fiber optic").astype(int)

    return df


# ── MUST MATCH TRAINING ─────────────────────────────
FEATURE_COLUMNS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "Tenure_to_Charges", "AvgMonthlyCharge", "HighSpender", "ServiceCount",
    "LowEngagement", "IsMonthToMonth", "FiberUser",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No_phone_service", "MultipleLines_Yes",
    "InternetService_Fiber_optic", "InternetService_No",
    "OnlineSecurity_No_internet_service", "OnlineSecurity_Yes",
    "OnlineBackup_No_internet_service", "OnlineBackup_Yes",
    "DeviceProtection_No_internet_service", "DeviceProtection_Yes",
    "TechSupport_No_internet_service", "TechSupport_Yes",
    "StreamingTV_No_internet_service", "StreamingTV_Yes",
    "StreamingMovies_No_internet_service", "StreamingMovies_Yes",
    "Contract_One_year", "Contract_Two_year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit_card_(automatic)",
    "PaymentMethod_Electronic_check",
    "PaymentMethod_Mailed_check",
    "TenureGroup_Short-Term", "TenureGroup_Mid-Term", "TenureGroup_Long-Term"
]


# ── PREP INPUT ──────────────────────────────────────
def prepare_input(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])
    df = feature_engineering(df)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded.columns = df_encoded.columns.str.replace(" ", "_")

    df_aligned = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df_aligned


# ── SHAP (FIXED) ────────────────────────────────────
def get_shap_values(model, df_aligned: pd.DataFrame) -> dict:
    try:
        import shap

        # Extract final model from pipeline if needed
        if hasattr(model, "named_steps"):
            final_model = list(model.named_steps.values())[-1]
        else:
            final_model = model

        # Try SHAP
        explainer = shap.Explainer(final_model)
        shap_vals = explainer(df_aligned)

        values = shap_vals.values[0]

        return {
            col: float(values[i]) if i < len(values) else 0.0
            for i, col in enumerate(FEATURE_COLUMNS)
        }

    except Exception as e:
        print(f"SHAP fallback used: {e}")

        # 🔥 SMART FALLBACK (NOT ZERO ANYMORE)
        base_prob = float(model.predict_proba(df_aligned)[:, 1][0])

        shap_approx = {}
        for col in FEATURE_COLUMNS:
            df_temp = df_aligned.copy()
            df_temp[col] = 0
            new_prob = float(model.predict_proba(df_temp)[:, 1][0])
            shap_approx[col] = base_prob - new_prob

        return shap_approx


# ── PREDICT ─────────────────────────────────────────
def predict(raw_input: dict) -> dict:
    from model_loader import load_model

    model, threshold = load_model()

    df = prepare_input(raw_input)

    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)

    if prob >= 0.7:
        risk = "HIGH RISK"
    elif prob >= 0.4:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    shap_dict = get_shap_values(model, df)

    return {
        "probability": round(prob, 4),
        "prediction": pred,
        "risk_level": risk,
        "threshold_used": threshold,
        "shap_values": shap_dict
    }