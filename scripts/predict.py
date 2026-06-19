"""predict.py — Phase 6: Predictions + single-row predict for FastAPI.

The seller provides item_title and item_description.
These are processed through the same TF-IDF vectorizer used during training
to produce 100 numeric features. Combined with structured features this gives
significantly better predictions than structured features alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from scripts.utils import load_object, report_metrics, OUTPUT_DIR
from scripts.preprocessing import (load_raw, step1_drop_duplicates, step2_drop_leakage,
    step3_keep_text, step4_fix_formatting, step5_handle_missing,
    step6_encode, step7_handle_outliers, step8_log_transform)
from scripts.feature_engineering import engineer_features

TARGET = "final_selling_price"

# ── Prediction interval config (80%) ──────────────────────────────────────────
# RMSLE is updated after every training run. z=1.282 → 80% two-sided interval.
# Interval computed in log-space then converted back to EGP so that it
# correctly captures multiplicative price uncertainty.
_RMSLE = 0.2699     # ← updated after training
_Z_80  = 1.282
_LOWER = float(np.exp(-_Z_80 * _RMSLE))   # ≈ 0.709
_UPPER = float(np.exp(+_Z_80 * _RMSLE))   # ≈ 1.411


# def load_pipeline():
#     return (
#         load_object("encoders.pkl"),
#         load_object("outlier_caps.pkl"),
#         load_object("log_cols.pkl"),
#         load_object("scaler.pkl"),
#         load_object("selected_features.pkl"),
#         load_object("best_model.pkl"),
#         load_object("tfidf_vectorizer.pkl"),
#     )

def load_pipeline():
    try:
        encoders = load_object("encoders.pkl")
        outlier_caps = load_object("outlier_caps.pkl")
        log_cols = load_object("log_cols.pkl")
        scaler = load_object("scaler.pkl")
        selected_features = load_object("selected_features.pkl")
        bundle = load_object("best_model.pkl")
        tfidf_vectorizer = load_object("tfidf_vectorizer.pkl")
        
        # Critical verification check for Railway logs
        if hasattr(tfidf_vectorizer, 'idf_') is False:
            raise ValueError("The loaded tfidf_vectorizer.pkl is missing fitted attributes ('idf_').")
            
        return (
            encoders,
            outlier_caps,
            log_cols,
            scaler,
            selected_features,
            bundle,
            tfidf_vectorizer,
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading pipeline elements: {str(e)}")
        raise e


def _raw_predictions(bundle, X):
    """Return individual and ensemble predictions in EGP."""
    rf, lgbm, xgb = bundle["rf"], bundle["lgbm"], bundle["xgb"]
    rf_p   = np.maximum(np.expm1(rf.predict(X)),   0)
    lgbm_p = np.maximum(np.expm1(lgbm.predict(X)), 0)
    xgb_p  = np.maximum(np.expm1(xgb.predict(X)),  0)
    ensemble = (rf_p + lgbm_p + xgb_p) / 3
    return ensemble, rf_p, lgbm_p, xgb_p


def _price_range(point: float, rf_p: float, lgbm_p: float, xgb_p: float):
    """
    80% prediction interval, widened by model disagreement.

    - Base interval from calibrated RMSLE in log-space.
    - Spread penalty: coefficient of variation across 3 models, capped at ±10%.
    - Both bounds rounded to nearest 50 EGP.
    - Guaranteed: low < point < high.
    """
    low  = point * _LOWER
    high = point * _UPPER

    preds = np.array([rf_p, lgbm_p, xgb_p])
    cv    = preds.std() / (preds.mean() + 1e-9)
    extra = min(cv * 0.20, 0.10)
    low   = low  * (1.0 - extra)
    high  = high * (1.0 + extra)

    low  = max(low,  0.0)
    high = max(high, point)

    low  = round(low  / 50) * 50
    high = round(high / 50) * 50

    return float(low), float(high)


def run_predictions():
    print("\n=== PHASE 6 — FINAL PREDICTIONS ===")
    encoders, caps, log_cols, scaler, selected, bundle, vectorizer = load_pipeline()
    print(f"  Model: {load_object('best_model_name.pkl')}")

    df = load_raw()
    df = step1_drop_duplicates(df); df = step2_drop_leakage(df)
    df = step3_keep_text(df);       df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, _ = step6_encode(df, fit=False, encoders=encoders)
    df, _ = step7_handle_outliers(df, fit=False, caps=caps)
    df, _ = step8_log_transform(df, log_cols=log_cols)
    df, _ = engineer_features(df, fit=False, vectorizer=vectorizer)

    X_all = df.drop(columns=[TARGET]); y_all = df[TARGET]
    _, X_te_r, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_te_s = pd.DataFrame(scaler.transform(X_te_r), columns=X_te_r.columns, index=X_te_r.index)
    X_test = X_te_s[selected]

    y_pred = _raw_predictions(bundle, X_test)[0]

    print("\n  Performance:")
    metrics = report_metrics(y_test, y_pred, "Ensemble_Test")
    pct     = np.abs(y_test.values - y_pred) / y_test.values * 100
    print(f"  Median % error : {np.median(pct):.1f}%")
    print(f"  Within 20%     : {(pct<=20).mean()*100:.1f}% of test samples")

    comp = pd.DataFrame({
        "True_Value": y_test.values,
        "Predicted":  y_pred.round(0).astype(int),
        "Abs_Error":  np.abs(y_test.values - y_pred).round(0).astype(int),
        "Pct_Error":  pct.round(2),
    })
    comp.to_csv(os.path.join(OUTPUT_DIR, "predictions_comparison.csv"), index=False)
    return y_test, y_pred, metrics, comp


def predict_single_with_interval(input_dict: dict) -> dict:
    """
    Main prediction function called by FastAPI.

    input_dict must include item_title and item_description (for TF-IDF)
    plus all 13 structured fields.

    Returns:
        predicted_price   — point estimate in EGP
        price_range_low   — lower bound of 80% interval
        price_range_high  — upper bound of 80% interval
    """
    encoders, caps, log_cols, scaler, selected, bundle, vectorizer = load_pipeline()

    df = pd.DataFrame([input_dict])
    df = step4_fix_formatting(df)
    df, _ = step6_encode(df, fit=False, encoders=encoders)
    df, _ = step7_handle_outliers(df, fit=False, caps=caps)
    df, _ = step8_log_transform(df, log_cols=log_cols)
    df, _ = engineer_features(df, fit=False, vectorizer=vectorizer)
    df    = df.drop(columns=[TARGET], errors="ignore")

    X_sc  = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_sel = X_sc[selected]

    ensemble, rf_p, lgbm_p, xgb_p = _raw_predictions(bundle, X_sel)
    point = round(float(ensemble[0]), 2)
    low, high = _price_range(point, float(rf_p[0]), float(lgbm_p[0]), float(xgb_p[0]))

    return {
        "predicted_price":  point,
        "price_range_low":  low,
        "price_range_high": high,
    }

if __name__ == "__main__":
    run_predictions()