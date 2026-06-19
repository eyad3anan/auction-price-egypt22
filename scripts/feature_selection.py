"""feature_selection.py — Phase 3: Feature selection on train only.

TF-IDF features are treated specially: the RF importance threshold is relaxed
for them because their individual importance is diluted across 100 features,
but collectively they explain significant variance in price.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scripts.utils import save_object

def select_features(X_train, X_test, y_train):
    print("\n=== PHASE 3 — FEATURE SELECTION (train-only) ===")

    tfidf_cols     = [c for c in X_train.columns if c.startswith("tfidf_")]
    structured_cols= [c for c in X_train.columns if not c.startswith("tfidf_")]

    # ── Signal 1: near-zero variance (structured cols only) ───────────────────
    nzv = [c for c in structured_cols
           if X_train[c].value_counts(normalize=True).iloc[0] >= 0.95]

    # ── Signal 2: mutual information ──────────────────────────────────────────
    mi = pd.Series(
        mutual_info_regression(X_train, y_train, random_state=42),
        index=X_train.columns,
    )

    # ── Signal 3: high correlation — drop lower-MI partner ────────────────────
    # Only among structured cols (TF-IDF cols are already sparse/low-corr)
    corr = X_train[structured_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > 0.90].tolist()
        for p in partners:
            corr_drop.add(col if mi[col] < mi[p] else p)

    # ── Signal 4: RF importance ────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)

    # Structured: drop if importance < 0.001
    struct_drop = set(rf_imp[structured_cols][rf_imp[structured_cols] < 0.001].index)

    # TF-IDF: keep the top 30 by RF importance (they share importance across 100 cols)
    tfidf_imp     = rf_imp[tfidf_cols].sort_values(ascending=False)
    tfidf_keep    = set(tfidf_imp.head(30).index)
    tfidf_drop    = set(tfidf_cols) - tfidf_keep

    to_drop = set(nzv) | corr_drop | struct_drop | tfidf_drop
    selected = [c for c in X_train.columns if c not in to_drop]

    n_tfidf_kept = len([c for c in selected if c.startswith("tfidf_")])
    n_struct_kept= len([c for c in selected if not c.startswith("tfidf_")])
    print(f"  KEPT ({len(selected)}): {n_struct_kept} structured + {n_tfidf_kept} TF-IDF")
    print(f"  Structured kept: {[c for c in selected if not c.startswith('tfidf_')]}")
    print(f"  DROPPED ({len(to_drop)})")

    save_object(selected, "selected_features.pkl")
    return X_train[selected].copy(), X_test[selected].copy(), selected

if __name__ == "__main__":
    print("Called from train.py or notebook")