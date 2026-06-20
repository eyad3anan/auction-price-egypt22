"""feature_selection.py — Phase 3: Select features on train only, apply mask to test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scripts.utils import save_object

def select_features(X_train, X_test, y_train):
    print("\n=== PHASE 3 — FEATURE SELECTION (train-only) ===")
    # Signal 1: near-zero variance
    nzv = [c for c in X_train.columns
           if X_train[c].value_counts(normalize=True).iloc[0] >= 0.95]
    # Signal 2: mutual information
    mi = pd.Series(mutual_info_regression(X_train, y_train, random_state=42),
                   index=X_train.columns)
    # Signal 3: high correlation — drop lower-MI partner
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > 0.90].tolist()
        for p in partners:
            corr_drop.add(col if mi[col] < mi[p] else p)
    # Signal 4: RF importance
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    rf_drop = rf_imp[rf_imp < 0.001].index.tolist()

    to_drop = set(nzv) | corr_drop | set(rf_drop)
    selected = [c for c in X_train.columns if c not in to_drop]
    print(f"  KEPT ({len(selected)}): {selected}")
    print(f"  DROPPED ({len(to_drop)}): {sorted(to_drop)}")
    save_object(selected, "selected_features.pkl")
    return X_train[selected].copy(), X_test[selected].copy(), selected

if __name__ == "__main__":
    print("Called from train.py or notebook")