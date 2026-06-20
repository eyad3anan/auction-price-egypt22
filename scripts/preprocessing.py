"""preprocessing.py — Phase 2: 10-step preprocessing pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from scripts.utils import DATA_PATH, save_object
from scripts.feature_engineering import engineer_features

TARGET       = "final_selling_price"
LEAKAGE_COLS = ["reserve_price", "buy_now_price"]
TEXT_COLS    = ["item_title", "item_description"]
LOG_COLS     = ["starting_price", "seller_total_sales", "seller_account_age"]
CAP_COLS     = ["starting_price", "product_age", "seller_total_sales",
                "seller_account_age", "seller_rating"]

def load_raw():
    return pd.read_csv(DATA_PATH)

def step1_drop_duplicates(df):
    b = len(df); df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Step1  Dropped {b-len(df)} duplicates ({len(df)} remain)")
    return df

def step2_drop_leakage(df):
    df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    print(f"  Step2  Dropped leakage cols: {LEAKAGE_COLS}")
    return df

def step3_drop_text(df):
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    print(f"  Step3  Dropped text cols: {TEXT_COLS}")
    return df

def step4_fix_formatting(df):
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    if "condition" in df.columns:
        df["condition"] = df["condition"].str.title()
    print("  Step4  Fixed formatting (strip whitespace, title-case condition)")
    return df

def step5_handle_missing(df):
    miss = df.isnull().sum().sum()
    print(f"  Step5  Missing values total: {miss}")
    return df

def step6_encode(df, fit=True, encoders=None):
    if encoders is None: encoders = {}
    cond_cats = [["For Parts","Poor","Fair","Good","Very Good","Excellent","Like New","New"]]
    if fit:
        enc = OrdinalEncoder(categories=cond_cats,
                             handle_unknown="use_encoded_value", unknown_value=-1)
        df["condition"] = enc.fit_transform(df[["condition"]])
        encoders["condition"] = enc
        encoders["day_map"] = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                               "Friday":4,"Saturday":5,"Sunday":6}
        encoders["cat_map"] = {v:i for i,v in enumerate(sorted(df["category"].unique()))}
        encoders["freq_sub"]   = df["subcategory"].value_counts().to_dict()
        encoders["freq_brand"] = df["brand"].value_counts().to_dict()
    else:
        df["condition"] = encoders["condition"].transform(df[["condition"]])
    df["listing_day_of_week"] = df["listing_day_of_week"].map(encoders["day_map"]).fillna(0).astype(int)
    df["category"]    = df["category"].map(encoders["cat_map"]).fillna(-1).astype(int)
    df["subcategory"] = df["subcategory"].map(encoders["freq_sub"]).fillna(0).astype(float)
    df["brand"]       = df["brand"].map(encoders["freq_brand"]).fillna(0).astype(float)
    print("  Step6  Encoded: condition(ordinal), day(ordinal), category(label), subcategory/brand(freq)")
    return df, encoders

def step7_handle_outliers(df, fit=True, caps=None):
    if caps is None: caps = {}
    for col in CAP_COLS:
        if col not in df.columns: continue
        if fit: caps[col] = (df[col].quantile(0.01), df[col].quantile(0.99))
        df[col] = df[col].clip(*caps[col])
    print("  Step7  Clipped outliers at 1st-99th percentile")
    return df, caps

def step8_log_transform(df, log_cols=None):
    lc = [c for c in (log_cols or LOG_COLS) if c in df.columns]
    for col in lc: df[col] = np.log1p(df[col])
    print(f"  Step8  log1p applied to: {lc}")
    return df, lc

def run_preprocessing(save=True):
    print("\n=== PHASE 2 — PREPROCESSING ===")
    df = load_raw()
    df = step1_drop_duplicates(df); df = step2_drop_leakage(df)
    df = step3_drop_text(df);       df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, encoders = step6_encode(df, fit=True)
    df, caps     = step7_handle_outliers(df, fit=True)
    df, log_cols = step8_log_transform(df)
    # IMPORTANT: feature engineering must run BEFORE the scaler is fit,
    # because train.py / predict.py load this saved scaler and call it
    # on the post-engineer_features() frame (13 raw cols + 10 engineered
    # cols = 23 total). Fitting the scaler here on only the 13 raw
    # columns (the old behaviour) caused a feature-name mismatch at
    # train/predict time -> ValueError -> Railway deploy crash.
    df = engineer_features(df)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    X_tr_r, X_te_r, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_tr_r), columns=X_tr_r.columns, index=X_tr_r.index)
    X_test  = pd.DataFrame(scaler.transform(X_te_r),     columns=X_te_r.columns, index=X_te_r.index)
    if save:
        save_object(encoders, "encoders.pkl"); save_object(caps, "outlier_caps.pkl")
        save_object(log_cols, "log_cols.pkl"); save_object(scaler, "scaler.pkl")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, encoders, caps, log_cols, scaler

if __name__ == "__main__":
    run_preprocessing()