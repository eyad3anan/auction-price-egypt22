"""preprocessing.py — Phase 2: Preprocessing pipeline (structured features only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from scripts.utils import DATA_PATH, save_object

TARGET       = "final_selling_price"
LEAKAGE_COLS = ["reserve_price", "buy_now_price"]
# NOTE: item_title and item_description are NOT dropped here.
# They are processed by feature_engineering.py into TF-IDF features,
# then dropped. This is correct — seller must provide them.
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

# step3 intentionally does NOT drop text columns anymore.
# Text is now converted to TF-IDF features in feature_engineering.py.
def step3_keep_text(df):
    present = [c for c in TEXT_COLS if c in df.columns]
    print(f"  Step3  Text cols kept for TF-IDF engineering: {present}")
    return df

def step4_fix_formatting(df):
    for col in df.select_dtypes("object").columns:
        if col not in TEXT_COLS:          # don't strip/title-case free text
            df[col] = df[col].str.strip()
    if "condition" in df.columns:
        df["condition"] = df["condition"].str.strip().str.title()
    print("  Step4  Fixed formatting (strip whitespace, title-case condition)")
    return df

def step5_handle_missing(df):
    # Fill missing text with empty string
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    miss = df.drop(columns=TEXT_COLS, errors="ignore").isnull().sum().sum()
    print(f"  Step5  Missing values in structured cols: {miss}")
    return df

def step6_encode(df, fit=True, encoders=None):
    if encoders is None: encoders = {}
    cond_cats = [["For Parts","Poor","Acceptable","Fair","Good","Very Good","Excellent","Like New","New"]]
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
    df = step1_drop_duplicates(df)
    df = step2_drop_leakage(df)
    df = step3_keep_text(df)           # keep text for TF-IDF
    df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, encoders = step6_encode(df, fit=True)
    df, caps     = step7_handle_outliers(df, fit=True)
    df, log_cols = step8_log_transform(df)
    # NOTE: do NOT split or scale here — feature_engineering needs text cols
    # The train/test split + scaling happens in train.py after TF-IDF
    if save:
        save_object(encoders, "encoders.pkl")
        save_object(caps,     "outlier_caps.pkl")
        save_object(log_cols, "log_cols.pkl")
    print(f"  Preprocessing done. Shape: {df.shape}")
    return df, encoders, caps, log_cols

if __name__ == "__main__":
    run_preprocessing()