"""feature_engineering.py — Phase 4: Structured features + TF-IDF text features.

The seller provides item_title and item_description. These are converted into
100 TF-IDF features (trained on the full dataset), which meaningfully improve
prediction accuracy (combined R² >> structured-only R²). The raw text strings
are dropped after this step — only the numeric TF-IDF columns remain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.utils import save_object, load_object

TARGET    = "final_selling_price"
TEXT_COLS = ["item_title", "item_description"]

# ── Structured feature engineering ────────────────────────────────────────────

def _add_structured_features(df):
    """9 interaction / transformation features from structured columns."""
    if "starting_price" in df.columns:
        df["starting_price_sq"] = df["starting_price"] ** 2

    if {"seller_rating","seller_total_sales"}.issubset(df.columns):
        df["seller_credibility"] = df["seller_rating"] * df["seller_total_sales"]

    if {"seller_total_sales","seller_account_age"}.issubset(df.columns):
        df["seller_activity_rate"] = df["seller_total_sales"] / (df["seller_account_age"] + 1e-6)

    if "auction_duration" in df.columns:
        df["is_short_auction"] = (df["auction_duration"] <= 3).astype(int)
        df["is_long_auction"]  = (df["auction_duration"] >= 14).astype(int)

    if "listing_day_of_week" in df.columns:
        df["is_weekend_listing"] = (df["listing_day_of_week"] >= 4).astype(int)

    if "listing_hour" in df.columns:
        df["is_primetime"] = ((df["listing_hour"] >= 18) & (df["listing_hour"] <= 23)).astype(int)

    if "product_age" in df.columns:
        df["product_freshness"] = np.log1p(1.0 / (df["product_age"] + 1))

    if {"category","condition"}.issubset(df.columns):
        key  = df["category"].astype(str) + "_" + df["condition"].astype(str)
        freq = key.value_counts().to_dict()
        df["cat_x_condition_freq"] = key.map(freq)

    if {"verified_seller","seller_rating"}.issubset(df.columns):
        df["verified_rating"] = df["verified_seller"] * df["seller_rating"]

    return df

# ── TF-IDF text features ───────────────────────────────────────────────────────

N_TFIDF = 100   # number of TF-IDF features to keep

def _build_text(df):
    """Combine title + description into a single string column."""
    title = df["item_title"].fillna("").astype(str).str.lower()
    desc  = df["item_description"].fillna("").astype(str).str.lower()
    return (title + " " + desc)

def fit_tfidf(df):
    """Fit TfidfVectorizer on the training corpus. Returns fitted vectorizer."""
    corpus   = _build_text(df)
    vectorizer = TfidfVectorizer(
        max_features  = N_TFIDF,
        stop_words    = "english",
        ngram_range   = (1, 2),
        sublinear_tf  = True,
    )
    vectorizer.fit(corpus)
    save_object(vectorizer, "tfidf_vectorizer.pkl")
    print(f"  TF-IDF vectorizer fitted: {N_TFIDF} features")
    return vectorizer

def _apply_tfidf(df, vectorizer):
    """Transform text and add tfidf_* columns to df."""
    corpus      = _build_text(df)
    tfidf_mat   = vectorizer.transform(corpus).toarray()
    feat_names  = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    tfidf_df    = pd.DataFrame(tfidf_mat, columns=feat_names, index=df.index)
    return pd.concat([df, tfidf_df], axis=1)

# ── Public API ────────────────────────────────────────────────────────────────

def engineer_features(df, fit=False, vectorizer=None):
    """
    Apply all feature engineering.

    Parameters
    ----------
    df          : DataFrame (must contain item_title and item_description)
    fit         : True during training (fits TF-IDF), False during inference
    vectorizer  : pre-fitted TfidfVectorizer (used when fit=False)

    Returns
    -------
    df          : DataFrame with text cols dropped and all engineered features added
    vectorizer  : the (possibly newly fitted) TfidfVectorizer
    """
    df   = df.copy()
    orig = set(df.columns)

    # 1. Structured features
    df = _add_structured_features(df)

    # 2. TF-IDF features
    if fit:
        vectorizer = fit_tfidf(df)
    elif vectorizer is None:
        vectorizer = load_object("tfidf_vectorizer.pkl")
    df = _apply_tfidf(df, vectorizer)

    # 3. Drop raw text — model doesn't need strings
    df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])

    new = [c for c in df.columns if c not in orig and c != TARGET]
    print(f"  FE: Added {len(new)} features "
          f"({len([c for c in new if c.startswith('tfidf_')])} TF-IDF + "
          f"{len([c for c in new if not c.startswith('tfidf_')])} structured)")
    return df, vectorizer

if __name__ == "__main__":
    print("Import and call engineer_features(df, fit=True/False)")