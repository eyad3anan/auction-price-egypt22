"""feature_engineering.py — Phase 4: Create 9 new features from listing-time data only."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

TARGET = "final_selling_price"

def engineer_features(df):
    df = df.copy()
    orig = set(df.columns)

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
        key = df["category"].astype(str) + "_" + df["condition"].astype(str)
        freq = key.value_counts().to_dict()
        df["cat_x_condition_freq"] = key.map(freq)

    if {"verified_seller","seller_rating"}.issubset(df.columns):
        df["verified_rating"] = df["verified_seller"] * df["seller_rating"]

    new = [c for c in df.columns if c not in orig and c != TARGET]
    print(f"  FE: Added {len(new)} features: {new}")
    return df

if __name__ == "__main__":
    print("Import and call engineer_features(df)")