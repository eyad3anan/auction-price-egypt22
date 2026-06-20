"""
data_exploration.py — Phase 1: Exhaustive EDA for the Egyptian Auction dataset.
Run standalone:  python scripts/data_exploration.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scripts.utils import DATA_PATH, OUTPUT_DIR, CONDITION_ORDER

sns.set_theme(style="whitegrid", palette="muted")
FIGDIR = os.path.join(OUTPUT_DIR, "eda_figures")
os.makedirs(FIGDIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, name), dpi=120, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════
def load():
    df = pd.read_csv(DATA_PATH)
    print(f"\n{'='*60}")
    print(" PHASE 1 — EXHAUSTIVE EDA")
    print(f"{'='*60}")
    return df


# ══════════════════════════════════════════════════════════
#  1. BASIC INFO
# ══════════════════════════════════════════════════════════
def basic_info(df):
    print("\n── 1. Basic Info ──")
    print(f"  Shape         : {df.shape}")
    print(f"  Memory usage  : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("\n  Dtypes:")
    print(df.dtypes.to_string())
    print("\n  Head (3 rows):")
    print(df.head(3).to_string())


# ══════════════════════════════════════════════════════════
#  2. MISSING VALUES
# ══════════════════════════════════════════════════════════
def missing_values(df):
    print("\n── 2. Missing Values ──")
    miss = df.isnull().sum()
    miss_pct = miss / len(df) * 100
    miss_df = pd.DataFrame({"count": miss, "pct": miss_pct}).sort_values("pct", ascending=False)
    print(miss_df[miss_df["count"] > 0].to_string() if miss_df["count"].sum() > 0 else "  ✓ No missing values")

    # Heatmap regardless (shows structure)
    plt.figure(figsize=(14, 4))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    savefig("01_missing_heatmap.png")


# ══════════════════════════════════════════════════════════
#  3. DUPLICATES
# ══════════════════════════════════════════════════════════
def duplicates(df):
    print("\n── 3. Duplicates ──")
    exact = df.duplicated().sum()
    print(f"  Exact duplicate rows : {exact}")
    # Column-level: identical columns
    dup_cols = []
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                dup_cols.append((cols[i], cols[j]))
    print(f"  Identical column pairs: {dup_cols if dup_cols else 'None'}")


# ══════════════════════════════════════════════════════════
#  4. CONSTANT / QUASI-CONSTANT FEATURES
# ══════════════════════════════════════════════════════════
def constant_features(df):
    print("\n── 4. Constant / Quasi-constant Features ──")
    for col in df.columns:
        top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
        if top_pct >= 95:
            print(f"  ⚠  {col}: top value covers {top_pct:.1f}% of rows")


# ══════════════════════════════════════════════════════════
#  5. CARDINALITY
# ══════════════════════════════════════════════════════════
def cardinality(df):
    print("\n── 5. Cardinality Analysis ──")
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        n = df[col].nunique()
        tag = "LOW" if n <= 10 else ("HIGH" if n <= 50 else "UNIQUE-ID-LIKE")
        print(f"  {col:30s}: {n:5d} unique  [{tag}]")
        if n <= 20:
            print(f"    Values: {df[col].unique()[:20].tolist()}")


# ══════════════════════════════════════════════════════════
#  6. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════
def outliers(df):
    print("\n── 6. Outlier Detection ──")
    num_cols = df.select_dtypes("number").columns.tolist()
    outlier_report = {}
    for col in num_cols:
        s = df[col].dropna()
        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_iqr = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        # Z-score
        z = np.abs(stats.zscore(s))
        n_z = (z > 3).sum()
        outlier_report[col] = {"IQR": n_iqr, "Zscore": n_z}
        severity = "HIGH" if n_iqr/len(s) > 0.05 else ("MED" if n_iqr/len(s) > 0.01 else "LOW")
        print(f"  {col:25s}: IQR={n_iqr:5d}  Z={n_z:5d}  [{severity}]")

    # Boxplots
    n = len(num_cols)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6))
        axes[i].set_title(col, fontsize=9)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Boxplots — Numeric Features", fontsize=12)
    savefig("02_boxplots.png")
    return outlier_report


# ══════════════════════════════════════════════════════════
#  7. DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════
def distributions(df):
    print("\n── 7. Distribution Analysis (skew / kurtosis) ──")
    num_cols = df.select_dtypes("number").columns.tolist()
    stats_rows = []
    for col in num_cols:
        s = df[col].dropna()
        sk = s.skew()
        ku = s.kurtosis()
        flag = "SKEWED" if abs(sk) > 1 else ("MOD" if abs(sk) > 0.5 else "OK")
        print(f"  {col:25s}: skew={sk:+.3f}  kurt={ku:+.3f}  [{flag}]")
        stats_rows.append({"col": col, "skew": sk, "kurtosis": ku, "flag": flag})

    # Histograms + KDE
    n = len(num_cols)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=50, alpha=0.6, density=True, color="steelblue", edgecolor="none")
        df[col].dropna().plot.kde(ax=axes[i], color="red", lw=1.5)
        axes[i].set_title(f"{col}\nskew={df[col].skew():.2f}", fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Histograms + KDE — Numeric Features", fontsize=12)
    savefig("03_histograms.png")
    return pd.DataFrame(stats_rows)


# ══════════════════════════════════════════════════════════
#  8. TARGET ANALYSIS
# ══════════════════════════════════════════════════════════
def target_analysis(df, target="final_selling_price"):
    print(f"\n── 8. Target Analysis: {target} ──")
    t = df[target]
    print(f"  Min={t.min()}  Max={t.max()}  Mean={t.mean():.0f}  Median={t.median():.0f}")
    print(f"  Skew={t.skew():.3f}  Kurt={t.kurtosis():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(t, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("Target Distribution (raw)")
    axes[1].hist(np.log1p(t), bins=80, color="salmon", edgecolor="none", alpha=0.8)
    axes[1].set_title("Target Distribution (log1p)")
    savefig("04_target_distribution.png")


# ══════════════════════════════════════════════════════════
#  9. DATA LEAKAGE DETECTION
# ══════════════════════════════════════════════════════════
def leakage_detection(df, target="final_selling_price"):
    print("\n── 9. Data Leakage Detection ──")
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    corrs = df[num_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    print("  Pearson |correlation| with target:")
    print(corrs.to_string())
    print()
    print("  ⚠ LEAKAGE FLAGS:")
    print("  reserve_price  — set at listing time with knowledge of expected final price (proxy for target)")
    print("  buy_now_price  — set to anchor the final price; directly encodes seller's valuation")
    print("  starting_price — partial leakage (correlated ~0.9+ with target in many datasets)")
    print("  → These will be DROPPED in Phase 2 (leakage columns).")


# ══════════════════════════════════════════════════════════
#  10. MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════
def multicollinearity(df, target="final_selling_price"):
    print("\n── 10. Multicollinearity ──")
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    corr_mat = df[num_cols].corr().abs()

    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Matrix — Numeric Features")
    savefig("05_correlation_matrix.png")

    # VIF — only after dropping leakage cols
    safe_cols = [c for c in num_cols if c not in ["reserve_price", "buy_now_price"]]
    X_vif = df[safe_cols].dropna()
    from numpy.linalg import matrix_rank
    if matrix_rank(X_vif.values) == X_vif.shape[1]:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                           for i in range(X_vif.shape[1])]
        print("\n  VIF scores (after removing known leakage):")
        print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))
    else:
        print("  VIF: matrix is singular — likely perfect multicollinearity in some columns.")


# ══════════════════════════════════════════════════════════
#  11. BIVARIATE — FEATURE vs TARGET
# ══════════════════════════════════════════════════════════
def bivariate(df, target="final_selling_price"):
    print("\n── 11. Bivariate Analysis ──")
    num_cols = [c for c in df.select_dtypes("number").columns if c != target]
    cat_cols  = df.select_dtypes("object").columns.tolist()

    # Numeric vs target: scatter
    n = len(num_cols)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].scatter(df[col], df[target], alpha=0.05, s=4, color="steelblue")
        pr = df[[col, target]].corr().iloc[0, 1]
        axes[i].set_title(f"{col}\nr={pr:.2f}", fontsize=8)
        axes[i].set_xlabel(col, fontsize=7)
        axes[i].set_ylabel(target, fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Features vs Target (scatter)", fontsize=11)
    savefig("06_scatter_vs_target.png")

    # Categorical vs target: violin
    for col in cat_cols:
        if df[col].nunique() <= 20:
            plt.figure(figsize=(12, 5))
            order = df.groupby(col)[target].median().sort_values().index
            sns.violinplot(data=df, x=col, y=target, order=order,
                           inner="quartile", palette="muted")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{col} vs {target}")
            savefig(f"07_violin_{col}.png")

    # Pearson & Spearman
    print("\n  Pearson & Spearman correlations with target:")
    for col in num_cols:
        p_r, _ = stats.pearsonr(df[col].dropna(), df[target].loc[df[col].dropna().index])
        s_r, _ = stats.spearmanr(df[col].dropna(), df[target].loc[df[col].dropna().index])
        print(f"  {col:25s}: Pearson={p_r:+.3f}  Spearman={s_r:+.3f}")

    # Chi-square for categoricals
    print("\n  Chi-square + Cramér V (categorical vs target bins):")
    df_tmp = df.copy()
    df_tmp["target_bin"] = pd.qcut(df_tmp[target], q=5, labels=False, duplicates="drop")
    for col in cat_cols:
        ct = pd.crosstab(df_tmp[col], df_tmp["target_bin"])
        chi2, p, dof, _ = chi2_contingency(ct)
        n  = ct.values.sum()
        min_dim = min(ct.shape) - 1
        v = np.sqrt(chi2 / (n * max(min_dim, 1)))
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        print(f"  {col:30s}: χ²={chi2:.1f}  p={p:.4f}  V={v:.3f}  {sig}")


# ══════════════════════════════════════════════════════════
#  12. INCONSISTENT FORMATTING
# ══════════════════════════════════════════════════════════
def formatting_issues(df):
    print("\n── 12. Formatting Issues ──")
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        # leading/trailing whitespace
        has_ws = (df[col] != df[col].str.strip()).sum()
        # mixed case
        vals = df[col].dropna().unique()
        mixed = any(v != v.lower() and v != v.upper() and v != v.title() for v in vals[:200])
        print(f"  {col:30s}: whitespace_issues={has_ws}  mixed_case={mixed}")


# ══════════════════════════════════════════════════════════
#  13. IMPOSSIBLE VALUES
# ══════════════════════════════════════════════════════════
def impossible_values(df):
    print("\n── 13. Impossible / Illogical Values ──")
    checks = {
        "product_age < 0":       (df["product_age"] < 0).sum(),
        "starting_price <= 0":   (df["starting_price"] <= 0).sum(),
        "reserve_price <= 0":    (df["reserve_price"] <= 0).sum(),
        "buy_now_price <= 0":    (df["buy_now_price"] <= 0).sum(),
        "final_selling_price<=0":(df["final_selling_price"] <= 0).sum(),
        "seller_rating > 5":     (df["seller_rating"] > 5).sum(),
        "seller_rating < 0":     (df["seller_rating"] < 0).sum(),
        "listing_hour > 23":     (df["listing_hour"] > 23).sum(),
        "auction_duration <= 0": (df["auction_duration"] <= 0).sum(),
    }
    for k, v in checks.items():
        status = "✓" if v == 0 else f"⚠ {v} rows"
        print(f"  {k:35s}: {status}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def run_eda():
    df = load()
    basic_info(df)
    missing_values(df)
    duplicates(df)
    constant_features(df)
    cardinality(df)
    outlier_report = outliers(df)
    dist_stats = distributions(df)
    target_analysis(df)
    leakage_detection(df)
    multicollinearity(df)
    bivariate(df)
    formatting_issues(df)
    impossible_values(df)
    print(f"\n{'='*60}")
    print(" EDA COMPLETE — figures saved to outputs/eda_figures/")
    print(f"{'='*60}\n")
    return df, dist_stats, outlier_report


if __name__ == "__main__":
    run_eda()