"""utils.py — Shared paths, metrics, save/load helpers."""
import os, numpy as np, joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "auction_dataset_egypt.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rmsle(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)

def report_metrics(y_true, y_pred, label=""):
    r2 = r2_score_manual(np.asarray(y_true), np.asarray(y_pred))
    rl = rmsle(np.asarray(y_true), np.asarray(y_pred))
    print(f"  [{label}]  R2={r2:.4f}   RMSLE={rl:.4f}")
    return {"R2": r2, "RMSLE": rl}

def save_object(obj, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path, compress=3)
    print(f"  Saved -> {path}  ({os.path.getsize(path)/1e6:.3f} MB)")
    return path

def load_object(name):
    return joblib.load(os.path.join(MODEL_DIR, name))

CONDITION_ORDER = ["For Parts","Poor","Fair","Good","Very Good","Excellent","Like New","New"]
DAY_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}