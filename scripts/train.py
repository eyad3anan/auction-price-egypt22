"""
train.py — Phase 5: Optuna tuning + training of RF, LGBM, XGBoost.

Model sizes (compressed, joblib compress=3):
  Random Forest : ~4.0 MB  (capped: n_estimators<=120, max_depth<=20)
  LightGBM      : ~0.5 MB
  XGBoost       : ~0.1 MB
  TOTAL BUNDLE  : ~4.6 MB  — well under GitHub 100 MB limit

Why size-cap RF:
  RF with max_depth=None serialises every leaf of every tree.
  300 uncapped trees = 431 MB. Capped at 120 trees + depth 20 = 4 MB,
  with negligible test performance loss because the ensemble averages 3 models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

from lightgbm  import LGBMRegressor
from xgboost   import XGBRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing   import RobustScaler
from sklearn.metrics         import make_scorer

from scripts.utils import save_object, load_object, report_metrics, rmsle
from scripts.preprocessing import (load_raw, step1_drop_duplicates, step2_drop_leakage,
    step3_drop_text, step4_fix_formatting, step5_handle_missing,
    step6_encode, step7_handle_outliers, step8_log_transform)
from scripts.feature_engineering import engineer_features
from scripts.feature_selection   import select_features

TARGET = "final_selling_price"

def _neg_rmsle_log(y_true, y_pred):
    return -rmsle(np.expm1(y_true), np.expm1(np.maximum(y_pred, 0)))

def cv_rmsle(model, X, y, cv=3):
    scorer = make_scorer(_neg_rmsle_log)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    return -cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1).mean()

def tune_lgbm(Xtr, log_y, n_trials=30):
    print(f"\n  Tuning LightGBM ({n_trials} trials)...")
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",600,2000),
                 num_leaves=trial.suggest_int("num_leaves",31,200),
                 learning_rate=trial.suggest_float("learning_rate",0.005,0.08,log=True),
                 max_depth=trial.suggest_int("max_depth",5,12),
                 subsample=trial.suggest_float("subsample",0.5,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.4,1.0),
                 reg_alpha=trial.suggest_float("reg_alpha",1e-6,5.0,log=True),
                 reg_lambda=trial.suggest_float("reg_lambda",1e-6,5.0,log=True),
                 min_child_samples=trial.suggest_int("min_child_samples",5,80),
                 random_state=42,n_jobs=-1,verbose=-1)
        return cv_rmsle(LGBMRegressor(**p), Xtr, log_y)
    s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best CV RMSLE: {s.best_value:.4f}")
    print(f"    Best params  : {s.best_params}")
    return s.best_params

def tune_xgb(Xtr, log_y, n_trials=30):
    print(f"\n  Tuning XGBoost ({n_trials} trials)...")
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",400,1500),
                 max_depth=trial.suggest_int("max_depth",3,10),
                 learning_rate=trial.suggest_float("learning_rate",0.005,0.1,log=True),
                 subsample=trial.suggest_float("subsample",0.5,1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree",0.4,1.0),
                 colsample_bylevel=trial.suggest_float("colsample_bylevel",0.4,1.0),
                 reg_alpha=trial.suggest_float("reg_alpha",1e-6,5.0,log=True),
                 reg_lambda=trial.suggest_float("reg_lambda",1e-6,5.0,log=True),
                 min_child_weight=trial.suggest_int("min_child_weight",1,20),
                 gamma=trial.suggest_float("gamma",0.0,2.0),
                 random_state=42,n_jobs=-1,verbosity=0)
        return cv_rmsle(XGBRegressor(**p), Xtr, log_y)
    s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best CV RMSLE: {s.best_value:.4f}")
    print(f"    Best params  : {s.best_params}")
    return s.best_params

def tune_rf(Xtr, log_y, n_trials=20):
    print(f"\n  Tuning Random Forest ({n_trials} trials, size-constrained)...")
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators",60,120),
                 max_depth=trial.suggest_int("max_depth",10,20),
                 min_samples_split=trial.suggest_int("min_samples_split",2,20),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf",1,15),
                 max_features=trial.suggest_float("max_features",0.3,0.9),
                 max_samples=trial.suggest_float("max_samples",0.6,1.0),
                 random_state=42,n_jobs=-1)
        return cv_rmsle(RandomForestRegressor(**p), Xtr, log_y)
    s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best CV RMSLE: {s.best_value:.4f}")
    print(f"    Best params  : {s.best_params}")
    return s.best_params

def run_training():
    print("\n=== PHASE 5 — HYPERPARAMETER TUNING + TRAINING ===")
    encoders=load_object("encoders.pkl"); caps=load_object("outlier_caps.pkl")
    log_cols=load_object("log_cols.pkl"); scaler=load_object("scaler.pkl")

    df = load_raw()
    df = step1_drop_duplicates(df); df = step2_drop_leakage(df)
    df = step3_drop_text(df);       df = step4_fix_formatting(df)
    df = step5_handle_missing(df)
    df, _ = step6_encode(df,fit=False,encoders=encoders)
    df, _ = step7_handle_outliers(df,fit=False,caps=caps)
    df, _ = step8_log_transform(df,log_cols=log_cols)
    df    = engineer_features(df)

    X_all=df.drop(columns=[TARGET]); y_all=df[TARGET]
    X_tr_r,X_te_r,y_train,y_test=train_test_split(X_all,y_all,test_size=0.2,random_state=42)
    X_tr_s=pd.DataFrame(scaler.transform(X_tr_r),columns=X_tr_r.columns,index=X_tr_r.index)
    X_te_s=pd.DataFrame(scaler.transform(X_te_r),columns=X_te_r.columns,index=X_te_r.index)
    Xtr,Xte,_=select_features(X_tr_s,X_te_s,y_train)
    log_y=np.log1p(y_train); results={}

    # Baselines
    print("\n--- BASELINES (raw target) ---")
    lgbm_b=LGBMRegressor(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.85,
                          colsample_bytree=0.7,random_state=42,n_jobs=-1,verbose=-1)
    lgbm_b.fit(Xtr,y_train)
    results["LGBM_Baseline_Train"]=report_metrics(y_train,lgbm_b.predict(Xtr),"LGBM_Base_Train")
    results["LGBM_Baseline_Test"] =report_metrics(y_test, lgbm_b.predict(Xte),"LGBM_Base_Test")

    xgb_b=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.05,subsample=0.85,
                        colsample_bytree=0.7,random_state=42,n_jobs=-1,verbosity=0)
    xgb_b.fit(Xtr,y_train)
    results["XGB_Baseline_Train"]=report_metrics(y_train,xgb_b.predict(Xtr),"XGB_Base_Train")
    results["XGB_Baseline_Test"] =report_metrics(y_test, xgb_b.predict(Xte),"XGB_Base_Test")

    rf_b=RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1)
    rf_b.fit(Xtr,y_train)
    results["RF_Baseline_Train"]=report_metrics(y_train,rf_b.predict(Xtr),"RF_Base_Train")
    results["RF_Baseline_Test"] =report_metrics(y_test, rf_b.predict(Xte),"RF_Base_Test")

    # Tuning
    print("\n--- OPTUNA TUNING ---")
    lgbm_p=tune_lgbm(Xtr,log_y,30); save_object(lgbm_p,"lgbm_best_params.pkl")
    xgb_p =tune_xgb(Xtr,log_y,30);  save_object(xgb_p, "xgb_best_params.pkl")
    rf_p  =tune_rf(Xtr,log_y,20);   save_object(rf_p,  "rf_best_params.pkl")

    # Tuned models
    print("\n--- TUNED MODELS (log-target) ---")
    lgbm_t=LGBMRegressor(**lgbm_p,random_state=42,n_jobs=-1,verbose=-1)
    lgbm_t.fit(Xtr,log_y)
    results["LGBM_Tuned_Train"]=report_metrics(y_train,np.expm1(lgbm_t.predict(Xtr)),"LGBM_Tuned_Train")
    results["LGBM_Tuned_Test"] =report_metrics(y_test, np.expm1(lgbm_t.predict(Xte)),"LGBM_Tuned_Test")

    xgb_t=XGBRegressor(**xgb_p,random_state=42,n_jobs=-1,verbosity=0)
    xgb_t.fit(Xtr,log_y)
    results["XGB_Tuned_Train"]=report_metrics(y_train,np.expm1(xgb_t.predict(Xtr)),"XGB_Tuned_Train")
    results["XGB_Tuned_Test"] =report_metrics(y_test, np.expm1(xgb_t.predict(Xte)),"XGB_Tuned_Test")

    rf_t=RandomForestRegressor(**rf_p,random_state=42,n_jobs=-1)
    rf_t.fit(Xtr,log_y)
    results["RF_Tuned_Train"]=report_metrics(y_train,np.expm1(rf_t.predict(Xtr)),"RF_Tuned_Train")
    results["RF_Tuned_Test"] =report_metrics(y_test, np.expm1(rf_t.predict(Xte)),"RF_Tuned_Test")

    # Ensemble
    print("\n--- ENSEMBLE (RF + LGBM + XGB log-blend) ---")
    def ens(X):
        p=(rf_t.predict(X)+lgbm_t.predict(X)+xgb_t.predict(X))/3
        return np.maximum(np.expm1(p),0)
    results["Ensemble_Train"]=report_metrics(y_train,ens(Xtr),"Ensemble_Train")
    results["Ensemble_Test"] =report_metrics(y_test, ens(Xte),"Ensemble_Test")

    rows=[{"Run":k,"R2":round(v["R2"],4),"RMSLE":round(v["RMSLE"],4)} for k,v in results.items()]
    res_df=pd.DataFrame(rows)
    print("\n"+"="*58+"\n FULL COMPARISON TABLE\n"+"="*58)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "outputs","model_results.csv"),index=False)

    bundle={"rf":rf_t,"lgbm":lgbm_t,"xgb":xgb_t,"log_target":True,"type":"rf_lgbm_xgb_log_blend"}
    save_object(bundle,"best_model.pkl")
    save_object("RF_LGBM_XGB_Ensemble","best_model_name.pkl")
    sz=os.path.getsize(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "models","best_model.pkl"))/1e6
    print(f"\nBundle size: {sz:.2f} MB (GitHub limit = 100 MB)")
    return lgbm_t,xgb_t,rf_t,results

if __name__ == "__main__":
    run_training()