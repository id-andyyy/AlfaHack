import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective_catboost(trial, X_train, X_val, y_train, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-4, 10, log=True),
        "od_type": "Iter",
        "task_type": "GPU",
        "verbose": 0,
        "random_seed": 42,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Trial {trial.number} | CatBoost ROC AUC: {score}")
    return score

def objective_lgbm(trial, X_train, X_val, y_train, y_val):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "random_state": 42,
        "device": "gpu",
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Trial {trial.number} | LightGBM ROC AUC: {score}")
    return score

def objective_histgb(trial, X_train, X_val, y_train, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 200),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": 42,
    }
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Trial {trial.number} | HistGradientBoosting ROC AUC: {score}")
    return score

def tune_models(X, y, n_trials=20):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.3
    )
    
    sampler = TPESampler(seed=42)
    
    study_catboost = optuna.create_study(direction="maximize", sampler=sampler)
    study_catboost.optimize(
        lambda trial: objective_catboost(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials
    )
    
    study_lgbm = optuna.create_study(direction="maximize", sampler=sampler)
    study_lgbm.optimize(
        lambda trial: objective_lgbm(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials
    )
    
    study_histgb = optuna.create_study(direction="maximize", sampler=sampler)
    study_histgb.optimize(
        lambda trial: objective_histgb(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials
    )
    
    return study_catboost.best_params, study_lgbm.best_params, study_histgb.best_params