from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def train_models(X, y, best_params_catboost, best_params_lgbm, best_params_histgb):
    catboost_clf = CatBoostClassifier(**best_params_catboost)
    catboost_clf.fit(X, y)
    
    lgbm_clf = LGBMClassifier(**best_params_lgbm)
    lgbm_clf.fit(X, y)
    
    histgb_clf = HistGradientBoostingClassifier(**best_params_histgb)
    histgb_clf.fit(X, y)
    
    return catboost_clf, lgbm_clf, histgb_clf