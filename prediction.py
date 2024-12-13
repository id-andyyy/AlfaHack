def make_predictions(models, test_data):
    catboost_clf, lgbm_clf, histgb_clf = models
    X_test = test_data.drop('smpl', axis=1)
    
    catboost_pred = catboost_clf.predict_proba(X_test)[:, 1]
    lgbm_pred = lgbm_clf.predict_proba(X_test)[:, 1]
    histgb_pred = histgb_clf.predict_proba(X_test)[:, 1]
    
    ensemble_pred = (catboost_pred + lgbm_pred + histgb_pred) / 3
    
    return {
        'catboost': catboost_pred,
        'lgbm': lgbm_pred,
        'histgb': histgb_pred,
        'ensemble': ensemble_pred
    }

def save_submissions(predictions, test_data):
    for model_name, preds in predictions.items():
        test_data['target'] = preds
        test_data[['id', 'target']].to_csv(f'{model_name}_submission.csv', index=False)