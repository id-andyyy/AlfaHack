from data_preprocessing import load_and_preprocess_data
from model_tuning import tune_models
from model_training import train_models
from prediction import make_predictions, save_submissions

def main():
    print("Загрузка и предобработка данных...")
    train_data, test_data = load_and_preprocess_data()
    
    X = train_data.drop(['target', 'smpl'], axis=1)
    y = train_data['target']
    
    print("Оптимизация гиперпараметров...")
    best_params_catboost, best_params_lgbm, best_params_histgb = tune_models(X, y)
    
    print("Обучение моделей...")
    models = train_models(X, y, best_params_catboost, best_params_lgbm, best_params_histgb)
    
    print("Получение предсказаний и сохранение результатов...")
    predictions = make_predictions(models, test_data)
    save_submissions(predictions, test_data)
    
    print("Готово!")

if __name__ == "__main__":
    main()