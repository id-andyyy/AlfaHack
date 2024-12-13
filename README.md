![Art](https://i.postimg.cc/brkSNynq/art.png)

![GitHub Created At](https://img.shields.io/github/created-at/id-andyyy/AlfaHack?style=flat&color=8fff00)
![Top Language](https://img.shields.io/github/languages/top/id-andyyy/AlfaHack?style=flat)

# AlfaHack&nbsp;&#128200;

Investment propensity prediction model&nbsp;&#128202;. Developed as part of [Alfa-Bank ML Hackathon&nbsp;&#128142;](https://alfabank.ru/alfastudents/event/hack/).

## Description

Machine learning model for predicting individuals' propensity to invest&nbsp;&#128181;. The main objective of the project is to determine the probability with which a client will decide to invest their funds, based on the [provided data](https://drive.google.com/drive/folders/1JgdIgCJwwy3HrMaTN0X860rxzzw0iJ6o?usp=sharing)&nbsp;&#128452;.

Key stages of work:
- Data collection and preprocessing: feature combination and compression using Principal Component Analysis (PCA)
- Hyperparameter optimization: finding optimal settings for models using the Optuna library
- Model training: using CatBoost, LightGBM, and HistGradientBoostingClassifier algorithms to achieve high accuracy
- Ensembling: combining predictions from all models to improve result quality
- Model evaluation: ROC-AUC metric was the main success criterion

## Technologies and Tools

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffffff)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=ffffff)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=ffffff)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=ffffff)
![CatBoost](https://img.shields.io/badge/CatBoost-1E2952?style=for-the-badge&logo=catboost&logoColor=ffffff)
![LightGBM](https://img.shields.io/badge/LightGBM-017FFD?style=for-the-badge&logo=lightgbm&logoColor=ffffff)
![Optuna](https://img.shields.io/badge/Optuna-73D2DE?style=for-the-badge&logo=optuna&logoColor=000000)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=ffffff)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white&color=#6CeA8C)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white&color=f14e32)

Development features:

- Project was written in 12 days during the [AlfaHack](https://alfabank.ru/alfastudents/event/hack/) hackathon
- `Python` with `Pandas` and `NumPy` libraries for data processing
- `scikit-learn`, `CatBoost`, and `LightGBM` for creating and training machine learning models
- `Optuna` for hyperparameter optimization
- `Jupyter Notebook` for development and testing
- Dataset used for model implementation is available at this [link](https://drive.google.com/drive/folders/1JgdIgCJwwy3HrMaTN0X860rxzzw0iJ6o?usp=sharing)
- `Git` for project management and version control

## Project Structure

- `data_preprocessing.py` — module for data loading and preprocessing. Principal Component Analysis (PCA) is used to reduce feature dimensionality.
- `model_tuning.py` — module for optimizing model hyperparameters (CatBoost, LightGBM, HistGradientBoosting) using the Optuna library.
- `model_training.py` — module for training models using the best hyperparameters.
- `prediction.py` — module for making predictions, ensembling them, and saving results in CSV format.
- `main.py` — main script that runs the entire process of processing, model training, and predictions.
- `requirements.txt` — list of all dependencies required for the project.

## Getting Started

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&duration=2500&color=F7F7F7&background=000000&multiline=true&width=625&height=165&lines=git+clone+https%3A%2F%2Fgithub.com%2Fid-andyyy%2FAlfaHack.git;cd+AlfaHack;pip+install+-r+requirements.txt;python+main.py)](https://git.io/typing-svg)

```sh
git clone https://github.com/id-andyyy/AlfaHack.git
cd AlfaHack
pip install -r requirements.txt
python main.py
```

## Feedback

I would greatly appreciate it if you give the project a star&nbsp;&#11088;. If you find a bug or have suggestions for improvement, please use the [Issues](https://github.com/id-andyyy/AlfaHack/issues) section.

## Team

Development team [Mojarung](https://t.me/mojarung):

- [Andrey Obrezkov](https://github.com/id-andyyy) (Data Scientist)
- [Kirill Veriyalov](https://github.com/verikirill) (ML Engineer)
- [Vladislav Politsyn](https://t.me/wasbyy7) (ML Engineer)
- [Artem Melikhov](https://github.com/Amkaus) (Data Scientist)

Read in [Russian&nbsp;&#127479;&#127482;](README-ru.md)