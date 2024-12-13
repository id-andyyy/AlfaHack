![Арт](https://i.postimg.cc/brkSNynq/art.png)

![GitHub Created At](https://img.shields.io/github/created-at/id-andyyy/AlfaHack?style=flat&color=8fff00)
![Top Language](https://img.shields.io/github/languages/top/id-andyyy/AlfaHack?style=flat)

# AlfaHack&nbsp;&#128200;

Модель для предсказания склонности к инвестициям&nbsp;&#128202;. Разработана в рамках [ML-хакатона Альфа-Банка&nbsp;&#128142;](https://alfabank.ru/alfastudents/event/hack/).

## Описание

Модель машинного обучения для предсказания склонности физических лиц к инвестициям&nbsp;&#128181;. Основная задача проекта — определить вероятность, с которой клиент примет решение инвестировать свои средства, основываясь на [предоставленных данных](https://drive.google.com/drive/folders/1JgdIgCJwwy3HrMaTN0X860rxzzw0iJ6o?usp=sharing)&nbsp;&#128452;.

Ключевые этапы работы:
- Сбор и предобработка данных: объединение и сжатие признаков с помощью анализа главных компонент (PCA)
- Оптимизация гиперпараметров: подбор оптимальных настроек для моделей с использованием библиотеки Optuna
- Обучение моделей: использование алгоритмов CatBoost, LightGBM и HistGradientBoostingClassifier для достижения высокой точности
- Ансамблирование: объединение предсказаний всех моделей для повышения качества результатов
- Оценка модели: основным критерием успешности решения стала метрика ROC-AUC

## Технологии и инструменты

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

Особенности разработки:

- Проект написан за 12 дней в рамках хакатона [AlfaHack](https://alfabank.ru/alfastudents/event/hack/)
- `Python` с библиотеками `Pandas` и `NumPy` для обработки данных
- `scikit-learn`, `CatBoost` и `LightGBM` для создания и обучения моделей машинного обучения
- `Optuna` для оптимизации гиперпараметров
- `Jupyter Notebook` для разработки и тестирования
- Для реализации модели использовался набор данных, доступный по [ссылке](https://drive.google.com/drive/folders/1JgdIgCJwwy3HrMaTN0X860rxzzw0iJ6o?usp=sharing)
- `Git` для управления проектом и контроля версий

## Структура проекта

- `data_preprocessing.py` — модуль для загрузки и предобработки данных. Используется метод главных компонент (PCA) для уменьшения размерности признаков.
- `model_tuning.py` — модуль для оптимизации гиперпараметров моделей (CatBoost, LightGBM, HistGradientBoosting) с помощью библиотеки Optuna.
- `model_training.py` — модуль для обучения моделей с использованием лучших гиперпараметров.
- `prediction.py` — модуль для создания предсказаний, их ансамблирования и сохранения результатов в формате CSV.
- `main.py` — главный скрипт, который запускает весь процесс обработки, обучения моделей и предсказаний.
- `requirements.txt` — список всех зависимостей, необходимых для работы проекта.

## Начало работы

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&duration=2500&color=F7F7F7&background=000000&multiline=true&width=655&height=165&lines=%25+git+clone+https%3A%2F%2Fgithub.com%2Fid-andyyy%2FAlfaHack.git;%25+cd+AlfaHack;%25+pip+install+-r+requirements.txt;%25+python+main.py)](https://git.io/typing-svg)

```sh
git clone https://github.com/id-andyyy/AlfaHack.git
cd AlfaHack
pip install -r requirements.txt
python main.py
```

## Обратная связь

Буду признателен, если вы поставите звезду&nbsp;&#11088;. Если вы нашли баг или у вас есть предложения по улучшению,
используйте раздел [Issues](https://github.com/id-andyyy/AlfaHack/issues).

## Команда

Команда разработчиков [Mojarung](https://t.me/mojarung):

- [Андрей Обрезков](https://github.com/id-andyyy) (Data Scientist)
- [Кирилл Вериялов](https://github.com/verikirill) (ML Engineer)
- [Владислав Полицын](https://t.me/wasbyy7) (ML Engineer)
- [Артем Мелихов](https://github.com/Amkaus) (Data Scientist)

Читать на [английском&nbsp;&#127468;&#127463;](README.md)