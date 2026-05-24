# SentiScan — Быстрый старт

## Что уже готово

✅ Структура проекта создана  
✅ Модуль предобработки текста (`utils/preprocess.py`)  
✅ Скрипт обучения модели (`train.py`)  
✅ Flask API сервер (`app.py`)  
✅ Веб-интерфейс (HTML + CSS + JS)  
✅ Тесты (`tests/test_api.py`)  

## Следующие шаги

### 1. Установите зависимости

```bash
cd "/home/levi/Developer/SentiScan AI"
pip install -r requirements.txt
```

### 2. Загрузите датасет IMDB

Скачайте датасет с Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Поместите файл `IMDB Dataset.csv` в папку `data/`

Подробная инструкция: `data/README.md`

### 3. Обучите модель

```bash
python train.py
```

Это займет 2-3 минуты. Модель будет сохранена в `models/`

### 4. Запустите сервер

```bash
python app.py
```

Сервер запустится на http://localhost:5000

### 5. Откройте в браузере

Перейдите на http://localhost:5000 и начните анализировать тексты!

## Тестирование

Запустите тесты предобработки:

```bash
python tests/test_api.py
```

## Структура проекта

```
SentiScan AI/
├── app.py                    # Flask API сервер
├── train.py                  # Скрипт обучения модели
├── requirements.txt          # Зависимости Python
├── README.md                 # Документация
├── .gitignore               # Git ignore файл
│
├── utils/
│   ├── __init__.py
│   └── preprocess.py         # Предобработка текста
│
├── models/                   # Сохраненные модели (создается при обучении)
│   ├── vectorizer.pkl
│   └── model.pkl
│
├── data/                     # Датасет (нужно скачать)
│   ├── README.md
│   └── IMDB Dataset.csv      # (скачать с Kaggle)
│
├── static/                   # Фронтенд
│   ├── index.html
│   ├── app.js
│   └── style.css
│
└── tests/
    └── test_api.py           # Тесты
```

## Технологии

- **Backend:** Python 3.11+, Flask, scikit-learn
- **ML:** TF-IDF + Logistic Regression
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Датасет:** IMDB (50,000 отзывов)

## Ожидаемые метрики

После обучения модель должна показать:
- Accuracy: ~89-91%
- Precision: ~90%
- Recall: ~88%
- F1-Score: ~89%

## Возможные проблемы

**Проблема:** `ModuleNotFoundError: No module named 'flask'`  
**Решение:** Установите зависимости: `pip install -r requirements.txt`

**Проблема:** `FileNotFoundError: data/IMDB Dataset.csv`  
**Решение:** Скачайте датасет с Kaggle (см. `data/README.md`)

**Проблема:** Сервер не запускается  
**Решение:** Сначала обучите модель: `python train.py`

## Контакты

Учебный проект по курсу «Основы искусственного интеллекта»
