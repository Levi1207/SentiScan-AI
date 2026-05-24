# SentiScan — Веб-сервис анализа тональности текста

> **Учебный проект по предмету «Основы искусственного интеллекта»**  
> Реализация собственной ML-модели без использования готовых AI API

## Описание

**SentiScan** — это веб-сервис машинного обучения, который автоматически определяет тональность текстовых отзывов: позитивная или негативная.

Пользователь вводит произвольный текст (отзыв о товаре, фильме, ресторане) — система обрабатывает его через обученную ML-модель и возвращает:

- метку тональности (`positive` / `negative`)
- уровень уверенности модели в процентах
- топ слов, которые повлияли на решение

## Технологии

- **Backend:** Python 3.11+, Flask, scikit-learn
- **ML:** TF-IDF + Logistic Regression
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Датасет:** IMDB Movie Reviews (50,000 отзывов)

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Загрузка датасета

Скачайте `IMDB Dataset.csv` с [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) и поместите в папку `data/`.

### 3. Обучение модели

```bash
python train.py
```

Выводит метрики и сохраняет `models/vectorizer.pkl` и `models/model.pkl`.  
Занимает ~2–3 минуты.

### 4. Запуск сервера

```bash
python app.py
```

Сервер запустится на `http://localhost:5000`

### 5. Открыть в браузере

Перейдите на `http://localhost:5000` и используйте интерфейс.

## Структура проекта

```
SentiScan/
├── train.py              # Скрипт обучения модели
├── app.py                # Flask-сервер
├── requirements.txt      # Зависимости Python
├── utils/
│   └── preprocess.py     # Функции предобработки текста
├── models/               # Сохранённые артефакты обучения
│   ├── vectorizer.pkl
│   └── model.pkl
├── data/
│   └── IMDB Dataset.csv  # Исходный датасет
├── static/               # Фронтенд
│   ├── index.html
│   ├── app.js
│   └── style.css
└── tests/
    └── test_api.py       # Тесты эндпоинтов
```

## Метрики качества

После обучения на IMDB ожидаются следующие показатели:

- **Accuracy:** ~89–91%
- **Precision:** ~90%
- **Recall:** ~88%
- **F1-Score:** ~89%

## API Endpoints

### POST /predict

Анализ тональности текста.

**Request:**
```json
{
  "text": "This movie was absolutely amazing!"
}
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 0.943,
  "top_words": [
    { "word": "amazing", "weight": 0.312 }
  ],
  "probabilities": {
    "positive": 0.943,
    "negative": 0.057
  }
}
```

### GET /health

Проверка работоспособности сервера.

**Response:**
```json
{
  "status": "ok",
  "model": "logistic_regression_tfidf"
}
```

## Автор

Учебный проект по курсу «Основы искусственного интеллекта»

## Лицензия

MIT
