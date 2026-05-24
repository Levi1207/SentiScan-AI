# SentiScan — Веб-сервис анализа тональности текста

> **Учебный проект по предмету «Основы искусственного интеллекта»**
> Реализация собственной ML-модели без использования готовых AI API

---

## Содержание

1. [Описание проекта](#1-описание-проекта)
2. [Название и концепция](#2-название-и-концепция)
3. [Алгоритм работы ИИ](#3-алгоритм-работы-ии)
4. [Используемые технологии](#4-используемые-технологии)
5. [Архитектура проекта](#5-архитектура-проекта)
6. [Структура файлов](#6-структура-файлов)
7. [Этапы разработки](#7-этапы-разработки-план-по-неделям)
8. [Датасет](#8-датасет)
9. [Предобработка текста](#9-предобработка-текста)
10. [Обучение модели](#10-обучение-модели)
11. [Flask API](#11-flask-api)
12. [Фронтенд](#12-фронтенд)
13. [Метрики качества](#13-метрики-качества)
14. [Расширения и улучшения](#14-расширения-и-улучшения)
15. [Запуск проекта](#15-запуск-проекта)
16. [Зависимости](#16-зависимости)

---

## 1. Описание проекта

**SentiScan** — это веб-сервис машинного обучения, который автоматически определяет тональность текстовых отзывов: позитивная, негативная или нейтральная.

Пользователь вводит произвольный текст (отзыв о товаре, фильме, ресторане) — система обрабатывает его через обученную ML-модель и возвращает:

- метку тональности (`positive` / `negative`)
- уровень уверенности модели в процентах
- топ слов, которые повлияли на решение

Весь ИИ реализован самостоятельно: данные → предобработка → векторизация → обучение модели → сохранение → инференс. Никакие внешние AI API (OpenAI, Hugging Face Inference API и т.д.) не используются.

---

## 2. Название и концепция

| Поле            | Значение                                         |
| --------------- | ------------------------------------------------ |
| **Название**    | SentiScan                                        |
| **Расшифровка** | Sentiment + Scan — "сканирование настроения"     |
| **Слоган**      | _"Понимаем текст — видим настроение"_            |
| **Тип**         | Веб-приложение (SPA + REST API)                  |
| **Аудитория**   | Бизнес (анализ отзывов), студенты, исследователи |

**Почему SentiScan:**
Название коротко и точно описывает задачу — система «сканирует» текст и определяет его эмоциональную окраску (sentiment). Легко запоминается и звучит профессионально.

---

## 3. Алгоритм работы ИИ

Алгоритм состоит из двух фаз: **обучение** (один раз, оффлайн) и **инференс** (каждый запрос пользователя).

### Фаза 1 — Обучение модели

```
IMDB датасет (50 000 отзывов)
        │
        ▼
Предобработка текста
  ├── Приведение к нижнему регистру
  ├── Удаление HTML-тегов (<br>, <p> и т.д.)
  ├── Удаление специальных символов и пунктуации
  └── Токенизация (разбивка на слова)
        │
        ▼
TF-IDF Векторизация
  ├── Term Frequency (TF): насколько часто слово встречается в тексте
  ├── Inverse Document Frequency (IDF): насколько редко слово в корпусе
  ├── max_features = 10 000 (топ слов по всему датасету)
  └── stop_words = 'english' (удаление мусорных слов: the, is, a...)
        │
        ▼
Logistic Regression (бинарный классификатор)
  ├── Обучение на 80% датасета (40 000 отзывов)
  ├── Валидация на 20% датасета (10 000 отзывов)
  └── max_iter = 1000
        │
        ▼
Сохранение артефактов
  ├── vectorizer.pkl (обученный TF-IDF)
  └── model.pkl (обученная модель)
```

### Фаза 2 — Инференс (предсказание по запросу)

```
Пользователь вводит текст в браузере
        │
        ▼
POST /predict { "text": "..." }
        │
        ▼
Загрузка модели из vectorizer.pkl + model.pkl
        │
        ▼
Предобработка входного текста (те же шаги, что при обучении)
        │
        ▼
Векторизация через сохранённый TF-IDF
        │
        ▼
model.predict() → 'positive' или 'negative'
model.predict_proba() → [0.12, 0.88] → уверенность: 88%
        │
        ▼
JSON-ответ: { label, confidence, top_words }
        │
        ▼
Отображение результата на фронтенде
```

### Почему именно TF-IDF + Logistic Regression

| Критерий             | Naive Bayes | **Logistic Regression** | LSTM / BERT |
| -------------------- | ----------- | ----------------------- | ----------- |
| Точность на IMDB     | ~85%        | **~89–91%**             | ~94–96%     |
| Скорость обучения    | Секунды     | Минуты                  | Часы        |
| Интерпретируемость   | Высокая     | **Высокая**             | Низкая      |
| Сложность реализации | Низкая      | **Низкая**              | Высокая     |
| Объяснение на защите | Просто      | **Просто**              | Сложно      |

Logistic Regression — оптимальный баланс точности, скорости и понятности для учебного проекта.

---

## 4. Используемые технологии

### Backend (Python)

| Библиотека     | Версия  | Зачем                                             |
| -------------- | ------- | ------------------------------------------------- |
| `scikit-learn` | ≥1.3    | TF-IDF векторизация, Logistic Regression, метрики |
| `pandas`       | ≥2.0    | Загрузка и обработка датасета CSV                 |
| `numpy`        | ≥1.24   | Числовые операции                                 |
| `flask`        | ≥3.0    | REST API сервер                                   |
| `flask-cors`   | ≥4.0    | Разрешение кросс-доменных запросов от фронтенда   |
| `pickle`       | встроен | Сохранение/загрузка обученной модели              |
| `re`           | встроен | Очистка текста регулярными выражениями            |

### Frontend (JavaScript)

| Технология        | Зачем                                          |
| ----------------- | ---------------------------------------------- |
| HTML5             | Разметка страницы                              |
| CSS3              | Стилизация (без фреймворков)                   |
| Vanilla JS (ES6+) | Логика: fetch API, DOM-манипуляции             |
| Chart.js (CDN)    | Визуализация: confidence bar, история анализов |

### Инструменты разработки

| Инструмент   | Зачем                           |
| ------------ | ------------------------------- |
| Python 3.11+ | Основной язык                   |
| VS Code      | IDE                             |
| Git + GitHub | Контроль версий                 |
| Postman      | Тестирование API                |
| Kali Linux   | Операционная система разработки |

---

## 5. Архитектура проекта

```
┌─────────────────────────────────────────────────────────┐
│                     БРАУЗЕР                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              index.html + app.js                │   │
│  │  [Поле ввода текста]  [Кнопка "Анализировать"]  │   │
│  │  [Результат: Positive 88%]                      │   │
│  │  [Топ слов]  [График истории]                   │   │
│  └───────────────────┬─────────────────────────────┘   │
└──────────────────────│─────────────────────────────────┘
                       │ HTTP POST /predict
                       │ { "text": "This movie was amazing!" }
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  FLASK SERVER (app.py)                  │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  /predict    │───▶│      predict_sentiment()     │   │
│  │  POST route  │    │  1. clean_text(text)         │   │
│  └──────────────┘    │  2. vectorizer.transform()   │   │
│                      │  3. model.predict()          │   │
│  ┌──────────────┐    │  4. model.predict_proba()    │   │
│  │  /stats      │    │  5. get_top_words()          │   │
│  │  GET route   │    └──────────────────────────────┘   │
│  └──────────────┘               │                       │
│                                 ▼                       │
│               ┌─────────────────────────────┐           │
│               │    ML Model Layer           │           │
│               │  vectorizer.pkl (TF-IDF)    │           │
│               │  model.pkl (LogReg)         │           │
│               └─────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
                       │ JSON ответ
                       │ { "label": "positive",
                       │   "confidence": 0.88,
                       │   "top_words": ["amazing","great"] }
                       ▼
                   Браузер отображает результат
```

### Компоненты и их ответственность

| Компонент        | Файл                  | Ответственность                                                 |
| ---------------- | --------------------- | --------------------------------------------------------------- |
| Обучение модели  | `train.py`            | Загрузка датасета, предобработка, обучение, сохранение .pkl     |
| API сервер       | `app.py`              | REST-эндпоинты, загрузка модели, возврат предсказаний           |
| Предобработка    | `utils/preprocess.py` | Общие функции очистки текста (используются в train.py и app.py) |
| Веб-интерфейс    | `static/index.html`   | UI форма, отображение результатов                               |
| Логика фронтенда | `static/app.js`       | Fetch запросы к API, рендер результатов, история                |
| Стили            | `static/style.css`    | Визуальное оформление                                           |

---

## 6. Структура файлов

```
SentiScan/
│
├── train.py                  # Скрипт обучения модели (запускается один раз)
├── app.py                    # Flask-сервер (основное приложение)
├── requirements.txt          # Зависимости Python
├── README.md                 # Краткое описание для GitHub
│
├── utils/
│   └── preprocess.py         # Функции предобработки текста
│
├── models/                   # Сохранённые артефакты обучения
│   ├── vectorizer.pkl        # Обученный TF-IDF векторайзер
│   └── model.pkl             # Обученная Logistic Regression
│
├── data/
│   └── IMDB Dataset.csv      # Исходный датасет (скачать отдельно)
│
├── notebooks/
│   └── exploration.ipynb     # Jupyter notebook: EDA, эксперименты
│
├── static/                   # Фронтенд (Flask отдаёт как статику)
│   ├── index.html            # Главная страница
│   ├── app.js                # JavaScript логика
│   └── style.css             # Стили
│
└── tests/
    └── test_api.py           # Тесты эндпоинтов
```

---

## 7. Этапы разработки (план по неделям)

### Неделя 1–2: Данные и обучение модели

**Цель:** Рабочая обученная модель, сохранённая в .pkl файлах.

- [ ] Скачать IMDB датасет с Kaggle
- [ ] Изучить структуру данных (EDA в Jupyter notebook)
- [ ] Написать функцию `clean_text()` в `utils/preprocess.py`
- [ ] Реализовать TF-IDF векторизацию
- [ ] Обучить Logistic Regression
- [ ] Измерить accuracy, precision, recall, F1
- [ ] Сохранить `vectorizer.pkl` и `model.pkl`
- [ ] Построить confusion matrix

**Ожидаемый результат:** Точность ~89% на тестовой выборке.

### Неделя 3–4: Flask REST API

**Цель:** Рабочий API сервер с эндпоинтами.

- [ ] Создать `app.py` с Flask
- [ ] Реализовать эндпоинт `POST /predict`
- [ ] Реализовать эндпоинт `GET /health` (проверка работоспособности)
- [ ] Реализовать эндпоинт `GET /stats` (статистика запросов)
- [ ] Добавить CORS заголовки (flask-cors)
- [ ] Тестировать через Postman
- [ ] Обработка ошибок: пустой текст, слишком длинный текст

**Ожидаемый результат:** API отвечает корректным JSON на запросы.

### Неделя 5–6: Фронтенд

**Цель:** Красивый и интуитивный веб-интерфейс.

- [ ] Создать HTML-страницу с формой ввода
- [ ] Реализовать fetch-запрос к API через `app.js`
- [ ] Отображение результата: цветной бейдж + текст
- [ ] Шкала уверенности (прогресс-бар)
- [ ] Подсветка топ-слов, повлиявших на решение
- [ ] История последних 10 анализов
- [ ] Адаптивная вёрстка (мобильные устройства)

**Ожидаемый результат:** Полностью рабочий веб-интерфейс.

### Неделя 7–8: Улучшения и финализация

**Цель:** Полированный продукт, готовый к демонстрации.

- [ ] Страница "О модели" (accuracy, confusion matrix, описание алгоритма)
- [ ] Пакетный анализ: загрузить CSV с отзывами → получить таблицу результатов
- [ ] Графики: распределение pos/neg по истории запросов
- [ ] Написать README.md с инструкцией запуска
- [ ] Финальное тестирование
- [ ] Подготовка к презентации/защите

---

## 8. Датасет

**IMDB Movie Reviews Dataset**

| Параметр           | Значение                                                    |
| ------------------ | ----------------------------------------------------------- |
| Источник           | Kaggle: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` |
| Размер             | 50 000 отзывов                                              |
| Классы             | `positive` (25 000), `negative` (25 000)                    |
| Формат             | CSV (columns: `review`, `sentiment`)                        |
| Размер файла       | ~66 MB                                                      |
| Сбалансированность | Идеально сбалансирован (50/50)                              |

**Пример записи:**

```csv
review,sentiment
"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right...",positive
"A wonderful little production. The filming technique is very unassuming...",positive
"I thought this was a wonderful way to spend time on a too hot summer evening...",negative
```

**Как скачать:**

```bash
# Вариант 1: Вручную с https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Вариант 2: Через Kaggle CLI
pip install kaggle
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/
```

---

## 9. Предобработка текста

Файл: `utils/preprocess.py`

```python
import re

def clean_text(text: str) -> str:
    """
    Очистка текста для подачи в модель.
    Применяется одинаково при обучении и инференсе.
    """
    # 1. Приведение к нижнему регистру
    text = text.lower()

    # 2. Удаление HTML-тегов (отзывы IMDB часто содержат <br />, <p> и т.д.)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Удаление URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Удаление небуквенных символов (оставляем только буквы и пробелы)
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Замена множественных пробелов на один
    text = re.sub(r'\s+', ' ', text).strip()

    return text
```

**Важно:** функция `clean_text()` должна быть идентична и в `train.py`, и в `app.py`. Для этого она вынесена в отдельный модуль `utils/preprocess.py`.

---

## 10. Обучение модели

Файл: `train.py`

```python
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from utils.preprocess import clean_text

# ── 1. Загрузка данных ─────────────────────────────────────────
df = pd.read_csv('data/IMDB Dataset.csv')
print(f"Датасет загружен: {len(df)} записей")
print(df['sentiment'].value_counts())

# ── 2. Предобработка ───────────────────────────────────────────
print("Очистка текстов...")
df['clean_review'] = df['review'].apply(clean_text)

# ── 3. Разбивка на train/test ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']   # сохраняем баланс классов
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. TF-IDF векторизация ─────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=10_000,       # топ 10 000 слов по TF-IDF
    ngram_range=(1, 2),        # учитываем биграммы ("not good", "very bad")
    stop_words='english',      # убираем стоп-слова
    sublinear_tf=True          # применяем log-нормализацию TF
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"Размер матрицы признаков: {X_train_vec.shape}")

# ── 5. Обучение модели ─────────────────────────────────────────
print("Обучение Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    C=1.0,           # регуляризация (меньше = сильнее)
    solver='lbfgs',
    n_jobs=-1        # использовать все CPU ядра
)
model.fit(X_train_vec, y_train)

# ── 6. Оценка ──────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 7. Сохранение артефактов ───────────────────────────────────
import os
os.makedirs('models', exist_ok=True)
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
pickle.dump(model,      open('models/model.pkl',      'wb'))
print("\nМодель сохранена в models/")
```

---

## 11. Flask API

Файл: `app.py`

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from utils.preprocess import clean_text

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Загрузка модели при старте сервера (один раз)
print("Загрузка модели...")
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
model      = pickle.load(open('models/model.pkl',      'rb'))
feature_names = vectorizer.get_feature_names_out()
print("Модель готова!")

def get_top_words(text_vec, n=5):
    """Топ-N слов с наибольшим вкладом в предсказание."""
    feature_weights = model.coef_[0]           # веса модели
    text_weights = text_vec.toarray()[0]       # TF-IDF веса текста
    combined = feature_weights * text_weights  # вклад каждого слова
    top_idx = np.argsort(np.abs(combined))[-n:][::-1]
    return [
        {"word": feature_names[i], "weight": round(float(combined[i]), 4)}
        for i in top_idx if text_weights[i] > 0
    ]

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Текст не может быть пустым'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Текст слишком длинный (макс. 5000 символов)'}), 400

    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    confidence = float(proba.max())
    top_words  = get_top_words(vec)

    return jsonify({
        'label':      label,
        'confidence': round(confidence, 3),
        'top_words':  top_words,
        'probabilities': {
            'positive': round(float(proba[list(model.classes_).index('positive')]), 3),
            'negative': round(float(proba[list(model.classes_).index('negative')]), 3),
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'logistic_regression_tfidf'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Пример запроса и ответа

**Запрос:**

```http
POST http://localhost:5000/predict
Content-Type: application/json

{
  "text": "This movie was absolutely amazing! The acting was superb and the story kept me engaged throughout."
}
```

**Ответ:**

```json
{
  "label": "positive",
  "confidence": 0.943,
  "top_words": [
    { "word": "amazing", "weight": 0.312 },
    { "word": "superb",  "weight": 0.278 },
    { "word": "kept engaged", "weight": 0.201 }
  ],
  "probabilities": {
    "positive": 0.943,
    "negative": 0.057
  }
}
```

---

## 12. Фронтенд

Файл: `static/app.js` (ключевая логика)

```javascript
const API_URL = 'http://localhost:5000';
const history = [];

async function analyze() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) return showError('Введите текст для анализа');

  setLoading(true);

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error('Ошибка сервера');
    const data = await res.json();

    displayResult(data);
    addToHistory(text, data);
  } catch (err) {
    showError('Не удалось подключиться к серверу');
  } finally {
    setLoading(false);
  }
}

function displayResult(data) {
  const isPositive = data.label === 'positive';

  // Основной результат
  document.getElementById('result-label').textContent =
    isPositive ? 'Позитивный' : 'Негативный';
  document.getElementById('result-card').className =
    `result-card ${isPositive ? 'positive' : 'negative'}`;

  // Шкала уверенности
  const pct = Math.round(data.confidence * 100);
  document.getElementById('confidence-bar').style.width = `${pct}%`;
  document.getElementById('confidence-text').textContent = `Уверенность: ${pct}%`;

  // Топ слов
  const wordsContainer = document.getElementById('top-words');
  wordsContainer.innerHTML = data.top_words
    .map(w => `<span class="word-badge">${w.word}</span>`)
    .join('');

  document.getElementById('result-section').style.display = 'block';
}
```

### Экраны интерфейса

**Главная страница:**

- Логотип и слоган
- Текстовое поле (многострочное, placeholder с примером)
- Кнопка "Анализировать"
- Секция результата (скрыта до первого запроса):
  - Цветной бейдж: зелёный = Позитивный, красный = Негативный
  - Прогресс-бар уверенности
  - Чипы топ-слов
- История последних 10 анализов (список)

**Страница "О модели":**

- Описание алгоритма (TF-IDF + Logistic Regression)
- Метрики: Accuracy, Precision, Recall, F1
- Confusion matrix (таблица)
- Описание датасета

---

## 13. Метрики качества

После обучения на IMDB ожидаются следующие показатели:

| Метрика              | Значение |
| -------------------- | -------- |
| Accuracy             | ~89–91%  |
| Precision (positive) | ~90%     |
| Recall (positive)    | ~88%     |
| F1-Score             | ~89%     |
| AUC-ROC              | ~0.96    |

**Confusion Matrix (примерная, на 10 000 тестовых):**

```
                Predicted
                Positive    Negative
Actual Positive   4 420         580
       Negative     520       4 480
```

**Как интерпретировать:**

- Из 5 000 позитивных отзывов модель правильно определила 4 420 (88.4%)
- Из 5 000 негативных отзывов модель правильно определила 4 480 (89.6%)
- Ошибки минимальны и симметричны

---

## 14. Расширения и улучшения

Если базовая версия готова, можно добавить следующее:

### Техническое улучшение модели

- **Нейросетевая версия:** заменить Logistic Regression на простую LSTM-сеть (PyTorch) — точность вырастет до ~93%
- **Тональность в 3 классах:** добавить класс `neutral` (требует другой датасет)
- **Мультиязычность:** обучить отдельную модель на русскоязычных отзывах

### Улучшения продукта

- **Batch-анализ:** загрузить CSV с колонкой отзывов → получить CSV с результатами
- **API-ключи:** система аутентификации для публичного API
- **Telegram-бот:** интеграция модели в Telegram через python-telegram-bot
- **Дашборд аналитики:** если подключить к базе (SQLite), показывать тренды тональности по времени

---

## 15. Запуск проекта

### Шаг 1: Установка зависимостей

```bash
git clone https://github.com/yourusername/sentiscan.git
cd sentiscan
pip install -r requirements.txt
```

### Шаг 2: Загрузка датасета

Скачать `IMDB Dataset.csv` с Kaggle и поместить в папку `data/`.

### Шаг 3: Обучение модели

```bash
python train.py
```

Выводит метрики и сохраняет `models/vectorizer.pkl` и `models/model.pkl`.
Занимает ~2–3 минуты.

### Шаг 4: Запуск сервера

```bash
python app.py
```

Сервер запустится на `http://localhost:5000`

### Шаг 5: Открыть в браузере

Перейти на `http://localhost:5000` и использовать интерфейс.

---

## 16. Зависимости

Файл: `requirements.txt`

```txt
flask>=3.0.0
flask-cors>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

Установка одной командой:

```bash
pip install -r requirements.txt
```

---

## Авторы и учебный контекст

| Поле                  | Значение                                    |
| --------------------- | ------------------------------------------- |
| Предмет               | Основы искусственного интеллекта            |
| Тип работы            | Учебный проект                              |
| Язык программирования | Python 3.11, JavaScript ES6                 |
| Тип ИИ                | Supervised Machine Learning (классификация) |
| Алгоритм              | TF-IDF + Logistic Regression                |
| Внешние AI API        | Не используются                             |

---

_Документ создан в рамках проекта SentiScan. Все компоненты системы реализованы самостоятельно на базе открытых ML-библиотек._
