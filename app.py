"""
Flask API сервер для SentiScan.
Предоставляет REST API для анализа тональности текста.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import clean_text, detect_language

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Глобальные переменные для модели
vectorizer = None
model = None
feature_names = None


def load_model():
    """Загрузка модели при старте сервера."""
    global vectorizer, model, feature_names

    print("=" * 60)
    print("SentiScan API Server")
    print("=" * 60)
    print("\nЗагрузка модели...")

    if not os.path.exists('models/vectorizer.pkl'):
        print("❌ ОШИБКА: models/vectorizer.pkl не найден!")
        print("Сначала обучите модель: python train.py")
        sys.exit(1)

    if not os.path.exists('models/model.pkl'):
        print("❌ ОШИБКА: models/model.pkl не найден!")
        print("Сначала обучите модель: python train.py")
        sys.exit(1)

    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()

    print("✓ Модель загружена успешно!")
    print(f"  Словарь: {len(feature_names)} слов")
    print(f"  Классы: {model.classes_}")
    print("\nСервер готов к работе!")
    print("=" * 60)


def get_top_words(text_vec, n=5):
    """
    Получить топ-N слов с наибольшим вкладом в предсказание.

    Args:
        text_vec: Векторизованный текст (sparse matrix)
        n: Количество топ-слов

    Returns:
        Список словарей с словами и их весами
    """
    feature_weights = model.coef_[0]           # веса модели
    text_weights = text_vec.toarray()[0]       # TF-IDF веса текста
    combined = feature_weights * text_weights  # вклад каждого слова

    # Сортируем по абсолютному значению вклада
    top_idx = np.argsort(np.abs(combined))[-n:][::-1]

    return [
        {
            "word": feature_names[i],
            "weight": round(float(combined[i]), 4)
        }
        for i in top_idx if text_weights[i] > 0
    ]


@app.route('/')
def index():
    """Главная страница."""
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Эндпоинт для анализа тональности текста.

    Request JSON:
        {
            "text": "Your review text here"
        }

    Response JSON:
        {
            "label": "positive" | "negative",
            "confidence": 0.943,
            "language": "en" | "ru" | "unknown",
            "top_words": [{"word": "amazing", "weight": 0.312}, ...],
            "probabilities": {"positive": 0.943, "negative": 0.057}
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Не передан JSON'}), 400

        text = data.get('text', '').strip()

        # Валидация
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400

        if len(text) > 5000:
            return jsonify({'error': 'Текст слишком длинный (макс. 5000 символов)'}), 400

        # Предобработка
        cleaned = clean_text(text)

        if not cleaned:
            return jsonify({'error': 'После очистки текст пустой. Попробуйте другой текст.'}), 400

        # Определение языка
        language = detect_language(text)

        # Векторизация
        vec = vectorizer.transform([cleaned])

        # Предсказание
        label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = float(proba.max())

        # Топ слов
        top_words = get_top_words(vec, n=5)

        # Формирование ответа
        response = {
            'label': label,
            'confidence': round(confidence, 3),
            'language': language,
            'top_words': top_words,
            'probabilities': {
                'positive': round(float(proba[list(model.classes_).index('positive')]), 3),
                'negative': round(float(proba[list(model.classes_).index('negative')]), 3),
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Проверка работоспособности сервера.

    Response JSON:
        {
            "status": "ok",
            "model": "logistic_regression_tfidf",
            "vocabulary_size": 10000
        }
    """
    return jsonify({
        'status': 'ok',
        'model': 'logistic_regression_tfidf',
        'vocabulary_size': len(feature_names) if feature_names is not None else 0
    })


@app.route('/stats', methods=['GET'])
def stats():
    """
    Статистика модели.

    Response JSON:
        {
            "model_type": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
            "vocabulary_size": 10000,
            "classes": ["negative", "positive"]
        }
    """
    return jsonify({
        'model_type': type(model).__name__,
        'vectorizer': type(vectorizer).__name__,
        'vocabulary_size': len(feature_names),
        'classes': model.classes_.tolist()
    })


if __name__ == '__main__':
    load_model()
    print("\n🚀 Запуск сервера на http://localhost:5000")
    print("Нажмите Ctrl+C для остановки\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
