"""
Скрипт обучения модели для SentiScan.
Загружает IMDB датасет, обучает TF-IDF + Logistic Regression, сохраняет артефакты.
"""

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import clean_text


def main():
    print("=" * 60)
    print("SentiScan — Обучение модели анализа тональности")
    print("=" * 60)

    # ── 1. Загрузка данных ─────────────────────────────────────────
    print("\n[1/7] Загрузка датасета...")

    # Проверяем наличие объединенного датасета
    combined_path = 'data/Combined_Dataset.csv'
    imdb_path = 'data/IMDB Dataset.csv'

    if os.path.exists(combined_path):
        dataset_path = combined_path
        print("[OK] Используем объединенный датасет (EN + RU)")
    elif os.path.exists(imdb_path):
        dataset_path = imdb_path
        print("[WARNING] Используем только IMDB датасет (EN)")
        print("  Для мультиязычной модели запустите: python merge_datasets.py")
    else:
        print(f"[ERROR] ОШИБКА: Датасет не найден!")
        print("\nСкачайте IMDB датасет с Kaggle:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        print(f"И поместите его в папку data/")
        print("\nЗатем запустите: python merge_datasets.py")
        return

    df = pd.read_csv(dataset_path)
    print(f"[OK] Датасет загружен: {len(df):,} записей")
    print(f"\nРаспределение классов:")
    print(df['sentiment'].value_counts())

    # Показываем языковую статистику если есть
    if 'language' in df.columns:
        print(f"\nРаспределение по языкам:")
        print(df['language'].value_counts())

    # ── 2. Предобработка ───────────────────────────────────────────
    print("\n[2/7] Предобработка текстов...")
    df['clean_review'] = df['review'].apply(clean_text)

    # Проверка на пустые тексты после очистки
    empty_count = (df['clean_review'].str.len() == 0).sum()
    if empty_count > 0:
        print(f"[WARNING] Найдено {empty_count} пустых текстов после очистки, удаляем...")
        df = df[df['clean_review'].str.len() > 0]

    print(f"[OK] Очистка завершена. Осталось {len(df):,} записей")

    # ── 3. Разбивка на train/test ──────────────────────────────────
    print("\n[3/7] Разбивка на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']   # сохраняем баланс классов
    )
    print(f"[OK] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 4. TF-IDF векторизация ─────────────────────────────────────
    print("\n[4/7] TF-IDF векторизация...")

    # Русские стоп-слова (самые частые)
    russian_stopwords = [
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все',
        'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
        'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему',
        'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть',
        'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
        'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для',
        'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз',
        'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому',
        'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
        'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно',
        'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот',
        'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три',
        'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть'
    ]

    # Английские стоп-слова (базовые)
    english_stopwords = [
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
        'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once'
    ]

    # Объединяем стоп-слова обоих языков
    combined_stopwords = list(set(russian_stopwords + english_stopwords))

    vectorizer = TfidfVectorizer(
        max_features=15_000,       # увеличили для двух языков
        ngram_range=(1, 2),        # учитываем биграммы ("not good", "очень плохо")
        stop_words=combined_stopwords,  # стоп-слова для обоих языков
        sublinear_tf=True,         # применяем log-нормализацию TF
        min_df=2                   # слово должно встретиться минимум в 2 документах
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"[OK] Размер матрицы признаков: {X_train_vec.shape}")
    print(f"  Словарь: {len(vectorizer.get_feature_names_out())} слов")

    # ── 5. Обучение модели ─────────────────────────────────────────
    print("\n[5/7] Обучение Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,           # регуляризация (меньше = сильнее)
        solver='lbfgs',
        n_jobs=-1,       # использовать все CPU ядра
        random_state=42
    )
    model.fit(X_train_vec, y_train)
    print("[OK] Обучение завершено")

    # ── 6. Оценка ──────────────────────────────────────────────────
    print("\n[6/7] Оценка качества модели...")
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test == 'positive', y_proba)

    print(f"\n{'='*60}")
    print(f"МЕТРИКИ КАЧЕСТВА")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nИнтерпретация:")
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")

    # ── 7. Сохранение артефактов ───────────────────────────────────
    print(f"\n[7/7] Сохранение модели...")
    os.makedirs('models', exist_ok=True)

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("[OK] Модель сохранена в models/")
    print("  - vectorizer.pkl")
    print("  - model.pkl")

    # Тестовый пример
    print(f"\n{'='*60}")
    print("ТЕСТОВЫЙ ПРИМЕР")
    print(f"{'='*60}")
    test_texts = [
        "This movie was absolutely amazing! Best film I've ever seen.",
        "Terrible waste of time. I hated every minute of it.",
        "It was okay, nothing special but not bad either."
    ]

    for text in test_texts:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        label = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = proba.max()

        print(f"\nТекст: {text}")
        print(f"Результат: {label.upper()} (уверенность: {confidence*100:.1f}%)")

    print(f"\n{'='*60}")
    print("[OK] ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print(f"{'='*60}")
    print("\nТеперь запустите сервер: python app.py")


if __name__ == '__main__':
    main()
