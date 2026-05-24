"""
Скрипт объединения английского (IMDB) и русского датасетов.
Создает единый датасет для обучения мультиязычной модели.
"""

import pandas as pd
import os
from utils.preprocess import detect_language


def main():
    print("=" * 60)
    print("Объединение датасетов EN + RU")
    print("=" * 60)

    # ── 1. Загрузка английского датасета ────────────────────────────
    print("\n[1/4] Загрузка IMDB датасета...")
    imdb_path = 'data/IMDB Dataset.csv'

    if not os.path.exists(imdb_path):
        print(f"[ERROR] Файл {imdb_path} не найден!")
        print("\nСкачайте датасет с Kaggle:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        return

    df_en = pd.read_csv(imdb_path)
    print(f"[OK] IMDB загружен: {len(df_en):,} записей")

    # ── 2. Загрузка русского датасета ───────────────────────────────
    print("\n[2/4] Загрузка русского датасета...")
    russian_path = 'data/Russian_Reviews.csv'

    if not os.path.exists(russian_path):
        print(f"[WARNING] Файл {russian_path} не найден!")
        print("Создаем тестовый русский датасет...")
        os.system('python create_russian_dataset.py')

        if not os.path.exists(russian_path):
            print("[ERROR] Не удалось создать русский датасет")
            return

    df_ru = pd.read_csv(russian_path)
    print(f"[OK] Русский датасет загружен: {len(df_ru):,} записей")

    # ── 3. Объединение ──────────────────────────────────────────────
    print("\n[3/4] Объединение датасетов...")

    # Добавляем метку языка
    df_en['language'] = 'en'
    df_ru['language'] = 'ru'

    # Объединяем
    df_combined = pd.concat([df_en, df_ru], ignore_index=True)

    # Перемешиваем
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[OK] Объединено: {len(df_combined):,} записей")
    print(f"\nРаспределение по языкам:")
    print(df_combined['language'].value_counts())
    print(f"\nРаспределение по тональности:")
    print(df_combined['sentiment'].value_counts())

    # 4. Сохранение
    print("\n[4/4] Сохранение объединенного датасета...")
    output_path = 'data/Combined_Dataset.csv'
    df_combined.to_csv(output_path, index=False, encoding='utf-8')

    print(f"[OK] Датасет сохранен: {output_path}")

    # Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОБЪЕДИНЕННОГО ДАТАСЕТА")
    print("=" * 60)
    print(f"Всего записей: {len(df_combined):,}")
    print(f"\nПо языкам:")
    print(f"  Английский: {len(df_combined[df_combined['language'] == 'en']):,}")
    print(f"  Русский:    {len(df_combined[df_combined['language'] == 'ru']):,}")
    print(f"\nПо тональности:")
    print(f"  Positive: {len(df_combined[df_combined['sentiment'] == 'positive']):,}")
    print(f"  Negative: {len(df_combined[df_combined['sentiment'] == 'negative']):,}")

    # Примеры
    print(f"\n{'='*60}")
    print("ПРИМЕРЫ ИЗ ДАТАСЕТА")
    print(f"{'='*60}")

    print("\n[EN] Английский (positive):")
    sample_en_pos = df_combined[(df_combined['language'] == 'en') &
                                 (df_combined['sentiment'] == 'positive')]['review'].iloc[0]
    print(f"  {sample_en_pos[:100]}...")

    print("\n[EN] Английский (negative):")
    sample_en_neg = df_combined[(df_combined['language'] == 'en') &
                                 (df_combined['sentiment'] == 'negative')]['review'].iloc[0]
    print(f"  {sample_en_neg[:100]}...")

    print("\n[RU] Русский (positive):")
    sample_ru_pos = df_combined[(df_combined['language'] == 'ru') &
                                 (df_combined['sentiment'] == 'positive')]['review'].iloc[0]
    print(f"  {sample_ru_pos[:100]}...")

    print("\n[RU] Русский (negative):")
    sample_ru_neg = df_combined[(df_combined['language'] == 'ru') &
                                 (df_combined['sentiment'] == 'negative')]['review'].iloc[0]
    print(f"  {sample_ru_neg[:100]}...")

    print("\n" + "=" * 60)
    print("[OK] ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print("\nТеперь запустите обучение: python train.py")


if __name__ == '__main__':
    main()
