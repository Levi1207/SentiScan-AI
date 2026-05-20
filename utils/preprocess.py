"""
Модуль предобработки текста для SentiScan.
Функции применяются одинаково при обучении и инференсе.
Поддержка английского и русского языков.
"""

import re


def clean_text(text: str) -> str:
    """
    Очистка текста для подачи в модель.
    Применяется одинаково при обучении и инференсе.
    Поддерживает английский и русский языки.

    Args:
        text: Исходный текст отзыва

    Returns:
        Очищенный текст (lowercase, без HTML, URL, спецсимволов)
    """
    # 1. Приведение к нижнему регистру
    text = text.lower()

    # 2. Удаление HTML-тегов (отзывы IMDB часто содержат <br />, <p> и т.д.)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Удаление URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Удаление email адресов
    text = re.sub(r'\S+@\S+', '', text)

    # 5. Удаление небуквенных символов (оставляем латиницу, кириллицу и пробелы)
    text = re.sub(r'[^a-zа-яё\s]', '', text)

    # 6. Замена множественных пробелов на один
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def detect_language(text: str) -> str:
    """
    Определение языка текста (русский или английский).

    Args:
        text: Исходный текст

    Returns:
        'ru' для русского, 'en' для английского, 'unknown' если не определен
    """
    # Подсчет кириллических и латинских букв
    cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    total = cyrillic_count + latin_count

    if total == 0:
        return 'unknown'

    # Если больше 70% кириллицы — русский
    if cyrillic_count / total > 0.7:
        return 'ru'
    # Если больше 70% латиницы — английский
    elif latin_count / total > 0.7:
        return 'en'
    else:
        return 'unknown'
