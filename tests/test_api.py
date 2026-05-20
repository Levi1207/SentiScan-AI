"""
Простые тесты для API эндпоинтов SentiScan.
Запуск: python -m pytest tests/test_api.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Проверка, что все модули импортируются."""
    from utils.preprocess import clean_text
    assert callable(clean_text)


def test_clean_text():
    """Тест функции предобработки текста."""
    from utils.preprocess import clean_text

    # Тест 1: HTML теги
    text = "This is <br> a test <p>paragraph</p>"
    result = clean_text(text)
    assert '<' not in result
    assert '>' not in result

    # Тест 2: Приведение к нижнему регистру
    text = "HELLO World"
    result = clean_text(text)
    assert result == "hello world"

    # Тест 3: Удаление спецсимволов
    text = "Hello! How are you? I'm fine."
    result = clean_text(text)
    assert result == "hello how are you im fine"

    # Тест 4: Множественные пробелы
    text = "Hello    world"
    result = clean_text(text)
    assert result == "hello world"

    print("✓ Все тесты предобработки пройдены")


if __name__ == '__main__':
    test_imports()
    test_clean_text()
    print("\n✓ Все тесты успешно пройдены!")
