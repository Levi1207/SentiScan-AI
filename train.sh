#!/bin/bash
# Скрипт для обучения модели SentiScan

echo "=========================================="
echo "  SentiScan - Обучение модели"
echo "=========================================="

# Проверка виртуального окружения
if [ ! -d "myenv" ] && [ ! -d "venv" ]; then
    echo "⚠ Виртуальное окружение не найдено. Создаём..."
    python3 -m venv venv
    echo "✓ Виртуальное окружение создано"
fi

# Активация виртуального окружения
if [ -d "myenv" ]; then
    source myenv/bin/activate
    echo "✓ Активировано окружение: myenv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Активировано окружение: venv"
fi

# Проверка зависимостей
echo ""
echo "Проверка зависимостей..."
python -c "import sklearn, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ Зависимости не установлены. Устанавливаем..."
    pip install -r requirements.txt
    echo "✓ Зависимости установлены"
else
    echo "✓ Все зависимости установлены"
fi

# Проверка датасета
echo ""
if [ ! -f "data/IMDB Dataset.csv" ]; then
    echo "❌ ОШИБКА: Датасет не найден!"
    echo ""
    echo "Скачайте датасет IMDB с Kaggle:"
    echo "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    echo ""
    echo "Поместите файл 'IMDB Dataset.csv' в папку data/"
    echo "Подробная инструкция: data/README.md"
    exit 1
fi

# Запуск обучения
echo ""
echo "=========================================="
echo "  Запуск обучения модели..."
echo "=========================================="
python train.py
