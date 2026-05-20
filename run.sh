#!/bin/bash
# Скрипт для быстрого запуска SentiScan

echo "=========================================="
echo "  SentiScan - Запуск сервера"
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
python -c "import flask, sklearn, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ Зависимости не установлены. Устанавливаем..."
    pip install -r requirements.txt
    echo "✓ Зависимости установлены"
else
    echo "✓ Все зависимости установлены"
fi

# Проверка модели
echo ""
if [ ! -f "models/model.pkl" ] || [ ! -f "models/vectorizer.pkl" ]; then
    echo "⚠ Модель не обучена!"
    echo ""
    echo "Сначала обучите модель:"
    echo "  python train.py"
    echo ""
    echo "Для этого нужен датасет IMDB Dataset.csv в папке data/"
    echo "Инструкция по загрузке: data/README.md"
    exit 1
fi

# Запуск сервера
echo ""
echo "=========================================="
echo "  Запуск Flask сервера..."
echo "=========================================="
python app.py
