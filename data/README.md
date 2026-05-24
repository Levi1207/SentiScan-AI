# Инструкция по загрузке датасета IMDB

Для обучения модели необходим датасет IMDB Movie Reviews.

## Вариант 1: Загрузка с Kaggle (рекомендуется)

1. Перейдите на страницу датасета:
   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

2. Нажмите кнопку "Download" (требуется аккаунт Kaggle)

3. Распакуйте скачанный архив и поместите файл `IMDB Dataset.csv` в папку `data/`

## Вариант 2: Через Kaggle CLI

```bash
# Установка Kaggle CLI
pip install kaggle

# Настройка API токена (требуется создать на kaggle.com/settings)
# Поместите kaggle.json в ~/.kaggle/

# Загрузка датасета
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Распаковка
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/
```

## Структура датасета

Файл: `IMDB Dataset.csv`
- Размер: ~66 MB
- Записей: 50,000
- Колонки: `review`, `sentiment`
- Классы: `positive` (25,000), `negative` (25,000)

## Проверка

После загрузки убедитесь, что файл находится по пути:
```
SentiScan AI/
└── data/
    └── IMDB Dataset.csv
```

Затем запустите обучение:
```bash
python train.py
```
