# 🌍 SentiScan — Мультиязычная версия (RU + EN)

## ✅ Что изменено

### 1. Предобработка текста (`utils/preprocess.py`)
- ✅ Добавлена поддержка кириллицы (русские буквы)
- ✅ Добавлена функция `detect_language()` для автоопределения языка
- ✅ Обновлена регулярка: `[^a-zа-яё\s]` вместо `[^a-z\s]`

### 2. Обучение модели (`train.py`)
- ✅ Добавлены русские стоп-слова (150+ слов)
- ✅ Объединены стоп-слова EN + RU
- ✅ Увеличено количество признаков: 15,000 (было 10,000)
- ✅ Добавлен параметр `min_df=2` для фильтрации редких слов
- ✅ Поддержка объединенного датасета `Combined_Dataset.csv`

### 3. API сервер (`app.py`)
- ✅ Импорт функции `detect_language`
- ✅ Определение языка текста перед анализом
- ✅ Возврат поля `language` в JSON ответе

### 4. Веб-интерфейс
- ✅ Отображение определенного языка (🇬🇧 / 🇷🇺)
- ✅ Обновлен placeholder с примерами на обоих языках
- ✅ Добавлен CSS стиль для индикатора языка

### 5. Новые скрипты
- ✅ `create_russian_dataset.py` — генерация тестового русского датасета (1000 отзывов)
- ✅ `merge_datasets.py` — объединение IMDB + русский датасет

## 🚀 Как запустить

### Шаг 1: Создать русский датасет
```bash
cd "/home/levi/Developer/SentiScan AI"
source myenv/bin/activate
python create_russian_dataset.py
```

Это создаст файл `data/Russian_Reviews.csv` с 1000 тестовыми отзывами.

### Шаг 2: Объединить датасеты
```bash
python merge_datasets.py
```

Это создаст `data/Combined_Dataset.csv` (IMDB 50,000 + Russian 1,000 = 51,000 записей).

### Шаг 3: Переобучить модель
```bash
python train.py
```

Модель обучится на объединенном датасете (~3-4 минуты).

### Шаг 4: Запустить сервер
```bash
python app.py
```

Откройте http://localhost:5000 и тестируйте!

## 📊 Ожидаемые результаты

### Метрики модели
- **Accuracy:** ~88-90% (немного ниже из-за меньшего русского датасета)
- **Поддержка языков:** Английский (50k примеров) + Русский (1k примеров)

### Примеры работы

**Английский текст:**
```
Input: "This movie was absolutely amazing! Best film ever."
Output: {
  "label": "positive",
  "confidence": 0.94,
  "language": "en",
  "top_words": ["amazing", "best", "film"]
}
```

**Русский текст:**
```
Input: "Отличный фильм! Очень понравился, рекомендую всем."
Output: {
  "label": "positive",
  "confidence": 0.89,
  "language": "ru",
  "top_words": ["отличный", "понравился", "рекомендую"]
}
```

## 🔧 Технические детали

### TF-IDF Vectorizer
```python
TfidfVectorizer(
    max_features=15_000,           # увеличено для двух языков
    ngram_range=(1, 2),            # биграммы: "not good", "очень плохо"
    stop_words=combined_stopwords, # EN + RU стоп-слова
    sublinear_tf=True,             # log-нормализация
    min_df=2                       # минимум 2 документа
)
```

### Определение языка
```python
def detect_language(text: str) -> str:
    cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    
    if cyrillic_count / total > 0.7:
        return 'ru'
    elif latin_count / total > 0.7:
        return 'en'
    else:
        return 'unknown'
```

## 📝 Улучшения для продакшена

Если хочешь улучшить качество для русского языка:

1. **Больше данных:** Скачай реальный датасет с Kaggle
   - [Отзывы Кинопоиска](https://www.kaggle.com/datasets/alexandersemiletov/kinopoisk-reviews)
   - [Russian Product Reviews](https://www.kaggle.com/datasets/d0rj3228/russian-sentiment-dataset)

2. **Лемматизация:** Добавь `pymorphy2` для русского языка
   ```python
   import pymorphy2
   morph = pymorphy2.MorphAnalyzer()
   lemma = morph.parse(word)[0].normal_form
   ```

3. **Балансировка:** Сделай равное количество EN и RU примеров

4. **Раздельные модели:** Обучи две модели (EN и RU) для максимальной точности

## 🎓 Для защиты проекта

Теперь можешь сказать:
- ✅ Мультиязычная модель (EN + RU)
- ✅ Автоматическое определение языка
- ✅ Единая модель для двух языков (TF-IDF + LogReg)
- ✅ Веб-интерфейс с индикацией языка
- ✅ REST API с полной документацией

## 📞 Быстрые команды

```bash
# Полный цикл с нуля
cd "/home/levi/Developer/SentiScan AI"
source myenv/bin/activate
python create_russian_dataset.py
python merge_datasets.py
python train.py
python app.py

# Или используй скрипты
./train.sh
./run.sh
```

---

**Обновлено:** 2026-05-20  
**Статус:** ✅ Готово к использованию  
**Поддержка языков:** 🇬🇧 English + 🇷🇺 Русский
