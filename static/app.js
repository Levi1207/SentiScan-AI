/**
 * SentiScan Frontend Logic
 * Обработка запросов к API и отображение результатов
 */

const API_URL = "http://localhost:5000";
const MAX_HISTORY = 10;
let history = [];

let hiddenMSG = "Stego";
console.log(hiddenMSG);
// Загрузка истории из localStorage при старте
document.addEventListener("DOMContentLoaded", () => {
  loadHistory();
  checkServerHealth();
});

/**
 * Проверка работоспособности сервера
 */
async function checkServerHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (res.ok) {
      console.log("✓ Сервер доступен");
    }
  } catch (err) {
    console.warn("⚠ Сервер недоступен:", err);
  }
}

/**
 * Основная функция анализа текста
 */
async function analyze() {
  const textInput = document.getElementById("text-input");
  const text = textInput.value.trim();

  // Валидация
  if (!text) {
    showError("Введите текст для анализа");
    return;
  }

  if (text.length > 5000) {
    showError("Текст слишком длинный (максимум 5000 символов)");
    return;
  }

  // Скрыть предыдущие ошибки
  hideError();

  // Показать загрузку
  setLoading(true);

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.error || "Ошибка сервера");
    }

    const data = await res.json();

    // Отображение результата
    displayResult(data);

    // Добавление в историю
    addToHistory(text, data);
  } catch (err) {
    console.error("Ошибка при анализе:", err);
    showError(
      err.message ||
        "Не удалось подключиться к серверу. Убедитесь, что сервер запущен.",
    );
  } finally {
    setLoading(false);
  }
}

/**
 * Отображение результата анализа
 */
function displayResult(data) {
  const isPositive = data.label === "positive";

  // Основной результат
  const resultLabel = document.getElementById("result-label");
  resultLabel.textContent = isPositive ? "Позитивный" : "Негативный";

  const resultCard = document.getElementById("result-card");
  resultCard.className = `result-card ${isPositive ? "positive" : "negative"}`;

  // Иконка
  const iconSvg = document.getElementById("result-icon-svg");
  if (isPositive) {
    iconSvg.innerHTML = `
            <circle cx="24" cy="24" r="22" stroke="currentColor" stroke-width="2" fill="none"/>
            <path d="M16 20C16 20 18 23 24 23C30 23 32 20 32 20" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="18" cy="18" r="2" fill="currentColor"/>
            <circle cx="30" cy="18" r="2" fill="currentColor"/>
        `;
  } else {
    iconSvg.innerHTML = `
            <circle cx="24" cy="24" r="22" stroke="currentColor" stroke-width="2" fill="none"/>
            <path d="M16 28C16 28 18 26 24 26C30 26 32 28 32 28" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="18" cy="18" r="2" fill="currentColor"/>
            <circle cx="30" cy="18" r="2" fill="currentColor"/>
        `;
  }

  // Язык текста
  const languageMap = {
    'en': '🇬🇧 Английский',
    'ru': '🇷🇺 Русский',
    'unknown': '❓ Не определен'
  };
  const languageText = languageMap[data.language] || '❓ Не определен';
  document.getElementById("language-text").textContent = `Язык: ${languageText}`;

  // Уверенность
  const confidencePct = Math.round(data.confidence * 100);
  document.getElementById("confidence-text").textContent =
    `Уверенность: ${confidencePct}%`;
  document.getElementById("confidence-bar").style.width = `${confidencePct}%`;

  // Вероятности
  document.getElementById("prob-positive").textContent =
    `${Math.round(data.probabilities.positive * 100)}%`;
  document.getElementById("prob-negative").textContent =
    `${Math.round(data.probabilities.negative * 100)}%`;

  // Топ слов
  const wordsContainer = document.getElementById("top-words");
  if (data.top_words && data.top_words.length > 0) {
    wordsContainer.innerHTML = data.top_words
      .map(
        (w) =>
          `<span class="word-badge" title="Вес: ${w.weight}">${w.word}</span>`,
      )
      .join("");
  } else {
    wordsContainer.innerHTML =
      '<span class="no-words">Нет ключевых слов</span>';
  }

  // Показать секцию результата
  document.getElementById("result-section").style.display = "block";

  // Плавная прокрутка к результату
  document
    .getElementById("result-section")
    .scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Добавление результата в историю
 */
function addToHistory(text, data) {
  const historyItem = {
    id: Date.now(),
    text: text.substring(0, 100) + (text.length > 100 ? "..." : ""),
    label: data.label,
    confidence: data.confidence,
    timestamp: new Date().toLocaleString("ru-RU"),
  };

  history.unshift(historyItem);

  // Ограничение размера истории
  if (history.length > MAX_HISTORY) {
    history = history.slice(0, MAX_HISTORY);
  }

  saveHistory();
  renderHistory();
}

/**
 * Отображение истории
 */
function renderHistory() {
  if (history.length === 0) {
    document.getElementById("history-section").style.display = "none";
    return;
  }

  const historyList = document.getElementById("history-list");
  historyList.innerHTML = history
    .map((item) => {
      const isPositive = item.label === "positive";
      const confidencePct = Math.round(item.confidence * 100);

      return `
            <div class="history-item ${isPositive ? "positive" : "negative"}">
                <div class="history-header">
                    <span class="history-label">${isPositive ? "😊 Позитивный" : "😞 Негативный"}</span>
                    <span class="history-confidence">${confidencePct}%</span>
                </div>
                <p class="history-text">${item.text}</p>
                <span class="history-time">${item.timestamp}</span>
            </div>
        `;
    })
    .join("");

  document.getElementById("history-section").style.display = "block";
}

/**
 * Сохранение истории в localStorage
 */
function saveHistory() {
  try {
    localStorage.setItem("sentiscan_history", JSON.stringify(history));
  } catch (err) {
    console.warn("Не удалось сохранить историю:", err);
  }
}

/**
 * Загрузка истории из localStorage
 */
function loadHistory() {
  try {
    const saved = localStorage.getItem("sentiscan_history");
    if (saved) {
      history = JSON.parse(saved);
      renderHistory();
    }
  } catch (err) {
    console.warn("Не удалось загрузить историю:", err);
    history = [];
  }
}

/**
 * Очистка истории
 */
function clearHistory() {
  if (confirm("Вы уверены, что хотите очистить историю?")) {
    history = [];
    saveHistory();
    renderHistory();
  }
}

/**
 * Очистка поля ввода
 */
function clearInput() {
  document.getElementById("text-input").value = "";
  hideError();
  document.getElementById("text-input").focus();
}

/**
 * Показать состояние загрузки
 */
function setLoading(isLoading) {
  const btn = document.getElementById("analyze-btn");
  const btnText = btn.querySelector(".btn-text");
  const btnLoader = btn.querySelector(".btn-loader");

  if (isLoading) {
    btn.disabled = true;
    btnText.style.display = "none";
    btnLoader.style.display = "flex";
  } else {
    btn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display = "none";
  }
}

/**
 * Показать ошибку
 */
function showError(message) {
  const errorEl = document.getElementById("error-message");
  errorEl.textContent = message;
  errorEl.style.display = "block";
}

/**
 * Скрыть ошибку
 */
function hideError() {
  const errorEl = document.getElementById("error-message");
  errorEl.style.display = "none";
}

/**
 * Обработка Enter в textarea (Ctrl+Enter для отправки)
 */
document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("text-input");
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      analyze();
    }
  });
});
