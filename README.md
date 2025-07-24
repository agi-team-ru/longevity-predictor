# Longevity Research Predictor 🧬

**Система анализа и прогнозирования приоритетных направлений исследований в области биологии старения и долголетия на основе обработки научных статей с использованием графов знаний и машинного обучения.**

## Пример работы сервиса

<div style="display: flex; gap: 24px; align-items: flex-start;">
  <div style="flex: 1; text-align: center;">
    <img src="assets\graph_100_better.png" alt="Граф знаний исследований долголетия" style="width:100%; max-width:350px;"/>
    <br/>
    <span><b>Визуализация графа знаний, показывающая связи между исследовательскими задачами, научными статьями и биомедицинскими терминами</b></span>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="research\neo4j_visuals\zoom_clusters.png" alt="Анализ кластеров задач" style="width:100%; max-width:350px;"/>
    <br/>
    <span><b>Кластеризация исследовательских задач по семантической близости для выявления основных направлений исследований</b></span>
  </div>
</div>

## Развертывание сервиса

### Переменные окружения
```bash
# LLM API настройки
LLM_API_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-dummy-key
LLM_MODEL=llama-3.3-70b-instruct

# Порт для frontend
LISTEN_PORT=8501

# Логирование
LOG_LEVEL=INFO
```

*Перед запуском настройте необходимые переменные окружения в `.env`, или скопируйте `.env.example` в `.env`*

### Загрузка моделей

Скачайте архивы моделей из `https://drive.google.com/drive/folders/1JlpaZjutgf6oe0ln_d58Ml54pwmPnUb8` и распакуйте их в папку `./models` в корне проекта


### Bash запуск
```bash
# Запуск через docker
docker compose up -d
```

*Приложение микросервисное на основе Docker, запуск в одну команду.*

Открыть в браузере `http://localhost/`

## Архитектура

<img src="assets/service_structure.jpg" style="width: 50%">

### Докер контейнеры

На сервисе развернуты следующие компоненты:

<div style="display: flex; gap: 32px; margin-bottom: 24px;">
  <div style="flex: 1; background: #f5f7fa; border-radius: 18px; padding: 24px; min-width: 220px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
    <p style="margin: 0; text-align: center; color: #000000;">
        <b>Frontend (Streamlit)</b>
        <p style="color: #000;">Веб-интерфейс для визуализации и анализа данных. Порт 80.</p>
    </p>
  </div>
  <div style="flex: 1; background: #f5f7fa; border-radius: 18px; padding: 24px; min-width: 220px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
    <p style="margin: 0; text-align: center; color: #000000;">
        <b>Backend (FastAPI)</b>
        <p style="color: #000;">API сервис с NER и embedding моделями. Порт 3000.</p>
    </p>
  </div>
  <div style="flex: 1; background: #f5f7fa; border-radius: 18px; padding: 24px; min-width: 220px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
    <p style="margin: 0; text-align: center; color: #000000;">
        <b>Qdrant (Vector DB)</b>
        <p style="color: #000;">Векторная база данных для хранения эмбеддингов. Порт 6333.</p>
    </p>
  </div>
  <div style="flex: 1; background: #f5f7fa; border-radius: 18px; padding: 24px; min-width: 220px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
    <p style="margin: 0; text-align: center; color: #000000;">
        <b>Neo4j (Graph DB)</b>
        <p style="color: #000;">Графовая база данных для хранения связей. Порт 7474/7687.</p>
    </p>
  </div>
</div>

**ML компоненты в Backend:**
- **2 spaCy модели для NER**: `en_ner_bc5cdr_md` и `en_ner_bionlp13cg_md` для извлечения именованных сущностей
- **SentenceTransformer**: `BAAI/bge-large-en-v1.5` для создания эмбеддингов
- **UMLS линкинг**: планируется интеграция для нормализации биомедицинских терминов

---

### Принцип работы

<img src="assets/work_schema.jpg" style="width: 70%">

**Принцип работы сервиса:**

1. **Обработка статей**: Получаем статьи из PubMed/BioRxiv и анализируем абстракты
2. **Фильтрация**: Проверяем релевантность статей к проблематике долголетия
3. **Извлечение сущностей**: Используем NER модели для выделения биомедицинских терминов
4. **Создание эмбеддингов**: Генерируем векторные представления текстов
5. **Хранение данных**: 
   - Эмбеддинги сохраняем в Qdrant (векторная БД)
   - Связи между сущностями храним в Neo4j (графовая БД)
6. **Визуализация**: Streamlit интерфейс для анализа и отображения результатов

**Текущий статус**: Базовая инфраструктура развернута, ML модели загружаются, планируется интеграция с LLM для извлечения исследовательских задач.

---

### Структура файлов

#### Микросервисная архитектура
```
longevity-predictor/
├── frontend/                    # Streamlit приложение для визуализации
│   ├── src/
│   │   └── app.py              # основной файл Streamlit приложения
│   ├── Dockerfile
│   └── requirements.txt
├── backend/                     # FastAPI сервис с ML компонентами
│   ├── src/
│   │   └── main.py             # API эндпоинты для NER и эмбеддингов
│   ├── Dockerfile
│   └── requirements.txt
├── qdrant/                      # конфигурация векторной базы данных
├── neo4j/                       # конфигурация графовой базы данных
├── models/                      # директория для ML моделей (spaCy, SentenceTransformer)
├── assets/                      # изображения для документации и визуализации
├── docker-compose.yml           # оркестрация всех сервисов
└── .env                         # переменные окружения (создать из .env.example)
```

#### Устаревшие компоненты (из старой версии)
- **`graph_making.py`** - обработка статей и построение графа знаний
- **`llm_task_clustering.py`** - кластеризация задач с LLM
- **`reporting_agent.py`** - генерация отчетов
- **`scorer_v2/`** - система оценки приоритетности

---

## Команда

**Команда AGI Team:**

|     | Имя | GitHub | Позиция |
|-----|-----|--------|---------|
| 1. | Борисов Никита | [**nizier193**](https://github.com/Nizier193) | ML (NLP) |
| 2. | Анна Чифранова | [**amsurex**](https://github.com/amsurex) | ML (NLP) |
| 3. | Дашевский Илья | [**idashevskii**](https://github.com/idashevskii) | Fullstack |
| 4. | Янышевская Карина | [**fanot**](https://github.com/fanot) | ML (NLP) |
| 5. | Вадим Баталев | [**d0zya**](https://github.com/d0zya) | ML (NLP) |
| 6. | Семенов Дмитрий | [**Sem-dmitry**](https://github.com/Sem-dmitry) | ML (NLP) |
