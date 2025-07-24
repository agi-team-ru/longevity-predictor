FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка зависимостей
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Скачивание spaCy моделей
RUN python -m spacy download en_core_web_sm

# Копирование приложения
COPY streamlit_app.py .

# Копирование файлов данных
COPY networkx_graphs/ ./
COPY parsed_articles/ ./

# Копирование базового файла tasks_db.json
COPY tasks_db.json .

# Открытие порта
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.runOnSave=true"] 