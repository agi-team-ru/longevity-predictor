#!/bin/bash

# Активация виртуального окружения (если есть)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Установка зависимостей
echo "Installing dependencies..."
pip install -r requirements.txt

# Запуск Streamlit приложения
echo "Starting Streamlit app..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

echo "Open your browser and go to: http://localhost:8501" 