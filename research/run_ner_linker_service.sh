#!/bin/bash

# Запуск FastAPI сервиса для NER и EntityLinker
# Используйте: bash run_ner_linker_service.sh

# Можно указать свой путь к python, если нужно
PYTHON=python

$PYTHON -m uvicorn ner_linker_service:app --host 0.0.0.0 --port 8001 