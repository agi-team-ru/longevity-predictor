FROM python:3.12.11

WORKDIR /app

RUN apt -y update && \
  apt install -y --no-install-recommends curl && \
  rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN --mount=type=cache,mode=0755,target=/root/.cache \
  pip install -r requirements.txt
  
CMD ["python", "-Xfrozen_modules=off", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "streamlit", "run", "src/app.py"]

