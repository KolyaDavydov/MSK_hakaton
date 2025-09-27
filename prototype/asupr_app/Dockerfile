FROM python:3.10-slim

# Устанавливаем переменные окружения для корректного отображения прогресса
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm

WORKDIR /app

# # Устанавливаем системные зависимости для clickhouse-driver
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "-u", "/app/src/generate_data.py"]