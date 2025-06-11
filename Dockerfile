
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Konfigurasi variabel lingkungan Flask
ENV FLASK_APP=app.py

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app