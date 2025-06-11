
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Konfigurasi variabel lingkungan Flask
ENV FLASK_APP=app.py