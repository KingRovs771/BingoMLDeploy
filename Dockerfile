
FROM python:3.9-slim-buster

WORKDIR /app

COPY ./requirements.txt /code/requirements

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]