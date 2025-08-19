FROM python:3.13.6-slim

WORKDIR /app

RUN pip install pipenv

COPY data/data.csv ../data/data.csv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --ignore-pipfile --system

RUN hf download Qwen/Qwen3-1.7B --local-dir ../model/Qwen3-1.7B

COPY app .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
