FROM python:3.12.12-slim
WORKDIR /eidos
COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-lock.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "reproduce.py"]
CMD ["--suite", "all", "--output", "/results", "--workers", "4"]
