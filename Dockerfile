FROM python:3.12-slim
WORKDIR /opt/eidos
COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-lock.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
CMD ["run_benchmarks.py", "--suite", "smoke", "--output", "/results", "--workers", "2"]
