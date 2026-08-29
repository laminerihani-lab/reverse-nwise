# Deterministic runtime for the Reverse N-Wise artifact.
# Build:  docker build -t rnwise .
# Run:    docker run --rm -v "$PWD/results:/app/results" rnwise \
#             python -m experiments.exp04_significance --runs 30 --json results/exp04.json
FROM python:3.11-slim

# OpenMP runtime required by XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY rnwise ./rnwise
COPY baselines ./baselines
COPY benchmarks ./benchmarks
COPY experiments ./experiments
COPY analysis ./analysis
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[analysis,dev]"

# Smoke-test the install at build time (fast tests only; slow tests train models
# and fetch datasets, which need network not available during `docker build`).
RUN pytest -q -m "not slow"

CMD ["python", "-m", "experiments.exp01_table1_adult", "--seed", "0"]
