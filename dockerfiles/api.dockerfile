# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Copy files from local to docker image
COPY requirements.txt .
COPY pyproject.toml .
COPY src/mlops_project_tcs/ src/mlops_project_tcs/
COPY data/ data/

RUN pip install --upgrade pip
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["uvicorn", "src/mlops_project_tcs/api:app", "--host", "0.0.0.0", "--port", "8000"]
