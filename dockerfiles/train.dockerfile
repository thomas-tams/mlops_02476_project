# Base image
FROM python:3.9-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists

WORKDIR /app

# Copy files from local to docker image
COPY requirements.txt .
COPY pyproject.toml .
COPY src/mlops_project_tcs/ src/mlops_project_tcs/
COPY data/ data/

RUN pip install --upgrade pip
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir


# Entrypoint: Application to run when the image is being executed
# -u sends to local terminal from image
#ENTRYPOINT ["python", "-u", "src/mlops_project_tcs/vae_train.py", "experiment.hyperparameter.n_epochs=5"]