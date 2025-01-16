# Base image
# TODO: W.I.P
FROM nvcr.io/nvidia/pytorch:22.07-py3 AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists

# Copy files from local to docker image
COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY requirements_gpu.txt requirements_gpu.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY data/ data/

RUN pip install --upgrade pip
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements_gpu.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

## Entrypoint: Application to run when the image is being executed
## -u sends to local terminal from image
#ENTRYPOINT ["python", "-u", "src/mlops_project_tcs/vae_train.py", "experiment.hyperparameter.n_epochs=5"]
