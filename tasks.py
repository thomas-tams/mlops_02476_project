import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project_tcs"
PYTHON_VERSION = "3.9"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def launch_api(ctx: Context) -> None:
    """Launches a uvicorn api server accessible through 'http://localhost:8000/'"""
    ctx.run("uvicorn --reload --port 8000 src.mlops_project_tcs.api:app")


@task
def launch_streamlit(ctx: Context) -> None:
    """Launches a streamlit frontend app"""
    ctx.run("streamlit run src/mlops_project_tcs/frontend.py --server.port 8050 --server.address 0.0.0.0")


@task
def prepare_data(ctx: Context) -> None:
    """Prepare dataset from download to processed and ready to use for training."""
    ctx.run("dvc pull")
    ctx.run(f"python src/{PROJECT_NAME}/data.py balance", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py split", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py augment", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py preprocess", echo=True, pty=not WINDOWS)


@task
def dataset_statistics(ctx: Context) -> None:
    """Creates statistics for the prepared processed data"""
    ctx.run("python src/mlops_project_tcs/data.py dataset-statistics -i data/processed/ -o reports/dataset_statistics/")


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task()
def docker_build_train(ctx: Context) -> None:
    """Build local image (used for training). Requires wandb_api.txt file in home of repository, containing WANDB api key"""    
    if not os.path.exists("data/"):
        ctx.run("invoke prepare-data")
    
    res = ctx.run("cat wandb_api.txt", hide=True, pty=not WINDOWS)
    ctx.run(
        f"docker build --build-arg WANDB_API_KEY={res.stdout.strip()} -t mlopsdtu_local_train:latest . -f dockerfiles/mlopsdtu_local_train.dockerfile",
        echo=True,
        pty=not WINDOWS,
    )

@task()
def docker_train_interactive(ctx: Context) -> None:
    """Runs an interactive session of the mlopsdtu_local_train with gpus and mounted models/ outputs/ directories."""
    ctx.run(
        "docker run --rm --gpus all -it -v $(pwd)/outputs:/app/outputs -v $(pwd)/models:/app/models mlopsdtu_local_train sh",
        echo=True,
        pty=not WINDOWS,
    )


@task()
def docker_remove(ctx: Context) -> None:
    """Removes mlopsdtu_local_train"""
    ctx.run("docker rmi mlopsdtu_local_train")


@task()
def docker_purge_all(ctx: Context) -> None:
    """Purges all docker images, volumes and cache. Use with caution"""
    ctx.run("docker system prune -a --volumes")


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
