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
def prepare_data(ctx: Context) -> None:
    """Prepare dataset from download to processed and ready to use for training."""
    ctx.run("dvc pull")
    ctx.run(f"python src/{PROJECT_NAME}/data.py balance", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py split", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py augment", echo=True, pty=not WINDOWS)
    ctx.run(f"python src/{PROJECT_NAME}/data.py preprocess", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run(
        "docker build -t dtumlops_baseimage:latest . -f dockerfiles/dtumlops_baseimage.dockerfile",
        echo=True,
        pty=not WINDOWS,
    )


@task()
def docker_build_gpu(ctx: Context) -> None:
    """Build docker images with gpu support"""
    ctx.run(
        "docker build -t dtumlops_cudaimage:latest . -f dockerfiles/dtumlops_cudaimage.dockerfile",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task()
def docker_remove(ctx: Context) -> None:
    """Removes dtumlops_baseimage"""
    ctx.run("docker rmi dtumlops_baseimage")


@task()
def docker_interactive(ctx: Context) -> None:
    """Runs dtumlops_baseimage"""
    ctx.run("docker run --rm -it dtumlops_baseimage")


@task()
def docker_purge_all(ctx: Context) -> None:
    """Purges all docker images, volumes and cache. Use with caution"""
    ctx.run("docker system prune -a --volumes")
