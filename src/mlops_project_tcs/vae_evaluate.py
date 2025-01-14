import logging
import os
from pathlib import Path

from typing import Annotated
import typer
import hydra
from omegaconf import OmegaConf, DictConfig
import contextlib

from vae_model import Decoder, Encoder, Model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO or DEBUG for more verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("evaluate.log"),  # Log to a file
    ],
)
log = logging.getLogger(__name__)


def load_model(model_path: Path, config: DictConfig):
    model_conf = config.experiment.model

    encoder = Encoder(
        input_dim=model_conf["x_dim"],
        hidden_dim=model_conf["hidden_dim"],
        latent_dim=model_conf["latent_dim"],
    )
    decoder = Decoder(
        latent_dim=model_conf["latent_dim"],
        hidden_dim=model_conf["hidden_dim"],
        output_dim=model_conf["x_dim"],
    )

    model = Model(encoder=encoder, decoder=decoder)

    # Load weights
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path))

    return model


def evaluate_model(model: Model, config: DictConfig, outdir: Path):
    # Load config
    dataset_conf = config.experiment.dataset
    model_conf = config.experiment.model
    evaluation_profiling = config.profiling["evaluation_profiling"]

    log.info(f"Evaluating model {model_conf['model_name']} on {dataset_conf['dataset_name']}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(f"Evaluating on device type: {device} ({torch.cuda.get_device_name(device)})")
    else:
        log.info(f"Evaluating on device type: {device}")

    # Setup test data
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST(dataset_conf["dataset_path"], transform=mnist_transform, train=False, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=dataset_conf["batch_size"], shuffle=False)

    # Generate reconstructions
    model.to(device)
    model.eval()

    # Setup profiling
    if evaluation_profiling:
        trace_path = outdir / "evaluation_profiling_trace"
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(trace_path),
        )

        log.info(f"Profiling training loops: {evaluation_profiling}")
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context as prof:
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_loader):
                x = x.view(x.size(0), -1)
                x = x.to(device)
                x_hat, _, _ = model(x)

                if evaluation_profiling is not None:
                    prof.step()


# Typer CLI function
@app.command()
def evaluate(
    train_run_output_dir: Annotated[
        str,
        typer.Option(
            "--train_output", "-to", help="Output directory of a training run. Created from running vae_train.py"
        ),
    ] = "outputs/2025-01-09/18-47-30/",
) -> None:
    """
    Command-line entry point to load the model using Hydra configuration.
    """
    train_run_output_dir = Path(train_run_output_dir)

    config = OmegaConf.load(train_run_output_dir / ".hydra" / "config.yaml")

    torch.manual_seed(config.experiment.hyperparameter["seed"])

    model_path = train_run_output_dir / ".hydra" / "config.yaml" / f"{config.experiment.model.model_name}_trained.pt"
    model = load_model(model_path=model_path, config=config)

    evaluate_model(config=config, model=model, out_dir=train_run_output_dir)


if __name__ == "__main__":
    app()
