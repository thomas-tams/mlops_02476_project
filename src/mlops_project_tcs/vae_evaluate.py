import logging
import os
from pathlib import Path

from typing import Annotated
import typer

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vae_model import Decoder, Encoder, Model
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
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

def load_model(model_path:Path, config: DictConfig):
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

def evaluate_model(model: Model, config: DictConfig, profiler: profile=None):
    
    # Load config
    dataset_conf = config.experiment.dataset
    model_conf = config.experiment.model

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

    if profiler is not None:
        profiler.step()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(x.size(0), -1)
            x = x.to(device)
            x_hat, _, _ = model(x)

            if profiler is not None:
                profiler.step()


# Typer CLI function
@app.command()
def evaluate(
    train_run_output_dir: Annotated[str, typer.Option("--train_output", "-to", help="Output directory of a training run. Created from running vae_train.py")] = "outputs/2025-01-09/18-47-30/",
    profiling: Annotated[bool, typer.Option("--profiling", "-p", help="Return profiling from running the evaluation")] = False
) -> None:
    """
    Command-line entry point to load the model using Hydra configuration.
    """
    config_dir = Path(train_run_output_dir) / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_dir)

    torch.manual_seed(config.experiment.hyperparameter["seed"])

    model_path = Path(train_run_output_dir) / f"{config.experiment.model.model_name}_trained.pt"
    model = load_model(model_path=model_path, config=config)
    
    if profiling:
        trace_path = Path(train_run_output_dir) / 'evaluation_profiling_trace'
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(trace_path)
        ) as prof:
            log.info("Profiling evaluation run")
            evaluate_model(config=config, model=model, profiler=prof)
    else:
        evaluate_model(config=config, model=model)

if __name__ == "__main__":
    app()
