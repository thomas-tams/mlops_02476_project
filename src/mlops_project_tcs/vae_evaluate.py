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
from torch.profiler import profile, ProfilerActivity

app = typer.Typer()
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

def evaluate_model(model: Model, config: DictConfig):
    # Load config
    dataset_conf = config.experiment.dataset
    model_conf = config.experiment.model

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
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(dataset_conf["batch_size"], model_conf["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)

# Typer CLI function
@app.command()
def evaluate(
    train_run_output_dir: Annotated[str, typer.Option("--train_output", "-to", help="Output directory of a training run. Created from running vae_train.py")]
) -> None:
    """
    Command-line entry point to load the model using Hydra configuration.
    """
    config_dir = Path(train_run_output_dir) / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_dir)

    torch.manual_seed(config.experiment.hyperparameter["seed"])

    model_path = Path(train_run_output_dir) / f"{config.experiment.model.model_name}_trained.pt"

    model = load_model(model_path=model_path, config=config)

    evaluate_model(model = model, config=config)

if __name__ == "__main__":
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        app()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
