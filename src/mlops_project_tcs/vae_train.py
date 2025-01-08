"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vae_model import Decoder, Encoder, Model
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.3")
def train(config) -> None:
    """Train VAE on MNIST."""
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dataset_conf = config.experiment.dataset
    model_conf = config.experiment.model
    hparams = config.experiment.hyperparameter
    torch.manual_seed(hparams["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_conf["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_conf["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=dataset_conf["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=dataset_conf["batch_size"], shuffle=False)

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

    model = Model(encoder=encoder, decoder=decoder).to(device)

    from torch.optim import Adam

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld

    optimizer = Adam(model.parameters(), lr=hparams["lr"])

    log.info("Start training VAE...")
    model.train()
    for epoch in range(hparams["n_epochs"]):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 500 == 0:
                print(batch_idx)
            x = x.view(dataset_conf["batch_size"], model_conf["x_dim"])
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete! Average Loss: {overall_loss / (batch_idx*dataset_conf['batch_size'])}")
    log.info("Finish!!")

    # save weights
    model_path = hydra_output_dir / f"{model_conf['model_name']}_trained.pt"
    torch.save(model, model_path)

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(dataset_conf["batch_size"], model_conf["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)
            break
    
    orig_data_img_path = hydra_output_dir / f"{model_conf['model_name']}_orig_data.png"
    recon_data_img_path = hydra_output_dir / f"{model_conf['model_name']}_reconstructions.png"
    save_image(x.view(dataset_conf["batch_size"], 1, 28, 28), orig_data_img_path)
    save_image(x_hat.view(dataset_conf["batch_size"], 1, 28, 28), recon_data_img_path)

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(dataset_conf["batch_size"], model_conf["latent_dim"]).to(device)
        generated_images = decoder(noise)

    generated_img_path = hydra_output_dir / f"{model_conf['model_name']}_generated_sample.png"
    save_image(generated_images.view(dataset_conf["batch_size"], 1, 28, 28), generated_img_path)


if __name__ == "__main__":
    train()
