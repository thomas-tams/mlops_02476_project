import logging
from pathlib import Path
import typer
import contextlib
import hydra
from omegaconf import OmegaConf, DictConfig
from mlops_project_tcs.vae_model import Decoder, Encoder, Model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


log = logging.getLogger(__name__)
app = typer.Typer()

# Global variable to store profiling flag
TRAIN_PROFILING = False


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(f"Training on device type: {device} ({torch.cuda.get_device_name(device)})")
    else:
        log.info(f"Training on device type: {device}")

    return device


def loss_function(x, x_hat, mean, log_var):
    """Loss function for VAE (reconstruction + kl-divergence)"""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


def setup_model(config: DictConfig) -> None:
    """Setting up VAE model based on hydra configs"""
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

    return model


def setup_dataloaders(config: DictConfig) -> None:
    """Setting up train and test dataloaders for mnist dataset"""
    dataset_conf = config.experiment.dataset

    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_conf["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_conf["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=dataset_conf["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=dataset_conf["batch_size"], shuffle=False)

    return train_loader, test_loader


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.3")
def train_model(config) -> None:
    """Train VAE on MNIST."""
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Load hydra into variables
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dataset_conf = config.experiment.dataset
    model_conf = config.experiment.model
    hparams = config.experiment.hyperparameter
    train_profiling = config.profiling["training_profiling"]

    torch.manual_seed(hparams["seed"])

    device = get_device()

    train_loader, test_loader = setup_dataloaders(config)

    model = setup_model(config)
    model = model.to(device)

    # TODO: Setup hydra parsing of optimizer
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=hparams["lr"])

    # Setup profiling
    if train_profiling:
        trace_path = hydra_output_dir / "training_profiling_trace"
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(trace_path),
        )

        log.info(f"Profiling training loops: {train_profiling}")
    else:
        profiler_context = contextlib.nullcontext()

    log.info("Start training VAE...")
    model.train()
    loaded_samples = 0
    with profiler_context as prof:
        for epoch in range(hparams["n_epochs"]):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                loaded_samples += x.shape[0]
                print(round(batch_idx / len(train_loader) * 100, 2), "%")

                x = x.view(x.size(0), -1)  # Flatten each sample
                x = x.to(device)
                x_hat, mean, log_var = model(x)

                optimizer.zero_grad()
                loss = loss_function(x, x_hat, mean, log_var)
                loss.backward()
                optimizer.step()

                overall_loss += loss.item()

                if train_profiling:
                    if loaded_samples > 5000:
                        log.info("Ending training early to not fill memory with huge amount of logging data")
                        break
                    prof.step()

            log.info(
                f"Epoch {epoch + 1} complete! Average Loss: {overall_loss / (batch_idx * dataset_conf['batch_size'])}"
            )
            if train_profiling:
                break
        log.info("Finish!!")

    # Save weights
    model_path = hydra_output_dir / f"{model_conf['model_name']}_trained.pt"
    torch.save(model.state_dict(), model_path)

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
        generated_images = model.decoder(noise)

    generated_img_path = hydra_output_dir / f"{model_conf['model_name']}_generated_sample.png"
    save_image(generated_images.view(dataset_conf["batch_size"], 1, 28, 28), generated_img_path)


if __name__ == "__main__":
    """ Train VAE on MNIST using hydra configurations """
    train_model()
