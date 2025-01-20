from loguru import logger
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from mlops_project_tcs.model import VGG16Classifier
from mlops_project_tcs.data import BrainMRIDataModule
import wandb


def get_accelerator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Training on device type: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logger.info(f"Training on device type: {device}")

    return device


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.3")
def train_model(config) -> dict:
    """Train VAE on MNIST."""
    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.add(hydra_output_dir / "training.log", level="DEBUG")
    pl.seed_everything(config.experiment.hyperparameter["seed"], workers=True)

    wandb.config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(project=config.wandbconf.project, )

    model = VGG16Classifier(
        input_size=config.experiment.model["input_size"],
        hidden_size=config.experiment.model["hidden_size"],
        num_classes=config.experiment.dataset["num_classes"],
        dropout_p=config.experiment.model["dropout_p"],
        criterion=hydra.utils.instantiate(config.experiment.model.loss_fn),
    )
    model.configure_optimizers(
        optimizer=hydra.utils.instantiate(config.experiment.hyperparameter.optimizer, params=model.parameters())
    )
    data_module = BrainMRIDataModule(
        datadir=config.experiment.dataset["processed_dir"],
        batch_size=config.experiment.dataset["batch_size"],
        val_split=config.experiment.dataset["val_split"],
        test_split=config.experiment.dataset["test_split"],
        num_workers=1,
    )

    accelerator = get_accelerator()

    # Setup trainer
    if config.profiling["training_profiling"]:
        profiling = "simple"
    else:
        profiling = None
    wand_logger = pl.loggers.WandbLogger(project="dtu_mlops")
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=config.experiment.hyperparameter["patience"], verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir=hydra_output_dir,
        accelerator=accelerator,
        log_every_n_steps=10,
        max_epochs=config.experiment.hyperparameter["n_epochs"],
        profiler=profiling,
        logger=wand_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Train
    logger.info("Start training ...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Finish!!")

    # Collect training results
    results = {
        "status": "success",
        "final_epoch": trainer.current_epoch,
        "best_val_loss": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        "total_epochs": config.experiment.hyperparameter["n_epochs"],
    }

    return results


if __name__ == "__main__":
    """ Train VGG16 using hydra configurations """
    train_model()
