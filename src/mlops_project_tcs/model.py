import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

from loguru import logger
from omegaconf import OmegaConf, DictConfig
import hydra

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src.mlops_project_tcs.data import setup_dataloaders


class VGG16Classifier(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(VGG16Classifier, self).__init__()
        self.criterion = None
        self.optimizer = None

        self.cfg = config
        self.criterion = hydra.utils.instantiate(self.cfg.experiment.model.loss_fn)
        self.train_loader, self.val_loader = setup_dataloaders(
            data_dir="data/processed", batch_size=self.cfg.experiment.dataset["batch_size"]
        )

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze the feature extractor layers to prevent updates during training
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Setup dnn for interpretation of feature extractor output
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(self.cfg.experiment.model["input_size"], self.cfg.experiment.model["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.experiment.model["dropout_p"]),
            nn.Linear(self.cfg.experiment.model["hidden_size"], self.cfg.experiment.model["hidden_size"]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.experiment.model["dropout_p"]),
            nn.Linear(self.cfg.experiment.model["hidden_size"], self.cfg.experiment.dataset.num_classes),
        )

    def forward(self, x):
        return self.vgg16(x)

    def training_step(self, batch):
        """Training step."""
        img, target = batch
        preds = self(img)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return self.criterion(preds, target)

    def validation_step(self, batch) -> None:
        """Validation step."""
        img, target = batch
        preds = self(img)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        self.optimizer = hydra.utils.instantiate(self.cfg.experiment.hyperparameter.optimizer, params=self.parameters())
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


# Example usage
@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.3")
def main(config):
    logger.add("logs/setup_model_example.log", level="DEBUG")
    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")

    model = VGG16Classifier(config=config)

    # Test the model with dummy input
    input_tensor = torch.randn(4, 3, 224, 224)  # Batch size 4, 3 channels, 224x224 image
    output = model(input_tensor)

    print("Model output shape:", output.shape)  # Should be (4, num_classes)


if __name__ == "__main__":
    main()
