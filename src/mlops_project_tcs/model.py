import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from loguru import logger
from omegaconf import OmegaConf
import hydra


class VGG16Classifier(pl.LightningModule):
    def __init__(
        self, input_size: int, hidden_size: int, num_classes: int, dropout_p: float, criterion: nn.Module
    ) -> None:
        super(VGG16Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.criterion = criterion

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Freeze the feature extractor layers to prevent updates during training
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Setup dnn for interpretation of feature extractor output
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_size, self.num_classes),
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

    def configure_optimizers(self, optimizer: torch.optim.Optimizer = None):
        """Configure optimizer."""
        if optimizer is not None:
            self.optimizer = optimizer
        return self.optimizer


# Example usage
@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.3")
def main(config):
    logger.add("logs/setup_model_example.log", level="DEBUG")
    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")

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

    # Test the model with dummy input
    input_tensor = torch.randn(4, 3, 224, 224)  # Batch size 4, 3 channels, 224x224 image
    output = model(input_tensor)

    print("Model output shape:", output.shape)  # Should be (4, num_classes)


if __name__ == "__main__":
    main()
