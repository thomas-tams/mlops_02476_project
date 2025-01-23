import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from typing import Any, Optional


class VGG16Classifier(pl.LightningModule):
    """
    VGG16-based classifier using PyTorch Lightning.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        num_classes (int): Number of output classes.
        dropout_p (float): Dropout probability.
        criterion (nn.Module): Loss function.
    """

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
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.vgg16(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        img, target = batch
        preds = self(img)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Validation step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Batch index.
        """
        img, target = batch
        preds = self(img)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self, optimizer: Optional[torch.optim.Optimizer] = None) -> torch.optim.Optimizer:
        """
        Configure optimizer.

        Args:
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to use.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        return self.optimizer
