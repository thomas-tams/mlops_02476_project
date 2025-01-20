import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import Accuracy

class MyModel(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.001):
        super().__init__()
        self.layer = nn.Linear(28 * 28, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 10)
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the image
        x = self.layer(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
