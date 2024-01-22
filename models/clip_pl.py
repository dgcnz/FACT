import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from models.clip_encoder import CLIPClassifier
import torch
from torch import Tensor

class CLIPClassifierTrainer(L.LightningModule):
    def __init__(self, model_name: str, n_classes: int, lr: float):
        super().__init__()
        self.model = CLIPClassifier(model_name=model_name, n_classes=n_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        y_hat = self.model(x)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        return optimizer
    

