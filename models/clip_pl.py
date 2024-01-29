import torch
import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from models.clip_encoder import CLIPClassifier, AudioCLIPClassifier
from torch import Tensor


class AudioCLIPClassifierTrainer(L.LightningModule):
    def __init__(self, pretrained: bool, n_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = AudioCLIPClassifier(pretrained=pretrained, n_classes=n_classes)
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        val_accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', val_accuracy, prog_bar=True)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        return [optimizer], [scheduler]
    

class CLIPClassifierTrainer(L.LightningModule):
    def __init__(self, model_name: str, n_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        val_accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', val_accuracy, prog_bar=True)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.lr,
                                    nesterov=True, dampening=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        return [optimizer], [scheduler]
    