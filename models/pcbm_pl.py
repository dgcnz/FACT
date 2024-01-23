import torch
import lightning as L


class PCBMClassifier(torch.nn.Module):
    def __init__(self, n_concepts: int, n_classes: int = 5):
        super(PCBMClassifier, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.classifier = torch.nn.Linear(self.n_concepts, self.n_classes)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)

    def prune(self, concept_ix: int, class_ix: int):
        with torch.no_grad():
            self.classifier.weight[class_ix, concept_ix] = 0.0


class PCBMClassifierTrainer(PCBMClassifier, L.LightningModule):
    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        lr: float,
        weight_decay: float,
        pruned_concept_class_pairs: list[tuple[int, int]] = [],
        normalize: bool = False,
    ):
        super(PCBMClassifierTrainer, self).__init__(
            n_concepts=n_concepts, n_classes=n_classes
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.pruned_concept_class_pairs = pruned_concept_class_pairs
        self.normalize = normalize
        assert (
            not self.normalize or self.pruned_concept_class_pairs
        ), "If normalize is true, pruned_concept_class_pairs must be nonempty"
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("train_accuracy", accuracy)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self(x)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs)
        for concept_ix, class_ix in self.pruned_concept_class_pairs:
            previous_weight = self.classifier.weight[class_ix, concept_ix].item()
            self.prune(concept_ix=concept_ix, class_ix=class_ix)
            zero_weight = self.classifier.weight[class_ix, concept_ix].item()
            assert zero_weight == 0.0, zero_weight
            assert previous_weight != 0.0, previous_weight
        if self.normalize:
            raise NotImplementedError("Normalized not implemented")
