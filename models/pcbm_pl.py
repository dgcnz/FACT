import torch
import lightning as L
from typing import Literal, Any, Callable
import logging
from concepts import ConceptBank
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)
import matplotlib.pyplot as plt
from torch import Tensor
from models.metrics.spurious_metrics import PrecisionForClass, AccuracyForClass, RecallForClass


class ConceptProjectionModule(torch.nn.Module):
    vectors: Tensor
    intercepts: Tensor
    norms: Tensor

    def __init__(self, pickle_path: str):
        super().__init__()
        concept_bank = ConceptBank.from_pickle(
            pickle_path=pickle_path, device=None, sort_by_keys=True
        )
        self.register_buffer("vectors", concept_bank.vectors, persistent=True)
        self.register_buffer("intercepts", concept_bank.intercepts, persistent=True)
        self.register_buffer("norms", concept_bank.norms, persistent=True)

    def forward(self, x: Tensor):
        with torch.no_grad():
            concept_weights = ((self.vectors @ x.T) + self.intercepts) / (self.norms)
            return concept_weights.T


def normalized_prune(a: torch.Tensor, zeroed_indices: list[int]):
    """
    Given tensor a and list of `zeroed_indices`, return tensor b
    such that b[i] = 0 for all `i` in `zeroed_indices` and
    the l1 norm of b is the same as the l1 norm of a.
    """
    with torch.no_grad():
        b = a.clone()
        b[zeroed_indices] = 0.0

        l1_norm_a = a.abs().sum()
        l1_norm_b = b.abs().sum()

        if l1_norm_a != 0:
            scaling_factor = l1_norm_a / l1_norm_b
            b *= scaling_factor
        return b


class PCBMClassifier(torch.nn.Module):
    def __init__(self, n_concepts: int, n_classes: int):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.classifier = torch.nn.Linear(self.n_concepts, self.n_classes)
        # self.classifier.weight.data.fill_(0.00)
        # self.classifier.bias.data.fill_(0.00)

    def forward(self, x: Tensor):
        return self.classifier(x)

    def prune(self, concept_ix: int, class_ix: int):
        with torch.no_grad():
            self.classifier.weight[class_ix, concept_ix] = 0.0

    def prune_and_normalize(self, concept_ix: int, class_ix: int):
        with torch.no_grad():
            previous_norm = self.classifier.weight[class_ix].abs().sum()
            self.classifier.weight[class_ix] = normalized_prune(
                self.classifier.weight[class_ix], [concept_ix]
            )
            rescaled_norm = self.classifier.weight[class_ix].abs().sum()
            assert torch.allclose(
                rescaled_norm, previous_norm
            ), f"Rescaled norm {rescaled_norm} != previous norm {previous_norm}, diff {abs(rescaled_norm - previous_norm)}"


class PCBMClassifierV2(torch.nn.Module):
    def __init__(
        self, n_concepts: int, n_classes: int, concept_bank: ConceptProjectionModule
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.concept_bank = concept_bank
        self.classifier = torch.nn.Linear(self.n_concepts, self.n_classes)

    def forward(self, x: Tensor):
        with torch.no_grad():
            x = self.concept_bank(x)
        return self.classifier(x)

    def prune(self, concept_ix: int, class_ix: int):
        with torch.no_grad():
            self.classifier.weight[class_ix, concept_ix] = 0.0

    def prune_and_normalize(self, concept_ix: int, class_ix: int):
        norm_ord = 1
        with torch.no_grad():
            previous_norm = torch.linalg.vector_norm(
                self.classifier.weight[class_ix], ord=norm_ord
            ).item()
            pruned_norm = self.classifier.weight[class_ix, concept_ix].item()
            self.classifier.weight[class_ix, concept_ix] = 0.0
            unpruned_norm = torch.linalg.vector_norm(
                self.classifier.weight[class_ix], ord=norm_ord
            ).item()
            self.classifier.weight[class_ix] = self.classifier.weight[class_ix] * (
                1 + pruned_norm / unpruned_norm
            )
            rescaled_norm = torch.linalg.vector_norm(
                self.classifier.weight[class_ix], ord=1
            ).float()
            EPS = 1e-1
            assert (
                abs(rescaled_norm - previous_norm) < EPS
            ), f"Rescaled norm {rescaled_norm} != previous norm {previous_norm}, diff {abs(rescaled_norm - previous_norm)}"


def analyze_classifier(
    model: PCBMClassifierV2,
    class_names: list[str],
    concept_names: list[str],
    k=5,
    print_lows=False,
):
    weights = model.classifier.weight.clone().detach()
    output = []

    if len(class_names) == 2:
        weights = [weights.squeeze(), weights.squeeze()]

    for idx, cls in enumerate(class_names):
        cls_weights = weights[idx]
        topk_vals, topk_indices = torch.topk(cls_weights, k=k)
        topk_indices = topk_indices.detach().cpu().numpy()
        topk_concepts = [concept_names[j] for j in topk_indices]
        analysis_str = [f"Class : {cls}"]
        for j, c in enumerate(topk_concepts):
            analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
        analysis_str = "\n".join(analysis_str)
        output.append(analysis_str)

        if print_lows:
            topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [concept_names[j] for j in topk_indices]
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

    analysis = "\n".join(output)
    return analysis


class PCBMLoss(torch.nn.Module):
    def __init__(
        self,
        lam: float,
        n_concepts: int,
        n_classes: int,
        alpha: float,
        classifier: torch.nn.Module,
        reduction: Literal["sum", "mean"] = "sum",
    ):
        """
        Calculates the PCBM loss with elastic net regularization
        weighted by lam / (n_classes * n_concepts)

        Args:
        :param lam: numerator in elastic net constant
        :param n_concepts: number of concepts used
        :param n_classes: number of classes used
        :param alpha: l1 to l2 ratio in elastic net
        :param classifier: classifier to optimize
        :param reduction: reduction to use for loss function
        """
        super(PCBMLoss, self).__init__()
        self.lam = lam
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.alpha = alpha
        self.classifier = classifier
        self.reduction = reduction
        self.reduce_fn = torch.sum if self.reduction == "sum" else torch.mean
        self.elastic_net_constant = self.lam / (self.n_classes * self.n_concepts)
        assert 0 <= self.alpha and self.alpha <= 1, self.alpha

    def _per_param_l1_norm(self, param: Tensor):
        return param.abs().sum()

    def _per_param_squared_l2_norm(self, param: Tensor):
        return param.square().sum()

    def _per_sample_elastic_net(self, param: Tensor):
        """
        Calculates the elastic net for a single parameter
        """
        l1_norm = self._per_param_l1_norm(param)
        squared_l2_norm = self._per_param_squared_l2_norm(param)
        return self.alpha * l1_norm + (1 - self.alpha) * squared_l2_norm

    def _aggregate_param_fun(self, per_param_fun: Callable[[Tensor], Tensor]) -> Tensor:
        return self.reduce_fn(
            torch.stack(
                [per_param_fun(param) for param in self.classifier.parameters()]
            )
        )

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Calculates the PCBM loss
        """
        base_loss = self.base_loss(y_hat=y_hat, y=y)
        elastic_net = self._aggregate_param_fun(self._per_sample_elastic_net)
        return base_loss + self.elastic_net_constant * elastic_net

    def l1_loss(self):
        return self.alpha * self._aggregate_param_fun(self._per_param_l1_norm)

    def l2_loss(self):
        return (1 - self.alpha) * self._aggregate_param_fun(
            self._per_param_squared_l2_norm
        )

    def elastic_net_loss(self):
        return self.elastic_net_constant * self._aggregate_param_fun(
            self._per_sample_elastic_net
        )

    def base_loss(self, y_hat: Tensor, y: Tensor):
        return torch.nn.functional.cross_entropy(y_hat, y, reduction=self.reduction)



class PCBMClassifierTrainerV2(L.LightningModule):
    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        lr: float,
        lam: float,
        alpha: float,
        concept_bank_path: str,  # Any allows classmethods for jsonargparse
        pruned_concept_class_pairs: list[tuple[int, int]] = [],
        normalize: bool = False,
    ):
        super().__init__()
        self.concept_projection = ConceptProjectionModule(pickle_path=concept_bank_path)
        self.model = PCBMClassifierV2(
            n_concepts=n_concepts,
            n_classes=n_classes,
            concept_bank=self.concept_projection,
        )
        self.lr = lr
        self.pruned_concept_class_pairs = pruned_concept_class_pairs
        self.normalize = normalize
        self.lam = lam
        self.alpha = alpha
        self.n_classes = n_classes
        self.loss = PCBMLoss(
            lam=self.lam,
            n_concepts=n_concepts,
            n_classes=n_classes,
            alpha=self.alpha,
            classifier=self.model.classifier,
        )
        assert (
            not self.normalize or self.pruned_concept_class_pairs
        ), "If normalize is true, pruned_concept_class_pairs must be nonempty"
        self.save_hyperparameters()

        self._setup_metrics()

    def _setup_metrics(self):
        self.train_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.cm = {
            "train": MulticlassConfusionMatrix(num_classes=self.n_classes),
            "test": MulticlassConfusionMatrix(num_classes=self.n_classes),
        }
        self.precision = {
            "test": MulticlassPrecision(num_classes=self.n_classes),
        }
        self.recall = {
            "test": MulticlassRecall(num_classes=self.n_classes),
        }

    def forward(self, batch):
        # x = batch["concept_weights"]
        x = batch["input"]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("train_accuracy", self.train_accuracy)
        self.log("train_loss", loss)
        self.log("train_l1_loss", self.loss.l1_loss())
        self.log("train_l2_loss", self.loss.l2_loss())
        self.log("train_elastic_net_loss", self.loss.elastic_net_loss())
        self.log("train_base_loss", self.loss.base_loss(y_hat, y))
        self.cm["train"].update(y_hat.cpu().detach(), y.cpu().detach())
        return loss

    def on_train_epoch_start(self):
        # Reset on each epoch start so that only the last epoch is logged
        self.cm["train"].reset()

    def on_train_end(self):
        self._log_confusion_matrix_and_reset("train")

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        self.test_accuracy(y_hat, y)
        self.log("test_accuracy", self.test_accuracy)
        self.cm["test"].update(y_hat.cpu().detach(), y.cpu().detach())
        self.precision["test"].update(y_hat.cpu().detach(), y.cpu().detach())
        self.test_recall.update(y_hat.cpu().detach(), y.cpu().detach())

    def _log_confusion_matrix_and_reset(self, stage: str):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.title.set_text(f"{stage} confusion matrix")
        self.cm[stage].plot(ax=ax)
        self.logger.experiment.add_figure(
            f"{stage}_confusion_matrix",
            fig,
        )
        plt.close(fig)
        self.cm[stage].reset()

    def on_test_end(self):
        # log histogram of classifier weights to tensorboard
        weights = self.model.classifier.weight.detach().cpu().reshape(-1).numpy()
        self.logger.experiment.add_histogram("classifier_weights", weights, 0)
        self._log_confusion_matrix_and_reset("test")
        self.log("test_precision", self.precision["test"])
        self.log("test_recall", self.test_recall)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        return optimizer

    def load_state_dict(self, *args, **kwargs):
        logging.info(
            f"Calling custom load_state_dict with {self.pruned_concept_class_pairs}"
        )
        super().load_state_dict(*args, **kwargs)
        if not self.normalize:
            for concept_ix, class_ix in self.pruned_concept_class_pairs:
                previous_weight = self.model.classifier.weight[
                    class_ix, concept_ix
                ].item()
                self.model.prune(concept_ix=concept_ix, class_ix=class_ix)
                zero_weight = self.model.classifier.weight[class_ix, concept_ix].item()
                logging.info(
                    f"Pruned {concept_ix} {class_ix} from {previous_weight} to {zero_weight}"
                )
                assert zero_weight == 0.0, zero_weight
                assert previous_weight != 0.0, previous_weight
        else:
            if len(self.pruned_concept_class_pairs) == 1:
                concept_ix, class_ix = self.pruned_concept_class_pairs[0]
                previous_weight = self.model.classifier.weight[
                    class_ix, concept_ix
                ].item()
                self.model.prune_and_normalize(
                    concept_ix=concept_ix,
                    class_ix=class_ix,
                )
                zero_weight = self.model.classifier.weight[class_ix, concept_ix].item()
                logging.info(
                    f"Pruned {concept_ix} {class_ix} from {previous_weight} to {zero_weight}"
                    " and normalized remaining weights."
                )
            else:
                raise NotImplementedError(
                    f"Currently prune_and_normalize only handles one prune."
                )


class PCBMClassifierTrainer(L.LightningModule):
    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        lr: float,
        lam: float,
        alpha: float,
        pruned_concept_class_pairs: list[tuple[int, int]] = [],
        normalize: bool = False,
        spurious_class: int = None,
    ):
        super().__init__()
        self.model = PCBMClassifier(
            n_concepts=n_concepts,
            n_classes=n_classes,
        )
        self.lr = lr
        self.pruned_concept_class_pairs = pruned_concept_class_pairs
        self.normalize = normalize
        self.lam = lam
        self.alpha = alpha
        self.n_classes = n_classes
        self.spurious_class = spurious_class
        self.loss = PCBMLoss(
            lam=self.lam,
            n_concepts=n_concepts,
            n_classes=n_classes,
            alpha=self.alpha,
            classifier=self.model.classifier,
        )
        assert (
            not self.normalize or self.pruned_concept_class_pairs
        ), "If normalize is true, pruned_concept_class_pairs must be nonempty"
        self.save_hyperparameters()

        self._setup_metrics()

    def _setup_metrics(self):
        self.train_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.cm = {
            "train": MulticlassConfusionMatrix(num_classes=self.n_classes),
            "test": MulticlassConfusionMatrix(num_classes=self.n_classes),
        }
        self.test_precision = MulticlassPrecision(
            num_classes=self.n_classes, average="macro"
        )
        self.test_recall = MulticlassRecall(num_classes=self.n_classes, average="macro")
        if self.spurious_class is not None:
            self.test_spurious_class_precision = PrecisionForClass(
                num_classes=self.n_classes, class_index=self.spurious_class
            )
            self.test_spurious_class_recall = RecallForClass(
                num_classes=self.n_classes, class_index=self.spurious_class
            )
            self.test_spurious_class_accuracy = AccuracyForClass(
                num_classes=self.n_classes, class_index=self.spurious_class
            )

    def forward(self, batch):
        # x = batch["concept_weights"]
        x = batch["input"]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("train_accuracy", self.train_accuracy)
        self.log("train_loss", loss)
        self.log("train_l1_loss", self.loss.l1_loss())
        self.log("train_l2_loss", self.loss.l2_loss())
        self.log("train_elastic_net_loss", self.loss.elastic_net_loss())
        self.log("train_base_loss", self.loss.base_loss(y_hat, y))
        self.cm["train"].update(y_hat.cpu().detach(), y.cpu().detach())
        return loss

    def on_train_epoch_start(self):
        # Reset on each epoch start so that only the last epoch is logged
        self.cm["train"].reset()

    def on_train_end(self):
        self._log_confusion_matrix_and_reset("train")

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        loss = self.loss(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.forward(batch)
        self.test_accuracy.update(y_hat, y)
        self.cm["test"].update(y_hat.cpu().detach(), y.cpu().detach())
        self.test_precision.update(y_hat.cpu().detach(), y.cpu().detach())
        self.test_recall.update(y_hat.cpu().detach(), y.cpu().detach())
        self.test_spurious_class_precision.update(
            y_hat.cpu().detach(), y.cpu().detach()
        )
        self.test_spurious_class_recall.update(y_hat.cpu().detach(), y.cpu().detach())
        self.test_spurious_class_accuracy.update(y_hat.cpu().detach(), y.cpu().detach())

    
        self.log_dict(
            {
                "test_accuracy": self.test_accuracy,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
            }
        )
        if self.spurious_class is not None:
            self.log_dict(
                {
                    "test_spurious_class_precision": self.test_spurious_class_precision,
                    "test_spurious_class_recall": self.test_spurious_class_recall,
                    "test_spurious_class_accuracy": self.test_spurious_class_accuracy,
                },
            )

    def _log_confusion_matrix_and_reset(self, stage: str):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.title.set_text(f"{stage} confusion matrix")
        self.cm[stage].plot(ax=ax)
        self.logger.experiment.add_figure(
            f"{stage}_confusion_matrix",
            fig,
        )
        plt.close(fig)
        self.cm[stage].reset()

    def on_test_end(self):
        # log histogram of classifier weights to tensorboard
        weights = self.model.classifier.weight.detach().cpu().reshape(-1).numpy()
        self.logger.experiment.add_histogram("classifier_weights", weights, 0)
        self._log_confusion_matrix_and_reset("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        return optimizer

    def load_state_dict(self, *args, **kwargs):
        logging.info(
            f"Calling custom load_state_dict with {self.pruned_concept_class_pairs}"
        )
        super().load_state_dict(*args, **kwargs)
        if not self.normalize:
            for concept_ix, class_ix in self.pruned_concept_class_pairs:
                previous_weight = self.model.classifier.weight[
                    class_ix, concept_ix
                ].item()
                self.model.prune(concept_ix=concept_ix, class_ix=class_ix)
                zero_weight = self.model.classifier.weight[class_ix, concept_ix].item()
                logging.info(
                    f"Pruned {concept_ix} {class_ix} from {previous_weight} to {zero_weight}"
                )
                assert zero_weight == 0.0, zero_weight
                assert previous_weight != 0.0, previous_weight
        else:
            if len(self.pruned_concept_class_pairs) == 1:
                concept_ix, class_ix = self.pruned_concept_class_pairs[0]
                previous_weight = self.model.classifier.weight[
                    class_ix, concept_ix
                ].item()
                self.model.prune_and_normalize(
                    concept_ix=concept_ix,
                    class_ix=class_ix,
                )
                zero_weight = self.model.classifier.weight[class_ix, concept_ix].item()
                logging.info(
                    f"Pruned {concept_ix} {class_ix} from {previous_weight} to {zero_weight}"
                    " and normalized remaining weights."
                )
            else:
                raise NotImplementedError(
                    f"Currently prune_and_normalize only handles one prune."
                )
