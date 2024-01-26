import torch
import lightning as L
from typing import Literal
import logging


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
            EPS = 1e-4
            assert (
                abs(rescaled_norm - previous_norm) < EPS
            ), f"Rescaled norm {rescaled_norm} != previous norm {previous_norm}, diff {abs(rescaled_norm - previous_norm)}"


def analyze_classifier(
    model: PCBMClassifier,
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

    def _per_sample_elastic_net(self, param: torch.Tensor):
        """
        Calculates the elastic net for a single parameter
        """
        l1_norm = param.abs().sum()
        squared_l2_norm = param.square().sum()
        return self.alpha * l1_norm + (1 - self.alpha) * squared_l2_norm

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the PCBM loss
        """
        base_loss = torch.nn.functional.cross_entropy(
            y_hat, y, reduction=self.reduction
        )
        elastic_net = self.reduce_fn(
            torch.stack(
                [
                    self._per_sample_elastic_net(param)
                    for param in self.classifier.parameters()
                ]
            )
        )
        return base_loss + self.elastic_net_constant * elastic_net


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
    ):
        super().__init__()
        self.model = PCBMClassifier(n_concepts=n_concepts, n_classes=n_classes)
        self.lr = lr
        self.pruned_concept_class_pairs = pruned_concept_class_pairs
        self.normalize = normalize
        self.lam = lam
        self.alpha = alpha
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

    def training_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("train_accuracy", accuracy)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["concept_weights"], batch["label"]
        y_hat = self.model(x)
        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
        self.log("test_accuracy", accuracy)

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
