from torchmetrics import Metric
import torch
from torch import Tensor
from torchmetrics.functional.classification.stat_scores import multiclass_stat_scores


class RecallForClass(Metric):
    num_classes: int
    class_index: int
    stat_scores: Tensor

    def __init__(self, num_classes: int, class_index: int, eps: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_index = class_index
        self.eps = eps
        self.add_state(
            "stat_scores",
            default=torch.zeros(self.num_classes, 5),
            dist_reduce_fx="sum",
        )

    def compute(self):
        true_positives = self.stat_scores[self.class_index, 0]
        false_negatives = self.stat_scores[self.class_index, 3]
        recall = true_positives / (true_positives + false_negatives + self.eps)
        return recall

    def update(self, preds: Tensor, target: Tensor):
        self.stat_scores += multiclass_stat_scores(
            preds, target, num_classes=self.num_classes, average="none"
        )

    def reset(self):
        self.stat_scores = torch.zeros(self.num_classes, 5)


class PrecisionForClass(Metric):
    num_classes: int
    class_index: Tensor
    stat_scores: Tensor

    def __init__(self, num_classes: int, class_index: int, eps: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_index = class_index
        self.eps = eps
        self.add_state(
            "stat_scores",
            default=torch.zeros(self.num_classes, 5),
            dist_reduce_fx="sum",
        )

    def compute(self):
        true_positives = self.stat_scores[self.class_index, 0]
        false_positives = self.stat_scores[self.class_index, 1]
        precision = true_positives / (true_positives + false_positives + self.eps)
        return precision

    def update(self, preds: Tensor, target: Tensor):
        self.stat_scores += multiclass_stat_scores(
            preds, target, num_classes=self.num_classes, average="none"
        )

    def reset(self):
        self.stat_scores = torch.zeros(self.num_classes, 5)


class AccuracyForClass(Metric):
    num_classes: int
    class_index: Tensor
    stat_scores: Tensor

    def __init__(self, num_classes: int, class_index: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_index = class_index
        self.add_state(
            "stat_scores",
            default=torch.zeros(self.num_classes, 5),
            dist_reduce_fx="sum",
        )
        # stat_scores:
        # 0: true positives, 1: false positives, 2: true negatives, 3: false negatives and the 4:support.

    def compute(self):
        true_positives = self.stat_scores[self.class_index, 0]
        true_negatives = self.stat_scores[self.class_index, 2]
        false_positives = self.stat_scores[self.class_index, 1]
        false_negatives = self.stat_scores[self.class_index, 3]
        accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_positives + false_negatives
        )
        return accuracy


    def update(self, preds: Tensor, target: Tensor):
        self.stat_scores += multiclass_stat_scores(
            preds, target, num_classes=self.num_classes, average="none"
        )

    def reset(self):
        self.stat_scores = torch.zeros(self.num_classes, 5)
