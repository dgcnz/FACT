from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from utils.skl.module import SKLModule
from skops.io import dump, load


class PCBMClassifierSKL(SKLModule):
    SAVE_FORMAT_EXT: str = ".skops"

    def __init__(
        self,
        seed: int,
        n_classes: int,
        lam: float,
        alpha: float,
        max_iter: int,
        n_concepts: int,
        spurious_class: int | None = None,
        pruned_concept_class_pairs: list[tuple[int, int]] = [],
        normalize: bool = False,
    ):
        super().__init__()
        self.classifier = SGDClassifier(
            random_state=seed,
            loss="log_loss",
            alpha=lam / (n_classes * n_concepts),
            l1_ratio=alpha,
            verbose=0,
            penalty="elasticnet",
            max_iter=max_iter,
        )
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        self.spurious_class = spurious_class
        self.pruned_concept_class_pairs = pruned_concept_class_pairs
        self.normalize = normalize
        self.setup_metrics()

    def dump(self, path: str):
        dump(self.classifier, path)

    def load(self, path: str):
        self.classifier = load(path, trusted=True)
        if self.pruned_concept_class_pairs:
            if self.normalize:
                self.prune_and_normalize(self.pruned_concept_class_pairs)
            else:
                self.prune(self.pruned_concept_class_pairs)

    def setup_metrics(self):
        self.train_accuracy = accuracy_score
        self.test_accuracy = accuracy_score

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        self.classifier.fit(xs, ys)
        yps = self.classifier.predict(xs)
        self.log("train_accuracy", self.train_accuracy(ys, yps))

    def test(self, xs: np.ndarray, ys: np.ndarray):
        yps = self.classifier.predict(xs)
        self.log("test_accuracy", self.test_accuracy(ys, yps))

    def prune(self, concept_class_pairs: list[tuple[int, int]]):
        for concept_ix, class_ix in concept_class_pairs:
            self.classifier.coef_[class_ix, concept_ix] = 0.0

    def prune_and_normalize(self, concept_class_pairs: list[tuple[int, int]]):
        for concept_ix, class_ix in concept_class_pairs:
            previous_norm = np.sum(np.abs(self.classifier.coef_[class_ix]))
            self.classifier.coef_[class_ix] = normalized_prune_np(
                self.classifier.coef_[class_ix], [concept_ix]
            )
            rescaled_norm = np.sum(np.abs(self.classifier.coef_[class_ix]))
            assert np.allclose(
                rescaled_norm, previous_norm
            ), f"Rescaled norm {rescaled_norm} != previous norm {previous_norm}, diff {abs(rescaled_norm - previous_norm)}"


def normalized_prune_np(a: np.ndarray, zeroed_indices: list[int]):
    """
    Given array a and list of `zeroed_indices`, return array b
    such that b[i] = 0 for all `i` in `zeroed_indices` and
    the l1 norm of b is the same as the l1 norm of a.
    """
    b = a.copy()
    b[zeroed_indices] = 0.0

    l1_norm_a = np.sum(np.abs(a))
    l1_norm_b = np.sum(np.abs(b))

    if l1_norm_b != 0:
        scaling_factor = l1_norm_a / l1_norm_b
        b *= scaling_factor
    return b
