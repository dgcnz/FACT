import torch.nn as nn
from torch import Tensor
import torch
from models.pcbm_pl import PCBMClassifier
from concepts.concept_utils import ConceptBank

class PCBMClassifierV2(nn.Module):
    def __init__(self, n_concepts: int, n_classes: int = 5, concept_bank: ConceptBank):
        super(PCBMClassifier, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def forward(self, x: Tensor):
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



class PosthocHybridCBM(nn.Module):
    def __init__(self, classifier: PCBMClassifier, concept_bank: ConceptBank):
        super().__init__()
        self.classifier = classifier
        self.cb_vectors = concept_bank.vectors
        self.emedding_dim
        self.embedding_dim = concept_bank
        self.n_classes = self.classifier.n_classes
        self.residual_classifier = nn.Linear(self.embedding_dim, self.n_classes)

    def forward(self, emb, return_dist=False):
        out = self.classifier(x) + self.residual_classifier(emb)
        if return_dist:
            return out, x
        return out


