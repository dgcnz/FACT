import torch
import torch.nn as nn


class PCBMUserStudy(nn.Module):
    def __init__(self, concept_bank, backbone_name, idx_to_class=None, n_classes=5):
        """
        PosthocCBM Linear Layer.
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            backbone_name (str): Name of the backbone, e.g. clip:RN50.
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(PCBMUserStudy, self).__init__()
        # Get the concept information from the bank
        self.backbone_name = backbone_name
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        self.n_classes = n_classes
        # Will be used to plot classifier weights nicely
        self.idx_to_class = (
            idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}
        )

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def prune(self, concept_ix: int, class_ix: int):
        with torch.no_grad():
            self.classifier.weight.data[class_ix, concept_ix] = 0.0

    def prune_and_normalize(self, concept_ix: int, class_ix: int):
        norm_ord = 1
        with torch.no_grad():
            previous_norm = torch.linalg.vector_norm(
                self.classifier.weight.data[class_ix], ord=norm_ord
            ).item()
            pruned_norm = self.classifier.weight.data[class_ix, concept_ix].item()
            self.classifier.weight.data[class_ix, concept_ix] = 0.0
            unpruned_norm = torch.linalg.vector_norm(
                self.classifier.weight.data[class_ix], ord=norm_ord
            ).item()
            self.classifier.weight.data[class_ix] *= (
                1 + pruned_norm / unpruned_norm
            )
            rescaled_norm = torch.linalg.vector_norm(
                self.classifier.weight.data[class_ix], ord=1
            ).float()
            EPS = 1e-4
            assert (
                abs(rescaled_norm - previous_norm) < EPS
            ), f"Rescaled norm {rescaled_norm} != previous norm {previous_norm}, diff {abs(rescaled_norm - previous_norm)}"

    def compute_dist(self, emb):
        # Computing the geometric margin to the decision boundary specified by CAV.
        margins = (torch.matmul(self.cavs, emb.T) + self.intercepts) / (self.norms)
        return margins.T

    def forward(self, emb, return_dist=False):
        x = self.compute_dist(emb)
        out = self.classifier(x)
        if return_dist:
            return out, x
        return out

    def forward_projs(self, projs):
        return self.classifier(projs)

    def trainable_params(self):
        return self.classifier.parameters()

    def classifier_weights(self):
        return self.classifier.weight

    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).float().to(
            self.classifier.weight.device
        )
        self.classifier.bias.data = torch.tensor(bias).float().to(self.classifier.weight.device)
        return 1

    def analyze_classifier(self, k=5, print_lows=False):
        weights = self.classifier.weight.clone().detach()
        output = []

        if len(self.idx_to_class) == 2:
            weights = [weights.squeeze(), weights.squeeze()]

        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis
    
    def analyze_classifier_withResults(self, k=5, print_lows=False):
        weights = self.classifier.weight.clone().detach()
        output = []
        analysis_data = []

        if len(self.idx_to_class) == 2:
            weights = [weights.squeeze(), weights.squeeze()]
        
        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            for j, c in enumerate(topk_concepts):
                analysis_data.append({
                    'class': cls,
                    'rank': j+1,
                    'concept': c,
                    'weight': topk_vals[j].item()
                })
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis, analysis_data
    
    def test_step(self, batch, device):
        ''' Calculate test accuracy and per class accuracy'''
        features, labels = batch
        features = torch.tensor(features).float().to(device)
        labels = torch.tensor(labels).long().to(device)
        with torch.no_grad():
            outputs = self.forward_projs(features)
            _, predicted = torch.max(outputs.data, 1)

            correct_predictions = torch.zeros(labels.max() + 1).to(device)
            total_predictions = torch.zeros_like(correct_predictions)

            for i in range(len(labels)):
                total_predictions[labels[i]] += 1
                if predicted[i] == labels[i]:
                    correct_predictions[labels[i]] += 1

            class_accuracies = (correct_predictions / total_predictions) * 100
            print(class_accuracies)
            overall_accuracy = torch.sum(predicted == labels).item() / labels.size(0) * 100

        return overall_accuracy, class_accuracies

    def get_sparsity(self):
        return (self.classifier.weight > 0).sum().item()


class PosthocHybridCBM(nn.Module):
    def __init__(self, bottleneck: PCBMUserStudy):
        """
        PosthocCBM Hybrid Layer.
        Takes an embedding as the input, outputs class-level predictions.
        Uses both the embedding and the concept predictions.
        Args:
            bottleneck (PosthocLinearCBM): [description]
        """
        super(PosthocHybridCBM, self).__init__()
        # Get the concept information from the bank
        self.bottleneck = bottleneck
        # A single linear layer will be used as the classifier
        self.d_embedding = self.bottleneck.cavs.shape[1]
        self.n_classes = self.bottleneck.n_classes
        self.residual_classifier = nn.Linear(self.d_embedding, self.n_classes)

    def forward(self, emb, return_dist=False):
        x = self.bottleneck.compute_dist(emb)
        out = self.bottleneck.classifier(x) + self.residual_classifier(emb)
        if return_dist:
            return out, x
        return out

    def trainable_params(self):
        return self.residual_classifier.parameters()

    def classifier_weights(self):
        return self.residual_classifier.weight

    def analyze_classifier(self):
        return self.bottleneck.analyze_classifier()
