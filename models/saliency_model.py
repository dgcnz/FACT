import torch.nn as nn
import torch
from collections import OrderedDict


class SaliencyModel(nn.Module):
    """
    This class implements a seperate module to get gradients for a saliency map
    """

    def __init__(self, concept_bank, backbone, backbone_name, concept_names = None):

        super().__init__()
        
        #set concept bank

        if concept_bank == None:
            self.cavs = concept_bank.vectors
            self.intercepts = concept_bank.intercepts
            self.norms = concept_bank.norms
            self.names = concept_bank.concept_names.copy()
            self.n_concepts = self.cavs.shape[0]
        else:
            indices = [i for i, x in enumerate(concept_bank.concept_names) if x in concept_names]
            self.cavs = torch.stack([concept_bank.vectors[i] for i in indices])
            self.intercepts = torch.stack([concept_bank.intercepts[i] for i in indices])
            self.norms = torch.stack([concept_bank.norms[i] for i in indices])
            #self.names = [concept_bank.concept_names[i].clone() for i in indices]
            self.n_concepts = self.cavs.shape[0]

        #set backbone
        self.backbone = backbone
        self.backbone_name = backbone_name
        if "clip" in self.backbone_name.lower():
            self.features = self.backbone.visual
        else:
            raise NotImplementedError(f"Backbone {self.backbone_name} features name is not specified.")
                

    def forward(self, x):
        
        #encode image with the chosen backbone
        if "clip" in self.backbone_name.lower():
            embeddings = self.backbone.encode_image(x).float()
        elif "audio" in self.backbone_name.lower():
            raise ValueError("saliency map not implemented for audio")
        else:
            embeddings = self.backbone(x)

        #perfrom the projection
        print(self.cavs.dtype)
        print(embeddings.dtype)
        margins = (torch.matmul(self.cavs, embeddings.T) + self.intercepts) / (self.norms)
        print('concept features values for the given concepts are:', margins)
        
        return margins.T

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
