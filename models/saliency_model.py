import torch.nn as nn
import torch
from collections import OrderedDict


class SaliencyModel(nn.Module):
    """
    This class implements a seperate module to get gradients for a saliency map
    """

    def __init__(self, concept_bank, backbone, backbone_name):

        super().__init__()
        
        #set concept bank
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        #set backbone
        self.backbone = backbone
        self.backbone_name = backbone_name
                

    def forward(self, x):
        
        #encode image with the chosen backbone
        if "clip" in self.backbone_name.lower():
            embeddings = self.backbone.encode_image(x)
        elif "audio" in args.backbone_name.lower():
            raise ValueError("saliency map not implemented for audio")
        else:
            embeddings = self.backbone(x)

        #perfrom the projection
        margins = (torch.matmul(self.cavs, embeddings.T) + self.intercepts) / (self.norms)
        
        return margins.T

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
