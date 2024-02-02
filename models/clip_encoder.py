import torch
import clip
import os
from clip.model import CLIP
from AudioCLIP import AudioCLIP


class CLIPImageEncoder(torch.nn.Module):
    clip_model: CLIP

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        clip_model, _ = clip.load(model_name, jit=False, device=device)
        self.model = clip_model

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Return image embeddings from batch of images."""
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                return self.model.encode_image(x)
            else:
                return torch.stack([self.model.encode_image(i) for i in x], dim=0)
            

class CLIPAudioEncoder(torch.nn.Module):

    def __init__(self, pretrained:bool=False):
        super().__init__()
        if pretrained == True:
            filedir = os.path.abspath(__file__)
            filedir = os.path.dirname(filedir)
            pt_path = os.path.join(filedir, "AudioCLIP/assets/audioclip.pt")
            model = AudioCLIP(pretrained=pt_path)
        
        else:
            model = AudioCLIP(pretrained=False)
        
        self.model = model

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Return image embeddings from batch of images."""
        with torch.no_grad():
            ((embeddings, _, _), _), _ = self.model(audio=x)
            if isinstance(x, torch.Tensor):
                return embeddings
            else:
                stack = []
                for i in x:
                    ((embeddings, _, _), _), _ = self.model(audio=i)
                    stack.append(embeddings)
                return torch.stack(stack, dim=0)


class CLIPClassifier(torch.nn.Module):
    def __init__(self, model_name: str, n_classes: int, grads=False):
        super().__init__()
        self.backbone = CLIPImageEncoder(model_name=model_name)
        # if grads == False:
        #     for param in self.backbone.model.parameters():
        #         param.requires_grad = False

        self.features_dtype = self.backbone.model.ln_final.weight.dtype
        self.classifier = torch.nn.Linear(
            self.backbone.model.ln_final.weight.size(0) * 2, n_classes
        ).to(dtype=self.features_dtype)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        features = features.to(self.features_dtype)
        logits = self.classifier(features)
        return logits


class AudioCLIPClassifier(torch.nn.Module):
    def __init__(self, pretrained: bool, n_classes: int, grads=False):
        super().__init__()
        self.backbone = CLIPAudioEncoder(pretrained=pretrained)
        # self.features_dtype = self.backbone.model.audio.fc
        self.classifier = torch.nn.Linear(
            self.backbone.model.audio.fc.out_features, n_classes
        )#.to(dtype=self.features_dtype)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        # features = features.to(self.features_dtype)
        logits = self.classifier(features)
        return logits
    