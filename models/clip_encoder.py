import torch
import clip
from clip.model import CLIP


class CLIPImageEncoder(torch.nn.Module):
    clip_model: CLIP

    def __init__(self, model_name: str):
        super().__init__()
        clip_model, _ = clip.load(model_name, jit=False)
        self.model = clip_model

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Return image embeddings from batch of images."""
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                return self.model.encode_image(x)
            else:
                return torch.stack([self.model.encode_image(i) for i in x], dim=0)


class CLIPClassifier(torch.nn.Module):
    def __init__(self, model_name: str, n_classes: int):
        super().__init__()
        self.backbone = CLIPImageEncoder(model_name=model_name)
        self.classifier = torch.nn.Linear(
            self.backbone.clip_model.transformer.width, n_classes
        )

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits

