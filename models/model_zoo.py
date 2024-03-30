import sys
sys.path.append("./models")

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if "clip" in backbone_name:
        import clip
        # We assume clip models are passed of the form : clip:RN50
        clip_backbone_name = backbone_name.split(":")[1]
        backbone, preprocess = clip.load(clip_backbone_name, device=args.device, download_root=args.out_dir)
        backbone = backbone.eval()
        model = None
    
    elif backbone_name.lower() == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, root=args.out_dir)
        backbone, _ = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, _ = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
        
    elif backbone_name.lower() == "resnet18_imagenet1k_v1":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone, _ = ResNetBottom(model), ResNetTop(model)
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])

    # For some of the extension experiments
    elif backbone_name.lower() == "audio":
        from models.AudioCLIP.model import AudioCLIP
        from models.AudioCLIP.utils.transforms import ToTensor1D

        # Done like this to ensure that it does not do relative imports w.r.t. from where
        # the user is running the script
        filedir = os.path.abspath(__file__)
        filedir = os.path.dirname(filedir)
        pt_path = os.path.join(filedir, "AudioCLIP/assets/audioclip.pt")
        
        # loading the model and transforms (only audio is used)
        backbone = AudioCLIP(pretrained=pt_path, multilabel=False)
        backbone.eval()
        preprocess = transforms.Compose([
                        ToTensor1D()
                      ])
        model = None
        
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess

# For testing
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-name", default="audio", type=str)
    parser.add_argument("--out-dir", default="artifacts/outdir", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    backbone, preprocess = get_model(args, args.backbone_name)

    print(backbone)
