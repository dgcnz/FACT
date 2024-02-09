import argparse
import os
import pickle
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from concepts import ConceptBank
from data import get_dataset
from models import SaliencyModel, get_model
from utils.saliency.gradients import SmoothGrad, VanillaGrad
from utils.saliency.image_utils import save_as_gray_image


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept-bank", required=True, type=str, help="Path to the concept bank"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        default="artifacts/outdir",
        help="Output folder for model/run info.",
    )
    # For the above: Please make sure to output the COCO-Stuff results in "outdir/coco-stuff"
    parser.add_argument("--dataset", default="cub", type=str)

    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--sort-concepts", default=False, type=bool)
    parser.add_argument(
        "--img-path", default=None, help="img to compute saliency map for"
    )
    parser.add_argument("--concept-names", nargs="+", type=str, default=None)
    parser.add_argument("--targetclass", default="bicycle", type=str)
    parser.add_argument("--concept-ix", type=int)
    parser.add_argument("--method", default="smoothgrad", type=str)

    return parser.parse_args()


def saliencyv2(
    input_img: Image.Image,
    input: torch.Tensor,
    model: torch.nn.Module,
    out_dir: str,
    concept_ix: int,
    method: Literal["vanilla", "smoothgrad"] = "smoothgrad",
) -> np.ndarray:
    """
    input_img: raw image
    input: preprocessed image
    model: torch.nn.Module
    """
    input = input.unsqueeze(0)

    if method == "vanilla":
        vanilla_grad = VanillaGrad(pretrained_model=model, cuda=False)
        vanilla_saliency = vanilla_grad(input, index=concept_ix)
        save_as_gray_image(vanilla_saliency, os.path.join(out_dir, "vanilla_grad.jpg"))
    elif method == "smoothgrad":
        N_SAMPLES_SMOOTHGRAD: int = 25
        smooth_grad = SmoothGrad(
            pretrained_model=model,
            cuda=False,
            n_samples=N_SAMPLES_SMOOTHGRAD,
            magnitude=True,
        )
        smooth_saliency = smooth_grad(input, index=concept_ix)
        save_as_gray_image(smooth_saliency, os.path.join(out_dir, "smooth_grad.jpg"))
        print("Saved smooth gradient image")
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def saliency(input_img, input, model, out_dir):

    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()

    # transoform input PIL image to torch.Tensor and normalize
    input.unsqueeze_(0)

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # plot image and its saliency map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    # plt.imshow(np.transpose(input_img.numpy(), (1, 2, 0)))\
    plt.imshow(input_img)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.cpu().numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f"{args.out_dir}/saliency.png")
    plt.show()
    print(f"figure save in {args.out_dir}/saliency.png")


if __name__ == "__main__":
    args = config()

    if args.sort_concepts:
        concept_bank = ConceptBank.from_pickle(
            args.concept_bank, sort_by_keys=True, device=args.device
        )
    else:
        all_concepts = pickle.load(open(args.concept_bank, "rb"))
        all_concept_names = list(all_concepts.keys())
        print(
            f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used."
        )
        concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    # initialize the saliency model
    saliency_model = SaliencyModel(
        concept_bank=concept_bank,
        backbone=backbone,
        backbone_name=args.backbone_name,
        concept_names=args.concept_names,
    )

    (input, label), (input_img, input_label), class_name = get_dataset(
        args, preprocess, single_image=True
    )

    input = input.to(args.device)
    saliency_model = saliency_model.to(args.device)

    print(input)
    print(input_img)
    print("saliency map for image with label", label, input_label)
    print("which is class_name", class_name)
    # get a single preprocessed and non-preprossed image from the dataloader
    # img = Image.open(args.img_path).convert('RGB')
    # input_img = preprocess(img)

    # saliency(input_img, input, saliency_model, args.out_dir)
    saliencyv2(
        input_img,
        input,
        saliency_model,
        args.out_dir,
        concept_ix=args.concept_ix,
        method=args.method,
    )
