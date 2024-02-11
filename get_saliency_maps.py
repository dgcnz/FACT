import argparse
import os
import pickle
from typing import Literal
import copy

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
        "--concept-bank1", required=True, type=str, help="Path to the concept bank"
    )
    parser.add_argument(
        "--concept-bank2", required=True, type=str, help="Path to the concept bank"
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
    parser.add_argument(
        "--img-path", default=None, help="img to compute saliency map for"
    )
    parser.add_argument("--concept-names1", nargs="+", type=str, default=None)
    parser.add_argument("--concept-names2", nargs="+", type=str, default=None)

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
    


    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()

    input = input.unsqueeze(0)

    input.requires_grad = True

    

    if method == "vanilla":
        vanilla_grad = VanillaGrad(pretrained_model=model, cuda=True)
        vanilla_saliency = vanilla_grad(input, index=concept_ix)
        img = save_as_gray_image(vanilla_saliency, os.path.join(out_dir, "vanilla_grad.jpg"))
        return img
    elif method == "smoothgrad":
        N_SAMPLES_SMOOTHGRAD: int = 25#25
        smooth_grad = SmoothGrad(
            pretrained_model=model,
            cuda=True,
            n_samples=N_SAMPLES_SMOOTHGRAD,
            magnitude=True,
        )
        smooth_saliency = smooth_grad(input, index=concept_ix)
        img = save_as_gray_image(smooth_saliency, os.path.join(out_dir, "smooth_grad.jpg"))
        
        print("Saved smooth gradient image")
        return img
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

    return slc.cpu().numpy()

def plot_maps(img, maps1, maps2, concept_names1, concept_names2):
    plt.figure(figsize=(15, 5))
    plt.subplot(2, len(concept_names1)+1, 1)
    # plt.imshow(np.transpose(input_img.numpy(), (1, 2, 0)))\
    plt.imshow(img)
    plt.title('Input image')
    plt.xticks([])
    plt.yticks([])

    for i in range(len(concept_names1)):
        plt.subplot(2, len(concept_names1)+1, i + 2)
        if i == 0:
            plt.ylabel('CLIP concepts')
        plt.title(concept_names1[i])
        plt.imshow(maps1[i], cmap=plt.cm.gray) #maybe I should make this greyscale instead idk 
        plt.xticks([])
        plt.yticks([])

    for i in range(len(concept_names1)):
        plt.subplot(2, len(concept_names1)+1, len(concept_names1) + i + 3)
        if i == 0:
            plt.ylabel('CAV concepts')
        plt.title(concept_names2[i])
        plt.imshow(maps2[i], cmap=plt.cm.gray) #maybe I should make this greyscale instead idk 
        plt.xticks([])
        plt.yticks([])

    plt.savefig(f"{args.out_dir}/saliency.png")
    plt.show()
    print(f"figure save in {args.out_dir}/saliency.png")



if __name__ == "__main__":
    args = config()

    #get concepts for the first concept bank
    all_concepts1 = pickle.load(open(args.concept_bank1, "rb"))
    all_concept_names1 = list(all_concepts1.keys())
    print(
        f"Bank path: {args.concept_bank1}. {len(all_concept_names1)} concepts will be used."
    )
    concept_bank1 = ConceptBank(all_concepts1, args.device)

    #get concepts for the second concept bank
    all_concepts2 = pickle.load(open(args.concept_bank2, "rb"))
    all_concept_names2 = list(all_concepts2.keys())
    print(
        f"Bank path: {args.concept_bank2}. {len(all_concept_names2)} concepts will be used."
    )
    concept_bank2 = ConceptBank(all_concepts2, args.device)


    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    # initialize the saliency model
    

    (input, label), (input_img, input_label), class_name = get_dataset(
        args, preprocess, single_image=True
    )

    input = input.to(args.device)
    

    print(input)
    print(input_img)
    print("saliency map for image with label", label, input_label)
    print("which is class_name", class_name)
    # get a single preprocessed and non-preprossed image from the dataloader
    # img = Image.open(args.img_path).convert('RGB')
    # input_img = preprocess(img)

    '''
    saliency_model = SaliencyModel(
                concept_bank=concept_bank,
                backbone=backbone,
                backbone_name=args.backbone_name,
                concept_names=args.concept_names,
            )

    saliency(input_img, input, saliency_model, args.out_dir)
    
    '''
    maps1 = []
    maps2 = []

    for concept_name1, concept_name2 in zip(args.concept_names1, args.concept_names2):
        # get the map for both the first concept bank 
        saliency_model1 = SaliencyModel(
                concept_bank=concept_bank1,
                backbone=backbone,
                backbone_name=args.backbone_name,
                concept_names=[concept_name1],
            )
        
        saliency_model1 = saliency_model1.to(args.device)

        #second bank 
        saliency_model2 = SaliencyModel(
                concept_bank=concept_bank2,
                backbone=backbone,
                backbone_name=args.backbone_name,
                concept_names=[concept_name2],
            )
        
        saliency_model2 = saliency_model2.to(args.device)

        if args.method in ['smoothgrad', 'vanilla']:

            map1 = saliencyv2(
                    input_img,
                    input,
                    saliency_model1,
                    args.out_dir,
                    concept_ix=args.concept_ix,
                    method=args.method,
                )
            
            map2 = saliencyv2(
                    input_img,
                    input,
                    saliency_model2,
                    args.out_dir,
                    concept_ix=args.concept_ix,
                    method=args.method,
                )
        else:
            map1 = saliency(input_img, copy.copy(input), saliency_model1, args.out_dir)
            map2 = saliency(input_img, copy.copy(input), saliency_model2, args.out_dir)

        
        maps1.append(map1)
        maps2.append(map2)
    
    plot_maps(img = input_img, maps1 = maps1, maps2 = maps2, concept_names1=args.concept_names1, concept_names2 = args.concept_names2)
    
