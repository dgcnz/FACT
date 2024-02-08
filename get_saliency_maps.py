import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from data import get_dataset
from concepts import ConceptBank
from models import get_model, SaliencyModel

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    # For the above: Please make sure to output the COCO-Stuff results in "outdir/coco-stuff"
    parser.add_argument("--dataset", default="cub", type=str)

    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--sort-concepts", default=False, type=bool)
    parser.add_argument("--targets", default=[3, 6, 31, 35, 36, 37, 40, 41, \
                                             43, 46, 47, 50, 53, 64, 75, 76, 78, 80, 85, 89], \
                                             type=int, nargs='+', help="target indexes for cocostuff")
    parser.add_argument("--img_path", default=None, help="img to compute saliency map for")

    return parser.parse_args()

def saliency(input_img, input, model, out_dir):
    #we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    
    #transoform input PIL image to torch.Tensor and normalize
    input.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input 
    input.requires_grad = True
    #forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    #plot image and its saliency map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(f"{args.out_dir}/saliency.png")
    plt.show()
    print(f'figure save in {args.out_dir}/saliency.png')

if __name__ == "__main__":
    args = config()

    if args.sort_concepts:
        concept_bank = ConceptBank.from_pickle(args.concept_bank, sort_by_keys=True,  device=args.device)
    else:
        all_concepts = pickle.load(open(args.concept_bank, 'rb'))
        all_concept_names = list(all_concepts.keys())
        print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
        concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    
    #initialize the saliency model
    saliency_model = SaliencyModel(concept_bank = concept_bank, backbone=backbone, backbone_name=args.backbone_name)

    #get a single preprocessed and non-preprossed image from the dataloader
    img = Image.open(args.img_path).convert('RGB')
    input_img = preprocess(img)

    saliency(input_img, input, saliency_model, args.out_dir)


    
    