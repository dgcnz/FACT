import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms


def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()
    

@torch.no_grad()
def get_projections(args, backbone, posthoc_layer, loader, n_batches = np.inf):
    all_projs, all_embs, all_lbls = None, None, None
    batches = 0
    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        batch_X = batch_X.to(args.device)
        if "clip" in args.backbone_name.lower():
            embeddings = backbone.encode_image(batch_X).detach().float()
        elif "audio" in args.backbone_name.lower():
            ((embeddings, _, _), _), _ = backbone(audio=batch_X)
        else:
            embeddings = backbone(batch_X).detach()

        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)

        batches += 1
        if batches == n_batches:
          break

    return all_embs, all_projs, all_lbls

@torch.no_grad()
def get_projections_self_supervised(args, backbone, posthoc_layer, loader):
    all_projs, all_embs = None, None
    for batch in tqdm(loader):
        batch_X = batch
        batch_X = batch_X.to(args.device)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        elif "audio" in args.backbone_name.lower():
            ((embeddings, _, _), _), _ = backbone(audio=batch_X)
        else:
            embeddings = backbone(batch_X).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
    return all_embs, all_projs


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader, compute = False, self_supervised = False, n_batches = np.inf):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    
    # To make it easier to analyze results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"train-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_file = f"test-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_proj_file = f"train-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_proj_file = f"test-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_lbls_file = f"train-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    test_lbls_file = f"test-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    
    train_file = os.path.join(args.out_dir, train_file)
    test_file = os.path.join(args.out_dir, test_file)
    train_proj_file = os.path.join(args.out_dir, train_proj_file)
    test_proj_file = os.path.join(args.out_dir, test_proj_file)
    train_lbls_file = os.path.join(args.out_dir, train_lbls_file)
    test_lbls_file = os.path.join(args.out_dir, test_lbls_file)

    if os.path.exists(train_proj_file) and not compute and not self_supervised:
        train_embs = np.load(train_file)
        test_embs = np.load(test_file)
        train_projs = np.load(train_proj_file)
        test_projs = np.load(test_proj_file)
        train_lbls = np.load(train_lbls_file)
        test_lbls = np.load(test_lbls_file)

        return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls

    elif not self_supervised:
        train_embs, train_projs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader, n_batches = n_batches)
        test_embs, test_projs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader, n_batches = n_batches)

        np.save(train_file, train_embs)
        np.save(test_file, test_embs)
        np.save(train_proj_file, train_projs)
        np.save(test_proj_file, test_projs)
        np.save(train_lbls_file, train_lbls)
        np.save(test_lbls_file, test_lbls)

        return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls
    
    else:
        train_embs, train_projs = get_projections_self_supervised(args, backbone, posthoc_layer, train_loader)

        return train_embs, train_projs
    



def split_image_into_snippets(image_tensor):
    """
    Split the input image tensor into 3x3 snippet images.

    Args:
    - image_tensor (torch.Tensor): Input image tensor

    Returns:
    - snippet_images (list): List of 3x3 snippet images as tensors
    """

    snippet_images = []

    # Convert to PIL image
    image_pil = transforms.ToPILImage()(image_tensor)

    # Calculate snippet size
    width, height = image_pil.size
    snippet_width = width // 3
    snippet_height = height // 3

    # Split the image into 3x3 snippet images
    for i in range(3):
        for j in range(3):
            left = j * snippet_width
            upper = i * snippet_height
            #right = (j + 1) * snippet_width
            #lower = (i + 1) * snippet_height
            snippet = transforms.functional.crop(image_pil, left, upper, snippet_width, snippet_height)
            snippet_images.append(transforms.functional.to_tensor(snippet))

    return snippet_images

import torch
import numpy as np
from tqdm import tqdm

def split_batch_into_snippets(batch_tensor):
    """
    Split the input batch tensor into 3x3 snippet images.

    Args:
    - batch_tensor (torch.Tensor): Batch of input images tensor

    Returns:
    - snippet_images (torch.Tensor): Batch of 3x3 snippet images as tensors
    """

    batch_size, channels, height, width = batch_tensor.shape

    # Calculate snippet size
    snippet_height = height // 3
    snippet_width = width // 3

    # Split each image in the batch into 3x3 snippet images
    snippet_images = []
    for i in range(3):
        for j in range(3):
            left = j * snippet_width
            upper = i * snippet_height
            right = (j + 1) * snippet_width
            lower = (i + 1) * snippet_height
            snippet = batch_tensor[:, :, upper:lower, left:right]
            snippet_images.append(snippet)

    # Stack the snippet images along a new dimension
    snippet_images = torch.stack(snippet_images, dim=1)

    return snippet_images

def compute_aggregate_similarity(S_local_vectors, threshold):
    """
    Compute the unified local similarity vector S_aggregate based on the aggregation strategy.

    Args:
    - S_local_vectors (np.ndarray): Array containing local similarity vectors for each snippet
    - threshold (float): Threshold parameter ζ

    Returns:
    - S_aggregate (np.ndarray): Unified local similarity vector for each image
    """
    max_scores = np.max(S_local_vectors, axis=1)
    min_scores = np.min(S_local_vectors, axis=1)
    gamma = (max_scores >= threshold).astype(float)
    S_aggregate = gamma * max_scores + (1 - gamma) * min_scores
    return S_aggregate

def get_aggregate_projections(loader, backbone, posthoc_layer, args, n_batches, threshold):
    """
    Process batches from a data loader using the specified backbone and posthoc layer.

    Args:
    - loader: DataLoader object providing batches of data
    - backbone: Backbone model
    - posthoc_layer: Posthoc layer for computing distances or projections
    - args: Additional arguments
    - n_batches (int): Number of batches to process
    - threshold (float): Threshold parameter ζ for aggregation strategy

    Returns:
    - S_final (np.ndarray): Final similarity vector for each image
    """

    S_final = []

    for batch in tqdm(loader):
        batch_X, batch_Y = batch
        batch_X = batch_X.to(args.device)

        with torch.no_grad():
            # Split each image into 3x3 snippet images
            snippet_images = split_batch_into_snippets(batch_X)

            # Initialize list to store aggregate similarity vectors for each image in the batch
            S_aggregate_batch = []

            # Process each snippet image individually
            for snippet_batch in snippet_images:
                if "clip" in args.backbone_name.lower():
                    embeddings = backbone.encode_image(snippet_batch.view(-1, *snippet_batch.shape[2:])).float()
                elif "audio" in args.backbone_name.lower():
                    ((embeddings, _, _), _), _ = backbone(audio=snippet_batch.view(-1, *snippet_batch.shape[2:]))
                else:
                    embeddings = backbone(snippet_batch.view(-1, *snippet_batch.shape[2:]))

                embeddings = embeddings.view(snippet_batch.shape[0], snippet_batch.shape[1], -1)
                projs = posthoc_layer.compute_dist(embeddings).cpu().numpy()

                # Compute aggregate similarity vector for the current image
                S_aggregate_image = compute_aggregate_similarity(projs, threshold)
                S_aggregate_batch.append(S_aggregate_image)

            # Concatenate aggregate similarity vectors for all images in the batch
            S_aggregate_batch = np.concatenate(S_aggregate_batch)

        if len(S_final) >= n_batches:
            break

    return S_aggregate_batch


def compute_aggregate_projections(args, backbone, posthoc_layer, train_loader, test_loader, self_supervised=True):
    """
    Compute aggregate projections based on train and test loaders.

    Args:
    - args: Additional arguments (if any)
    - backbone: Backbone model (e.g., CNN)
    - posthoc_layer: Posthoc layer for projections
    - train_loader (torch.utils.data.DataLoader): Training data loader
    - test_loader (torch.utils.data.DataLoader): Test data loader
    - concept_matrix (torch.Tensor): Concept matrix for aggregation
    - self_supervised (bool): Whether the model is self-supervised or not

    Returns:
    - train_aggregate_projections (torch.Tensor): Aggregate projections for training data
    - test_aggregate_projections (torch.Tensor): Aggregate projections for test data
    """
    # Get the full dataset from the train and test_loader as a torch tensor
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Compute aggregate projections for training data
    train_aggregate_projections = get_aggregate_projections(args, backbone, posthoc_layer, train_dataset, args.threshold)

    return train_aggregate_projections  



        

    
    
