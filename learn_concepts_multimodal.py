import sys
sys.path.append("./models")

import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
import pandas as pd
from models.AudioCLIP import AudioCLIP
from re import sub
from tqdm import tqdm
from data.data_zoo import get_dataset


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--classes", default="cifar10", type=str)
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--recurse", default=1, type=int, help="How many times to recurse on the conceptnet graph")
    parser.add_argument('--class_names', type=str, default = 'airplane,bed,car,cow,keyboard')
    return parser.parse_args()


def get_single_concept_data(cls_name):
    if cls_name in concept_cache:
        return concept_cache[cls_name]
    
    all_concepts = []
    
    # Has relations
    has_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasA&start=/c/en/{}"
    obj = requests.get(has_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Made of relations
    madeof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/MadeOf&start=/c/en/{}"
    obj = requests.get(madeof_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Properties of things
    property_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasProperty&start=/c/en/{}"
    obj = requests.get(property_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Categorization concepts
    is_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
    obj = requests.get(is_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        if edge["weight"] <= 1:
            continue
        all_concepts.append(edge['end']['label'])
    
    # Parts of things
    parts_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/PartOf&end=/c/en/{}"
    obj = requests.get(parts_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['start']['label'])
    
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [c.replace("a ", "") for c in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c!=""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    
    concept_cache[cls_name] = all_concepts
    
    return all_concepts


def get_concept_data(all_classes):
    all_concepts = set()
    # Collect concepts that are relevant to each class
    for cls_name in all_classes:
        print(f"Pulling concepts for '{cls_name}'...")
        all_concepts |= get_single_concept_data(cls_name)
        
    return all_concepts


def clean_concepts(scenario_concepts):
    """
    Clean the plurals, trailing whitespaces etc.
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    # We use nltk to handle plurals, multiples of the same words etc.
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        c = c.strip()
        c_subwords = c.split(" ")
        # If a concept is made of more than 2 words, we drop it.
        if len(c_subwords) > 2:
            print(f"Skipping long concept: '{c_prev}'")
            continue
        # Lemmatize words to help eliminate non-unique concepts etc.
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))

    return scenario_concepts_rec


@torch.no_grad()
def learn_conceptbank(args, concept_list, scenario, model):
    assert args.device is not None, "Please specify a device."
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.
        if args.backbone_name.lower() == "audio":
            text = [[concept]]
            ((_, _, text_features), _), _ = model(text=text)
        else:
            text = clip.tokenize(f"{concept}").to(args.device)
            text_features = model.encode_text(text).cpu().numpy()
        
        text_features = text_features / np.linalg.norm(text_features)
        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"\nNumber of Concepts: {len(concept_dict)}")
    # This part of the code ensures that there are no colons in the backbone name (as it causes exporting errors):
    if ":" in args.backbone_name:
        args.backbone_name = sub(":", "", args.backbone_name)

    concept_dict_path = os.path.join(args.out_dir, f"mmc_{args.backbone_name}_{scenario}_recurse_{args.recurse}.pkl")

    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to {concept_dict_path}!\n")


if __name__ == "__main__":
    args = config()
    # Determine if we use a visual or audio model
    if "clip" in args.backbone_name.lower():
        model, _ = clip.load(args.backbone_name.split(":")[1], device=args.device, download_root=args.out_dir)
    elif "audio" in args.backbone_name.lower():
        # Done like this to ensure that it does not do relative imports w.r.t. from where
        # the user is running the script. Here we load the partially trained AudioCLIP model
        # which only uses the image-text embeddings as supervision
        filedir = os.path.abspath(__file__)
        filedir = os.path.dirname(filedir)
        pt_path = os.path.join(filedir, "models/AudioCLIP/assets/audioclip.pt")
        model = AudioCLIP(pretrained=pt_path)

    concept_cache = {}
    print(f"EXTRACTING CONCEPTS FOR {args.classes.upper()} CLASSES\n")
    
    if args.classes == "cifar10":
        # Pull CIFAR10 to get the class names.
        from torchvision import datasets
        cifar10_ds = datasets.CIFAR10(root=args.out_dir, train=True, download=True)
        # Get the class names.
        all_classes = list(cifar10_ds.classes)
        # Get the names of all concepts.
        all_concepts = get_concept_data(all_classes)
        # Clean the concepts for uniques, plurals etc. 
        all_concepts = clean_concepts(all_concepts)     
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # Generate the concept bank.
        learn_conceptbank(args, all_concepts, args.classes, model)
        
    elif args.classes == "cifar100":
        # Pull CIFAR100 to get the class names.
        from torchvision import datasets
        cifar100_ds = datasets.CIFAR100(root=args.out_dir, train=True, download=True)
        all_classes = list(cifar100_ds.classes)
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))

        learn_conceptbank(args, all_concepts, args.classes, model)
    
    elif args.classes == "cub":
        all_concepts = [
        # Physical Features
        'Plumage', 'Wing Shape', 'Beak Structure', 'Tail Length', 'Leg Length', 'Eye Color',
        
        # Habitat and Distribution
        'Geographic Location', 'Preferred Habitat', 'Altitude',
        
        # Behavioral Traits
        'Nesting', 'Migration', 'Feeding', 'Vocalization',
        
        # Size and Weight
        'Size', 'Wingspan', 'Weight',
        
        # Social Structure
        'Social Behavior', 'Mating', 'Hierarchy',
        
        # Adaptations
        'Specialization', 'Physical Traits',
        
        # Ecological Niche
        'Ecosystem Role', 'Species Interaction',
        
        # Conservation Status
        'Threat Level',
        
        # Taxonomic Information
        'Family', 'Genus', 'Species', 'Order', 'Class',
        
        # Notable Characteristics
        'Markings', 'Crest', 'Tufts', 'Facial Disk'
        ]

        learn_conceptbank(args, all_concepts, args.classes, model)

    # The below two if statement branches are part of the extension experiments
    # ESC50 has 50 labels in total
    elif args.classes == "esc50":
        # Use labels to generate concepts
        from data.constants import ESC_DIR
        meta_dir = os.path.join(ESC_DIR, "esc50.csv")
        df = pd.read_csv(meta_dir)

        all_classes = list(set(df['category']))
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))

        learn_conceptbank(args, all_concepts, args.classes, model)

    elif args.classes == "us8k":
        # The concepts are derived from the urban sound taxonomy defined by the authors
        # with adjustments to make it specific enough (no leaves to prevent label overlap,
        # vehicles too specific removedas well)
        all_concepts = [
        # The four main concepts (level one)
        'Human', 'Nature', 'Mechanical', 'Music',

        # Nodes directly below main concepts (level two)
        'Voice', 'Movement', 'Elements', 'Animals', 'Plants',
        'Construction', 'Ventilation', 'Non-motor Vehicle',
        'Signals', 'Motor Vehicle', 'Non-amplified Music', 'Amplified Music',

        # Nodes directly below level two concepts (level three)
        'Bicycle', 'Skateboard', 'Marine', 'Rail', 'Road', 'Air',
        'Live Music', 'Recorded Music',

        # Nodes directly below level three concepts (level four)
        'Boat', 'Train', 'Subway', 'Car', 'Motorcycle', 'Bus', 'Truck'
        ]

        learn_conceptbank(args, all_concepts, args.classes, model)
    
    elif args.classes == "audioset":
        import json

        # Read the JSON file
        with open('concepts/ontology.json', 'r') as file:
            data = json.load(file)

        # Extract the names into a list
        all_concepts = [entry['name'] for entry in data]

    elif args.classes == "audioset+us8k+esc50":
        import json
        from data.constants import ESC_DIR
        
        # Read the JSON file
        with open('concepts/ontology.json', 'r') as file:
            data = json.load(file)

        # Extract the names into a list
        all_concepts1 = [entry['name'] for entry in data]

        all_concepts1.extend([
        # The four main concepts (level one)
        'Human', 'Nature', 'Mechanical', 'Music',

        # Nodes directly below main concepts (level two)
        'Voice', 'Movement', 'Elements', 'Animals', 'Plants',
        'Construction', 'Ventilation', 'Non-motor Vehicle',
        'Signals', 'Motor Vehicle', 'Non-amplified Music', 'Amplified Music',

        # Nodes directly below level two concepts (level three)
        'Bicycle', 'Skateboard', 'Marine', 'Rail', 'Road', 'Air',
        'Live Music', 'Recorded Music',

        # Nodes directly below level three concepts (level four)
        'Boat', 'Train', 'Subway', 'Car', 'Motorcycle', 'Bus', 'Truck'
        ])

        meta_dir = os.path.join(ESC_DIR, "esc50.csv")
        df = pd.read_csv(meta_dir)

        all_classes = list(set(df['category']))
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)

        learn_conceptbank(args, all_concepts, args.classes, model)
    
    elif 'task' in args.classes :
        # Either get class names or pull the dataset ourselves(req token again) 
        # TODO decide on solution
        all_classes = args.class_names.split(',') if args.class_names else []
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.

        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))

        learn_conceptbank(args, all_concepts, args.classes)

    elif args.classes.lower() == "broden":
        from data.constants import BRODEN_CONCEPTS
        concept_loaders = {}
        concepts = [c for c in os.listdir(BRODEN_CONCEPTS) if os.path.isdir(os.path.join(BRODEN_CONCEPTS, c))]
        
        learn_conceptbank(args, concepts, args.classes, model)
        
    else:
        raise ValueError(f"Unknown classes: '{args.classes}'. Define your dataset here!")
