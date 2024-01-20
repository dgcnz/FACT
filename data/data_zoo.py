from torchvision import datasets
import torch
import os
import pandas as pd


def get_dataset(args, preprocess=None):
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.out_dir, train=True,
                                    download=True, transform=preprocess)
        testset  = datasets.CIFAR10(root=args.out_dir, train=False,
                                    download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_workers)
    
    
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root=args.out_dir, train=True,
                                     download=True, transform=preprocess)
        testset  = datasets.CIFAR100(root=args.out_dir, train=False,
                                     download=True, transform=preprocess)
        classes = trainset.classes
        class_to_idx = {c: i for (i,c) in enumerate(classes)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers)
        test_loader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_workers)


    elif args.dataset == "cub":
        from .cub import load_cub_data
        from .constants import CUB_PROCESSED_DIR, CUB_DATA_DIR
        from torchvision import transforms
        num_classes = 200
        TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
        TEST_PKL = os.path.join(CUB_PROCESSED_DIR, "test.pkl")
        normalizer = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        train_loader = load_cub_data([TRAIN_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False, 
            batch_size=args.batch_size, uncertain_label=False, image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
            n_classes=num_classes, resampling=True)

        classes = open(os.path.join(CUB_DATA_DIR, "classes.txt")).readlines()
        classes = [a.split(".")[1].strip() for a in classes]
        idx_to_class = {i: classes[i] for i in range(num_classes)}
        classes = [classes[i] for i in range(num_classes)]
        print(len(classes), "num classes for cub")
        print(len(train_loader.dataset), "training set size")
        print(len(test_loader.dataset), "test set size")
        

    elif args.dataset == "ham10000":
        from .derma_data import load_ham_data
        train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)
        class_to_idx = {v:k for k,v in idx_to_class.items()}
        classes = list(class_to_idx.keys())

    elif args.dataset == "coco_stuff":
        from .coco_stuff import load_coco_data, cid_to_class
        from .constants import COCO_STUFF_DIR

        return NotImplemented

        # The 20 most biased classes from Singh et al., 2020
        target_classes = ["cup", "wine glass", "handbag", "apple", "car",
                          "bus", "potted plant", "spoon", "microwave", "keyboard",
                          "skis", "clock", "sports ball", "remote", "snowboard",
                          "toaster", "hair drier", "tennis racket", "skateboard", "baseball glove"]
        
        label_path = os.path.join(COCO_STUFF_DIR, "labels.txt")
        train_path = os.path.join(COCO_STUFF_DIR, "train2017")
        test_path = os.path.join(COCO_STUFF_DIR, "val2017") # It is presumed that the validation set was used as the test one
        train_annot = os.path.join(COCO_STUFF_DIR, "annotations\instances_train2017.json")
        test_annot = os.path.join(COCO_STUFF_DIR, "annotations\instances_val2017.json")

        train_loader = load_coco_data(train_path, train_annot) # Not implemented yet ...
        test_loader  = load_coco_data(test_path, test_annot)
        idx_to_class = cid_to_class(label_path, target_classes)

    elif args.dataset == "siim_isic":
        from .siim_isic import load_siim_data
        from .constants import SIIM_DATA_DIR
        meta_dir = os.path.join(SIIM_DATA_DIR, "isic_metadata.csv")
        train_loader, test_loader = load_siim_data(meta_dir, 
                                                   batch_size=args.batch_size, 
                                                   seed=args.seed)
        
        # for the idx_to_class variable (need to read metadata table for this)
        classes = pd.read_csv(meta_dir)['benign_malignant']
        classes = sorted(list(set(classes))) # adjust so that 0:benign and 1:malignant as in the main dataset
        idx_to_class = {i: classes[i] for i in range(len(classes))}

    elif 'task' in args.dataset:
        from datasets import load_dataset
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms

        dataset = load_dataset("fact-40/pcbm_survey", name= args.dataset, use_auth_token=args.token)

        transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
        ])

        class PCBMSurveyDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                self.transform  = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]
                image = item['image']
                label = item['label']
                return image, label
            
        train_dataset = PCBMSurveyDataset(dataset['train'])
        test_dataset = PCBMSurveyDataset(dataset['test'])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        idx_to_class = {i: label for i, label in enumerate(dataset['train'].features['label'].names)}
        classes = dataset['train'].features['label'].names

    else:
        raise ValueError(args.dataset)
    
    return train_loader, test_loader, idx_to_class, classes