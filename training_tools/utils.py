# This is a function to help determine if we use multiple datasets (mainly COCO-Stuff)
# used for both train and verify files
import numpy as np
import torch
from re import sub


def test_runs(args, main, concept_bank, backbone, preprocess, mode:str="vdr"):
    """
    Arguments:
    args: argparser arguments which contains at least all the arguments from the other files (i.e., in "train_pcbm.py")
    main: the main function of the file
    concept_bank: concept bank to use
    backbone: the model backbone to use
    preprocess: the preprocessing to use on the data
    mode: codename for the file being used (i.e., in "verify_datasets_pcbm.py" -> vdr)
    """

    if args.dataset.lower() == "coco_stuff":
        print("Running 20-way Binary Classification on COCO-Stuff dataset. This may take a while...\n")
        tr_acc_list = []
        t_acc_list = []
        for target in args.targets:

            if mode == "vdr" or mode == "vcr": #  Verify Datasets PCBM | Verify Results Clip Concepts PCBM
                run_info = main(args, concept_bank, backbone, preprocess, **{'target': target})
                tr_acc_list.append(run_info['train_acc'])
                t_acc_list.append(run_info['test_acc'])
            
            elif mode == "vdh" or mode == "vch": # Verify Datasets PCBM-h | Verify Clip PCBM-h

                # We need to adjust the posthoc layer per class here
                args.pcbm_path = sub('.ckpt', f'_target_{target}.ckpt', args.pcbm_path)
                posthoc_layer = torch.load(args.pcbm_path)
                posthoc_layer = posthoc_layer.eval()
                print("Current Checkpoint:", args.pcbm_path)

                run_info = main(args, backbone, preprocess, posthoc_layer, **{'target': target})
                tr_acc_list.append(run_info['train_acc'].avg)
                # Below gives AP because the dataset is cocostuff which does classification
                t_acc_list.append(run_info['test_acc']) 

            else:
                print(f"Mode '{mode}' not supported. Use either 'r' or 'h' for regular and hybrid respectively.")
        
            print(f"Training Accuracy for Class {target} \t: {run_info['train_acc']}")
            print(f"Test Accuracy for Class {target} \t: {run_info['test_acc']}")

        #dataset is coco_stuff so compute the mean of the returned APs
        out_dict = {'train_acc': np.mean(tr_acc_list), 'test_acc': np.mean(t_acc_list)}

        return out_dict

    else:
        target = args.targets[0] # This argument does not matter as no other dataset should use this variable
                                 # (i.e., it acts as a dummy variable here)
        if mode == "vdr" or mode == "vcr": #  Verify Datasets PCBM | Verify Results Clip Concepts PCBM
            run_info = main(args, concept_bank, backbone, preprocess)
        elif mode == "vdh" or mode == "vch": # Verify Datasets PCBM-h | Verify Clip PCBM-h
            posthoc_layer = torch.load(args.pcbm_path)
            posthoc_layer = posthoc_layer.eval()
            
            run_info = main(args, backbone, preprocess, posthoc_layer, **{'target': target})
        else:
            print(f"Mode '{mode}' not supported. Use either 'r' or 'h' for regular and hybrid respectively.")
        
        return run_info

# for train_pcbm specifically as it requires the concept bank
def train_runs(args, main, concept_bank, backbone, preprocess, mode:str="r"):
    """
    Arguments:
    args: argparser arguments which contains at least all the arguments from the other files (i.e., in "train_pcbm.py")
    main: the main function of the file
    concept_bank: concept bank to use (not needed for PCBM-h)
    backbone: the model backbone to use
    preprocess: the preprocessing to use on the data
    mode: whether we're training a regular PCBM or a hybrid PCBM (inputs = 'r'/'h')
    """

    if args.dataset.lower() == "coco_stuff":
        print("Training 20 Model Instances for COCO-Stuff datasets. This may take a while...\n")
        for target in args.targets:
            if mode == 'r':
                main(args, concept_bank, backbone, preprocess, **{'target': target})
            elif mode == 'h':
                main(args, backbone, preprocess, **{'target': target})
            else:
                print(f"Mode '{mode}' not supported. Use either 'r' or 'h' for regular and hybrid respectively.")
        
    else:
        target = args.targets[0] # This argument does not matter as no other dataset should use this variable
                                 # (i.e., it acts as a dummy variable here)
        if mode == 'r':
            main(args, concept_bank, backbone, preprocess)
        elif mode == 'h':
            main(args, backbone, preprocess, **{'target': target})
        else:
            print(f"Mode '{mode}' not supported. Use either 'r' or 'h' for regular and hybrid respectively.")
