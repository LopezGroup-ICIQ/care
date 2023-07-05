"""Perform GNN ensemble model training."""

import argparse
from os.path import exists, isdir
from os import mkdir
from copy import deepcopy
import time
import sys
sys.path.insert(0, "../src")
import random

import torch 
import toml
from torch_geometric.loader import DataLoader

from gnn_eads.functions import create_loaders, scale_target, train_loop, test_loop, get_id
from gnn_eads.nets import FlexibleNet
from gnn_eads.post_training import create_model_report
from gnn_eads.create_graph_datasets import create_graph_datasets
from gnn_eads.constants import FG_RAW_GROUPS, loss_dict, pool_seq_dict, conv_layer, sigma_dict, pool_dict
from gnn_eads.paths import create_paths
from gnn_eads.processed_datasets import create_post_processed_datasets


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform a training process with the provided hyperparameter settings.")
    PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the ensemble learning process.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Output directory for the results.")
    ARGS = PARSER.parse_args()
    
    output_name = ARGS.i.split("/")[-1].split(".")[0]
    output_directory = ARGS.o
    if isdir("{}/{}".format(output_directory, output_name)):
        output_name = input("There is already a model with the chosen name in the provided directory, provide a new one: ")
    
    # Upload training hyperparameters from toml file
    HYPERPARAMS = toml.load(ARGS.i)  
    data_path = HYPERPARAMS["data"]["root"]    
    graph_settings = HYPERPARAMS["graph"]
    train = HYPERPARAMS["train"]
    architecture = HYPERPARAMS["architecture"]
        
    # Select device
    device_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name: {} (GPU)".format(torch.cuda.get_device_name(0)))
        device_dict["name"] = torch.cuda.get_device_name(0)
        device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
        device_dict["CUDNN_version"] = torch.backends.cudnn.version()
        device_dict["CUDA_version"] = torch.version.cuda
    else:
        print("Device name: CPU")
        device_dict["name"] = "CPU"    
     
    # Create FG-dataset from raw DFT data 
    graph_settings_identifier = get_id(graph_settings)
    family_paths = create_paths(FG_RAW_GROUPS, data_path, graph_settings_identifier)
    condition = [exists(data_path + "/" + family + "/pre_" + graph_settings_identifier) for family in FG_RAW_GROUPS]
    if False not in condition:
        FG_dataset = create_post_processed_datasets(graph_settings_identifier, family_paths)
    else:
        print("Creating graphs from raw data ...")  
        create_graph_datasets(graph_settings, family_paths)
        FG_dataset = create_post_processed_datasets(graph_settings_identifier, family_paths)
            
    # Create train/validation/test sets 
    # NB Test set is the same for all models in the ensemble, as well as the validation set
    train_big, val_loader, test_loader = create_loaders(FG_dataset,
                                                           batch_size=train["batch_size"],
                                                           split=train["splits"], 
                                                           test=train["test_set"])
    subset_size = len(train_big.dataset) // train["num_ensembles"]
    sizes = [subset_size] * train["num_ensembles"]
    remainder = len(train_big) % train["num_ensembles"]
    for i in range(remainder):
        sizes[i] += 1
    train_list = deepcopy(train_big.dataset)
    random.shuffle(train_list)  # Shuffle the dataset before splitting it into subsets
    train_datasets, train_loaders = [], []
    for ensemble in range(train["num_ensembles"]):
        train_datasets.append(train_list[ensemble*subset_size:ensemble*subset_size+sizes[ensemble]])
    for ensemble in range(train["num_ensembles"]):
        train_loaders.append(DataLoader(train_datasets[ensemble], 
                                        batch_size=train["batch_size"],
                                        shuffle=True))
    t0 = time.time()
    mkdir("{}/{}".format(output_directory, output_name))
    # Outer loop for k-ensembles
    for ensemble in range(train["num_ensembles"]):
        # Apply target scaling (standardization) 
        train_loader = train_loaders[ensemble]
        train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=train["target_scaling"], 
                                                                    test=train["test_set"]) 
        print(len(train_loader.dataset), len(val_loader.dataset))   
        # Instantiate the GNN architecture on the device
        model = FlexibleNet(dim=architecture["dim"],
                            N_linear=architecture["n_linear"], 
                            N_conv=architecture["n_conv"], 
                            adj_conv=architecture["adj_conv"],  
                            sigma=sigma_dict[architecture["sigma"]], 
                            bias=architecture["bias"], 
                            conv=conv_layer[architecture["conv_layer"]], 
                            pool=pool_dict[architecture["pool_layer"]], 
                            pool_ratio=architecture["pool_ratio"], 
                            pool_heads=architecture["pool_heads"], 
                            pool_seq=pool_seq_dict[architecture["pool_seq"]], 
                            pool_layer_norm=architecture["pool_layer_norm"]).to(device)     
        # Load optimizer and lr-scheduler
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=train["lr0"],
                                     eps=train["eps"], 
                                     weight_decay=train["weight_decay"],
                                     amsgrad=train["amsgrad"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=train["factor"],
                                                                  patience=train["patience"],
                                                                  min_lr=train["minlr"])    
        loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []         
        print("Ensemble model: {}".format(ensemble+1))
        print("-----------------------------------------------------------------------------------------")    
        # Inner loop for training
        for epoch in range(1, train["epochs"]+1):
            torch.cuda.empty_cache()
            lr = lr_scheduler.optimizer.param_groups[0]['lr']        
            loss, train_MAE = train_loop(model, device, train_loader, optimizer, loss_dict[train["loss_function"]])  
            val_MAE = test_loop(model, val_loader, device, std)  
            lr_scheduler.step(val_MAE)
            if train["test_set"]:
                test_MAE = test_loop(model, test_loader, device, std, mean)         
                test_list.append(test_MAE)
                print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                      'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
            else:
                print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV '
                      .format(epoch, lr, train_MAE*std, val_MAE))         
            loss_list.append(loss)
            train_list.append(train_MAE * std)
            val_list.append(val_MAE)
            lr_list.append(lr)
        print("-----------------------------------------------------------------------------------------")
        training_time = (time.time() - t0)/60  
        print("Training time: {:.2f} min".format(training_time))
        device_dict["training_time"] = training_time
        create_model_report("ensemble_{}".format(ensemble+1),
                            output_directory + "/" + output_name,
                            HYPERPARAMS,  
                            model, 
                            (train_loader, val_loader, test_loader),
                            (mean, std),
                            (train_list, val_list, test_list, lr_list), 
                            device_dict)