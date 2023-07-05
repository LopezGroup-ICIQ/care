"""Perform nested cross validation for GNN using the FG-dataset.
To run it, you can use the following command:
python nested_cross_validation_GNN.py -i input_hyperparams.toml -o ../results_dir
The input toml has the same structure as the one used for the training process."""

import argparse
from os.path import exists, isdir
from os import mkdir, listdir
import time
import sys
sys.path.insert(0, "../src")

import torch 
import toml
import numpy as np
import pandas as pd

from gnn_eads.functions import scale_target, train_loop, test_loop, get_id, create_loaders_nested_cv
from gnn_eads.nets import FlexibleNet
from gnn_eads.post_training import create_model_report
from gnn_eads.create_graph_datasets import create_graph_datasets
from gnn_eads.constants import FG_RAW_GROUPS, loss_dict, pool_seq_dict, conv_layer, sigma_dict, pool_dict
from gnn_eads.paths import create_paths
from gnn_eads.processed_datasets import create_post_processed_datasets


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform nested cross validation for GNN using the FG-dataset.")
    PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the nested cross validation.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Directory where the results will be saved.")
    ARGS = PARSER.parse_args()
    
    output_name = ARGS.i.split("/")[-1].split(".")[0]
    output_directory = ARGS.o
    if isdir("{}/{}".format(output_directory, output_name)):
        output_name = input("There is already a model with the chosen name in the provided directory, provide a new one: ")
    mkdir("{}/{}".format(output_directory, output_name))
    # Upload training hyperparameters from toml file
    HYPERPARAMS = toml.load(ARGS.i)  
    data_path = HYPERPARAMS["data"]["root"]    
    graph_settings = HYPERPARAMS["graph"]
    train = HYPERPARAMS["train"]
    architecture = HYPERPARAMS["architecture"]
        
    print("Nested cross validation for GNN using the FG-dataset")
    print("Number of splits: {}".format(train["splits"]))
    print("Total number of runs: {}".format(train["splits"]*(train["splits"]-1)))
    print("--------------------------------------------")
    # Select device (GPU/CPU)
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
    # Instantiate iterator for nested cross validation: Each iteration yields a different train/val/test set combination
    ncv_iterator = create_loaders_nested_cv(FG_dataset, split=train["splits"], batch_size=train["batch_size"])        
    MAE_outer = []
    counter = 0
    TOT_RUNS = train["splits"]*(train["splits"]-1)
    for outer in range(train["splits"]):
        MAE_inner = []
        for inner in range(train["splits"]-1):
            counter += 1
            train_loader, val_loader, test_loader = next(ncv_iterator)
            train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                            val_loader, 
                                                                            test_loader,
                                                                            mode=train["target_scaling"],
                                                                            test=True)  # True is necessary condition for nested CV
            # Instantiate model, optimizer and lr-scheduler
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
            t0 = time.time()
            # Run the learning    
            for epoch in range(1, train["epochs"]+1):
                torch.cuda.empty_cache()
                lr = lr_scheduler.optimizer.param_groups[0]['lr']        
                loss, train_MAE = train_loop(model, device, train_loader, optimizer, loss_dict[train["loss_function"]])  
                val_MAE = test_loop(model, val_loader, device, std)  
                lr_scheduler.step(val_MAE)
                test_MAE = test_loop(model, test_loader, device, std, mean)         
                print('{}/{}-Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                      'Test MAE: {:.4f} eV'.format(counter, TOT_RUNS, epoch, lr, train_MAE*std, val_MAE, test_MAE))
                test_list.append(test_MAE)       
                loss_list.append(loss)
                train_list.append(train_MAE * std)
                val_list.append(val_MAE)
                lr_list.append(lr)
                if epoch == train["epochs"]:
                    MAE_inner.append(test_MAE)
            print("-----------------------------------------------------------------------------------------")
            training_time = (time.time() - t0)/60  
            print("Training time: {:.2f} min".format(training_time))
            device_dict["training_time"] = training_time
            create_model_report("{}_{}".format(outer+1, inner+1),
                                output_directory+"/"+output_name,
                                HYPERPARAMS,  
                                model, 
                                (train_loader, val_loader, test_loader),
                                (mean, std),
                                (train_list, val_list, test_list, lr_list), 
                                device_dict)
            del model, optimizer, lr_scheduler, train_loader, val_loader, test_loader
            if device == "cuda":
                torch.cuda.empty_cache()
        MAE_outer.append(np.mean(MAE_inner))
    MAE = np.mean(MAE_outer)
    print("Nested CV MAE: {:.4f} eV".format(MAE))
    # Generate report of the whole experiment
    ncv_results = listdir(output_directory + "/" + output_name)
    df = pd.DataFrame()
    for run in ncv_results:
        results = pd.read_csv(output_directory + "/" + output_name + "/" + run + "/test_set.csv", sep="\t")
        # add column with run number
        results["run"] = run
        df = pd.concat([df, results], axis=0)
    df = df.reset_index(drop=True)
    df.to_csv(output_directory + "/" + output_name + "/summary.csv", index=False)



