"""Module with functions for creating plots for training report"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import torch

from GAMERNet.gnn_eads.src.gnn_eads.functions import split_percentage
from GAMERNet.gnn_eads.src.gnn_eads.constants import DPI


def hist_num_atoms(n_list:list[int]):
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=DPI)
    ax.bar(*np.unique(n_list, return_counts=True), color="#219ebc")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Count")
    zoomcol = "black"
    ax.spines['bottom'].set_color(zoomcol)
    ax.spines['top'].set_color(zoomcol)
    ax.spines['right'].set_color(zoomcol)
    ax.spines['left'].set_color(zoomcol)
    plt.ioff()
    return fig, ax


def violinplot_family(model,
                      loader,
                      std_tv: float,
                      dataset_labels: list,
                      device :str="cpu"):
    """Generate violinplot sorted by chemical family.

    Args:
        model (_type_): GNN model instantiation.
        loader (_type_): DataLoader.
        std_tv (_type_): standard deviation of the train+val dataset.
    """
    family_dict = {}
    error_dict = {}
    error_list = []
    family_list = []
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=DPI)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    x = "$\mathit{E}_{GNN}-\mathit{E}_{DFT}$ / eV"
    for label in dataset_labels:
        checker = lambda x: x.family == label
        family_dict[label] = list(filter(checker, loader.dataset)) 
        error = np.zeros(len(family_dict[label]))
        y_GNN = np.zeros(len(family_dict[label]))
        model.to(device)
        model.eval()
        family_loader = DataLoader(family_dict[label], batch_size=len(family_dict[label]), shuffle=False)
        for batch in family_loader:
            y_GNN = model(batch)
        y_GNN = y_GNN.detach().numpy()           
        for i in range(len(family_dict[label])): 
            y_DFT = family_dict[label][i].y           
            error[i] = -(y_DFT - y_GNN[i]) * std_tv
        error_dict[label] = list(error)
        for item in error_dict[label]:
            error_list.append(item) 
    for label in dataset_labels:
        for i in range(len(error_dict[label])):
            family_list.append(label)
    df = pd.DataFrame(data={x: np.asarray(error_list, dtype=float), "Chemical family": family_list})
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    order = ["$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}O_{(2,3)}$", "$C_{x}H_{y}N$", "$C_{x}H_{y}S$", "Amidines", "Amides",
             "Oximes", "Carbamates", "Aromatics"]
    ax = sns.violinplot(y=x, x="Chemical family", data=df, orient="v",
                        scale="width", linewidth=0.5, palette="pastel",
                        bw="scott", zorder=1, order=order)     
    ax.set_xlim([-0.5, 8.5])  
    ax.set_ylim([-1.75, 1.75])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.grid(False)
    #ax.set_xlabel(None)
    #ax.set_xticklabels([]) # Comment this line if you want to display the family names!
    ax.hlines(0, -1, 10, linestyles="dotted", zorder=0, lw=0.5, color="black")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    plt.ioff()
    return fig, ax  


def violinplot_metal(model,
                     loader,
                     std_tv: float,
                     metals: list[str],
                     device :str="cpu"):
    """"
    Generate violinplot sorted by metal.
    """
    metal_dict = {}
    error_dict = {}
    error_list = []
    metal_list = []
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=DPI)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    x = "$\mathit{E}_{GNN}-\mathit{E}_{DFT}$ / eV"
    for metal in metals:
        checker = lambda x: metal in x.formula
        metal_dict[metal] = list(filter(checker, loader.dataset)) 
        error = np.zeros(len(metal_dict[metal]))
        y_GNN = np.zeros(len(metal_dict[metal]))
        model.to(device)
        model.eval()
        metal_loader = DataLoader(metal_dict[metal], batch_size=len(metal_dict[metal]))
        for batch in metal_loader:
            y_GNN = model(batch)
        y_GNN = y_GNN.detach().numpy()           
        for i in range(len(metal_dict[metal])): 
            y_DFT = metal_dict[label][i].y           
            error[i] = -(y_DFT - y_GNN[i]) * std_tv
        error_dict[metal] = list(error)
        for item in error_dict[metal]:
            error_list.append(item) 
    for label in metals:
        for i in range(len(error_dict[label])):
            metal_list.append(label)
    df = pd.DataFrame(data={x: np.asarray(error_list, dtype=float), "Metal": metal_list})
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    ax = sns.violinplot(y=x, x="Metal", data=df, orient="v",
                        scale="width", linewidth=0.5, palette="pastel",
                        bw="scott", zorder=1)     
    ax.set_xlim([-0.5, 8.5])  
    ax.set_ylim([-1.75, 1.75])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.grid(False)
    #ax.set_xlabel(None)
    #ax.set_xticklabels([]) # Comment this line if you want to display the family names!
    ax.hlines(0, -1, 10, linestyles="dotted", zorder=0, lw=0.5, color="black")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    plt.ioff()
    return fig, ax  


def E_violinplot_train(model,
                       loader,
                       std_tv,
                       dataset_labels,
                       epoch):
    """Generate Violinplot during training
    Args:
        model (_type_): GNN model instantiation
        loader (_type_): DataLoader
        std_tv (_type_): standard deviation of the train+val dataset
    """
    family_dict = {}
    error_dict = {}
    error_list = []
    family_list = []
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=300)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    x = "$\mathit{E}_{GNN}-\mathit{E}_{DFT}$ / eV"
    for label in dataset_labels:
        checker = lambda x: x.family == label
        family_dict[label] = list(filter(checker, loader.dataset)) 
        error = np.zeros(len(family_dict[label]))
        y_GNN = np.zeros(len(family_dict[label]))
        family_loader = DataLoader(family_dict[label], batch_size=len(family_dict[label]), shuffle=False)
        for batch in family_loader:
            batch.to("cuda")
            y_GNN = model(batch)
        y_GNN = y_GNN.detach().cpu().numpy()           
        for i in range(len(family_dict[label])): 
            y_DFT = family_dict[label][i].y           
            error[i] = -(y_DFT - y_GNN[i]) * std_tv
        error_dict[label] = list(error)
        for item in error_dict[label]:
            error_list.append(item) 
    for label in dataset_labels:
        for i in range(len(error_dict[label])):
            family_list.append(label)
    df = pd.DataFrame(data={x: np.asarray(error_list, dtype=float), "Chemical family": family_list})
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    order = ["$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}O_{(2,3)}$", "$C_{x}H_{y}N$", "$C_{x}H_{y}S$", "Amidines", "Amides",
             "Oximes", "Carbamates", "Aromatics"]
    ax = sns.violinplot(y=x, x="Chemical family", data=df, orient="v",
                        scale="width", linewidth=0.5, palette="pastel",
                        bw="scott", zorder=1, order=order)     
    ax.set_xlim([-0.5, 8.5])  
    ax.set_ylim([-2.5, 2.5])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.set_xlabel(None)
    #ax.set_xticklabels([]) # Comment this line if you want to display the family names!
    ax.hlines(0, -1, 10, linestyles="dotted", zorder=0, lw=0.5, color="black")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.text(0.05, 0.95, "epoch {}".format(epoch), transform=ax.transAxes, verticalalignment='top')
    plt.ioff()
    return fig, ax    


def DFTvsGNN_plot(model,
                  loader,
                  mean_tv,
                  std_tv):
    """Generate plot of GNN prediction vs true DFT data.

    Args:
        model (_type_): GNN class
        loader (_type_): data loader (train, validation, test)
        mean_tv (float): mean energy of the train+val dataset
        std_tv (float): standard deviation of the train+val dataset
    """
    to_scal = lambda x: np.array([n.item() for n in x], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    ax.set_ylabel('$\mathit{E}_{GNN}$ / eV')
    ax.set_xlabel('$\mathit{E}_{DFT}$ / eV')
    predict_lst = []
    true_lst = []
    model.eval()
    for batch in loader:        
        batch = batch.to("cpu")
        predict_lst += model(batch)
        true_lst += batch.y
    predict_lst = to_scal(predict_lst)
    true_lst = to_scal(true_lst)
    error = [predict_lst[i] * std_tv - true_lst[i] * std_tv  for i in range(len(true_lst))]
    abs_error = [abs(error[i]) for i in range(len(error))]
    MAE = np.mean(abs_error) 
    #print(MAE)
    s_error = np.std(error)
    #print(s_error)
    ax.scatter(mean_tv + true_lst * std_tv, mean_tv + predict_lst * std_tv, 
               s=12, marker="v", alpha=0.8, c="red", edgecolors="black", linewidths=0.3, zorder=2)
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    zoomcol1 = "#83BD75"
    zoomcol = "#DDDDDD"
    x = np.arange(-120, 0, 0.01)
    ax.fill_between(x, -70, -65, where=(x < -65.0) & (x > - 70.0), alpha=0.8, color=zoomcol, zorder=0)
    info = 'MAE = {:.2f} eV\n$\mathit{{s}}_{{error}}$ = {:.2f} eV\n'.format(MAE, s_error)
    ax.annotate(info, [0.60, 0.15], xycoords='axes fraction')  
    l, b, h, w = .25, .6, .2, .2
    ax2 = fig.add_axes([l, b, w, h])
    ax2.scatter(mean_tv + predict_lst * std_tv, mean_tv + true_lst * std_tv,
                s=5, marker="v", alpha=0.8, c="red", edgecolors="black", linewidths=0.3)
    ax2.set_xlim([-70, -65])
    ax2.set_ylim([-70, -65])
    ax2.set_facecolor(zoomcol)
    ax2.yaxis.set_major_locator(MaxNLocator(1)) 
    ax2.xaxis.set_major_locator(MaxNLocator(1)) 
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.ioff()
    return fig, ax


def DFTvsGNN_plot_train(model,
                        loader,
                        mean_tv,
                        std_tv,
                        epoch):
    """Generate parity plot of GNN prediction vs DFT data.

    Args:
        model (_type_): GNN class
        loader (_type_): data loader (train, validation, test)
        mean_tv (float): mean energy of the train+val dataset
        std_tv (float): standard deviation of the train+val dataset
    """
    to_scal = lambda x: np.array([n.item() for n in x], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)        
    plt.rcParams.update({'mathtext.default': 'regular'})
    ax.set_ylabel('$\mathit{E}_{GNN}$ / eV')
    ax.set_xlabel('$\mathit{E}_{DFT}$ / eV')
    predict_lst = []
    true_lst = []
    for batch in loader:        
        batch = batch.to("cuda")
        predict_lst += model(batch)
        true_lst += batch.y
    predict_lst = to_scal(predict_lst)
    true_lst = to_scal(true_lst)
    error = [predict_lst[i] * std_tv - true_lst[i] * std_tv  for i in range(len(true_lst))]
    abs_error = [abs(error[i]) for i in range(len(error))]
    MAE = np.mean(abs_error) 
    R2 = r2_score(true_lst, predict_lst)
    ax.scatter(mean_tv + true_lst * std_tv, mean_tv + predict_lst * std_tv, 
               s=24, marker="v", alpha=0.8, c="red", edgecolors="black", linewidths=0.3, zorder=2)
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
    ax.set_aspect('equal')
    ax.set_xlim([-80, -70])
    ax.set_ylim([-80, -70])
    zoomcol1 = "#83BD75"
    zoomcol = "#DDDDDD"
    ax.spines['top'].set_color("black")
    ax.spines['right'].set_color("black")
    x = np.arange(-120, 0, 0.01)
    info = 'epoch {}\nMAE = {:.2f} eV\n'.format(epoch, MAE)
    ax.annotate(info, [0.60, 0.15], xycoords='axes fraction')  
    plt.ioff()
    return fig, ax


def pred_real(model,
              train_loader: DataLoader,
              val_loader: DataLoader,
              test_loader: DataLoader,
              split: int,
              mean_tv: float,
              std_tv: float):
    """
    Generate plots related to the training process.
    Args:
        model(Net): GNN model
        train_loader(DataLoader): training dataset
        val_loader(DataLoader): validation dataset
        test_loader(DataLoader): test dataset
        split(int): splits of the initial whole dataset
        mean_tv(float): mean of the train+val dataset
        std_tv(float): standard deviation of the train+val dataset
    Returns:
        fig, ax1, ax2, ax3: subplot
    """
    a, b, c = split_percentage(split)
    to_scal = lambda x: np.array([n.item() for n in x], dtype=float)
    model.to("cpu")
    N_tot = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 4), ncols=3, dpi=DPI)
    true_lst_lst = []
    ax1.set_ylabel('Predicted Energy / eV')
    ax2.set_xlabel('DFT Energy / eV')
    for axis, loader, name in zip((ax1, ax2, ax3),
                              (train_loader, val_loader, test_loader),
                              ('Training Set', 'Validation Set', 'Test Set')):
        predict_lst = []
        true_lst = []
        model.eval()
        for batch in loader:        
            batch = batch.to("cpu")
            predict_lst += model(batch)
            true_lst += batch.y
        predict_lst = to_scal(predict_lst)
        true_lst = to_scal(true_lst)
        true_lst_lst.append(true_lst)
        axis.scatter(mean_tv + predict_lst * std_tv, mean_tv + true_lst * std_tv, s=5)
        axis.set_title(name)
        #axis.grid()
        error = (true_lst - predict_lst) * std_tv 
        absolute_error = abs(error)
        info = 'MAE: {:.2f} eV \n'.format(np.mean(absolute_error))
        axis.annotate(info, [0.60, 0.15], xycoords='axes fraction')
    plt.ioff()
    return fig, ax1, ax2, ax3


def training_plot(train: list, val: list, test: list, split: int):
    """Returns the plot of the learning process (MAE vs Epoch).

    Args:
        train (list): _description_
        val (list): _description_
        test (list): _description_
    """
    epochs = list(range(len(train)))
    for item in epochs:
        item += 1
    fig, ax = plt.subplots(figsize=(15/2.54, 10/2.54), dpi=DPI, layout="constrained")
    ax.plot(epochs, train, label="Train", lw=2, color="#5fbcd3ff")
    ax.plot(epochs, [val[i] for i in range(len(val))], label="Validation", lw=2, color="#de8787ff")
    ax.plot(epochs, test, label="Test", lw=2, color="#ffd42aff")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_xlim([-1, len(train)+1])
    ax.set_ylabel("MAE / eV", fontsize=18)
    ax.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.grid()
    plt.ioff()
    return fig, ax    


def DFT_kdeplot(fg_dataset):
    g_list = []
    for dataset in fg_dataset:
        for graph in list(dataset):
            g_list.append(graph.ener.item())
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=600)
    sns.kdeplot(g_list, ax=ax, color="#008080")
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    plt.xlabel("$\mathit{E}_{DFT}$ / eV")
    plt.ylabel("Density")
    ax.yaxis.set_major_locator(MaxNLocator(3)) 
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    return fig, ax

def label_dist_train_val_test(train_loader, val_loader, test_loader=None):
    "Plot data labels distribution in the training, validation and test sets."
    train_y = [graph.y.item() for graph in train_loader.dataset]
    val_y = [graph.y.item() for graph in val_loader.dataset]
    fig, ax = plt.subplots(figsize=(9/2.54, 6/2.54), dpi=300)
    sns.kdeplot(train_y, ax=ax, color="red")
    sns.kdeplot(val_y, ax=ax, color="blue")
    if test_loader != None:
        test_y = [graph.y.item() for graph in test_loader.dataset]
        sns.kdeplot(test_y, ax=ax, color="orange")    
    ax.legend(["Train", "Val", "Test"], loc=1)
    plt.xlabel("Scaled energy / -")
    plt.ylabel("Density")
    return fig, ax
    


def E_violinplot_train_gif(model, loader, std_tv, dataset_labels, epoch):
    """Generate Violinplot during training
    Args:
        model (_type_): GNN model instantiation
        loader (_type_): DataLoader
        std_tv (_type_): standard deviation of the train+val dataset
    """
    family_dict = {}
    error_dict = {}
    error_list = []
    family_list = []
    fig, ax = plt.subplots(figsize=(8/2.54, 4.94/2.54), dpi=400)
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    x = "$\mathit{E}_{GNN}-\mathit{E}_{DFT}$ / eV"
    for label in dataset_labels:
        checker = lambda x: x.family == label
        family_dict[label] = list(filter(checker, loader.dataset)) 
        error = np.zeros(len(family_dict[label]))
        y_GNN = np.zeros(len(family_dict[label]))
        family_loader = DataLoader(family_dict[label], batch_size=len(family_dict[label]), shuffle=False)
        for batch in family_loader:
            batch.to("cuda")
            y_GNN = model(batch)
        y_GNN = y_GNN.detach().cpu().numpy()           
        for i in range(len(family_dict[label])): 
            y_DFT = family_dict[label][i].y           
            error[i] = -(y_DFT - y_GNN[i]) * std_tv
        error_dict[label] = list(error)
        for item in error_dict[label]:
            error_list.append(item) 
    for label in dataset_labels:
        for i in range(len(error_dict[label])):
            family_list.append(label)
    df = pd.DataFrame(data={x: np.asarray(error_list, dtype=float), "Chemical family": family_list})
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    order = ["$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}O_{(2,3)}$", "$C_{x}H_{y}N$", "$C_{x}H_{y}S$", "Amidines", "Amides",
             "Oximes", "Carbamates", "Aromatics"]
    ax = sns.violinplot(y=x, x="Chemical family", data=df, orient="v",
                        scale="width", linewidth=0.5, palette="pastel",
                        bw="scott", zorder=1, order=order)     
    ax.set_xlim([-0.5, 8.5])  
    ax.set_ylim([-2.5, 2.5])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tick_params(axis='x', which='both', bottom=False)
    ax.set_xlabel(None)
    #ax.set_xticklabels([]) # Comment this line if you want to display the family names!
    ax.hlines(0, -1, 10, linestyles="dotted", zorder=0, lw=0.5, color="black")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    #ax.text(0.05, 0.95, "epoch {}".format(epoch), transform=ax.transAxes, verticalalignment='top')
    plt.ioff()
    return fig, ax


def training_plot_gif(train: list, val: list, test: list):
    """Returns the plot of the learning process (MAE vs Epoch).

    Args:
        train (list): _description_
        val (list): _description_
        test (list): _description_
    """
    epochs = list(range(len(train)))
    for item in epochs:
        item += 1
    fig, ax = plt.subplots(figsize=(15/2.54, 10/2.54), dpi=400, layout="constrained")
    ax.plot(epochs, train, label="Train", lw=2, color="#5fbcd3ff")
    ax.plot(epochs, [val[i] for i in range(len(val))], label="Validation", lw=2, color="#de8787ff")
    ax.plot(epochs, test, label="Test", lw=2, color="#ffd42aff")
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_xlim([-1, 201])
    ax.set_ylabel("MAE / eV", fontsize=18)
    ax.set_ylim([0.0, 2.5])
    ax.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #ax.grid()
    plt.ioff()
    return fig, ax 

def error_dist_test_gif(test_loader, model, std):
    error_test = []
    loader = DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))
    for batch in loader:
        batch.to("cuda")
        ygnn = model(batch)
    ygnn = ygnn.detach().cpu().numpy() 
    torch.cuda.empty_cache()
    for i in range(len(test_loader.dataset)):
        error_test.append((test_loader.dataset[i].y.item() - ygnn[i])*std)
    fig, ax = plt.subplots(figsize=(12/2.54, 12/2.54), dpi=300)
    sns.displot(error_test, bins=50, kde=True)
    plt.xlim((-7.5, 7.5))
    plt.ylim((0, 35))
    plt.xlabel("Error / eV")
    plt.ylabel("Count / -") 
    plt.ioff()
    return fig, ax