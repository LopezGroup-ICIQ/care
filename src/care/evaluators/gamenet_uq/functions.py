"""This module contains functions used for loading a pre-trained GNN."""

from torch import load, device
from torch.nn import Module

from care.evaluators.gamenet_uq.nets import GameNetUQ

def get_mean_std_from_model(path: str) -> tuple[float]:
    """Get mean and standard deviation used for scaling the target values
       from the selected trained model.

    Args:
        model_name (str): GNN model path.

    Returns:
        mean, std (tuple[float]): mean and standard deviation for scaling the targets.
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "(train+val) mean" in line:
            mean = float(line.split()[-2])
        if "(train+val) standard deviation" in line:
            std = float(line.split()[-2])
    return mean, std


def get_graph_conversion_params(path: str) -> tuple:
    """Get the hyperparameters for geometry->graph conversion algorithm.
    Args:
        path (str): path to directory containing the GNN model.
    Returns:
        tuple: voronoi tolerance (float), scaling factor (float), metal nearest neighbours inclusion (bool)
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "Voronoi" in line:
            voronoi_tol = float(line.split()[-2])
        if "scaling factor" in line:
            scaling_factor = float(line.split()[-1])
        if "Second order" in line:
            if line.split()[-1] == "True":
                second_order_nn = True
            else:
                second_order_nn = False
    return voronoi_tol, scaling_factor, second_order_nn


def load_model(path: str) -> Module:
    """
    Load GAME-Net-UQ model.
    """
    from copy import deepcopy
    with open(path + "/input.txt", "r") as f:
        config_dict = eval(f.read())
        config_dict["graph"]["structure"]["surface_order"] = 2
    target_scaling_params = get_mean_std_from_model(path)
    model = GameNetUQ(20, 192)
    model.load_state_dict(load(path + "/GNN.pth", weights_only=True, map_location=device("cpu")))
    model.y_scale_params = {"mean": target_scaling_params[0], "std": target_scaling_params[1]}
    model.eval()
    model.graph_params = deepcopy(config_dict["graph"])
    return model
