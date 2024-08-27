"""This module contains functions used for loading a pre-trained GNN."""

from ase.db import connect
from torch import load
from torch.nn import Module

from care.evaluators.gamenet_uq.nets import GameNetUQ
from care import Surface
from care.evaluators.gamenet_uq import DB_PATH, METAL_STRUCT_DICT


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
    one_hot_encoder_elements = load(path + "/one_hot_encoder_elements.pth")
    with open(path + "/input.txt", "r") as f:
        config_dict = eval(f.read())
    graph_params = config_dict["graph"]
    scale_params = get_mean_std_from_model(path)
    node_feats_list = one_hot_encoder_elements.categories_[0].tolist()
    num_node_feats = len(node_feats_list) + sum(graph_params["features"].values())
    model = GameNetUQ(num_node_feats, config_dict["architecture"]["dim"])
    model.load_state_dict(load(path + "/GNN.pth"))
    model.y_scale_params = {"mean": scale_params[0], "std": scale_params[1]}
    model.eval()
    model.graph_params = graph_params
    return model


def load_surface(metal: str, hkl: str) -> Surface:
    """
    Load surface from ASE database.

    Args:
        metal (str): Metal symbol (e.g., "Ag")
        hkl (str): Miller index (e.g., "111", "0001")

    Note:
        The database should contain a surface with the given metal and Miller index.
        For hcp metals, the Miller index should be in the form "hkil", negative indices
        should be written as "mh-kil" (e.g. "10m11" stands for 10-11).
    """
    metal_db = connect(DB_PATH)
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    try:
        surface_ase = metal_db.get_atoms(
            calc_type="surface", metal=metal, facet=metal_structure
        )
    except:
        # Generate surface from scratch (possible with current implementation!!!)
        raise ValueError(f"{metal} surface {metal_structure} not found in the database.")

    return Surface(surface_ase, hkl)
