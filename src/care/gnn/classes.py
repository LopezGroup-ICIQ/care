"""Module containing the classes used for training and wrapping the GNN model."""


import os.path as osp
from os import listdir

import torch
from torch_geometric.data import Data
import numpy as np

from care.gnn.functions import get_graph_conversion_params, get_mean_std_from_model
from care.gnn.graph_tools import extract_adsorbate


class PreTrainedModel():
    def __init__(self, model_path: str):
        """Container class for loading pre-trained GNN models on the cpu.
        Args:
            model_path (str): path to model folder. It must contain:
                - model.pth: the model architecture
                - GNN.pth: the model weights
                - performance.txt: the model performance and settings                
        """
        self.model_path = model_path
        self.model = torch.load("{}/model.pth".format(self.model_path),
                                map_location=torch.device("cpu"))
        self.model.load_state_dict(torch.load("{}/GNN.pth".format(self.model_path), 
                                              map_location=torch.device("cpu")))
        self.model.eval()  
        self.model.to("cpu")
        # Scaling parameters (only standardization for now)
        self.mean, self.std = get_mean_std_from_model(self.model_path)
        # Graph conversion parameters
        self.g_tol, self.g_sf, self.g_metal_2nn = get_graph_conversion_params(self.model_path)
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
               
        param_size, buffer_size = 0, 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        self.size_all_mb = (param_size + buffer_size) / 1024**2
        
    def __repr__(self) -> str:
        string = "Pretrained graph neural network for DFT ground state energy prediction."
        string += "\nModel path: {}".format(osp.abspath(self.model_path))
        string += "\nNumber of parameters: {}".format(self.num_parameters)
        string += "\nModel size: {:.2f} MB".format(self.size_all_mb)
        return string
    
    def evaluate(self, graph: Data) -> float:
        """Evaluate graph energy

        Args:
            graph (Data): adsorption/molecular graph

        Returns:
            float: system energy in eV
        """
        with torch.no_grad():
            return self.model(graph).item() * self.std + self.mean
        
class EnsembleModel():
    def __init__(self, model_path: str):
        """Container class for loading multiple pre-trained GNN models on the cpu."""
        paths = [osp.join(model_path, model) for model in listdir(model_path)]
        self.num_ensembles = len(paths)
        self.models = [PreTrainedModel(path) for path in paths]
        self.g_tol, self.g_sf, self.g_metal_2nn = get_graph_conversion_params(paths[0])
    

    def __repr__(self) -> str:
        string = "Ensemble of {} pretrained GNNs for DFT ground state energy prediction.".format(self.num_ensembles)
        string += "\nModel paths: {}".format([osp.abspath(model.model_path) for model in self.models])
        string += "\nNumber of parameters: {}".format(sum([model.num_parameters for model in self.models]))
        string += "\nModel size: {:.2f} MB".format(sum([model.size_all_mb for model in self.models]))
        return string
    
    def evaluate(self, graph: Data) -> tuple[float]:
        """Evaluate graph energy and its standard deviation

        Args:
            graph (Data): adsorption/molecular graph

        Returns:
            tuple[float]: system energy in eV and its standard deviation
        """
        with torch.no_grad():
            preds = [model.evaluate(graph) for model in self.models]
            # Get normal distribution of predictions
            mean = sum(preds) / self.num_ensembles
            std = sum([(pred - mean)**2 for pred in preds]) / self.num_ensembles
            return mean, std
        
    def get_adsorption_energy(self, graph: Data) -> float:
        """Evaluate adsorption energy

        Args:
            graph (Data): adsorption graph. Must contain metal nodes.

        Returns:
            float: adsorption energy in eV, as well as its standard deviation

        Notes:
            The adsorption energy is computed as the difference between the energy of the adsorbate and the molecule energy.
            The molecule energy is predicted by the GNN.
        """
        with torch.no_grad():
            adsorption_energy = self.evaluate(graph)
            molecule_graph = extract_adsorbate(graph)
            molecule_energy = self.evaluate(molecule_graph)
            mean_adsorption_energy = adsorption_energy[0] - molecule_energy[0]
            std_adsorption_energy = (adsorption_energy[1]**2 + molecule_energy[1]**2)**0.5
            return mean_adsorption_energy, std_adsorption_energy