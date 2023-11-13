"""Functions for graphs handling and visualization purposes."""

import numpy as np
from networkx import Graph, get_node_attributes, kamada_kawai_layout, draw_networkx
import torch
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing._encoders import OneHotEncoder

from care.gnn_eads.constants import RGB_COLORS


def convert_gpytorch_to_networkx(graph: Data, 
                                 one_hot_encoder_elements: OneHotEncoder) -> Graph:
    """
    Convert graph in pytorch_geometric to NetworkX type.    
    For each node in the graph, the label corresponding to the atomic species 
    is added as attribute together with a corresponding color.
    Args:
        graph(torch_geometric.data.Data): torch_geometric graph object.
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements.
    Returns:
        nx_graph(networkx.classes.graph.Graph): NetworkX graph object.
    """
    node_features_matrix = graph.x.numpy()
    n_nodes = graph.num_nodes
    atom_list = []
    elements_list = list(one_hot_encoder_elements.categories_[0])
    for i in range(n_nodes):
        index = np.where(node_features_matrix[i,:] == 1)[0][0]
        atom_list.append(elements_list[index])
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    connections = list(g.edges)
    nx_graph = Graph()
    for i in range(n_nodes):
        nx_graph.add_node(i, atom=atom_list[i], rgb=RGB_COLORS[atom_list[i]])
    nx_graph.add_edges_from(connections, minlen=2)
    return nx_graph


def convert_networkx_to_gpytorch(graph_nx: Graph, 
                                 one_hot_encoder_elements: OneHotEncoder) -> Data:
    """
    Convert graph object from networkx to pytorch_geometric type.
    Args:
        graph(networkx.classes.graph.Graph): networkx graph object
    Returns:
        new_g(torch_geometric.data.Data): torch_geometric graph object        
    """
    n_nodes = graph_nx.number_of_nodes()
    n_edges = graph_nx.number_of_edges()
    node_features = torch.zeros((n_nodes, len(one_hot_encoder_elements)))
    edge_features = torch.zeros((n_edges, 1))
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    node_index = torch.zeros((n_nodes), dtype=torch.long)
    for i, node in enumerate(graph_nx.nodes):
        node_index[i] = node
        node_features[i, one_hot_encoder_elements[graph_nx.nodes[node]['atom']]] = 1
    for i, edge in enumerate(graph_nx.edges):
        edge_index[0, i] = edge[0]
        edge_index[1, i] = edge[1]
    graph_pyg = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=node_index)
    return graph_pyg


def graph_plotter(graph: Data,
                  one_hot_encoder_elements: OneHotEncoder,
                  node_size: int=320,
                  font_color: str="white",
                  font_weight: str="bold",
                  alpha: float=1.0, 
                  arrowsize: int=10,
                  width: float=1.2,
                  dpi: int=200,
                  figsize: tuple[int, int]=(4,4), 
                  node_index: bool=True, 
                  text: str=None):
    """
    Visualize graph with atom labels and colors. 
    Kamada_kawai_layout engine gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    graph = convert_gpytorch_to_networkx(graph, one_hot_encoder_elements)
    labels = get_node_attributes(graph, 'atom')
    colors = list(get_node_attributes(graph, 'rgb').values()) 
    plt.figure(figsize=figsize, dpi=dpi) 
    draw_networkx(graph, 
                  labels=labels, 
                  node_size=node_size,
                  font_color=font_color, 
                  font_weight=font_weight,
                  node_color=colors, 
                  alpha=alpha, 
                  arrowsize=arrowsize, 
                  width=width,
                  pos=kamada_kawai_layout(graph))
    if node_index:
        pos_dict = kamada_kawai_layout(graph)
        for node in graph.nodes:
            x, y = pos_dict[node]
            plt.text(x+0.05, y+0.05, node, fontsize=7)        
    if text != None:
        plt.text(0.03, 0.9, text, fontsize=10)
    # Remove frame
    plt.axis('off')
    plt.draw()                    


def extract_adsorbate(graph: Data, 
                      encoder: OneHotEncoder) -> Data:
    """
    Extract molecule from the adsorption graph, removing metals and connections between 
    metal and molecule. Node-featurization independent.
    
    Args:
        graph (torch_geometric.data.Data): Adsorption system in graph format
    Returns:
        adsorbate(torch_geometric.data.Data): Molecule in graph format
    """
    node_dim = graph.x.shape[1]
    CHONS = [list(encoder.categories_[0]).index(element) for element in ["C", "H", "O", "N", "S"]] 
    y = [None] * graph.num_nodes  # function for new indexing
    node_list, node_index, edge_list, edge_index = [], [], [], []  
    # 1) Node selection 
    counter = 0
    for atom in range(graph.num_nodes):
        index = torch.where(graph.x[atom, :] == 1)[0][0].item()
        if index in CHONS:
            y[atom] = counter
            node_index.append(atom)
            node_list.append(graph.x[atom, :])
            counter += 1
    def ff(num):  # new indexing for the new graph (important!)
        return y[num]
    # 2) Edge selection
    for link in range(graph.num_edges):
        nodes = graph.edge_index[:, link]
        switch = 0
        for node in nodes:
            if node not in node_index:
                switch = 1
        if switch == 0:
            edge_list.append(nodes)
            edge_index.append(link)
        switch = 0
    # 3) Graph construction
    x = torch.zeros((len(node_list), node_dim))
    edge = torch.zeros((2, len(edge_index)), dtype=torch.long)
    for i in range(x.shape[0]):
        x[i, :] = node_list[i]
    for j in range(2):
        for k in range(edge.shape[1]):
            edge[j, k] = ff(int(edge_list[k][j]))
    edge.to(torch.long)
    return Data(x, edge), y


