"""
Module containing a set of filter functions for graphs in the Geometric PyTorch format.
These filters are applied before the inclusion of the graphs in the HetGraphDataset objects.
"""

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing._encoders import OneHotEncoder
    

def single_fragment_filter(graph: Data, 
                           encoder: OneHotEncoder, 
                           molecule_elements: list[str]) -> bool:
    """
    Graph filter that checks if the adsorbate is fragmented in the graph.
    It is assumed that the input graph should represent a single adsorbate on the surface.
    Args: 
        graph(torch_geometric.data.Data): Adsorption graph.
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): OneHotEncoder for atomic elements
        molecule_elements(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Original adsorbate is not fragmented in the graph representation
                False = Original adsorbate is fragmented in the graph representation
    """
    # 1) Get node indices of C, H, O, N, S in the encoder
    molecule_elements_indices = [list(encoder.categories_[0]).index(element) for element in molecule_elements]  
    node_dim = graph.x.shape[1]
    switch = []
    # 2) Get C, H, O, N, S  nodes in the graph
    for node_index in range(graph.num_nodes):
        index = torch.where(graph.x[node_index, :] == 1)[0][0].item()
        switch.append(1 if index in molecule_elements_indices else 0)
    # 3) Build graph of the adsorbate
    x = torch.zeros((switch.count(1), node_dim))
    counter = 0
    func  = [None] * graph.num_nodes
    for node_index in range(graph.num_nodes):
        if switch[node_index] == 1:
            x[counter, :] = graph.x[node_index, :]
            func[node_index] = counter
            counter += 1             
    # Connections not involving metals are kept
    def ff(number):
        return func[number]
    connections = []
    switch = 0
    for edge_index in range(graph.num_edges):
        connection = list(graph.edge_index[:, edge_index])
        connection = [node.item() for node in connection]
        for j in connection:
            index = torch.where(graph.x[j, :] == 1)[0][0].item()
            if list(encoder.categories_[0])[index] not in molecule_elements:
                switch = 1
        if switch ==  0:
            connections.append(connection)
        else: # don't append connections, restart the switch
            switch = 0
    for bond in range(len(connections)):
        connections[bond] = [ff(i) for i in connections[bond]]
    connections = torch.Tensor(connections)

    graph_new = Data(x, connections.t().contiguous())
    graph_nx = to_networkx(graph_new, to_undirected=True)
    if graph_new.num_nodes != 1 and graph_new.num_edges != 0:
        return True if nx.is_connected(graph_nx) else False
    else:  # graph with one node and zero edges
        return True

    
def H_connectivity_filter(graph: Data, 
                          encoder: OneHotEncoder, 
                          molecule_elements: list[str]) -> bool:
    """
    Graph filter that checks the connectivity of H atoms whithin the adsorbate.
    Each H atoms must be connected to maximum one atom within the molecule.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        molecule_elements(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Correct connectivity for all H atoms in the molecule
                False = Bad connectivity for at least one H atom in the molecule
    """
    # 1) Get indices of molecule atoms in the one-hot encoder
    molecule_elements_indices = [list(encoder.categories_[0]).index(element) for element in molecule_elements]  
    H_index = list(encoder.categories_[0]).index('H')
    # 2) Find indices of hydrogen (H) nodes in the graph
    H_nodes_indexes = []
    for i in range(graph.num_nodes):
        if graph.x[i, H_index] == 1:
            H_nodes_indexes.append(i)
    # 3) Apply filter to H nodes (Just one bad connected H makes the whole graph wrong)
    for node_index in H_nodes_indexes:
        counter = 0  # edges between H and atoms in the molecule
        for j in range(graph.num_edges):
            if node_index == graph.edge_index[0, j]:  # NB: in PyG each edge repeated twice to have undirected graph
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0][0].item()  
                counter += 1 if other_atom in molecule_elements_indices else 0
        if counter > 1: 
            return False
    return True

def C_connectivity_filter(graph: Data, 
                          encoder: OneHotEncoder, 
                          molecule_elements: list[str]) -> bool:
    """
    Graph filter that checks the connectivity of C atoms whithin the adsorbate.
    Each C atom must be connected to maximum 4 atoms within the molecule.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        molecule_elements(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Correct connectivity for all C atoms in the molecule
                False = Bad connectivity for at least one C atom in the molecule
    """
    # 1) Get indices of molecule atoms in the one-hot encoder
    molecule_elements_indices = [list(encoder.categories_[0]).index(element) for element in molecule_elements]   
    C_index = molecule_elements_indices[0]
    # 2) Find indices of carbon (C) nodes in the graph
    C_nodes_indices = [index for index in range(graph.num_nodes) if graph.x[index, C_index] == 1]
    # 3) Apply filter to C nodes (Just one bad connected C makes the whole graph wrong)
    for node_index in C_nodes_indices:
        counter = 0  # number of edges between C and atoms belonging to molecule
        for j in range(graph.num_edges):
            if node_index == graph.edge_index[0, j]:  # NB: in PyG each edge repeated twice in order to have undirected graph
                other_atom = torch.where(graph.x[graph.edge_index[1, j], :] == 1)[0][0].item()  
                counter += 1 if other_atom in molecule_elements_indices else 0
        if counter > 4: 
            return False
    return True

    
def adsorption_filter(graph: Data, 
                      encoder: OneHotEncoder, 
                      molecule_elements: list[str]) -> bool:
    """
    Check presence of metal atoms in the adsorption graphs.
    sufficiency condition: if there is at least one atom different from C, H, O, N, S, 
    then the graph is considered as an adsorption graph.
    Args:
        graph(torch_geometric.data.Data): Graph object representation
        encoder(sklearn.preprocessing._encoders.OneHotEncoder): One-hot encoder for atomic elements
        molecule_elements(list[str]): List of atomic elements in the molecule
    Returns:
        (bool): True = Metal catalyst present in the adsorption graph
                False = No metal catalyst in the adsorption graph
    """
    if graph.type == "adsorption":
        molecule_elements_indices = [list(encoder.categories_[0]).index(element) for element in molecule_elements]
        for node_index in range(graph.num_nodes):
            index = torch.where(graph.x[node_index, :] == 1)[0][0].item()
            if index not in molecule_elements_indices:
                return True
        return False
    else:
        return True
    

def explode_graph(graph: Data, 
                  removed_node: int) -> Data:
    """Explode graph into fragments keeping out the selected node

    Args:
        graph (Data): Input adsorption/molecular graph.
        removed_node (int): index of the explosion node.

    Returns:
        exploded_graph (Data): Exploded graph.
    """
    # 1) Initialize graph feature matrix and connection for the new graph
    node_dim = graph.x.shape[1]
    x = torch.zeros((graph.num_nodes-1, node_dim))
    links = [graph.edge_index[1, i] for i in range(graph.num_edges) if graph.edge_index[0, i] == removed_node]
    edge = torch.zeros((2, graph.num_edges - 2 * len(links)))
    # 2) Find new indeces
    y = [None] * graph.num_nodes
    counter = 0
    for i in range(graph.num_nodes):
        if i != removed_node:
            y[i] = counter
            counter += 1
    def ff(node_index: int):
        return y[node_index]
    # 3) Remove connections between target node and other nodes
    edge_list = []
    edge_index = []
    for link in range(graph.num_edges):
        nodes = graph.edge_index[:, link]
        switch = 0
        for node in nodes:
            if node == removed_node:
                switch = 1
        if switch == 0:
            edge_list.append(nodes)
            edge_index.append(link)
        switch = 0
    # 4) Define new graph (delicate part)
    for node_index in range(x.shape[0]):
        if node_index < removed_node:
            xx = torch.where(graph.x[node_index, :] == 1)[0]
            x[node_index, xx] = 1
        elif node_index >= removed_node:
            xx = torch.where(graph.x[node_index+1, :] == 1)[0]
            x[node_index, xx] = 1
    for j in range(2):
        for k in range(edge.shape[1]):
            edge[j, k] = (ff(int(edge_list[k][j].item())))
    exploded_graph = Data(x, edge)
    return exploded_graph        