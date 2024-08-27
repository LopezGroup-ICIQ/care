"""Functions for graphs handling and visualization purposes."""

import matplotlib.pyplot as plt
import torch
import torch_geometric
from networkx import (
    Graph,
    draw_networkx,
    get_node_attributes,
    kamada_kawai_layout,
    set_edge_attributes,
)
from sklearn.preprocessing._encoders import OneHotEncoder
from torch_geometric.data import Data

from care.constants import RGB_COLORS


def pyg_to_nx(graph: Data) -> Graph:
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
    n_nodes = graph.num_nodes
    atom_list = [graph.elem[i] for i in range(n_nodes)]
    edge_ts_list = {}
    for i in range(graph.num_edges):
        node_idxs = graph.edge_index[:, i]
        node1 = node_idxs[0].item()
        node2 = node_idxs[1].item()
        if graph.edge_attr[i, 0] == 1:
            edge_ts_list[(node1, node2)] = 1
        else:
            edge_ts_list[(node1, node2)] = 0
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    connections = list(g.edges)
    nx_graph = Graph()
    for i in range(n_nodes):
        nx_graph.add_node(i, elem=atom_list[i], rgb=RGB_COLORS[atom_list[i]])
    nx_graph.add_edges_from(connections)
    set_edge_attributes(nx_graph, edge_ts_list, 'ts_edge')
    return nx_graph


def nx_to_pyg(graph_nx: Graph, one_hot_encoder_elements: OneHotEncoder) -> Data:
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
        node_features[i, one_hot_encoder_elements[graph_nx.nodes[node]["elem"]]] = 1
    for i, edge in enumerate(graph_nx.edges):
        edge_index[0, i] = edge[0]
        edge_index[1, i] = edge[1]

    graph_pyg = Data(
        x=node_features, edge_index=edge_index, edge_attr=edge_features, y=node_index
    )
    return graph_pyg


def graph_plotter(
    graph: Data,
    node_size: int = 320,
    font_color: str = "white",
    font_weight: str = "bold",
    alpha: float = 1.0,
    arrowsize: int = 10,
    width: float = 1.2,
    dpi: int = 200,
    figsize: tuple[int, int] = (4, 4),
    node_index: bool = True,
    text: str = None,
):
    """
    Visualize graph with atom labels and colors. Working also for TSs.
    Kamada_kawai_layout engine gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    nx_graph = pyg_to_nx(graph)
    labels = get_node_attributes(nx_graph, "elem")
    colors = list(get_node_attributes(nx_graph, "rgb").values())
    edge_colors = [
        "black" if nx_graph.edges[edge]["ts_edge"] == 0 else "red"
        for edge in nx_graph.edges
    ]
    plt.figure(figsize=figsize, dpi=dpi)
    draw_networkx(
        nx_graph,
        labels=labels,
        node_size=node_size,
        font_color=font_color,
        font_weight=font_weight,
        node_color=colors,
        edge_color=edge_colors,
        alpha=alpha,
        arrowsize=arrowsize,
        width=width,
        pos=kamada_kawai_layout(nx_graph),
        linewidths=0.5,
    )
    if node_index:
        pos_dict = kamada_kawai_layout(nx_graph)
        for node in nx_graph.nodes:
            x, y = pos_dict[node]
            plt.text(x + 0.05, y + 0.05, node, fontsize=7)
    if text != None:
        plt.text(0.03, 0.9, text, fontsize=10)
    plt.axis("off")
    plt.draw()
