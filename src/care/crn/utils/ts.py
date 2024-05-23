from copy import deepcopy

import networkx as nx
import numpy as np
from torch import where
from torch_geometric.data import Data

from care import ElementaryReaction
from care.gnn.graph_filters import extract_adsorbate
from care.gnn.graph_tools import pyg_to_nx


def ts_graph(step: ElementaryReaction) -> Data:
    """
    Given the bond-breaking reaction, detect the broken bond in the
    transition state and label the corresponding edge.
    Given A* + * -> B* + C* and the bond-breaking type X-Y, take the graph of A*,
    break all the potential X-Y bonds and perform isomorphism with B* + C*.
    When isomorphic, the broken edge is labelled.

    Args:
        graph (Data): adsorption graph of the intermediate which is fragmented in the reaction.
        reaction (ElementaryReaction): Bond-breaking reaction.

    Returns:
        Data: graph with the broken bond labeled.
    """

    if "-" not in step.r_type:
        raise ValueError("Input reaction must be a bond-breaking reaction.")
    bond = tuple(step.r_type.split("-"))

    # Select intermediate that is fragmented in the reaction (A*)
    inters_dict = {
        inter.code: inter
        for inter in list(step.reactants) + list(step.products)
        if not inter.is_surface
    }
    inters = {
        inter.code: inter.graph.number_of_edges()
        for inter in list(step.reactants) + list(step.products)
        if not inter.is_surface
    }
    inter_code = max(inters, key=inters.get)
    idx = min(
        inters_dict[inter_code].ads_configs,
        key=lambda x: inters_dict[inter_code].ads_configs[x]["mu"],
    )
    ts_graph = deepcopy(inters_dict[inter_code].ads_configs[idx]["pyg"])
    competitors = [
        inter
        for inter in list(step.reactants) + list(step.products)
        if not inter.is_surface and inter.code != inter_code
    ]

    # Build the nx graph of the competitors (B* + C*)
    if len(competitors) == 1:
        if abs(step.stoic[competitors[0].code]) == 2:  # A* -> 2B*
            nx_bc = [competitors[0].graph, competitors[0].graph]
            mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
            nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
            nx_bc = nx.compose(nx_bc[0], nx_bc[1])
        elif abs(step.stoic[competitors[0].code]) == 1:  # A* -> B* (ring opening)
            nx_bc = competitors[0].graph
        else:
            raise ValueError("Reaction stoichiometry not supported.")
    else:  # asymmetric fragmentation
        nx_bc = [competitors[0].graph, competitors[1].graph]
        mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
        nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
        nx_bc = nx.compose(nx_bc[0], nx_bc[1])

    # Lool for potential edges to break
    atom_symbol = lambda idx: ts_graph.node_feats[where(ts_graph.x[idx] == 1)[0].item()]
    potential_edges = []
    for i in range(ts_graph.edge_index.shape[1]):
        edge_idxs = ts_graph.edge_index[:, i]
        atom1, atom2 = atom_symbol(edge_idxs[0]), atom_symbol(edge_idxs[1])
        if (atom1, atom2) == bond or (atom2, atom1) == bond:
            potential_edges.append(i)
    counter = 0

    # Find correct one via isomorphic comparison
    while True:
        data = deepcopy(ts_graph)
        u, v = data.edge_index[:, potential_edges[counter]]
        mask = ~(
            (data.edge_index[0] == u) & (data.edge_index[1] == v)
            | (data.edge_index[0] == v) & (data.edge_index[1] == u)
        )
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]
        adsorbate = extract_adsorbate(data, ["C", "H", "O", "N", "S"])
        nx_graph = pyg_to_nx(adsorbate, data.node_feats)
        if nx.is_isomorphic(
            nx_bc, nx_graph, node_match=lambda x, y: x["elem"] == y["elem"]
        ):
            ts_graph.edge_attr[potential_edges[counter]] = 1
            idx = np.where(
                (ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u)
            )[0].item()
            ts_graph.edge_attr[idx] = 1
            break
        else:
            counter += 1
    step.ts_graph = ts_graph
