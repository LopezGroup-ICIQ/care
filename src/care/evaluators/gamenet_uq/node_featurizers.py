"""
Node featurization for adsorption graphs.
"""

import torch
from ase import Atoms
from torch_geometric.data import Data

from care.evaluators.gamenet_uq.graph import get_voronoi_neighbourlist


def get_gcn(
    graph: Data,
    atoms: Atoms,
    adsorbate_elements: list[str],
) -> Data:
    """
    Return the (normalized) generalized coordination number (gcn) for each surface atom in the ASE Atoms object.
    gcn is defined as the sum of the coordination numbers (cn) of the neighbours divided by the maximum coordination number.
    gcn=0 atom alone; gcn=1 bulk atom; 0<gcn<1=surface atom.

    graph must have as attributes:
    - atoms (Atoms): ASE atoms object containing a slab with an adsorbate
    - node_feats (list[str]): list of node features to be used in the graph

    Args:
        atoms (Atoms): ASE atoms object containing a slab with an adsorbate
        adsorbate_elements (list[str]): list of symbols of the adsorbate elements

    Returns:
        Data: PyG Data object with the gcn as a node feature. Data.x.shape[1] increases by 1.
                Data.node_feats is also updated.
    """
    if all(graph.elem[i] in adsorbate_elements for i in range(graph.num_nodes)):
        graph.x = torch.cat((graph.x, torch.zeros((graph.x.shape[0], 1))), dim=1)
    else:
        y = get_voronoi_neighbourlist(
            atoms, 0.5, 1.0, adsorbate_elements
        )  # focus only on catalyst atoms
        neighbour_dict = {}
        for idx, atom in enumerate(atoms):
            cn = 0
            neighbour_list = []
            for row in y:
                if idx in row:
                    neighbour_index = row[0] if row[0] != idx else row[1]
                    if atoms[neighbour_index].symbol not in adsorbate_elements:
                        cn += 1
                        neighbour_list.append(
                            (
                                atoms[neighbour_index].symbol,
                                neighbour_index,
                                atoms[neighbour_index].position[2],
                            )
                        )
                else:
                    continue
            neighbour_dict[idx] = (cn, atom.symbol, neighbour_list)
        max_cn = max([neighbour_dict[i][0] for i in neighbour_dict.keys()])
        gcn_dict = {}
        for idx in neighbour_dict.keys():
            if atoms[idx].symbol in adsorbate_elements:
                gcn_dict[idx] = (None, neighbour_dict[idx][0])
                continue
            cn_sum = 0.0
            for neighbour in neighbour_dict[idx][2]:
                cn_sum += neighbour_dict[neighbour[1]][0]
            gcn_dict[idx] = (cn_sum / max_cn**2, neighbour_dict[idx][0])

        gcn = torch.zeros((graph.x.shape[0], 1))
        for i, _ in enumerate(graph.x):
            if graph.elem[i] not in adsorbate_elements:
                gcn[i] = gcn_dict[graph.idx[i]][0]
        graph.x = torch.cat((graph.x, gcn), dim=1)
    return graph
