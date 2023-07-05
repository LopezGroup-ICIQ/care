# from GAMERXN.rnet.graphs.graph_utilities import edge_cutoffs, get_voronoi_neighbourlist
from GAMERNet.rnet.utilities import paths as pt
import networkx as nx
import ase
import numpy as np

# # ----- Graph generation from ASE atoms object ----- #

# # Graph generation containing the coordinates of the atoms
# # Coordinate data can be set off by setting the 'coord' parameter to False
# def ase_coord_2_graph(atoms: ase.Atoms, coords: bool) -> nx.Graph:
#     """Generates a NetworkX Graph from an ASE Atoms object.

#     Parameters
#     ----------
#     atoms : ase.Atoms
#         ASE Atoms object of the molecule.
#     coords : bool
#         Boolean indicating whether to include the atomic coordinates in the graph.

#     Returns
#     -------
#     nx.Graph
#         NetworkX Graph of the molecule (with atomic coordinates and bond lengths if 'coords' is True).
#     """

#     num_atom = list(range(len(atoms)))
#     elems_list = atoms.get_chemical_symbols()
#     xyz_coords = atoms.get_positions()

#     # Generating the graph
#     nx_graph = nx.Graph()
#     nx_graph.add_nodes_from(num_atom)

#     if coords:
#         node_attrs = {
#             num: {'elem': elems_list[i], 'xyz': xyz_coords[i]}
#                   for i, num in enumerate(num_atom)
#                   }
#     else:
#         node_attrs = {
#             num: {'elem': elems_list[i]}
#             for i, num in enumerate(num_atom)
#         }
#     nx.set_node_attributes(nx_graph, node_attrs)

#     # Adding the edges
#     edge_attrs = {}
#     for i in range(len(atoms)):
#         for j in range(i + 1, len(atoms)): 
#             cutoff = edge_cutoffs(atoms[i], atoms[j], tolerance=0.2)
#             bond_length = atoms.get_distance(i, j)
#             if bond_length < cutoff:
#                 edge_attrs[(i, j)] = {"length": bond_length}
    
#     edges = list(edge_attrs.keys())
#     nx_graph.add_edges_from(edges)
#     nx.set_edge_attributes(nx_graph, edge_attrs)

#     return nx_graph

# # Modified version of the original function provided by Santiago Morandi
# # DO NOT USE for systems with a single atom
# def ase_2_graph(atoms: ase.Atoms, voronoi_tolerance: float) -> nx.Graph:
#     """Generate a NetworkX Graph from an ASE Atoms object, representing the molecule through a connectivity annalysis.

#     Parameters
#     ----------
#     atoms : ase.Atoms
#         ASE Atoms object of the molecule.
#     voronoi_tolerance : float
#         Tolerance of the tessellation algorithm for edge creation.

#     Returns
#     -------
#     nx.Graph
#         NetworkX Graph representing the molecule.
#     """

#     # 1) Get connectivity list for the whole system
#     adsorbate_indexes = {atom.index for atom in atoms}
#     neighbour_list = get_voronoi_neighbourlist(atoms, voronoi_tolerance)
#     if len(neighbour_list) == 0:
#         return ase.Atoms()

#     # 2) Construct graph with the atoms in the ensemble
#     ensemble =  ase.Atoms(atoms[[*adsorbate_indexes]], pbc=atoms.pbc, cell=atoms.cell) #*metal_neighbours
#     nx_graph = nx.Graph()
#     nx_graph.add_nodes_from(range(len(ensemble)))
#     nx.set_node_attributes(nx_graph, {i: ensemble[i].symbol for i in range(len(ensemble))}, "elem")
#     ensemble_neighbour_list = get_voronoi_neighbourlist(ensemble, voronoi_tolerance)
#     ensemble_neighbour_list = np.concatenate((ensemble_neighbour_list, ensemble_neighbour_list[:, [1, 0]]))

#     nx_graph.add_edges_from(ensemble_neighbour_list)
#     return nx_graph
