from collections import namedtuple
import networkx as nx
from itertools import combinations, product
import networkx.algorithms.isomorphism as iso
from ase import Atoms
import numpy as np
from scipy.spatial import Voronoi
from copy import deepcopy

CORDERO = {"Ac": 2.15, "Al": 1.21, "Am": 1.80, "Sb": 1.39, "Ar": 1.06,
           "As": 1.19, "At": 1.50, "Ba": 2.15, "Be": 0.96, "Bi": 1.48,
           "B" : 0.84, "Br": 1.20, "Cd": 1.44, "Ca": 1.76, "C" : 0.76,
           "Ce": 2.04, "Cs": 2.44, "Cl": 1.02, "Cr": 1.39, "Co": 1.50,
           "Cu": 1.32, "Cm": 1.69, "Dy": 1.92, "Er": 1.89, "Eu": 1.98,
           "F" : 0.57, "Fr": 2.60, "Gd": 1.96, "Ga": 1.22, "Ge": 1.20,
           "Au": 1.36, "Hf": 1.75, "He": 0.28, "Ho": 1.92, "H" : 0.31,
           "In": 1.42, "I" : 1.39, "Ir": 1.41, "Fe": 1.52, "Kr": 1.16,
           "La": 2.07, "Pb": 1.46, "Li": 1.28, "Lu": 1.87, "Mg": 1.41,
           "Mn": 1.61, "Hg": 1.32, "Mo": 1.54, "Ne": 0.58, "Np": 1.90,
           "Ni": 1.24, "Nb": 1.64, "N" : 0.71, "Os": 1.44, "O" : 0.66,
           "Pd": 1.39, "P" : 1.07, "Pt": 1.36, "Pu": 1.87, "Po": 1.40,
           "K" : 2.03, "Pr": 2.03, "Pm": 1.99, "Pa": 2.00, "Ra": 2.21,
           "Rn": 1.50, "Re": 1.51, "Rh": 1.42, "Rb": 2.20, "Ru": 1.46,
           "Sm": 1.98, "Sc": 1.70, "Se": 1.20, "Si": 1.11, "Ag": 1.45,
           "Na": 1.66, "Sr": 1.95, "S" : 1.05, "Ta": 1.70, "Tc": 1.47,
           "Te": 1.38, "Tb": 1.94, "Tl": 1.45, "Th": 2.06, "Tm": 1.90,
           "Sn": 1.39, "Ti": 1.60, "Wf": 1.62, "U" : 1.96, "V" : 1.53,
           "Xe": 1.40, "Yb": 1.87, "Y" : 1.90, "Zn": 1.22, "Zr": 1.75}


MolPack = namedtuple('MolPack',['code', 'mol','graph', 'subs'])

def get_voronoi_neighbourlist(atoms: Atoms,
                              tolerance: float,
                              scaling_factor: float,
                              molecule_elements: list[str]) -> np.ndarray:
    """
    Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    Assumption: The catalyst surface does not contain elements present in the adsorbate.
    Args:
        atoms (Atoms): ASE Atoms object representing the adsorbate-metal system.
        tolerance (float): tolerance for the distance between two atoms to be considered connected.
        scaling_factor (float): scaling factor for the covalent radii of the metal atoms.
    Returns:
        np.ndarray: connectivity list of the system. Each row represents a pair of connected atoms.
    """

    if len(atoms) == 1:
        return np.array([])
    # First condition for two atoms to be connected: They must share a Voronoi facet
    coords_arr = np.repeat(np.expand_dims(np.copy(atoms.get_scaled_positions()), axis=0), 27, axis=0)
    mirrors = np.repeat(np.expand_dims(np.asarray(list(product([-1, 0, 1], repeat=3))), 1), coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(coords_arr + mirrors, (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]))
    # corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    pairs_corr = np.delete(pairs_corr, np.argwhere(pairs_corr[:, 0] == pairs_corr[:, 1]), axis=0)
    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their covalent radii plus a tolerance.
    # NB surface atoms covalent radii are scaled by a factor corresponding to scaling_factor.
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(pair[0], pair[1], mic=True)
        threshold = CORDERO[atoms[pair[0]].symbol] + CORDERO[atoms[pair[1]].symbol] + tolerance + \
                    (scaling_factor - 1.0) * ((atoms[pair[0]].symbol not in molecule_elements) * CORDERO[atoms[pair[0]].symbol] + \
                                              (atoms[pair[1]].symbol not in molecule_elements) * CORDERO[atoms[pair[1]].symbol])
        if distance <= threshold:
            pairs_lst.append(pair)
    return np.sort(np.array(pairs_lst), axis=1)

def generate_pack(ase_molecule_obj: Atoms, niter: int, group: int) -> dict[int, list[MolPack]]:
    ase_molecule_obj.arrays["conn_pairs"] = get_voronoi_neighbourlist(ase_molecule_obj, 0.25, 1.0, ['C', 'H', 'O']) 
    return generate_range(ase_molecule_obj, 'H', niter, group)

def edge_cutoffs(node_i: nx.Graph.nodes, node_j: nx.Graph.nodes, tolerance: float) -> float:
    """Get the cutoff distance for two atoms to be considered connected using Cordero's atomic radii.

    Parameters
    ----------
    node_i : nx.Graph.nodes
        Node i.
    node_j : nx.Graph.nodes
        Node j.
    tolerance : float
        Tolerance for the cutoff distance.

    Returns
    -------
    float
        Cutoff distance.
    """

    element_i = node_i.symbol
    element_j = node_j.symbol
    return CORDERO[element_i] + CORDERO[element_j] + tolerance

def digraph(atoms: Atoms, coords: bool) -> nx.DiGraph:
    """Generates a NetworkX Graph from an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object of the molecule.
    coords : bool
        Boolean indicating whether to include the atomic coordinates in the graph.

    Returns
    -------
    nx.Graph
        NetworkX Graph of the molecule (with atomic coordinates and bond lengths if 'coords' is True).
    """

    num_atom = list(range(len(atoms)))
    elems_list = atoms.get_chemical_symbols()
    xyz_coords = atoms.get_positions()

    # Generating the graph
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(num_atom)

    if coords:
        node_attrs = {
            num: {'elem': elems_list[i], 'xyz': xyz_coords[i]}
                  for i, num in enumerate(num_atom)
                  }
    else:
        node_attrs = {
            num: {'elem': elems_list[i]}
            for i, num in enumerate(num_atom)
        }
    nx.set_node_attributes(nx_graph, node_attrs)

    # Adding the edges
    edge_attrs = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)): 
            cutoff = edge_cutoffs(atoms[i], atoms[j], tolerance=0.2)
            bond_length = atoms.get_distance(i, j)
            if bond_length < cutoff:
                edge_attrs[(i, j)] = {"length": bond_length}
    
    edges = list(edge_attrs.keys())
    nx_graph.add_edges_from(edges)
    nx.set_edge_attributes(nx_graph, edge_attrs)

    return nx_graph

def code_name(molecule: Atoms, group: int, index: int) -> str: #name of the molecule
    name_elem = ('C', 'H', 'O')
    name_str = ''
    nC, nH, nO = 0, 0, 0
    for atom in molecule:
        if atom.symbol == 'C':
            nC += 1
        elif atom.symbol == 'H':
            nH += 1
        elif atom.symbol == 'O':
            nO += 1
    infor = {"C": nC, "H": nH, "O": nO}
    for item in name_elem:
        try:
            name_str += '{:01x}'.format(infor[item])
        except KeyError:
            name_str += str('0')
    name_str += '{:01x}'.format(group)
    name_str += '{:02x}'.format(index)
    return name_str

def decode(code: str) -> dict[str, int]: #storing of data?
    decoded = {'C': int(code[0], 16),
               'H': int(code[1], 16),
               'O': int(code[2], 16),
               'grp': int(code[3], 16),
               'iso': int(code[4:6], 16)}
    return decoded

def search_code(mg_pack: dict[int, list[MolPack]], code: str):
    for _, pack in mg_pack.items():
         for obj in pack:
            if str(obj.code) == str(code):
                return obj
    else:
        return False

def generate_map(packing: list[MolPack], elem: str):
    """Generate the map of the packing.

    Parameters
    ----------
    packing : list[MolPack]
        _description_
    elem : str
        chemical element. One between 'C', 'H', 'O'.

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph of the packing.
    """
    g = nx.DiGraph()
    em = iso.categorical_node_match('elem', elem)
    for index in range(len(packing) - 1):
        for father, son in product(packing[index], packing[index+1]):
            xx = nx.isomorphism.DiGraphMatcher(father.graph, son.graph, node_match=em)
            if xx.subgraph_is_isomorphic():
                g.add_edge(father.code, son.code)
    return g
    

def draw_dot(graph, filename):
    p=nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(filename)

# def print_pack(pack, surface, poscar='poscar', distance=None, atom=None, at_lst=None, point=None, origin=None, vector=None):
#     for _, item in pack.items():
#         for obj in item:
#             caty = CatalyticSystem('x')
#             caty.surface_set(surface)
#             caty.molecule_add(obj.mol)
#             if distance is not None:
#                 if origin is None:
#                     lowest = obj.mol.atom_obtain_lowest().coords
#                 else:
#                     lowest = origin 
#                 if atom is not None:
#                     caty.move_over_atom(obj.mol, atom, distance, origin=lowest)
#                 elif at_lst is not None:
#                     caty.move_over_multiple_atoms(obj.mol, at_lst, distance, origin=lowest)
#                 elif point is not None:
#                     obj.mol.move_to(point, origin='centroid')
#                     obj.mol.move_vector([0., 0., distance])
#                 else:
#                     caty.move_over_surface_center(obj.mol, distance=distance, origin=lowest)
#             if vector is not None:
#                 caty.molecules[0].move_vector(vector)  
                        
#             vaspio.print_vasp(caty, './{}/POSCAR.{}'.format(poscar, obj.code)) # storing the poscars

# def save_poscars(pack,folder): #write a function to save gas phase POSCARs
#     for _, item in pack.items():
#         for obj in item:
#             geomio.mol_to_file(obj.mol,'./{}/POSCAR_{}'.format(folder,obj.code),'contcar')


def generate_range(ase_molecule: Atoms, 
                   element: str, 
                   subs: int, 
                   group: int) -> dict[int, list]: #generating all the possibilities

    mg_pack = {0: [MolPack(code_name(ase_molecule, group, 1), ase_molecule, digraph(ase_molecule, coords=False), {})]}

    for index in range(subs):
        
        pack = get_all_subs(ase_molecule, index + 1, element)
        unique = get_unique(pack[1], element)
        
        tmp_pack = []
        for i_code, struct in enumerate(unique):
            mol_sel = pack[0][struct]
            tmp_pack.append(MolPack(code_name(mol_sel, group, i_code+1), mol_sel, pack[1][struct], {'H': index+1}))

        mg_pack[index+1] = tmp_pack

    return mg_pack



def get_all_subs(ase_molecule_obj: Atoms, 
                 n_sub: int, 
                 element: str) -> tuple[list[Atoms], list[nx.DiGraph]]:
    """function to what point you want to remove hydrogens
    """
    
    sel_atoms = [atom.index for atom in ase_molecule_obj if atom.symbol == element]
    mol_pack, graph_pack = [], []

    for comb in combinations(sel_atoms, n_sub):

        new_mol = deepcopy(ase_molecule_obj)
        del new_mol.arrays["conn_pairs"]

        int_list = []
        for index in comb:
            int_list.append(index)
        # Deleting the atoms which index is in int_list

        del new_mol[int_list]
        new_mol.arrays["conn_pairs"] = get_voronoi_neighbourlist(new_mol, 0.25, 1.0, ['C', 'H', 'O'])
        mol_pack.append(new_mol)
        graph_pack.append(digraph(new_mol, coords=False))
    return mol_pack, graph_pack

def get_unique(graph_pack: list[nx.DiGraph], element: str) -> list[int]: #function to get unqiue configs
    
    accepted = []
    em = iso.categorical_node_match('elem', element)
    for index, graph1 in enumerate(graph_pack):
        for graph2 in accepted:
            if nx.is_isomorphic(graph1.to_undirected(), graph_pack[graph2].to_undirected(), node_match=lambda x, y: x["elem"] == y["elem"]):
                break
        else:
            accepted.append(index)
    return accepted

