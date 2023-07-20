from ase import Atoms
import networkx as nx
from pubchempy import get_compounds, Compound
import copy

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

def ase_coord_2_graph(atoms: Atoms, coords: bool) -> nx.Graph:
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
    nx_graph = nx.Graph()
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

def add_O_2_molec(molecule_graph: nx.Graph) -> nx.Graph:
    molec_graph_copy = copy.deepcopy(molecule_graph)
    molec_graph_copy.neighbors(0)
    # Detecting the carbon atoms
    carbon_nodes = []
    for node in molec_graph_copy.nodes():
        if molec_graph_copy.nodes[node]['elem'] == 'C':
            neighb = molec_graph_copy.neighbors(node)
            oxy_flag = False
            for n in neighb:
                if molec_graph_copy.nodes[n]['elem'] == 'O':
                    oxy_flag = True
                    break
            carbon_nodes.append((node, oxy_flag))

    # Checking if the carbon atoms are connected to a hydrogen atom
    for node, oxy_flg in carbon_nodes:
        if oxy_flg == True:
            continue
        H_flag = False
        for neighbor in molec_graph_copy.neighbors(node):
            if molec_graph_copy.nodes[neighbor]['elem'] == 'H':
                while H_flag == False:
                    # Replacing the hydrogen atom with an oxygen atom
                    molec_graph_copy.nodes[neighbor]['elem'] = 'O'
                    H_flag = True
    
    return molec_graph_copy

def molecular_formula_from_graph(graph: nx.Graph) -> str:
    """Generates the molecular formula of a molecule from its graph representation.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX Graph representing the molecule.

    Returns
    -------
    str
        Molecular formula.
    """

    num_C, num_H, num_O, num_N = 0, 0, 0, 0
    for node in graph.nodes(data=True):
        elem_node = node[1]['elem']
        if elem_node == 'C':
            num_C += 1
        elif elem_node == 'H':
            num_H += 1
        elif elem_node == 'O':
            num_O += 1
        elif elem_node == 'N':
            num_N += 1

    # Generating the formula
    formula = ""
    if num_C > 0:
        formula += "C" + (str(num_C) if num_C > 1 else "")
    if num_H > 0:
        formula += "H" + (str(num_H) if num_H > 1 else "")
    if num_O > 0:
        formula += "O" + (str(num_O) if num_O > 1 else "")
    if num_N > 0:
        formula += "N" + (str(num_N) if num_N > 1 else "")
    return str(formula)

def compare_strucures_from_pubchem(molecular_formula: str, saturated_molecule_graph: nx.Graph) -> tuple[nx.Graph, str, Atoms]:
    """Compares the molecular formula and graph of a molecule with the PubChem database to obtain the
    molecular formula, ASE Atoms object, and NetworkX Graph of the saturated molecule.

    Parameters
    ----------
    molecular_formula : str
        Molecular formula of the molecule.
    saturated_molecule_graph : nx.Graph
        NetworkX Graph of the saturated molecule.

    Returns
    -------
    tuple(nx.Graph, str, Atoms)
        Tuple containing the NetworkX Graph, molecular formula, ASE Atoms object of the saturated molecule.
    """
    
    pubchem_compounds = get_compounds(molecular_formula, 'formula', record_type='3d', listkey_count=20)
    for compound in pubchem_compounds:
        pubchem_cid = compound.cid
        c = Compound.from_cid(pubchem_cid)
        compound_formula = c.molecular_formula
        
        compound_atoms = [atom.element for atom in compound.atoms]
        compound_coords = [(atom.x, atom.y, atom.z) for atom in compound.atoms]

        compound_ase_obj = Atoms(compound_atoms, positions=compound_coords, pbc=True)
        compound_graph = ase_coord_2_graph(compound_ase_obj, coords=True)

        if nx.is_isomorphic(saturated_molecule_graph, compound_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
            compound_name = c.iupac_name
            if compound_name == 'oxidane':
                compound_name = 'water'
            return  compound_graph, compound_formula, compound_ase_obj, compound_name



def sat_H_graph(molecule_nx_graph: nx.Graph) -> tuple[nx.Graph, bool]:
    """For the unsaturated molecules, it adds the missing H atoms to the nx.Graphs.

    Parameters
    ----------
    molecule_nx_graph : nx.Graph
        NetworkX Graph representing the molecule.

    Returns
    -------
    nx.Graph
        NetworkX Graph representing the molecule with the added H atoms.

    Examples
    --------
    C -> CH4
    CH -> CH4
    C2H2 -> C2H6
    """

    n_nodes = molecule_nx_graph.number_of_nodes()
    node_conn = nx.degree(molecule_nx_graph)
    node_attrs = nx.nodes(molecule_nx_graph)

    max_conns = {'C': 4, 'O': 2, 'N': 3, 'H': 1}
    copy_graph = molecule_nx_graph.copy()
    
    unsat_flag = False
    for idx_node, conn in node_conn:
        elem_node = dict(node_attrs)[idx_node]['elem']
        for element, max_conn in max_conns.items():
            if elem_node == element:
                while conn < max_conn:
                    unsat_flag = True
                    idx_updt = idx_node
                    copy_graph.add_node(n_nodes, elem='H')
                    copy_graph.add_edge(idx_updt, n_nodes)
                    n_nodes += 1
                    conn += 1

    return copy_graph, unsat_flag

# TODO: Redefine this function
def add_H_nodes(molecule_nx_graph: nx.Graph, name_molecule: str) -> tuple[nx.Graph, str, Atoms, str]:
    """For the unsaturated molecules, it adds the missing H atoms to the nx.Graphs.
    This is done to ensure a label consistency between the saturated and unsaturated molecules.
    Parameters
    ----------
    molecule_nx_graph : nx.Graph
        NetworkX Graph representing the molecule.
    Returns
    -------
    nx.Graph
        NetworkX Graph representing the molecule with the added H atoms.
    """
    copy_graph, unsat_flag = sat_H_graph(molecule_nx_graph)
    updated_molecular_formula, molecule_nx_graph, updated_ase_obj = None, None, None, 

    if unsat_flag:
        molecular_formula = molecular_formula_from_graph(copy_graph)
        molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule = compare_strucures_from_pubchem(molecular_formula, copy_graph)
    
    return molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule

def find_new_struct(molecule_graph: nx.Graph):
    molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule = None, None, None, None
    molecular_formula = molecular_formula_from_graph(molecule_graph)
    molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule = compare_strucures_from_pubchem(molecular_formula, molecule_graph)
    return molecule_nx_graph, updated_molecular_formula, updated_ase_obj, name_molecule