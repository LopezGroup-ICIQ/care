import networkx as nx
from ase import Atoms
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from care.constants import CORDERO

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def edge_cutoffs(
    node_i: nx.Graph.nodes, node_j: nx.Graph.nodes, tolerance: float
) -> float:
    """
    Get the cutoff distance for two atoms to be considered connected using Cordero's atomic radii.

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


def atoms_to_graph(atoms: Atoms, coords: bool = False) -> nx.Graph:
    """
    Generates a NetworkX Graph from an ASE Atoms object.

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
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(num_atom)

    if coords:
        node_attrs = {
            num: {"elem": elems_list[i], "xyz": xyz_coords[i]}
            for i, num in enumerate(num_atom)
        }
    else:
        node_attrs = {num: {"elem": elems_list[i]} for i, num in enumerate(num_atom)}
    nx.set_node_attributes(nx_graph, node_attrs)
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


def rdkit_to_graph(mol: Chem.Mol) -> nx.Graph:
    """
    Generates a NetworkX Graph from an RDKit molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.

    Returns
    -------
    nx.Graph
        NetworkX Graph of the molecule (with atomic coordinates and bond lengths).
    """

    # Adding Hs to the molecule
    mol = Chem.AddHs(mol)
    # Generate 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    nx_graph = nx.Graph()

    for i, atom in enumerate(mol.GetAtoms()):
        nx_graph.add_node(atom.GetIdx(), elem=atom.GetSymbol(), xyz=positions[i])

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = positions[i].Distance(positions[j])
        nx_graph.add_edge(i, j, length=length)

    return nx_graph
