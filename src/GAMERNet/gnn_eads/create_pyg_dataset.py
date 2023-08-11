""" Module containing the class for the generation of the PyG dataset from the ASE database."""

import os
from typing import Union

from torch_geometric.data import InMemoryDataset, Data
from torch import zeros, where, cat, load, save, tensor
import torch
from ase.db import connect
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from ase.atoms import Atoms
from ase.io import read
import networkx as nx
import pandas as pd
# from ase.data import covalent_radii as CR

from GAMERNet.gnn_eads.graph_filters import adsorption_filter, H_connectivity_filter, C_connectivity_filter, single_fragment_filter
from GAMERNet.gnn_eads.graph_tools import extract_adsorbate
from GAMERNet.gnn_eads.functions import atoms_to_pyggraph, get_voronoi_neighbourlist


def pyg_dataset_id(ase_database_path: str, 
                   graph_params: dict) -> str:
    """
    Provide dataset string identifier based on the provided graph parameters.
    
    Args:
        ase_database_path (str): Path to the ASE database containing the adsorption data.
        graph_params (dict): Dictionary containing the information for the graph generation 
                             in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "second_order_nn": int},
                             "features": {"encoder": OneHotEncoder, "adsorbate": bool, "ring": bool, "aromatic": bool, "radical": bool, "valence": bool, "facet": bool}}
    Returns:
        dataset_id (str): PyG dataset identifier.
    """
    # name of the ASE database (*.db file)
    id = ase_database_path.split("/")[-1].split(".")[0]
    # extract graph structure parameters
    structure_params = graph_params["structure"]
    tolerance = str(structure_params["tolerance"]).replace(".", "")
    scaling_factor = str(structure_params["scaling_factor"]).replace(".", "")
    metal_hops = str(structure_params["second_order_nn"])
    # extract node features parameters
    features_params = graph_params["features"]
    adsorbate = str(features_params["adsorbate"])
    ring = str(features_params["ring"])
    aromatic = str(features_params["aromatic"])
    radical = str(features_params["radical"])
    valence = str(features_params["valence"])
    facet = str(features_params["facet"])
    cn = str(features_params["gcn"])
    mag = str(features_params["magnetization"])
    target = graph_params["target"]
    # id convention: database name + target + all features. float values converted to strings and "." is removed
    dataset_id = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(id, target, tolerance, scaling_factor, metal_hops, adsorbate, ring, aromatic, radical, valence, cn, facet, mag)
    return dataset_id


def get_gcn(atoms_obj: Atoms, 
            adsorbate_elements: list[str]) -> dict[int, float]:
    """
    Given an ASE atoms object containing a slab with an adsorbate,
    returns a dictionary with the (normalized) generalized coordination number (gcn) for each atom.
    gcn is defined as the sum of the coordination numbers of the neighbours divided by the maximum coordination number.

    Args:
        atoms_obj (Atoms): ASE atoms object containing a slab with an adsorbate
        adsorbate_elements (list[str]): list of symbols of the adsorbate elements

    Returns:
        dict[int, tuple]: dictionary with the generalized coordination number (gcn) and traditional cn for each atom
    """

    y = get_voronoi_neighbourlist(atoms_obj, 0.25, 1.2, adsorbate_elements)  # keep these parameters fixed
    neighbour_dict = {}
    for atom_index, atom in enumerate(atoms_obj):
        coordination_number = 0
        neighbour_list = []
        for row in y:
            if atom_index in row:
                neighbour_index = row[0] if row[0] != atom_index else row[1]
                if atoms_obj[neighbour_index].symbol in adsorbate_elements:
                    coordination_number += 0  # Consider slab only (no adsorbate)
                else:
                    coordination_number += 1
                    neighbour_list.append((atoms_obj[neighbour_index].symbol, neighbour_index, atoms_obj[neighbour_index].position[2])) 
            else:
                continue
        neighbour_dict[atom_index] = (coordination_number, atom.symbol, neighbour_list)
    max_coordination_number = max([neighbour_dict[i][0] for i in neighbour_dict.keys()])
    gcn_dict = {}
    for atom_index in neighbour_dict.keys():
        if atoms_obj[atom_index].symbol in adsorbate_elements:
            gcn_dict[atom_index] = (None, neighbour_dict[atom_index][0])
            continue
        cn_sum = 0.0
        for neighbour in neighbour_dict[atom_index][2]:
            cn_sum += neighbour_dict[neighbour[1]][0]
        gcn_dict[atom_index] = (cn_sum / max_coordination_number ** 2, neighbour_dict[atom_index][0])
    return gcn_dict


def get_radical_atoms(atoms_obj: Atoms, 
                      adsorbate_elements: list[str]) -> list[int]:
    """
    Detect atoms in the molecule which are radicalswith RDKit.

    Args:
        atoms_obj (ase.Atoms): ASE atoms object of the adsorption structure
        adsorbate_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])
    
    Returns:
        radical_atoms (list[int]): List of indices of the radical atoms in the atoms_obj.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(conn_mol, Chem.SANITIZE_FINDRADICALS  ^ Chem.SANITIZE_SETHYBRIDIZATION)
    # num_radical_electrons = [atom.GetNumRadicalElectrons() for atom in conn_mol.GetAtoms()]
    radical_atoms = [atom.GetIdx() for atom in conn_mol.GetAtoms() if atom.GetNumRadicalElectrons() > 0]
    return radical_atoms


def get_atom_valence(atoms_obj: Atoms,
                     adsorbate_elements: list[str]) -> list[float]:
    """
    For each atom in the adsorbate, calculate the valence.
    Valence is defined as (x_max - x) / x_max, where x is the degree of the atom,
    and x_max is the maximum degree of the atom in the molecule. Bond order is not taken into account.

    Ref: https://doi.org/10.1103/PhysRevLett.99.016105

    Args:
        atoms_obj (ase.Atoms): ASE atoms object.
        molecule_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])
    Returns:
        valence (list[float]): List of valences for each atom in the molecule.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    Chem.SanitizeMol(conn_mol, Chem.SANITIZE_FINDRADICALS  ^ Chem.SANITIZE_SETHYBRIDIZATION)
    degree_vector = np.vectorize(lambda x: x.GetDegree())
    max_degree_vector = np.vectorize(lambda x: Chem.GetPeriodicTable().GetDefaultValence(x.GetAtomicNum()))
    atom_array = np.array([i for i in conn_mol.GetAtoms()]).reshape(-1, 1)
    degree_array = (max_degree_vector(atom_array) - degree_vector(atom_array)) / max_degree_vector(atom_array)
    valence = degree_array.reshape(-1, 1)
    return valence


def detect_ring_nodes(data: Data) -> set:
    """
    Return indices of the nodes in the PyG Data object that are part of a ring.
    To do so, the graph is converted to a networkx graph and the cycle basis is computed.

    Args:
        data (Data): PyG Data object.
    
    Returns:
        ring_nodes (set): Set of indices of the nodes that are part of a ring.    
    """
    edge_index = data.edge_index.cpu().numpy()
    edges = set()
    for i in range(edge_index.shape[1]):
        edge = tuple(sorted([edge_index[0, i], edge_index[1, i]]))
        edges.add(edge)
    nx_graph = nx.Graph()
    for edge in edges:
        nx_graph.add_edge(edge[0], edge[1])
    cycles = list(nx.cycle_basis(nx_graph))
    ring_nodes = set(node for cycle in cycles for node in cycle)
    return ring_nodes


def get_aromatic_atoms(atoms_obj: Atoms, 
                       adsorbate_elements: list[str]) -> list[int]:
    """
    Get indices of aromatic atoms in an ase atoms object with RDKit.

    Args:
        atoms_obj (ase.Atoms): ASE atoms object of the adsorption structure
        adsorbate_elements (list[str]): List of elements in the adsorbates (e.g. ["C", "H", "O", "N", "S"])

    Returns:
        aromatic_atoms (list[int]): List of indices of the aromatic atoms in the atoms_obj.
    """
    molecule_atoms_obj = Atoms()
    molecule_atoms_obj.set_cell(atoms_obj.get_cell())
    molecule_atoms_obj.set_pbc(atoms_obj.get_pbc())
    for atom in atoms_obj:
        if atom.symbol in adsorbate_elements:
            molecule_atoms_obj.append(atom)
    atomic_symbols = molecule_atoms_obj.get_chemical_symbols()
    coordinates = molecule_atoms_obj.get_positions()
    xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(atomic_symbols, coordinates))
    xyz = "{}\n\n{}".format(len(molecule_atoms_obj), xyz)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    conn_mol = Chem.Mol(rdkit_mol)
    rdDetermineBonds.DetermineBonds(conn_mol)
    aromatic_atoms = [atom.GetIdx() for atom in conn_mol.GetAtoms() if atom.GetIsAromatic()]
    return aromatic_atoms


def isomorphism_test(graph: Data, 
                     graph_list: list[Data], 
                     eps: float=0.01) -> bool:
    """
    Perform isomorphism test for the input graph before including it in the final dataset.
    Test based on graph formula and energy difference.

    Args:
        graph (Data): Input graph.
        graph_list (list[Data]): graph list against which the input graph is tested.
        eps (float): tolerance value for the energy difference in eV. Default to 0.01 eV.
        grwph: data graph as input
    Returns:
        (bool): Whether the graph passed the isomorphism test.
    """
    if len(graph_list) == 0:
        return True
    formula = graph.formula  # formula provided by ase 
    family = graph.family
    facet = graph.facet
    energy = graph.y
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    for rival_graph in graph_list:
        c1 = num_edges == rival_graph.num_edges
        c2 = num_nodes == rival_graph.num_nodes
        c3 = formula == rival_graph.formula
        c4 = np.abs(energy - rival_graph.y) < eps
        c5 = family == rival_graph.family
        c6 = facet == rival_graph.facet
        if c1 and c2 and c3 and c4 and c5 and c6:
            return False
        else:
            continue
    return True


class AdsorptionGraphDataset(InMemoryDataset):
    """
    Generate graph dataset representing molecules adsorbed on transition metal surfaces.
    It generates the graphs from the provided ASE database and conversion settings.
    Graphs are stored in the torch_geometric.data.Data type.
    When the dataset object is instantiated for the first time, two different files are created:
    1) a `processed` directory containing the additional information about the dataset
    2) a zip file containing the graphs in the torch_geometric.data.Data format. The name of the zip file is
        dependent on the conversion settings.

    Args:
        ase_database_name (str): Path to the ase database containing the adsorption data.
        graph_dataset_dir (str): Path to the directory where the graph dataset files are stored.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"structure": {"tolerance": float, "scaling_factor": float, "metal_hops": int},
                             "features": {"adsorbate": bool,
                                          "ring": bool,
                                           "aromatic": bool, 
                                           "radical": bool, 
                                           "valence": bool, 
                                           "facet": bool, 
                                           "gcn": bool}, 
                             "target": str}
        database_key (str): Key to access specific items of the ase database. Default to "calc_type=adsorption".
        
    Notes:
        - "target" in graph_params must be a key of the ASE database.
        - Each graph object has two labels: graph.y and graph.target. Originally they are the same, 
          but during the trainings graph.target represents the 
          original value (adsorption energy in eV), while graph.y is the scaled value (
          unitless scaled adsorption energy).

    Example:
        Generate graph dataset containing only adsorption systems on Pt(111) surface, 
        with adsorbate, ring, aromatic, radical and facet features, and e_ads_dft as target.
        >>> graph_params = {"structure": {"tolerance": 0.5, "scaling_factor": 1.5, "metal_hops": False},
                            "features": {"adsorbate": True, "ring": True, "aromatic": True, "radical": True, "valence": False, "facet": True},
                            "target": "e_ads_dft"}
        >>> ase_database_path = "path/to/ase/database"
        >>> graph_dataset_dir = "path/to/graph/dataset"
        >>> dataset = AdsorptionGraphDataset(ase_database_path, graph_dataset_dir, graph_params, "calc_type=adsorption,facet=fcc(111),metal=Pt")
    """

    def __init__(self,
                 ase_database_path: str,
                 graph_dataset_dir: str,
                 graph_params: dict[str, Union[dict, str]], 
                 database_key: str="calc_type=adsorption"):        
        self.dataset_id = pyg_dataset_id(ase_database_path, graph_params)
        self.ase_database_path = ase_database_path
        self.root = os.path.dirname(ase_database_path)
        self.graph_structure_params = graph_params["structure"]
        self.graph_features_params = graph_params["features"]    
        self.target = graph_params["target"]
        self.database_key = database_key
        self.output_path = os.path.join(os.path.abspath(graph_dataset_dir), self.dataset_id)
        # Construct one-hot encoders for chemical elements and surface orientation (based on the selected data defined by database_key)
        db = connect(self.ase_database_path)
        self.elements_list, self.surface_orientation_list = [], []
        for row in db.select(database_key):
            chemical_symbols = set(row.toatoms().get_chemical_symbols())    
            for element in chemical_symbols:
                if element not in self.elements_list:
                    self.elements_list.append(element)
            surface_orientation = row.get("facet")
            if surface_orientation not in self.surface_orientation_list:
                self.surface_orientation_list.append(surface_orientation)
        self.molecule_elements = [elem for elem in self.elements_list if elem in ["C", "H", "O", "N", "S"]]
        self.one_hot_encoder_elements = OneHotEncoder().fit(np.array(self.elements_list).reshape(-1, 1)) 
        self.one_hot_encoder_facets = OneHotEncoder().fit(np.array(self.surface_orientation_list).reshape(-1, 1))
        # Initialize counters
        self.database_size = 0
        self.graph_dataset_size = 0 
        # Filter counters
        self.counter_isomorphism = 0
        self.counter_H_filter = 0
        self.counter_C_filter = 0
        self.counter_fragment_filter = 0
        self.counter_adsorption_filter = 0
        # Filter bins
        self.bin_isomorphism = []
        self.bin_H_filter = []
        self.bin_C_filter = []
        self.bin_fragment_filter = []
        self.bin_adsorption_filter = []
        self.bin_unconverted_atoms_objects = [] 
        # Node features
        self.node_feature_list = list(self.one_hot_encoder_elements.categories_[0])
        self.node_dim = len(self.node_feature_list)
        if graph_params["features"]["adsorbate"]:
            self.node_dim += 1
            self.node_feature_list.append("Adsorbate")
        if graph_params["features"]["ring"]:
            self.node_dim += 1
            self.node_feature_list.append("Ring")
        if graph_params["features"]["aromatic"]:
            self.node_dim += 1
            self.node_feature_list.append("Aromatic")
        if graph_params["features"]["radical"]:
            self.node_dim += 1
            self.node_feature_list.append("Radical")
        if graph_params["features"]["valence"]:
            self.node_dim += 1
            self.node_feature_list.append("Valence")
        if graph_params["features"]["facet"]:
            self.node_dim += len(self.one_hot_encoder_facets.categories_[0])
            self.node_feature_list += list(self.one_hot_encoder_facets.categories_[0])
        if graph_params["features"]["gcn"]:
            self.node_dim += 1
            self.node_feature_list.append("gcn")
        if graph_params["features"]["magnetization"]:
            self.node_dim += 1
            self.node_feature_list.append("Magnetization")
        # self.additional_data_path = os.path.join(self.processed_dir, 'additional_data.pt')  # path for the additional data
        # if os.path.exists(self.additional_data_path):
        #     self.additional_data = torch.load(self.additional_data_path)  # load the additional data
        super().__init__(root=os.path.abspath(graph_dataset_dir))
        self.data, self.slices = load(self.processed_paths[0])    

    @property
    def raw_file_names(self): 
        return self.ase_database_path
    
    @property
    def processed_file_names(self): 
        """
        Return the name of the processed file containing the PyG data objects.
        """
        return self.output_path
    
    def download(self):
        pass
    
    def process(self):  
        db = connect(self.ase_database_path)    
        elements_list = list(self.one_hot_encoder_elements.categories_[0]) 
        molecule_elements_indices = [elements_list.index(element) for element in self.molecule_elements]
        data_list = []
        for row in db.select(self.database_key):
            self.database_size += 1
            atoms_obj = row.toatoms()
            calc_type = row.get("calc_type")
            formula = row.get("formula")
            family = row.get("family")
            metal = row.get("metal")
            facet = row.get("facet")
            # 1) Get primitive graph and filter out wrong structures
            try:
                graph, surface_neighbours = atoms_to_pyggraph(atoms_obj, 
                                      self.graph_structure_params["tolerance"], 
                                      self.graph_structure_params["scaling_factor"],
                                      self.graph_structure_params["second_order_nn"], 
                                      self.one_hot_encoder_elements, 
                                      self.molecule_elements)
            except:
                self.bin_unconverted_atoms_objects.append(atoms_obj)
                print("{} ({}) Error in graph structure generation.".format(formula, family))
                continue

            # Graph labelling
            y = tensor(float(row.get(self.target)), dtype=torch.float)
            graph.target, graph.y = y, y
            graph.formula = formula
            graph.family = family
            graph.type = calc_type
            graph.metal = metal
            graph.facet = facet
            graph.atoms_obj = atoms_obj

            if not adsorption_filter(graph, self.one_hot_encoder_elements, self.molecule_elements):
                print("{} ({}) filtered out: No catalyst representation.".format(formula, family))
                self.bin_adsorption_filter.append(graph)
                self.counter_adsorption_filter += 1
                continue    
            if not H_connectivity_filter(graph, self.one_hot_encoder_elements, self.molecule_elements):
                print("{} ({}) filtered out: Wrong H connectivity within the adsorbate.".format(formula, family))
                self.bin_H_filter.append(graph)
                self.counter_H_filter += 1  
                continue
            if not C_connectivity_filter(graph, self.one_hot_encoder_elements, self.molecule_elements):
                print("{} ({}) filtered out: Wrong C connectivity within the adsorbate.".format(formula, family))
                self.bin_C_filter.append(graph)
                self.counter_C_filter += 1  
                continue
            if not single_fragment_filter(graph, self.one_hot_encoder_elements, self.molecule_elements):
                print("{} ({}) filtered out: Fragmented adsorbate.".format(formula, family))
                self.bin_fragment_filter.append(graph)
                self.counter_fragment_filter += 1  
                continue
            # 2) Nodes' featurization
            if self.graph_features_params["adsorbate"]:
                x_adsorbate = zeros((graph.x.shape[0], 1))  # 1=adsorbate, 0=metal
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    x_adsorbate[i, 0] = 1 if index in molecule_elements_indices else 0
                graph.x = cat((graph.x, x_adsorbate), dim=1)
            if self.graph_features_params["ring"] and family:
                x_ring = zeros((graph.x.shape[0], 1))  # 1=ring, 0=no ring
                mol_graph, index_list = extract_adsorbate(graph, self.one_hot_encoder_elements)
                mol_ring_nodes = detect_ring_nodes(mol_graph)
                for node_index in mol_ring_nodes:
                    x_ring[index_list.index(node_index), 0] = 1
                graph.x = cat((graph.x, x_ring), dim=1)                
            if self.graph_features_params["aromatic"]:
                x_aromatic = torch.zeros((graph.x.shape[0], 1))  # 1=aromatic, 0=no aromatic/metal
                ring_descriptor_index = self.node_feature_list.index("Ring")  # aromatic atom -> ring atom
                if len(torch.where(graph.x[:, ring_descriptor_index] == 1)[0]) == 0:
                    graph.x = torch.cat((graph.x, x_aromatic), dim=1)
                else:  # Ring presence
                    try: 
                        aromatic_atoms = get_aromatic_atoms(atoms_obj, self.molecule_elements)
                    except:
                        print("{} ({}) Error in aromatic detection.".format(formula, family))
                        self.bin_unconverted_atoms_objects.append(atoms_obj)
                        continue
                    for index, node in enumerate(graph.x):
                        if node[ring_descriptor_index] == 0:  # atom not in a ring
                            x_aromatic[index, 0] = 0
                        else:  
                            if index in aromatic_atoms:
                                x_aromatic[index, 0] = 1 
                    graph.x = torch.cat((graph.x, x_aromatic), dim=1)
            if self.graph_features_params["radical"]:
                x_radical = torch.zeros((graph.x.shape[0], 1))  # 1=radical, 0=no radical/ metal
                radical_atoms = get_radical_atoms(atoms_obj, self.molecule_elements)
                for index, node in enumerate(graph.x):
                    if index in radical_atoms:
                        x_radical[index, 0] = 1
                graph.x = torch.cat((graph.x, x_radical), dim=1)
            if self.graph_features_params["valence"]:                
                try:
                    x_valence = torch.zeros((graph.x.shape[0], 1))
                    scaled_degree_vector = get_atom_valence(atoms_obj, self.molecule_elements)
                    for index, node in enumerate(scaled_degree_vector):
                        x_valence[index, 0] = scaled_degree_vector[index, 0]
                    graph.x = torch.cat((graph.x, x_valence), dim=1)
                except:
                    print("{} ({}) Error in valence detection.".format(formula, family))
                    self.bin_unconverted_atoms_objects.append(atoms_obj)
                    continue                
            if self.graph_features_params["facet"]:
                x_facet = zeros((graph.x.shape[0], len(self.one_hot_encoder_facets.categories_[0])))
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    facet_index = list(self.one_hot_encoder_facets.categories_[0]).index(facet)
                    if index not in molecule_elements_indices:
                        x_facet[i, facet_index] = 1
                graph.x = cat((graph.x, x_facet), dim=1)
            if self.graph_features_params["gcn"]:
                x_generalized_coordination_number = torch.zeros((graph.x.shape[0], 1))
                cn = get_gcn(atoms_obj, self.molecule_elements)
                counter = 0
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    if index not in molecule_elements_indices:
                        x_generalized_coordination_number[i, 0] = cn[surface_neighbours[counter]][0]
                        counter += 1
                graph.x = torch.cat((graph.x, x_generalized_coordination_number), dim=1)
            if self.graph_features_params["magnetization"]:
                x_magnetization = torch.zeros((graph.x.shape[0], 1))
                # magnetic atom (Fe,Ni,Co) = 1, non-magnetic atom = 0
                for i, node in enumerate(graph.x):
                    index = where(node == 1)[0][0].item()
                    element = elements_list[index]
                    if element in ("Fe", "Ni", "Co"):
                        x_magnetization[i, 0] = 1
                graph.x = torch.cat((graph.x, x_magnetization), dim=1)

            # if graph.x.shape[1] != self.node_dim:
            #     raise ValueError("Node dimension mismatch: {} vs {}".format(graph.x.shape[1], self.node_dim))

            # 3) Edge featurization 
            # TODO: implement edge features (and maybe hyperedges)
            # one-hot encoding for the kind of bond
            # adsorbate-adsorbate, adsorbate-surface, surface-surface, surface-hydrogen

            # 5) Create human-readable Pandas Dataframe of the node feature matrix of the graph
            df = pd.DataFrame(graph.x.numpy(), columns=self.node_feature_list)
            df["element"] = df.iloc[:, 0:len(self.elements_list)].idxmax(axis=1)
            df = df.drop(columns=df.columns[0:len(self.elements_list)])
            df = df[["element"] + [col for col in df.columns if col != "element"]]
            # Convert columns to boolean
            if "Ring" in df.columns:
                df["Ring"] = df["Ring"].astype(bool)
            if "Aromatic" in df.columns:
                df["Aromatic"] = df["Aromatic"].astype(bool)
            if "Radical" in df.columns:
                df["Radical"] = df["Radical"].astype(bool)
            graph.df = df

            # 6) Isomorphism test with graphs already in the dataset
            if not isomorphism_test(graph, data_list, 0.01):  
                print("{} ({}) filtered out: Isomorphic to another graph.".format(formula, family))
                self.bin_isomorphism.append(graph)
                self.counter_isomorphism += 1
                continue

            # # Convert one-hot encoded node element to atomic number
            # # meaning: rewrite x matrix with atomic numbers instead of one-hot encoded elements
            # x = zeros((graph.x.shape[0], 1))
            # num_elems = len(self.one_hot_encoder_elements.categories_[0])
            # for i, node in enumerate(graph.x):
            #     index = where(node == 1)[0][0].item()
            #     element = self.one_hot_encoder_elements.categories_[0][index]
            #     # get atomic number of element
            #     atomic_number = int(Chem.GetPeriodicTable().GetAtomicNumber(element))
            #     x[i, 0] = atomic_number
            
            # # subsitute first num_elems columns of x with atomic numbers
            # graph.x = cat((x, graph.x[:, num_elems:]), dim=1)

            data_list.append(graph)
            self.graph_dataset_size += 1
            print("{} ({}) added to dataset".format(formula, family))
        print("Graph dataset size: {}".format(len(data_list)))
        data, slices = self.collate(data_list)
        save((data, slices), self.processed_paths[0])
        # Save all other attributes in a dictionary
        # attributes_dict = {"bin_adsorption_filter": self.bin_adsorption_filter, 
        #               "bin_H_filter": self.bin_H_filter,
        #               "bin_C_filter": self.bin_C_filter,
        #               "bin_fragment_filter": self.bin_fragment_filter,
        #               "bin_isomorphism": self.bin_isomorphism,
        #               "counter_adsorption_filter": self.counter_adsorption_filter,
        #               "counter_H_filter": self.counter_H_filter,
        #               "counter_C_filter": self.counter_C_filter,
        #               "counter_fragment_filter": self.counter_fragment_filter,
        #               "counter_isomorphism": self.counter_isomorphism, 
        #               "bin_unconverted_atoms_objects": self.bin_unconverted_atoms_objects,
        #               "ase_database_size": self.database_size,
        #               "graph_dataset_size": self.graph_dataset_size}
        # save(attributes_dict, self.additional_data_path)
    
    
    # def print_summary(self):
    #     """
    #     Print a summary of the dataset.
    #     """
    #     database_name = "ASE database: {}\n".format(os.path.abspath(self.ase_database_path))
    #     selection_key = "Selection key: {}\n".format(self.database_key)
    #     database_size = "ASE database size: {}\n".format(self.additional_data["ase_database_size"])
    #     graph_dataset_size = "Graph dataset size: {}\n".format(self.additional_data["graph_dataset_size"])
    #     filtered_data = "Filtered data: {} ({:.2f}%)\n".format(self.additional_data["ase_database_size"] - self.additional_data["graph_dataset_size"], 100 * (1 - self.additional_data["graph_dataset_size"]/ self.additional_data["ase_database_size"]))
    #     graph_dataset_path = "Graph dataset path: {}\n".format(os.path.abspath(self.output_path))
    #     print(database_name + selection_key + database_size + graph_dataset_size + filtered_data + graph_dataset_path)


def atoms_to_data(structure: Union[Atoms, str], 
                  graph_params: dict[str, Union[float, int, bool]], 
                  model_elems: list[str], 
                  calc_type: str='adsorption') -> Data:
    """
    Convert ASE atoms object to PyG Data object based on the graph parameters.
    The implementation is similar to the one in the ASE to PyG converter class, but it is not a class method and 
    is used for inference. Target values are not included in the Data object.

    Args:
        structure (Atoms): ASE atoms object or file to POSCAR/CONTCAR file.
        graph_params (dict): Dictionary containing the information for the graph generation in the format:
                            {"tolerance": float, "scaling_factor": float, "metal_hops": int, "second_order_nn": bool}
        model_elems (list): List of chemical elements that can be processed by the model.
    Returns:
        graph (Data): PyG Data object.
    """
    
    if isinstance(structure, str):  
        structure = read(structure)
    elif not isinstance(structure, Atoms):
        raise TypeError("Structure must be of type ASE Atoms or POSCAR/CONTCAR file path.")
    
    # Get list of elements in the structure
    elements_list = list(set(structure.get_chemical_symbols()))
    if not all(elem in model_elems for elem in elements_list):
        raise ValueError("Not all chemical elements in the structure can be processed by the model.")
    
    # Read graph conversion parameters
    graph_structure_params = graph_params["structure"]
    graph_features_params = graph_params["features"]
    formula = structure.get_chemical_formula()

    # Construct one-hot encoder for elements
    adsorbate_elements = ["C", "H", "O", "N", "S"]  # hard-coded for now
    one_hot_encoder_elements = OneHotEncoder().fit(np.array(model_elems).reshape(-1, 1)) 
    elements_list = list(one_hot_encoder_elements.categories_[0])
    node_features_list = list(one_hot_encoder_elements.categories_[0]) 
    # append to node_features_list the key features whose value is True, in uppercase
    for key, value in graph_features_params.items():
        if value:
            node_features_list.append(key.upper())
    adsorbate_elements_indices = [elements_list.index(element) for element in adsorbate_elements]
    graph, surface_neighbours = atoms_to_pyggraph(structure, 
                                                  graph_structure_params["tolerance"], 
                                                  graph_structure_params["scaling_factor"],
                                                  graph_structure_params["second_order_nn"], 
                                                  one_hot_encoder_elements, 
                                                  adsorbate_elements)
    graph.type = calc_type
    if not adsorption_filter(graph, one_hot_encoder_elements, adsorbate_elements):
        raise ValueError("Adsorption filter failed for {}".format(formula))
    if not H_connectivity_filter(graph, one_hot_encoder_elements, adsorbate_elements):
        raise ValueError("H connectivity filter failed for {}".format(formula))
    if not C_connectivity_filter(graph, one_hot_encoder_elements, adsorbate_elements):
        raise ValueError("C connectivity filter failed for {}".format(formula))
    if not single_fragment_filter(graph, one_hot_encoder_elements, adsorbate_elements):
        raise ValueError("Single fragment filter failed for {}".format(formula))
    # node featurization
    if graph_features_params["adsorbate"]:
        x_adsorbate = zeros((graph.x.shape[0], 1))  # 1=adsorbate, 0=metal
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            x_adsorbate[i, 0] = 1 if index in adsorbate_elements_indices else 0
        graph.x = cat((graph.x, x_adsorbate), dim=1)
    if graph_features_params["ring"]:
        x_ring = zeros((graph.x.shape[0], 1))  # 1=ring, 0=no ring
        mol_graph, index_list = extract_adsorbate(graph, one_hot_encoder_elements)
        mol_ring_nodes = detect_ring_nodes(mol_graph)
        for node_index in mol_ring_nodes:
            x_ring[index_list.index(node_index), 0] = 1
        graph.x = cat((graph.x, x_ring), dim=1)                
    if graph_features_params["aromatic"]:
        x_aromatic = torch.zeros((graph.x.shape[0], 1))  # 1=aromatic, 0=no aromatic/metal
        ring_descriptor_index = node_features_list.index("Ring")  # aromatic atom -> ring atom
        if len(torch.where(graph.x[:, ring_descriptor_index] == 1)[0]) == 0:
            graph.x = torch.cat((graph.x, x_aromatic), dim=1)
        else: # presence of rings
            try: 
                aromatic_atoms = get_aromatic_atoms(structure, adsorbate_elements)
            except:
                raise ValueError("Aromatic atoms could not be detected. Check if the structure is valid.")
            for index, node in enumerate(graph.x):
                if node[ring_descriptor_index] == 0:  # atom not in a ring
                    x_aromatic[index, 0] = 0
                else:  
                    if index in aromatic_atoms:
                        x_aromatic[index, 0] = 1 
            graph.x = torch.cat((graph.x, x_aromatic), dim=1)
    if graph_features_params["radical"]:
        x_radical = torch.zeros((graph.x.shape[0], 1))  # 1=radical, 0=no radical/ metal
        radical_atoms = get_radical_atoms(structure, adsorbate_elements)
        for index, node in enumerate(graph.x):
            if index in radical_atoms:
                x_radical[index, 0] = 1
        graph.x = torch.cat((graph.x, x_radical), dim=1)
    if graph_features_params["valence"]:                
        try:
            x_valence = torch.zeros((graph.x.shape[0], 1))
            scaled_degree_vector = get_atom_valence(structure, adsorbate_elements)
            for index, node in enumerate(scaled_degree_vector):
                x_valence[index, 0] = scaled_degree_vector[index, 0]
            graph.x = torch.cat((graph.x, x_valence), dim=1)
        except:
            raise ValueError("{}: Error in valence detection.".format(formula))               
    # if graph_features_params["facet"]:
    #     x_facet = zeros((graph.x.shape[0], len(one_hot_encoder_facets.categories_[0])))
    #     for i, node in enumerate(graph.x):
    #         index = where(node == 1)[0][0].item()
    #         facet_index = list(one_hot_encoder_facets.categories_[0]).index(facet)
    #         if index not in adsorbate_elements_indices:
    #             x_facet[i, facet_index] = 1
    #     graph.x = cat((graph.x, x_facet), dim=1)
    if graph_features_params["gcn"]:
        x_generalized_coordination_number = torch.zeros((graph.x.shape[0], 1))
        cn = get_gcn(structure, adsorbate_elements)
        counter = 0
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            if index not in adsorbate_elements_indices:
                x_generalized_coordination_number[i, 0] = cn[surface_neighbours[counter]][0]
                counter += 1
        graph.x = torch.cat((graph.x, x_generalized_coordination_number), dim=1)
    if graph_features_params["magnetization"]:
        x_magnetization = torch.zeros((graph.x.shape[0], 1))
        # magnetic atom (Fe,Ni,Co) = 1, non-magnetic atom = 0
        for i, node in enumerate(graph.x):
            index = where(node == 1)[0][0].item()
            element = elements_list[index]
            if element in ("Fe", "Ni", "Co"):
                x_magnetization[i, 0] = 1
        graph.x = torch.cat((graph.x, x_magnetization), dim=1)
    
    graph.formula = formula
    graph.node_feats = node_features_list
    return graph