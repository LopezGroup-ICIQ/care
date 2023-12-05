# Functions to place the adsorbate on the surface

from numpy import max
import networkx as nx
from care.netgen.networks.intermediate import Intermediate
from care.netgen.networks.surface import Surface
from pymatgen.io.ase import AseAtomsAdaptor
import care.netgen.dock_ads_surf.dockonsurf.dockonsurf as dos
from ase import Atoms
import numpy as np
import itertools as it
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from care.netgen.intermediates_funcs import rdkit_to_ase

from care.netgen.data.constants_and_data import BOND_ORDER

def connectivity_analysis(graph: nx.Graph) -> list:
    """
    This function will return a list with the number of connections for each atom in the molecule.
    
    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the molecule.
    
    Returns
    -------
    list
        List with the number of connections for each atom in the molecule.
    """
    
    max_conns = BOND_ORDER

    unsat_elems = [node for node in graph.nodes() if graph.degree(node) < max_conns.get(graph.nodes[node]["elem"], 0)]
    if not unsat_elems:
        # If the molecule is H2, return the list of atoms
        if len(graph.nodes()) == 2 and graph.nodes[0]["elem"] == 'H' and graph.nodes[1]["elem"] == 'H':
            return list(set(graph.nodes()))
        else:
            sat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] != 'H']
            return list(set(sat_elems))
    
    # Specifying the carbon monoxide case
    elif len(graph.nodes()) == 2 and ((graph.nodes[0]["elem"] == 'C' and graph.nodes[1]["elem"] == 'O') or (graph.nodes[0]["elem"] == 'O' and graph.nodes[1]["elem"] == 'C')):
        # Extracting only the Carbon atom index
        unsat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] == 'C']
        return list(set(unsat_elems))
    
    # Specifying the case for CO2
    elif len(graph.nodes()) == 3 and ((graph.nodes[0]["elem"] == 'C' and graph.nodes[1]["elem"] == 'O' and graph.nodes[2]["elem"] == 'O') or (graph.nodes[0]["elem"] == 'O' and graph.nodes[1]["elem"] == 'O' and graph.nodes[2]["elem"] == 'C') or (graph.nodes[0]["elem"] == 'O' and graph.nodes[1]["elem"] == 'C' and graph.nodes[2]["elem"] == 'O')):
        # Extracting only the Carbon atom index
        unsat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] == 'C']
        return list(set(unsat_elems))
    
    # Specifying case for H
    elif len(graph.nodes()) == 1 and graph.nodes[0]["elem"] == 'H':
        return list(set(graph.nodes()))
    else:
        return list(set(unsat_elems))

def generate_inp_vars(adsorbate: Atoms, 
                      surface: Atoms,
                      ads_height: float,
                      max_structures: int, 
                      molec_ctrs: list,
                      sites: list,):
    """
    Generates the input variables for the dockonsurf screening.

    Parameters
    ----------
    adsorbate : ase.Atoms
        Atoms object of the adsorbate.
    surface : ase.Atoms
        Atoms object of the surface.
    ads_height : float
        Adsorption height.
    coll_thresh : float
        Collision threshold.
    max_structures : int
        Maximum number of structures.
    min_coll_height : float
        Minimum collision height.
    molec_ctrs : list
        Molecular centers of the adsorbate.
    sites : list
        Active sites of the surface.
    """ 
    adsorbate.set_cell(surface.get_cell().lengths())
    inp_vars = {
        "Global": True,
        "Screening":True,
        "run_type": 'Screening',
        "code": 'VASP',
        "batch_q_sys": 'False',
        "project_name": "test",
        "surf_file": surface,
        "use_molec_file": adsorbate,
        "sites": sites,
        "molec_ctrs": molec_ctrs,
        "min_coll_height": 1.5,
        "adsorption_height": float(ads_height),
        "collision_threshold": 1.2,
        "max_structures": max_structures,
        "set_angles": 'euler',
        'sample_points_per_angle': 3,
        "surf_norm_vect": 'z',
        'exclude_ads_ctr': False,
        'h_acceptor': 'all',
        'h_donor': False,
        'max_helic_angle': 180,
        'pbc_cell': surface.get_cell(),
        'select_magns': 'energy',
        'special_atoms': 'False',
        "potcar_dir": 'False',
    }
    return inp_vars

def adapt_surface(molec_ase: Atoms, surface: Surface, tolerance: float = 3.0) -> Atoms:
    """
    Adapts the surface depending on the size of the molecule.

    Parameters
    ----------
    molec_ase : Atoms
        Atoms object of the molecule.
    surface : Surface
        Surface instance of the surface.
    tolerance : float
        Toleration in Angstrom.

    Returns
    -------
    Atoms
        Atoms object of the surface.
    """
    molec_dist_mat = molec_ase.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    condition = surface.get_shortest_side() - tolerance > max_dist_molec
    if condition:
            new_slab = surface.slab
    else:
        counter = 1.0
        while not condition:
            counter += 1.0
            pymatgen_slab = AseAtomsAdaptor.get_structure(surface.slab)
            pymatgen_slab.make_supercell([counter, counter, 1])
            new_slab = AseAtomsAdaptor.get_atoms(pymatgen_slab)
            aug_surf = Surface(new_slab, surface.facet)
            condition = aug_surf.slab_diag - tolerance > max_dist_molec
    return new_slab

def ads_placement(intermediate: Intermediate, 
                 surface: Surface) -> tuple[str, list[Atoms]]:
    """
    Generate a set of adsorption structures for a given intermediate and surface.

    Parameters
    ----------
    intermediate : Intermediate
        Intermediate.
    surface : Surface
        Surface.

    Returns
    -------
    tuple(str, list[Atoms])
        Tuple containing the code of the intermediate and its list of adsorption configurations.
    """

    # Get the potential molecular centers for adsorption
    connect_sites_molec = connectivity_analysis(intermediate.graph)

    # Check if molecule fits on reference metal slab. If not, scale the surface
    slab = adapt_surface(intermediate.molecule, surface)
    
    # Generate input files for DockonSurf
    active_sites = {"{}".format(site["label"]): site["indices"] for site in surface.active_sites}

    total_config_list = []
    # If the chemical species is not a single atom, placing the molecule on the surface using DockonSurf
    if len(intermediate.molecule) > 1:
        for site_idxs in active_sites.values():
            ads_height = 2.2 if intermediate.molecule.get_chemical_formula() != 'H2' else 1.8
            if site_idxs != []:
                config_list = []
                while config_list == []:
                    inp_vars = generate_inp_vars(adsorbate=intermediate.molecule, 
                                            surface=slab,
                                            ads_height=ads_height,
                                            max_structures=1, 
                                            molec_ctrs=connect_sites_molec,
                                            sites=site_idxs,)
                    
                    config_list = dos.dockonsurf(inp_vars)
                        
                    if len(config_list) == 0:
                        ads_height += 0.2

                    total_config_list.extend(config_list)

        print(f'{intermediate.code} placed on the surface')
        return intermediate.code, total_config_list
    
    # If the chemical species is a single atom, placing the atom on the surface
    else:
        # for all sites, add the atom to the site
        for site in surface.active_sites:
            atoms = slab.copy()
            # append unique atom in the defined position in active_sites
            atoms.append(intermediate.molecule[0])
            atoms.positions[-1] = site['position']
            atoms.set_cell(surface.slab.get_cell())
            atoms.set_pbc(surface.slab.get_pbc())
            total_config_list.append(atoms)

        print(f'{intermediate.code} placed on the surface')
        return intermediate.code, total_config_list
    
def best_fit_plane(atom_coords):
    num_points = atom_coords.shape[0]
    centroid = np.mean(atom_coords, axis=0)

    if num_points == 2:
        # Create a vector from the first atom to the second
        vec_between_atoms = atom_coords[1] - atom_coords[0]

        # Choose a non-collinear vector (e.g., unit vector along z-axis)
        non_collinear_vec = np.array([0, 0, 1])

        # Calculate normal as cross product
        normal = np.cross(vec_between_atoms, non_collinear_vec)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
    else:
        # Subtract centroid
        coords_subtracted = atom_coords - centroid

        # Perform SVD
        u, s, vh = np.linalg.svd(coords_subtracted)

        # Plane normal is the last column of u
        normal = u[:, -1]

    # Calculate D using the centroid
    D = -np.dot(normal, centroid)

    return normal, D

def signed_distance_from_plane(atom_coords, plane_normal, D):
    # Normalize the plane normal vector
    norm = np.linalg.norm(plane_normal)
    normalized_normal = plane_normal / norm

    # Calculate the signed distance
    distances = np.dot(atom_coords, normalized_normal) + D / norm
    return distances

def atoms_on_one_side(atom_coords, plane_normal, D, side='negative'):
    distances = signed_distance_from_plane(atom_coords, plane_normal, D)
    
    if side == 'positive':
        selected_atoms = atom_coords[distances > 0]
    else:
        selected_atoms = atom_coords[distances <= 0]

    return selected_atoms

def ads_placement_graph(intermediate: Intermediate, 
                 surface: Surface) -> tuple[str, list[nx.Graph]]:
    """
    Generate a set of adsorption structures as graph representations for a given intermediate and surface.

    Parameters
    ----------
    intermediate : Intermediate
        Intermediate.
    surface : Surface
        Surface.

    Returns
    -------
    tuple(str, list[nx.Graph])
        Tuple containing the code of the intermediate and its list of adsorption configurations as graph representations.
    """

    # Get the potential molecular centers for adsorption
    anchoring_atoms = connectivity_analysis(intermediate.graph)

    # Dictionary containing for each facets (fcc and hcp) the corresponding coordination number of C, H, O

    facet_coord_dict = {
        'fcc': {
            'C': 3,
            'H': 1,
            'O': 4,
        },
        'hcp': {
            'C': 4,
            'H': 1,
            'O': 4,
        },
        'bcc': {
            'C': 4,
            'H': 1,
            'O': 4,
        },
    }

    # Getting the element of the surface
    surf_elem = surface.metal
    surf_struct = surface.crystal_structure

    # Getting the coordination number of the surface
    surf_coord = facet_coord_dict[surf_struct]

    rdkit_mol = intermediate.rdkit
    # Adding Hs to the molecule
    rdkit_mol = Chem.AddHs(rdkit_mol)
    formula = CalcMolFormula(rdkit_mol)

    num_conformers = 5
    conformers = AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=num_conformers)

    graph_config_list = []
    for conf_id in conformers:
        conf = rdkit_mol.GetConformer(conf_id)
        
        # Getting the coordinates of the molecule
        mol_coords = conf.GetPositions()

        # Converting the conformer to Chem.Mol object
        # Create a new molecule which is a copy of the original
        new_mol = Chem.Mol(rdkit_mol)

        # Assign the coordinates of the selected conformer to the new molecule
        new_conf = Chem.Conformer(conf)
        new_mol.RemoveAllConformers()  # Remove any existing conformers in the new molecule
        new_mol.AddConformer(new_conf, assignId=True)

        # Converting the molecule to an ASE atoms object
        ase_mol = rdkit_to_ase(new_mol)

        # Identifying the anchoring atoms in the molecule
        anchoring_atoms = connectivity_analysis(intermediate.graph)
        
        if len(anchoring_atoms) > 1:
            anchoring_atoms_elem = []
            for idx in anchoring_atoms:
                anchoring_atoms_elem.append(ase_mol[idx].symbol)

            # getting the coordinates of the anchoring atoms
            anchoring_atoms_coords = ase_mol.get_positions()[anchoring_atoms]
            
            # Extracting the coordinates as an array
            mol_coords = ase_mol.get_positions()
            
            normal, D = best_fit_plane(anchoring_atoms_coords)  # From the previous function
            selected_atoms = atoms_on_one_side(mol_coords, normal, D, side='negative')
            
            # Matching the coordinates of the selected atoms with the coordinates of the atoms in the molecule
            selected_atoms_idx = []
            for atom in selected_atoms:
                for idx, coord in enumerate(mol_coords):
                    if np.array_equal(coord, atom):
                        selected_atoms_idx.append(idx)


            
            # Getting the element of the selected atoms
            selected_atoms_elem = []
            for idx in selected_atoms_idx:
                selected_atoms_elem.append(ase_mol[idx].symbol)
            # If all the anchoring atoms are not in the selected atoms, skip this conformer
            if len(set(anchoring_atoms_elem) & set(selected_atoms_elem)) == 0:
                continue

        elif len(anchoring_atoms) == 0:
            print(anchoring_atoms)
            anchoring_atoms_elem = intermediate.graph.nodes[0]['elem']
            print('anchoring_atoms_elem: ', anchoring_atoms_elem)
            selected_atoms_idx = [0]
            selected_atoms_elem = anchoring_atoms_elem
            print('selected_atoms_elem: ', selected_atoms_elem)
        else:
            anchoring_atoms_elem = [ase_mol[anchoring_atoms[0]].symbol]
            selected_atoms_idx = anchoring_atoms
            selected_atoms_elem = anchoring_atoms_elem

        # Generating all possible combinations from 0 to the number of coordination, for each atom  in selected_atoms_elem
        comb_list = []
        for elem in selected_atoms_elem:
            comb_list.append(list(range(surf_coord[elem]+1)))
        comb_list = list(it.product(*comb_list))

        # Creating a mapping of combinations to atom indices and elements
        comb_mapping = []
        for combination in comb_list:
            atom_coordination = []
            for idx, coord_number in enumerate(combination):
                atom_index = selected_atoms_idx[idx]
                atom_element = selected_atoms_elem[idx]
                atom_coordination.append((atom_index, atom_element, coord_number))
            comb_mapping.append(atom_coordination)

        # For each combination in comb_mapping,
        # Generating a copy of the graph and, for each node (idx is the first element of the tuple),
        # Connect to it the number of nodes (which node attribute is the element of the surfacce) specified in the tuple (coord_number is the third element of the tuple)
        for combination in comb_mapping:
            print(intermediate.molecule.get_chemical_formula())
            new_graph = intermediate.graph.copy()
            # Adding for each node the number of connections specified in the combination)
            for node in combination:
                for i in range(node[2]):
                    # Adding the node  with the surf_elem element
                    new_graph.add_node(len(new_graph.nodes()), elem=surf_elem)
                    # Adding the edge between the node and the node with the surf_elem element
                    new_graph.add_edge(node[0], len(new_graph.nodes())-1)

            graph_config_list.append(new_graph)

    return intermediate.code, graph_config_list
