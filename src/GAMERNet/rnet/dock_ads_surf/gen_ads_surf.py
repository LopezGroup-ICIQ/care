import os
import ase
import time
import numpy as np
import multiprocessing
import resource
from torch_geometric.loader import DataLoader
from torch.nn import Module
from numpy import max, arange
from ase import Atoms
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from GAMERNet.rnet.graphs.graph_fn import connectivity_analysis, ase_coord_2_graph
from GAMERNet.gnn_eads.create_pyg_dataset import atoms_to_data
import GAMERNet.rnet.dock_ads_surf.dockonsurf.dockonsurf as dos
from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.surface import Surface


def process_chunk(args):
    chunk, graph_params, model_elems, calc_type = args
    return [atoms_to_data(structure, graph_params, model_elems, calc_type) for structure in chunk]

def atoms_to_data_parallel(atoms_list, graph_params, model_elems, calc_type='adsorption'):
    # Split atoms_list into chunks
    num_cores = multiprocessing.cpu_count()
    chunks = [(atoms_list[i::num_cores], graph_params, model_elems, calc_type) for i in range(num_cores)]

    # Use multiprocessing.Pool to parallelize the conversion
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten the list of results and return
    return [data for sublist in results for data in sublist]


def get_fragment_energy(structure: Atoms) -> float:
    """
    Get adsorbate gas-phase energy from reference closed-shell molecules.
    Args:
        structure (Atoms): Atoms object of the adsorption structure
    Returns:
        (float): gas-phase adsorbate energy in eV
    """ 
    elements = structure.get_chemical_symbols()
    ed = {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0}  # adsorbate elements
    for element in elements:
        if element in ed:
            ed[element] += 1
        else:
            ed[element] = 1
    # Reference DFT energy for C, H, O, N, S
    e_H2O = -14.21877278  # O
    e_H2 = -6.76639487    # H
    e_NH3 = -19.54236910  # N
    e_H2S = -11.20113092  # S
    e_CO2 = -22.96215586  # C
    # Count elemens in the structure
    n_C, n_H, n_O, n_N, n_S = ed["C"], ed["H"], ed["O"], ed["N"], ed["S"]
    #TODO: use numpy to solve this
    return n_C * e_CO2 + (n_O - 2*n_C) * e_H2O + (4*n_C + n_H - 2*n_O - 3*n_N - 2*n_S) * e_H2 * 0.5 + (n_N * e_NH3) + (n_S * e_H2S)


def generate_inp_vars(adsorbate: ase.Atoms, 
                      surface: ase.Atoms,
                      ads_height: float,
                      coll_thresh: float,
                      max_structures: int, 
                      min_coll_height: float,
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
        "min_coll_height": min_coll_height,
        "adsorption_height": float(ads_height),
        "collision_threshold": coll_thresh,
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


def run_docksurf(intermediate: Intermediate, 
                 surface: Surface, 
                 model: Module, 
                 graph_params:dict, 
                 model_elems:list, 
                 ) -> float:
    molec_ase_obj = intermediate.molecule
    surface_facet = surface.facet

    # Getting the distance between the furthest atoms of the molecule
    molec_dist_mat = molec_ase_obj.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    print('Distance between furthest atoms in the molecule: {:.2f} Angstrom'.format(max_dist_molec))

    # Convert ASE object to graph and get potential molecular centers for adsorption
    molec_graph = ase_coord_2_graph(molec_ase_obj, coords=True)
    connect_sites_molec = connectivity_analysis(molec_graph)

    print('Surface x-y extension: {:.2f} Angstrom'.format(surface.slab_diag))

    # Check if molecule fits on reference metal slab, if not scale the surface
    tolerance = 3.0   # Angstrom
    condition = surface.slab_diag - tolerance > max_dist_molec
    if condition:
        print('Molecule fits on reference metal slab')
        aug_slab = surface.slab
    else:
        print('Scaling reference metal slab...')
        counter = 1.0
        while not condition:
            counter += 1.0
            pymatgen_slab = AseAtomsAdaptor.get_structure(surface.slab)
            pymatgen_slab.make_supercell([counter, counter, 1])
            aug_slab = AseAtomsAdaptor.get_atoms(pymatgen_slab)
            aug_surf = Surface(aug_slab, surface_facet)
            condition = aug_surf.slab_diag - tolerance > max_dist_molec
        print('Reference metal slab scaled by factor {} on the x-y plane\n'.format(counter)) 
    
    # Generate input files for DockonSurf
    active_sites = {"Site_{}".format(site["label"]): site["indices"] for site in surface.active_sites}

    min_height = 2.5
    max_height = 2.8
    increment = 0.1

    t00 = time.time()
    total_config_list = []
    for ads_height in arange(min_height, max_height, increment):
        ads_height = '{:.2f}'.format(ads_height)
        for _, site_idxs in active_sites.items():
            if site_idxs != []:
                inp_vars = generate_inp_vars(molec_ase_obj, 
                                  aug_slab,
                                  ads_height,
                                  1.2,
                                  25, 
                                  1.2,
                                  connect_sites_molec,
                                  site_idxs,)
                # Run DockonSurf
                config_list = dos.dockonsurf(inp_vars)
                total_config_list.extend(config_list)

    print('DockonSurf run time: {:.2f} s'.format(time.time()-t00))
    print('Number of detected adsorption configurations: ', len(total_config_list))

    if len(total_config_list) == 0:
        return 0.0
    t_000 = time.time()
    # Removing the metal atoms with selective dynamics == False
    fixed_atms_idxs = surface.slab.todict().get('constraints', None)[0].get_indices()
    fixed_atms = surface.slab[np.isin(range(len(surface.slab)), fixed_atms_idxs)]
    for idx, atoms_obj in enumerate(total_config_list):
        # Removing the atoms which indices are in fixed_atms
        atoms_obj = atoms_obj[~np.isin(range(len(atoms_obj)), fixed_atms_idxs)]
        total_config_list[idx] = atoms_obj


    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096*2, rlimit[1]))
    
    ads_graph_list = atoms_to_data_parallel(total_config_list, graph_params, model_elems)

    # Setting back the default limit
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    loader = DataLoader(ads_graph_list, batch_size=len(ads_graph_list), shuffle=False)
    print('Time to convert adsorption configurations to graphs: {:.2f} s'.format(time.time()-t_000))
    t_00 = time.time()
    for batch in loader:
        energy_list = model(batch)  # unitless (scaled values)
        mean_tensor = energy_list.mean * model.scaling_params['std'] + model.scaling_params['mean'] # eV
        std_tensor = energy_list.scale * model.scaling_params['std'] # eV
    print('Time to evaluate adsorption configurations: {:.2f} s'.format(time.time()-t_00))
    best_poscar = total_config_list[mean_tensor.argmin().item()]
    best_poscar.extend(fixed_atms)
    last_idxs_best_poscar = len(best_poscar) - len(fixed_atms)
    
    last_idxs_list = list(range(last_idxs_best_poscar, len(best_poscar)))
    fixed_atms_constr = FixAtoms(indices=last_idxs_list)
    best_poscar.set_constraint(fixed_atms_constr)

    best_ensemble = mean_tensor.min().item()
    fragment_energy = get_fragment_energy(best_poscar)
    best_eads = best_ensemble - fragment_energy
    intermediate.energy = best_eads
    return best_eads

