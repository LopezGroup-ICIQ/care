import os
import ase
import time
import numpy as np
import multiprocessing
import resource
from torch_geometric.loader import DataLoader
from torch.nn import Module
from numpy import sqrt, max, arange
from numpy.linalg import norm
from ase import io, Atoms
from ase.constraints import FixAtoms
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from GAMERNet.rnet.utilities.functions import get_voronoi_neighbourlist
from GAMERNet.rnet.adsurf.functions.act_sites import get_act_sites
from GAMERNet.rnet.adsurf.functions.adsurf_fn import connectivity_analysis
from GAMERNet.rnet.adsurf.graphs.graph_utilities import ase_2_graph
from GAMERNet.gnn_eads.create_pyg_dataset import atoms_to_data
import GAMERNet.rnet.dock_ads_surf.dockonsurf.dockonsurf as dos

from GAMERNet import DOCK_DATA

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

def gen_docksurf_file(tmp_subdir: str, molecule_id: str, mol_obj: ase.Atoms, conn_idxs: list, slab_poscar_file: str, metal_lattice: list, activ_site: str, active_idxs: list, ads_height:float) -> None:
    """This function will generate the dockonsurf input file for the adsorbate-slab of interest.
    Parameters
    ----------
    molecule_id : str
        Name of the molecule.
    mol_obj : ase.Atoms
        ASE object of the molecule.
    conn_idxs : list
        List with the number of connections for each atom in the molecule.
    metal : str
        Name of the metal.
    metal_path : str
        Path to the metal directory.
    metal_lattice : list
        Lattice of the metal.
    run_path : str
        Path to the run directory.
    activ_site : str
        Name of the active site.
    active_idxs : list
        List with the indexes of the active sites.
    ads_height : float
        Height of the adsorbate used by DockOnSurf to screen potential configurations.
    Returns
    -------
    None
    """
    mol_obj.set_cell(metal_lattice)

    # files_path = os.path.abspath('./adsurf/data')
    files_path = os.path.abspath(DOCK_DATA)
    docksurf_template = f'{DOCK_DATA}/dockonsurf.inp'
    
    active_idxs = list(set(active_idxs))
    if len(active_idxs) == 1:
        active_idxs = f'{active_idxs[0]}'
    else:
        active_idxs = '(' + ' '.join([str(i) for i in active_idxs]) + ')'

    if len(conn_idxs) == 1:
        str_conn_idxs = f'{conn_idxs[0]}'
    else:
        str_conn_idxs = '(' + ' '.join([str(i) for i in conn_idxs]) + ')'
    
    # Generating the poscar directory with the dockonsurf poscars
    # poscar_directory = f"{tmp_subdir}/poscar_docksurf"
    poscar_directory = os.path.join(tmp_subdir, 'poscar_docksurf')
    os.makedirs(poscar_directory, exist_ok=True)
    # Generating the directories for the dockonsurf calculations
    # These folders contain the inputs and logs of the dockonsurf run and all the outputs generated on the dockonsurf run (screening)
    # docksurf_path = f"{tmp_subdir}/dos_outputs/{activ_site}"
    docksurf_path = os.path.join(tmp_subdir, 'dos_outputs', activ_site)
    os.makedirs(docksurf_path, exist_ok=True)
    io.write(f'{poscar_directory}/POSCAR', mol_obj, format='vasp')

    # Reading the template dockonsurf input and modifying it with information to run the analyses of interest
    with open(docksurf_template, 'r') as read:
        data = read.readlines()

    data[4] = f"project_name = {molecule_id}\n"
    data[15] = f"screen_inp_file = {files_path}/INCAR {files_path}/KPOINTS\n"
    data[16] = f"surf_file = {os.path.abspath(slab_poscar_file)}\n"
    data[17] = f"use_molec_file = {os.path.abspath(poscar_directory)}/POSCAR\n"
    data[19] = f"sites = {active_idxs}\n"
    data[22] = f"molec_ctrs = {str_conn_idxs}\n"
    # Check these two parameters
    data[23] = "min_coll_height = 1.5\n" # Default value
    data[26] = "adsorption_height = {}\n".format(ads_height) # Default value
    # Saving the modified dockonsurf input for each system
    str_ads_height = str(ads_height).replace('.','')
    with open(f"{docksurf_path}/dockonsurf_{molecule_id}_{str_ads_height}.inp", 'w') as write:
        write.writelines(data)
    return

def get_act_sites(slab_ase_obj: str, surface_facet: str) -> dict:
    # TODO: Refine this function to find the active sites of different metal facets
    """Finds the active sites of a metal surface. These can be ontop, bridge or hollow sites.

    Parameters
    ----------
    slab_ase_obj : Atoms
        ase.Atoms of the metal surface.

    Returns
    -------
    dict
        Dictionary with the atom indexes of each active site type.
    """
    # Converting the ase.Atoms object to a pymatgen.Structure object

    surface = AseAtomsAdaptor.get_structure(slab_ase_obj)
    surface.remove_site_property('selective_dynamics')
    
    surf_sites = AdsorbateSiteFinder(surface, selective_dynamics=True)
    most_active_sites = surf_sites.find_adsorption_sites()
    # TODO: Improve this part to find the active sites of the 110 facet
    if surface_facet == '110':
        # Getting only the first ontop site, 6 bridge sites and the first hollow sites
        most_active_sites['ontop'] = most_active_sites['ontop'][:1]
        most_active_sites['bridge'] = most_active_sites['bridge'][:6]
        most_active_sites['hollow'] = most_active_sites['hollow'][:1]
        # Updating the 'all' key
        most_active_sites['all'] = most_active_sites['ontop'] + most_active_sites['bridge'] + most_active_sites['hollow']
    
    count = 0
    active_site_dict = {}
    for coord_array in most_active_sites['all']:
        count += 1
        dict_label = f'Site_{count}'
        
        dummy_idx = surface.num_sites + 1
        surface.insert(dummy_idx, 'Li', coord_array, coords_are_cartesian=True)

        # Convertion to ASE atoms object
        surface_ase = AseAtomsAdaptor.get_atoms(surface)
        # Find the closest atoms to the dummy atom
        neigh_list = get_voronoi_neighbourlist(surface_ase, 0.25, 1, ['C', 'H', 'O'])
        
        # Looking for the Li dummy atom in the neighbour list
        active_site_dict[dict_label] = [i[0] for i in neigh_list if surface_ase.get_chemical_symbols().index('Li') in i]
        
        # Converting back to pymatgen structure
        new_surface = AseAtomsAdaptor.get_structure(surface_ase)

        # Removing the dummy atom by detecting the H atom
        for i in range(len(new_surface)):
            if new_surface[i].specie == Element('Li'):
                dummy_idx = i
                break
        surface.remove_sites([dummy_idx])
    return active_site_dict

def get_fragment_energy(structure: Atoms) -> float:
    """Calculate fragment energy from reference closed-shell molecules.
    This function allows to calculate the energy of both open- and closed-shell structures, 
    keeping the same reference.
    Args:
        structure (Atoms): Atoms object of the adsorption structure
    Returns:
        e_fragment (float): reference gas-phase energy in eV
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
    n_C = ed["C"]
    n_H = ed["H"]
    n_O = ed["O"]
    n_N = ed["N"]
    n_S = ed["S"]
    return n_C * e_CO2 + (n_O - 2*n_C) * e_H2O + (4*n_C + n_H - 2*n_O - 3*n_N - 2*n_S) * e_H2 * 0.5 + (n_N * e_NH3) + (n_S * e_H2S)

def run_docksurf(intermediate, slab_ase_obj: Atoms, surface_facet:str, model: Module, graph_params:dict, model_elems:list):
    molec_ase_obj = intermediate.molecule
    intermediate_code = intermediate.code
    res_folder = 'results/dockonsurf_screening'
    os.makedirs(res_folder, exist_ok=True)

    # Getting the distance between the furthest atoms of the molecule
    molec_dist_mat = molec_ase_obj.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    print('Distance between furthest atoms in the molecule: {:.2f} Angstrom'.format(max_dist_molec))

    # Convert ASE object to graph and get potential molecular centers for adsorption
    molec_graph = ase_2_graph(molec_ase_obj, coords=True)
    connect_sites_molec = connectivity_analysis(molec_graph)

    a, b, _ = slab_ase_obj.get_cell()
    slab_diagonal = sqrt(norm(a)**2 + norm(b)**2)
    print('Surface x-y extension: {:.2f} Angstrom'.format(slab_diagonal))

    # Check if molecule fits on reference metal slab, if not scale the surface
    tolerance = 3.0   # Angstrom
    condition = slab_diagonal - tolerance > max_dist_molec
    if condition:
        print('Molecule fits on reference metal slab\n')
    else:
        print('Scaling reference metal slab...')
        counter = 1.0
        while not condition:
            counter += 1.0
            pymatgen_slab = AseAtomsAdaptor.get_structure(slab_ase_obj)
            pymatgen_slab.make_supercell([counter, counter, 1])
            slab_ase_obj = AseAtomsAdaptor.get_atoms(pymatgen_slab)
            a, b, _ = slab_ase_obj.get_cell()
            slab_diagonal = sqrt(norm(a*counter)**2 + norm(b*counter)**2)
            condition = slab_diagonal - tolerance > max_dist_molec
        print('Reference metal slab scaled by factor {} on the x-y plane\n'.format(counter)) 
    
    tmp_subdir = os.path.join(res_folder, f'{intermediate_code}_{slab_ase_obj.get_chemical_formula()}{surface_facet}')
    os.makedirs(tmp_subdir, exist_ok=True)

    slab_poscar_file = os.path.join(tmp_subdir, 'POSCAR')
    io.write(slab_poscar_file, slab_ase_obj, format='vasp')

    # Generate input files for DockonSurf
    slab_active_sites = get_act_sites(slab_ase_obj, surface_facet)
    slab_lattice = slab_ase_obj.get_cell().lengths()
    if len(molec_ase_obj) <= 6:
        min_height = 1.7
        max_height = 1.8
        increment = 0.1
    elif 6 < len(molec_ase_obj) <= 8:
        min_height = 1.8
        max_height = 2.0
        increment = 0.1
    else:
        min_height = 2.4
        max_height = 3.2
        increment = 0.2
    t00 = time.time()
    total_config_list = []
    for ads_height in arange(min_height, max_height, increment):
        ads_height = '{:.2f}'.format(ads_height)
        for active_site, site_idxs in slab_active_sites.items():
            if site_idxs != []:
                gen_docksurf_file(tmp_subdir, 
                                intermediate_code, 
                                molec_ase_obj, 
                                connect_sites_molec, 
                                slab_poscar_file, 
                                slab_lattice, 
                                active_site, 
                                site_idxs, 
                                ads_height)
        # Run DockonSurf
        for root, _, files in os.walk(tmp_subdir):
            for file in files:
                if file.endswith(".inp"):
                    try:
                        file_path = os.path.join(root, file)
                        ads_list = dos.dockonsurf(os.path.abspath(file_path))
                        total_config_list.extend(ads_list)
                    except:
                        continue

    print('DockonSurf run time: {:.2f} s'.format(time.time()-t00))
    print('Number of detected adsorption configurations: ', len(total_config_list))  
    t_000 = time.time()
    # Removing the metal atoms with selective dynamics == False
    fixed_atms_idxs = slab_ase_obj.todict().get('constraints', None)[0].get_indices()
    fixed_atms = slab_ase_obj[np.isin(range(len(slab_ase_obj)), fixed_atms_idxs)]
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
    print('ads_graph_list: ', ads_graph_list[0])
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




# def bottom_poscar(axis: int, run_directory: str) -> None:
#     """Drags the structure to the bottom along the selected axis.

#     Parameters
#     ----------
#     axis : int
#         Axis along which the structure will be dragged to the bottom.
#         0 = x-axis,
#         1 = y-axis,
#         2 = z-axis
#     run_directory : str
#         Path to the directory (folder) where the POSCAR files are located.
#     Returns
#     -------
#     None
#     """
#     for root, dirs, files in os.walk(run_directory):
#         init_path = os.getcwd()
#         for file in files:
#             if file == "POSCAR":
#                 os.chdir(root)
#                 pos=io.read(file)
#                 pos.wrap()
#                 coord=pos.get_positions()[:,axis]
#                 bottom = min(coord)
#                 pos.positions[:,axis]=coord - bottom
#                 pos.write("POSCAR")
#                 os.chdir(init_path)
