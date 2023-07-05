"""This script is used to run GAME-Net on the web server."""

import os
import time
from multiprocessing import Pool
import resource

import GAMERNet.gnn_eads.new_web.dockonsurf.dockonsurf as dos
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from numpy import sqrt, max, arange
from numpy.linalg import norm
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.loader import DataLoader
import numpy as np

from GAMERNet.gnn_eads.new_web.adsurf.functions.act_sites import get_act_sites
from GAMERNet.gnn_eads.new_web.adsurf.functions.adsurf_fn import connectivity_analysis, gen_docksurf_file
from GAMERNet.gnn_eads.new_web.adsurf.graphs.graph_utilities import ase_2_graph
from GAMERNet.gnn_eads.src.gnn_eads.functions import atoms_to_pyggraph
from GAMERNet.gnn_eads.src.gnn_eads.nets import PreTrainedModel

PRG_FULL = "▰"
PRG_EMT = "▱"
TOT_ICN = 20

metal_structure_dict = {
    "Ag": "fcc",
    "Au": "fcc",
    "Cd": "hcp",
    "Co": "hcp",
    "Cu": "fcc",
    "Fe": "bcc",
    "Ir": "fcc",
    "Ni": "fcc",
    "Os": "hcp",
    "Pd": "fcc",
    "Pt": "fcc",
    "Rh": "fcc",
    "Ru": "hcp",
    "Zn": "hcp"
}



def wrap_atoms_to_pyggraph(args):
    return atoms_to_pyggraph(*args)

def gnn_eads_predict(molecule: Atoms,
                         metal: Atoms,
                         code: str,
                         surface_facet: str,
                         model: PreTrainedModel,
                         ) -> tuple:
    if code != '000000':
        tmp_subdir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(tmp_subdir, exist_ok=True)
        # Get the distance between the furthest atoms
        molec_dist_mat = molecule.get_all_distances(mic=True)
        max_dist_molec = max(molec_dist_mat)

        # Convert ASE object to graph and get potential molecular centers for adsorption
        molec_graph = ase_2_graph(molecule, coords=True)
        connect_sites_molec = connectivity_analysis(molec_graph)

        # Get metal slab from ASE database, given metal and surface facet
        code_path = os.path.join(tmp_subdir, code)
        os.makedirs(code_path, exist_ok=True)
        slab_poscar_file = os.path.join(f'{code_path}', "POSCAR")
        slab_ase_obj = metal
        a, b, _ = slab_ase_obj.get_cell()
        slab_diagonal = sqrt(norm(a)**2 + norm(b)**2)

        # Check if molecule fits on reference metal slab, if not scale the surface
        tolerance = 3.0   # Angstrom
        condition = slab_diagonal - tolerance > max_dist_molec
        if condition:
            ...
        else:
            counter = 1.0
            while not condition:
                counter += 1.0
                pymatgen_slab = AseAtomsAdaptor.get_structure(slab_ase_obj)
                pymatgen_slab.make_supercell([counter, counter, 1])
                slab_ase_obj = AseAtomsAdaptor.get_atoms(pymatgen_slab)
                a, b, _ = slab_ase_obj.get_cell()
                slab_diagonal = sqrt(norm(a*counter)**2 + norm(b*counter)**2)
                condition = slab_diagonal - tolerance > max_dist_molec
        
        # Write the slab in POSCAR format for DockonSurf
        write(slab_poscar_file, slab_ase_obj, format='vasp')

        # Generate input files for DockonSurf
        slab_active_sites = get_act_sites(slab_poscar_file, surface_facet)
        slab_lattice = slab_ase_obj.get_cell().lengths()
        
        total_config_list = []
        for ads_height in arange(1.5, 2.8, 0.2):
            ads_height = '{:.2f}'.format(ads_height)
            for active_site, site_idxs in slab_active_sites.items():
                if site_idxs != []:
                    gen_docksurf_file(code_path, 
                                    code, 
                                    molecule, 
                                    connect_sites_molec, 
                                    slab_poscar_file, 
                                    slab_lattice, 
                                    active_site, 
                                    site_idxs, 
                                    ads_height)
        
            # Run DockonSurf
            for root, _, files in os.walk(code_path):
                for file in files:
                    if file.endswith(".inp"):
                        try:
                            file_path = os.path.join(root, file)
                            ads_list = dos.dockonsurf(os.path.abspath(file_path))
                            total_config_list.extend(ads_list)
                        except:
                            continue

        print('Number of evaluated adsorption configurations: ', len(total_config_list))
        # Removing the metal atoms with selective dynamics == False
        fixed_atms_idxs = slab_ase_obj.todict().get('constraints', None)[0].get_indices()
        fixed_atms = slab_ase_obj[np.isin(range(len(slab_ase_obj)), fixed_atms_idxs)]
        for idx, atoms_obj in enumerate(total_config_list):
            # Removing the atoms which indices are in fixed_atms
            atoms_obj = atoms_obj[~np.isin(range(len(atoms_obj)), fixed_atms_idxs)]
            total_config_list[idx] = atoms_obj

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096*2, rlimit[1]))

        args_list = [(atoms_obj, model.g_tol, model.g_sf, model.g_metal_2nn) for atoms_obj in total_config_list]
        with Pool() as pool:
            ads_graph_list = pool.map(wrap_atoms_to_pyggraph, args_list)
        
        # Setting back the default limit
        resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

        loader = DataLoader(ads_graph_list, batch_size=len(ads_graph_list), shuffle=False)
        # Get molecule energy from GNN
        # gas_graph = atoms_to_pyggraph(
        #     molecule, model.g_tol, model.g_sf, model.g_metal_2nn)    
        # energy_molecule = model.evaluate(gas_graph)
        for batch in loader:
            energy_list = model.model(batch)* model.std + model.mean
        best_poscar = total_config_list[energy_list.argmin().item()]

        best_poscar.extend(fixed_atms)
        last_idxs_best_poscar = len(best_poscar) - len(fixed_atms)
        
        last_idxs_list = list(range(last_idxs_best_poscar, len(best_poscar)))
        fixed_atms_constr = FixAtoms(indices=last_idxs_list)
        best_poscar.set_constraint(fixed_atms_constr)

        best_ensemble = energy_list.min().item()
        eads_most_stable_conf = best_ensemble # - energy_molecule
        best_poscar.write(code_path + '/best_POSCAR')

        return eads_most_stable_conf
    else:
        return print('Surface not analyzed.')