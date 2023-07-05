"""This script is used to run GAME-Net on the web server."""

import argparse
import os
import sys
import time
from multiprocessing import Pool
import resource
sys.path.insert(0, "../src")
sys.path.insert(0, "./adsurf")

import dockonsurf.dockonsurf as dos
import ase
from ase.constraints import FixAtoms
from ase.db import connect
from ase.io import read, write
from matplotlib.pyplot import savefig
from numpy import sqrt, max, arange
from numpy.linalg import norm
from pubchempy import get_compounds, Compound
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.loader import DataLoader
import numpy as np

from adsurf.functions.act_sites import get_act_sites
from adsurf.functions.adsurf_fn import connectivity_analysis, gen_docksurf_file
from adsurf.graphs.graph_utilities import ase_2_graph
from GAMERNet.gnn_eads.src.gnn_eads.functions import atoms_to_pyggraph
from GAMERNet.gnn_eads.src.gnn_eads.graph_tools import plotter
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

# def gnn_predict(atoms_obj: ase.Atoms) -> tuple:
#     """This function will return the proxy adsorption energy of the molecule on the metal surface.
#     Parameters
#     ----------
#     atoms_obj : ase.Atoms
#         ASE object of the molecule on the metal surface.
#     Returns
#     -------
#     tuple
#         Proxy adsorption energy of the molecule on the metal surface.
#     """

#     ads_graph = atoms_to_pyggraph(
#         atoms_obj, model.g_tol, model.g_sf, model.g_metal_2nn)    
#     return model.evaluate(ads_graph)

def wrap_atoms_to_pyggraph(args):
    return atoms_to_pyggraph(*args)

def gen_dockonsurf_input(molecule: str,
                         metal: str,
                         surface_facet: str,
                         OUTPUT_DIR: str,
                         model: PreTrainedModel,
                         molecule_format: str = 'name',
                         ) -> tuple:
    """Collect the inputs for the generation of adsorption configurations.

    Parameters
    ----------
    molecule : str
        name of the molecule
    metal : str
        name of the metal
    surface_facet : str
        name of the surface facet
    molecule_format : str, optional
        Name of the molecule format, by default 'name'.
        Other options are 'smiles', 'inchi', 'inchikey', 'sdf', 'formula'.
        NB: 'name' refers to the IUPAC name of the molecule.
    calc_type : str, optional
        Type of calculation, by default 'adsorption'
    ads_height : float, optional
        Adsorption height (in Angstrom) used by DockonSurf to screen potential configurations, by default 2.5

    Returns
    -------
    tmp_subdir : str
        Path to the directory containing the results of the run
    iupac_name : str
        IUPAC name of the molecule
    canonical_smiles : str
        Canonical SMILES of the molecule   
    """
    # Retrieve the molecule from PubChem
    pubchem_molecule = get_compounds(
        molecule, molecule_format, record_type='3d', listkey_count=1)[0]    
    pubchem_cid = pubchem_molecule.cid
    c = Compound.from_cid(pubchem_cid)
    canonical_smiles = c.canonical_smiles
    
    # Create output directory for the studied adsorption system
    try:
        iupac_name = c.iupac_name
        iupac_file_name = iupac_name.replace(' ', '_').replace('(', '_').replace(')', '_').replace("'", "").replace(',', '_').replace('[', '_').replace(']', '_').replace('-', '_')
    except:
        iupac_name = molecule
        iupac_file_name = iupac_name.replace(' ', '_').replace('(', '_').replace(')', '_').replace("'", "").replace(',', '_').replace('[', '_').replace(']', '_').replace('-', '_')

    tmp_subdir = os.path.join(
        OUTPUT_DIR, f"{iupac_file_name}_{metal}{surface_facet.replace('(', '_').replace(')', '').replace(' ', '')}")
    os.makedirs(tmp_subdir, exist_ok=True)
    
    # Write the molecule in xyz format
    pubchem_atoms = pubchem_molecule.atoms
    n_atoms = len(pubchem_atoms)
    molecule_xyz_file = os.path.join(tmp_subdir, iupac_file_name + ".xyz")
    with open(molecule_xyz_file, "w") as f:
        f.write(f"{n_atoms}\n\n")
        for atom in pubchem_atoms:
            f.write(f"{atom.element}\t{atom.x} {atom.y} {atom.z}\n")

    # Convert xyz file to ASE object and get the distance between the furthest atoms
    molec_ase_obj = read(molecule_xyz_file)
    molec_dist_mat = molec_ase_obj.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    print('Distance between furthest atoms in the molecule: {:.2f} Angstrom'.format(max_dist_molec))

    # Convert ASE object to graph and get potential molecular centers for adsorption
    molec_graph = ase_2_graph(molec_ase_obj, coords=True)
    connect_sites_molec = connectivity_analysis(molec_graph)

    # Get metal slab from ASE database, given metal and surface facet
    surf_db_path = './adsurf/data/metal_surfaces.db'
    surf_db = connect(surf_db_path)
    slab_poscar_file = os.path.join(tmp_subdir, "POSCAR")
    metal_struct = metal_structure_dict[metal]
    full_facet = f"{metal_struct}({surface_facet})"
    slab_ase_obj = surf_db.get_atoms(metal=metal, facet=full_facet)
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
    
    # Write the slab in POSCAR format for DockonSurf
    write(slab_poscar_file, slab_ase_obj, format='vasp')

    # Generate input files for DockonSurf
    slab_active_sites = get_act_sites(slab_poscar_file, surface_facet)
    slab_lattice = slab_ase_obj.get_cell().lengths()
    total_config_list = []
    t00 = time.time()
    for ads_height in arange(2.5, 4.0, 0.15):
        ads_height = '{:.2f}'.format(ads_height)
        for active_site, site_idxs in slab_active_sites.items():
            if site_idxs != []:
                gen_docksurf_file(tmp_subdir, 
                                iupac_file_name, 
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
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args_list = [(atoms_obj, model.g_tol, model.g_sf, model.g_metal_2nn) for atoms_obj in total_config_list]
    with Pool() as pool:
        ads_graph_list = pool.map(wrap_atoms_to_pyggraph, args_list)
    
    # Setting back the default limit
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # ads_graph_list = [atoms_to_pyggraph(atoms_obj, model.g_tol, model.g_sf, model.g_metal_2nn) for atoms_obj in total_config_list]    
    loader = DataLoader(ads_graph_list, batch_size=len(ads_graph_list), shuffle=False)
    print('Time to convert adsorption configurations to graphs: {:.2f} s'.format(time.time()-t_000))
    # Get molecule energy from GNN
    gas_graph = atoms_to_pyggraph(
        molec_ase_obj, model.g_tol, model.g_sf, model.g_metal_2nn)    
    energy_molecule = model.evaluate(gas_graph)
    t_00 = time.time()
    for batch in loader:
        energy_list = model.model(batch)* model.std + model.mean
    print('Time to evaluate adsorption configurations: {:.2f} s'.format(time.time()-t_00))
    best_poscar = total_config_list[energy_list.argmin().item()]

    best_poscar.extend(fixed_atms)
    last_idxs_best_poscar = len(best_poscar) - len(fixed_atms)
    
    last_idxs_list = list(range(last_idxs_best_poscar, len(best_poscar)))
    fixed_atms_constr = FixAtoms(indices=last_idxs_list)
    best_poscar.set_constraint(fixed_atms_constr)

    best_ensemble = energy_list.min().item()
    eads_most_stable_conf = best_ensemble - energy_molecule

    return tmp_subdir, iupac_name, canonical_smiles, total_config_list, best_poscar, best_ensemble, eads_most_stable_conf

if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description='Provide GAME-Net prediction for potential adsorption configurations on a metal surface for a specific molecule')
    parser.add_argument('-id_format', type=str, dest="id_format", default='iupac', 
                        help='format of the molecule identifier (options: name, smiles, inchi, inchikey, sdf, formula)')
    parser.add_argument('-id', '--identifier', type=str, dest="id",
                        help='identifier of the molecule')
    parser.add_argument('-m', '--metal', type=str, dest="metal",
                        help='symbol of the metal (e.g. Au)')
    parser.add_argument('-sf', '--surface_facet', type=str, dest="surface_facet",
                        help='surface facet (e.g. 111)')
    args = parser.parse_args()

    # Load GNN model on CPU
    MODEL_PATH = "../models/GAME-Net"
    model = PreTrainedModel(MODEL_PATH)
    print("GAME-Net model loaded\n")


    OUTPUT_DIR = './results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_subdir, iupac_name, canonical_smiles, total_config_list, best_poscar, best_ensemble, ener_most_stable_conf = gen_dockonsurf_input(args.id, 
                                                                    args.metal, 
                                                                    args.surface_facet, 
                                                                    OUTPUT_DIR,
                                                                    model,  
                                                                    molecule_format=args.id_format)

    print("Screening adsorption configurations ...")

    # Remove all files with .xyz extension
    for root, dirs, files in os.walk(tmp_subdir):
        for file in files:
            if file.endswith(".xyz"):
                os.remove(os.path.join(root, file))
    os.remove(os.path.join(tmp_subdir, 'POSCAR'))

    metal_surface = args.metal + '(' + args.surface_facet + ')'
    ensemble_most_stable_conf = best_ensemble
    most_stable_conf_path = best_poscar
    system_title = iupac_name + ' on ' + metal_surface
    print('\n\nSystem: ', system_title) 
    print('Canonical SMILES: ', canonical_smiles)
    print('Metal surface: {}'.format(metal_surface))
    print('Number of evaluated adsorption configurations: ', len(total_config_list))
    print('Adsorption energy (most stable configuration): {:.2f} eV'.format(ener_most_stable_conf))
    # Saving the atomic structure of the most stable configuration
    res_poscar_path = os.path.join(tmp_subdir, 'POSCAR')
    best_poscar.write(res_poscar_path)

    ads_graph = atoms_to_pyggraph(
        most_stable_conf_path, model.g_tol, model.g_sf, model.g_metal_2nn)
    plotter(ads_graph, dpi=300)
    savefig(os.path.join(tmp_subdir,"most_stable_ads_graph.png"))
    print('Time elapsed: {:.2f} s'.format(time.time()-t0))