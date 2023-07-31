import os
import ase
from ase import io, Atoms
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from GAMERNet.rnet.utilities.functions import get_voronoi_neighbourlist
from pymatgen.io.ase import AseAtomsAdaptor


from GAMERNet import DOS_PATH, DOCK_DATA

    
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
    poscar_directory = f"{tmp_subdir}/poscar_docksurf"
    os.makedirs(poscar_directory, exist_ok=True)
    
    # Generating the directories for the dockonsurf calculations
    # These folders contain the inputs and logs of the dockonsurf run and all the outputs generated on the dockonsurf run (screening)
    docksurf_path = f"{tmp_subdir}/dos_outputs/{activ_site}"
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

def get_act_sites(metal_poscar: str, surface_facet: str) -> dict:
    # TODO: Refine this function to find the active sites of different metal facets
    """Finds the active sites of a metal surface. These can be ontop, bridge or hollow sites.

    Parameters
    ----------
    metal_poscar : str
        Path to the POSCAR of the metal surface.

    Returns
    -------
    dict
        Dictionary with the atom indexes of each active site type.
    """
    surface = Structure.from_file(metal_poscar)

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
        surface.insert(dummy_idx, 'H', coord_array, coords_are_cartesian=True)

        # Convertion to ASE atoms object
        surface_ase = AseAtomsAdaptor.get_atoms(surface)
        # Find the closest atoms to the dummy atom
        tol = 0.5
        scale_factor = 1.5
        neigh_list = get_voronoi_neighbourlist(surface_ase, tol, scale_factor)
        
        # Looking for the H atom in the neighbour list
        active_site_dict[dict_label] = [i[0] for i in neigh_list if surface_ase.get_chemical_symbols().index('H') in i]
        
        # Convertion back to pymatgen structure
        new_surface = AseAtomsAdaptor.get_structure(surface_ase)

        # Removing the dummy atom by detecting the H atom
        for i in range(len(new_surface)):
            if new_surface[i].specie == Element('H'):
                dummy_idx = i
                break
        surface.remove_sites([dummy_idx])
    return active_site_dict

def run_docksurf(ase_molec: Atoms, metal_slab: Atoms):
    return




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
