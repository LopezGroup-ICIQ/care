# Functions to place the adsorbate on the surface

import time
import ase
from numpy import max, arange
from GAMERNet.rnet.graphs.graph_fn import connectivity_analysis, ase_coord_2_graph
from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.surface import Surface
from pymatgen.io.ase import AseAtomsAdaptor
import GAMERNet.rnet.dock_ads_surf.dockonsurf.dockonsurf as dos
from ase import Atoms

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

def ads_placement(intermediate: Intermediate, 
                 surface: Surface) -> list[Atoms]:
    """Generate a set of adsorption structures for a given intermediate and surface.

    Parameters
    ----------
    intermediate : Intermediate
        Intermediate.
    surface : Surface
        Surface.

    Returns
    -------
    list
        List of adsorption structures.
    """

    # 1) Load the Intermediate ASE Atoms object and the Surface facet
    molec_ase_obj = intermediate.molecule
    surface_facet = surface.facet

    # 2) Convert the ASE Atoms object to a graph and get the potential molecular centers for adsorption
    molec_graph = ase_coord_2_graph(molec_ase_obj, coords=True)
    connect_sites_molec = connectivity_analysis(molec_graph)

    # Check if molecule fits on reference metal slab, if not scale the surface
    tolerance = 3.0   # Angstrom
    molec_dist_mat = molec_ase_obj.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    condition = surface.slab_diag - tolerance > max_dist_molec
    if condition:
            aug_slab = surface.slab
    else:
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
    total_config_list = []

    if len(molec_ase_obj) > 1:
        min_height = 2.5
        max_height = 2.8
        increment = 0.1

        if molec_ase_obj.get_chemical_formula() == 'H2':
            min_height = 1.8
            max_height = 2.4
            increment = 0.1

        t00 = time.time()
        for ads_height in arange(min_height, max_height, increment):
            ads_height = '{:.2f}'.format(ads_height)
            for _, site_idxs in active_sites.items():
                if site_idxs != []:
                    inp_vars = generate_inp_vars(molec_ase_obj, 
                                    aug_slab,
                                    ads_height,
                                    1.2,
                                    2, 
                                    1.5,
                                    connect_sites_molec,
                                    site_idxs,)
                    # Run DockonSurf
                    config_list = dos.dockonsurf(inp_vars)
                    total_config_list.extend(config_list)
        
        return total_config_list
    else:
        # for all sites, add the atom to the site
        for site in surface.active_sites:
            atoms = aug_slab.copy()
            # append unique atom in the defined position in active_sites
            atoms.append(molec_ase_obj[0])
            atoms.positions[-1] = site['position']
            atoms.set_cell(surface.slab.get_cell())
            atoms.set_pbc(surface.slab.get_pbc())
            total_config_list.append(atoms)
        return total_config_list