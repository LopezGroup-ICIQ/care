# Functions to place the adsorbate on the surface

from numpy import max, arange
from care.rnet.graphs.graph_fn import connectivity_analysis
from care.rnet.networks.intermediate import Intermediate
from care.rnet.networks.surface import Surface
from pymatgen.io.ase import AseAtomsAdaptor
import care.rnet.dock_ads_surf.dockonsurf.dockonsurf as dos
from ase import Atoms

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
    condition = surface.slab_diag - tolerance > max_dist_molec
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
        print('Reference metal slab scaled by factor {} on the x-y plane\n'.format(counter))
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

    #Get the potential molecular centers for adsorption
    connect_sites_molec = connectivity_analysis(intermediate.graph)

    # Check if molecule fits on reference metal slab. If not, scale the surface
    slab = adapt_surface(intermediate.molecule, surface)
    
    # Generate input files for DockonSurf
    active_sites = {"Site_{}".format(site["label"]): site["indices"] for site in surface.active_sites}

    # Min and max adsorption height for the adsorbate and the increment
    min_height = 2.5
    max_height = 2.8
    increment = 0.1
    # Number of configurations to generate per adsorption height
    max_structures = 1

    total_config_list = []
    # If the chemical species is not a single atom, placing the molecule on the surface using DockonSurf
    if len(intermediate.molecule) > 1:

        # For H2, the adsorption height is different
        if intermediate.molecule.get_chemical_formula() == 'H2':
            min_height = 1.8
            max_height = 2.4
            increment = 0.1

        for ads_height in arange(min_height, max_height, increment):
            ads_height = '{:.2f}'.format(ads_height)
            for site_idxs in active_sites.values():
                if site_idxs != []:
                    inp_vars = generate_inp_vars(adsorbate=intermediate.molecule, 
                                    surface=slab,
                                    ads_height=ads_height,
                                    max_structures=max_structures, 
                                    molec_ctrs=connect_sites_molec,
                                    sites=site_idxs,)
                    
                    # Run DockonSurf
                    config_list = dos.dockonsurf(inp_vars)
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