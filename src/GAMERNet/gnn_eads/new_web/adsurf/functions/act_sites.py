import sys
sys.path.insert(0, "../src")
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from GAMERNet.gnn_eads.src.gnn_eads.functions import get_voronoi_neighbourlist
from pymatgen.io.ase import AseAtomsAdaptor

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