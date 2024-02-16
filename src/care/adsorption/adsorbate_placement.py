# Functions to place the adsorbate on the surface

from typing import Any

import networkx as nx
from ase import Atoms
from numpy import max
from pymatgen.io.ase import AseAtomsAdaptor
import random

import care.adsorption.dockonsurf.dockonsurf as dos
from care import Intermediate, Surface
from care.constants import BOND_ORDER


def connectivity_analysis(graph: nx.Graph) -> list[int]:
    """
    Performs a connectivity analysis of the molecule. Returns a list of potential anchoring atoms.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the molecule.

    Returns
    -------
    list[int]
        List with the number of connections for each atom in the molecule.
    """

    max_conns = BOND_ORDER

    unsat_elems = [
        node
        for node in graph.nodes()
        if graph.degree(node) < max_conns.get(graph.nodes[node]["elem"], 0)
    ]
    if not unsat_elems:
        # If the molecule is H2, return the list of atoms
        if (
            len(graph.nodes()) == 2
            and graph.nodes[0]["elem"] == "H"
            and graph.nodes[1]["elem"] == "H"
        ):
            return list(set(graph.nodes()))
        else:
            sat_elems = [
                node for node in graph.nodes() if graph.nodes[node]["elem"] != "H"
            ]
            return list(set(sat_elems))

    # Specifying the carbon monoxide case
    elif len(graph.nodes()) == 2 and (
        (graph.nodes[0]["elem"] == "C" and graph.nodes[1]["elem"] == "O")
        or (graph.nodes[0]["elem"] == "O" and graph.nodes[1]["elem"] == "C")
    ):
        # Extracting only the Carbon atom index
        unsat_elems = [
            node for node in graph.nodes() if graph.nodes[node]["elem"] == "C"
        ]
        return list(set(unsat_elems))

    # Specifying the case for CO2
    elif len(graph.nodes()) == 3 and (
        (
            graph.nodes[0]["elem"] == "C"
            and graph.nodes[1]["elem"] == "O"
            and graph.nodes[2]["elem"] == "O"
        )
        or (
            graph.nodes[0]["elem"] == "O"
            and graph.nodes[1]["elem"] == "O"
            and graph.nodes[2]["elem"] == "C"
        )
        or (
            graph.nodes[0]["elem"] == "O"
            and graph.nodes[1]["elem"] == "C"
            and graph.nodes[2]["elem"] == "O"
        )
    ):
        # Extracting only the Carbon atom index
        unsat_elems = [
            node for node in graph.nodes() if graph.nodes[node]["elem"] == "C"
        ]
        return list(set(unsat_elems))

    # Specifying case for H
    elif len(graph.nodes()) == 1 and graph.nodes[0]["elem"] == "H":
        return list(set(graph.nodes()))
    else:
        return list(set(unsat_elems))


def generate_inp_vars(
    adsorbate: Atoms,
    surface: Atoms,
    ads_height: float,
    max_structures: int,
    molec_ctrs: list,
    sites: list,
) -> dict[str, Any]:
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

    Returns
    -------
    dict[str, Any]
        Dictionary with the input variables.
    """
    adsorbate.set_cell(surface.get_cell().lengths())
    inp_vars = {
        "Global": True,
        "Screening": True,
        "run_type": "Screening",
        "code": "VASP",
        "batch_q_sys": "False",
        "project_name": "test",
        "surf_file": surface,
        "use_molec_file": adsorbate,
        "sites": sites,
        "molec_ctrs": molec_ctrs,
        "min_coll_height": 1.5,
        "adsorption_height": float(ads_height),
        "collision_threshold": 1.0,
        "max_structures": max_structures,
        "set_angles": "euler",
        "sample_points_per_angle": 3,
        "surf_norm_vect": "z",
        "exclude_ads_ctr": False,
        "h_acceptor": "all",
        "h_donor": False,
        "max_helic_angle": 180,
        "pbc_cell": surface.get_cell(),
        "select_magns": "energy",
        "special_atoms": "False",
        "potcar_dir": "False",
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


def ads_placement(
    intermediate: Intermediate, surface: Surface
) -> tuple[str, list[Atoms]]:
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

    # Check if molecule fits on reference metal slab. If not, increase the surface
    slab = adapt_surface(intermediate.molecule, surface)

    # Generate input files for DockonSurf
    active_sites = {
        "{}".format(site["label"]): site["indices"] for site in surface.active_sites
    }

    total_config_list = []

    if intermediate['C'] > 3:
        
        # Getting all the indices of the all the active sites
        site_idx = [site["indices"] for site in surface.active_sites]
        site_idx = list(set([idx for sublist in site_idx for idx in sublist]))
        
        ads_height = 2.5
        config_list = []
        while config_list == []:
            inp_vars = generate_inp_vars(
                adsorbate=intermediate.molecule,
                surface=slab,
                ads_height=ads_height,
                max_structures=3,
                molec_ctrs=connect_sites_molec,
                sites=site_idx,
            )
            config_list = dos.dockonsurf(inp_vars)
            ads_height += 0.2
            total_config_list.extend(config_list)
        return total_config_list

    elif 2 <= len(intermediate.molecule) <= 3:
        ads_height = (
            2.2 if intermediate.molecule.get_chemical_formula() != "H2" else 1.8
        )
        for site_idxs in active_sites.values():
            if site_idxs != []:
                config_list = []
                while config_list == []:
                    inp_vars = generate_inp_vars(
                        adsorbate=intermediate.molecule,
                        surface=slab,
                        ads_height=ads_height,
                        max_structures=1,
                        molec_ctrs=connect_sites_molec,
                        sites=site_idxs,
                    )
                    config_list = dos.dockonsurf(inp_vars)
                    ads_height += 0.2
                    total_config_list.extend(config_list)

        return total_config_list

    # If the chemical species is a single atom, placing the atom on the surface
    else:
        # for all sites, add the atom to the site
        for site in surface.active_sites:
            atoms = slab.copy()
            # append unique atom in the defined position in active_sites
            atoms.append(intermediate.molecule[0])
            atoms.positions[-1] = site["position"]
            atoms.set_cell(surface.slab.get_cell())
            atoms.set_pbc(surface.slab.get_pbc())
            total_config_list.append(atoms)

        return total_config_list
