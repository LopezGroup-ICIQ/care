"""
Module containing tools to place molecules on surfaces.
These include DockOnSurf and ASE functionalities.
"""
from collections import defaultdict

from typing import Any

from acat.adsorption_sites import SlabAdsorptionSites
from acat.settings import CustomSurface
from ase import Atoms
from ase.build import add_adsorbate
from ase.constraints import FixAtoms
import networkx as nx
from numpy import max
from pymatgen.io.ase import AseAtomsAdaptor

import care.adsorption.dockonsurf.dockonsurf as dos
from care.crn.utils.species import atoms_to_graph
from care import Intermediate, Surface, BOND_ORDER, CORDERO
from care.crn.surface import bottom_half_indices


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

            # If there are oxygen atoms, return the list of oxygen atoms
            if "O" in [graph.nodes[node]["elem"] for node in graph.nodes()]:
                return [
                    node for node in graph.nodes()
                    if graph.nodes[node]["elem"] == "O"
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
        "min_coll_height": 0.1,
        "adsorption_height": float(ads_height),
        "collision_threshold": 1.05 if len(adsorbate) > 1 else 0.5,
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


def adapt_surface(molec_ase: Atoms, surface: Surface, tolerance: float = 2.0) -> Atoms:
    """
    Adapts the surface slab size to fit the adsorbate size
    by measuring the longest distance between atoms in the molecule and 
    the shortest side of the surface slab.

    Parameters
    ----------
    molec_ase : Atoms
        Atoms object of the molecule.
    surface : Surface
        Surface instance of the surface.
    tolerance : float
        Tolerance in Angstrom.

    Returns
    -------
    Atoms
        Atoms object of the surface.
    """
    molec_dist_mat = molec_ase.get_all_distances(mic=True)
    max_dist_molec = max(molec_dist_mat)
    condition = surface.shortest_side - tolerance > max_dist_molec
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


def place_adsorbate(
    intermediate: Intermediate, 
    surface: Surface, 
    num_configs: int,
) -> list[Atoms]:
    """
    Generate initial adsorption structures for a given intermediate/surface pair.

    Parameters
    ----------
    intermediate : Intermediate
        Intermediate.
    surface : Surface
        Surface.
    num_configs : int
        Number of configurations to generate. If set to -1, all configurations from
        DockOnSurf will be returned.

    Returns
    -------
    total_config_list : list[Atoms]
        List of Atoms objects with the initial adsorption structures.
    """
    adsorptions = []
    n_slab = len(surface.slab)
    n_adsorbate = len(intermediate.molecule)
    try:  # DockOnSurf + ACAT
        slab = adapt_surface(intermediate.molecule, surface)
        active_sites_acat = get_active_sites(surface)
        active_sites = {
            "{}".format(site["label"]): site["indices"] for site in active_sites_acat
        }
        if len(intermediate.molecule) > 10:
            site_idx = [site["indices"] for site in active_sites_acat]
            site_idx = list(set([idx for sublist in site_idx for idx in sublist]))
            ads_height = 2.0
            for site_idxs in active_sites.values():
                site_list = []
                if site_idxs != []:
                    try:
                        configs_to_place = intermediate.gen_gas_configs()
                    except AttributeError:
                        configs_to_place = [intermediate.molecule]
                    for config in configs_to_place:
                        config_graph = atoms_to_graph(config)
                        connect_sites_molec = connectivity_analysis(config_graph)
                        config_list_i = []
                        while config_list_i == []:
                            inp_vars = generate_inp_vars(
                                adsorbate=config,
                                surface=slab,
                                ads_height=ads_height,
                                max_structures=3,
                                molec_ctrs=connect_sites_molec,
                                sites=site_idxs,
                            )
                            config_list_i = dos.dockonsurf(inp_vars)
                            ads_height += 0.2
                            site_list.extend(config_list_i)
                            for ad in config_list_i:
                                ad.set_array('atom_tags', [0] * n_slab + [1] * n_adsorbate, dtype=int)
                                ad.set_constraint(FixAtoms(indices=bottom_half_indices(surface.slab)))
                adsorptions.append(site_list)
            if num_configs == -1:
                return [adsorption for sublist in adsorptions for adsorption in sublist]
            new_adsorptions = []
            index = 0
            while len(new_adsorptions) < num_configs:
                items_at_index = [lst[index] for lst in adsorptions if index < len(lst)]
                if not items_at_index:
                    break
                new_adsorptions.extend(items_at_index)
                if len(new_adsorptions) > num_configs:
                    new_adsorptions = new_adsorptions[:num_configs]
                    break
                index += 1
            return new_adsorptions            
        elif 2 <= len(intermediate.molecule) <= 10:
            ads_height = (
                1.8 if intermediate.molecule.get_chemical_formula() != "H2" else 1.5
            )
            for site_idxs in active_sites.values():
                site_list = []
                if site_idxs != []:
                    try:
                        configs_to_place = intermediate.gen_gas_configs()
                    except AttributeError:
                        configs_to_place = [intermediate.molecule]
                        
                    for config in configs_to_place:
                        config_graph = atoms_to_graph(config)
                        connect_sites_molec = connectivity_analysis(config_graph)
                        connect_sites_molec_comb = []
                        for i in range(1, len(connect_sites_molec) + 1):
                            for j in range(len(connect_sites_molec) - i + 1):
                                connect_sites_molec_comb.append(connect_sites_molec[j : j + i])
                        config_list_i = []
                        while config_list_i == []:
                            inp_vars = generate_inp_vars(
                                adsorbate=config,
                                surface=slab,
                                ads_height=ads_height,
                                max_structures=1,
                                molec_ctrs=connect_sites_molec,
                                sites=site_idxs,
                            )
                            config_list_i = dos.dockonsurf(inp_vars)
                            for ad in config_list_i:
                                ad.set_array('atom_tags', [0] * n_slab + [1] * n_adsorbate, dtype=int)
                                ad.set_constraint(FixAtoms(indices=bottom_half_indices(surface.slab)))
                            ads_height += 0.1
                            site_list.extend(config_list_i)
                adsorptions.append(site_list)
            if num_configs == -1:
                return [adsorption for sublist in adsorptions for adsorption in sublist]
            new_adsorptions = []
            index = 0
            while len(new_adsorptions) < num_configs:
                items_at_index = [lst[index] for lst in adsorptions if index < len(lst)]
                if not items_at_index:
                    break
                new_adsorptions.extend(items_at_index)
                if len(new_adsorptions) > num_configs:
                    new_adsorptions = new_adsorptions[:num_configs]
                    break
                index += 1
            return new_adsorptions                
        else:  # C*, H*, O*
            surface_atom_radii = CORDERO[slab.get_chemical_symbols()[0]]
            for site in active_sites_acat:
                atoms = slab.copy()
                atoms.append(intermediate.molecule[0])
                site_pos = site["position"] + [0, 0, surface_atom_radii]
                atoms.positions[-1] = site_pos
                atoms.set_array('atom_tags', list(atoms.get_array('atom_tags')) + [1])
                atoms.set_constraint(FixAtoms(indices=bottom_half_indices(surface.slab)))
                atoms.set_cell(surface.slab.get_cell())
                atoms.set_pbc(surface.slab.get_pbc())
                adsorptions.append(atoms)
            if num_configs == -1:
                return adsorptions
            return adsorptions[:num_configs]
    except:  # ASE (when DockOnSurf+ACAT fails on complex surfaces)
        if num_configs == -1:
            num_configs = 5  # hardcoded for now
        for configuration in range(num_configs):
            adsorption = surface.slab.copy()
            x_pos = adsorption.get_cell()[0, 0] / (num_configs+1) * configuration
            y_pos = adsorption.get_cell()[1, 1] / (num_configs+1) * configuration
            add_adsorbate(adsorption, intermediate.molecule, 2.0, position=(x_pos, y_pos))
            adsorption.set_array('atom_tags', [0] * n_slab + [1] * n_adsorbate, dtype=int)
            adsorption.set_constraint(FixAtoms(indices=bottom_half_indices(surface.slab)))
            adsorptions.append(adsorption)
        return adsorptions
    

def get_active_sites(surface) -> list[dict]:
        surf = surface.crystal_structure + surface.facet
        if surface.facet == "10m10":
            surf += "h"
        tol_dict = defaultdict(lambda: 0.5)
        tol_dict["Cd"] = 1.5
        tol_dict["Co"] = 0.75
        tol_dict["Os"] = 0.75
        tol_dict["Ru"] = 0.75
        tol_dict["Zn"] = 1.25
        if surface.facet == "10m11" or (
            surface.crystal_structure == "bcp" and surface.facet in ("111", "100")
        ):
            tol = 2.0
            sas = SlabAdsorptionSites(
                surface.slab, surface=surf, tol=tol, label_sites=True
            )
        elif surface.crystal_structure == "fcc" and surface.facet == "110":
            tol = 1.5
            sas = SlabAdsorptionSites(
                surface.slab, surface=surf, tol=tol, label_sites=True
            )
        else:
            try:
                sas = SlabAdsorptionSites(
                    surface.slab,
                    surface=surf,
                    tol=tol_dict[surface.metal],
                    label_sites=True,
                    optimize_surrogate_cell=True,
                )
            except ValueError:
                sas = SlabAdsorptionSites(
                    surface.slab,
                    surface=CustomSurface(surf),
                    tol=tol_dict[surface.metal],
                    label_sites=True,
                    optimize_surrogate_cell=True,
                )
        sas = sas.get_unique_sites()
        sas = [site for site in sas if site["position"][2] > 0.65 * surface.slab_height]
        return sas
