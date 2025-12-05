"""
Module containing tools to place molecules on surfaces.
These include DockOnSurf and ASE functionalities.
"""
from collections import defaultdict
from typing import Any, List

from acat.adsorption_sites import SlabAdsorptionSites
from acat.settings import CustomSurface
from ase import Atoms
from ase.build import add_adsorbate
from ase.constraints import FixAtoms
import networkx as nx
import numpy as np
from numpy import max
from pymatgen.io.ase import AseAtomsAdaptor

import care.adsorption.dockonsurf.dockonsurf as dos
from care.crn.utils.species import atoms_to_graph
from care import Intermediate, Surface, BOND_ORDER, CORDERO


def connectivity_analysis(graph: nx.Graph) -> List[int]:
    """
    Performs a connectivity analysis of the molecule. Returns a list of potential anchoring atoms.

    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the molecule.

    Returns
    -------
    list[int]
        List of atom indices that are potential anchoring atoms.
    """
    unsat_elems = [
        node
        for node in graph.nodes()
        if graph.degree(node) < BOND_ORDER.get(graph.nodes[node]["elem"], 0)
    ]
    
    # H
    if len(graph) == 1 and graph.nodes[0]["elem"] == "H":
        return list(graph.nodes())
    
    if unsat_elems:
        if len(graph) == 2 and set(graph.nodes[n]["elem"] for n in graph.nodes()) == {"C", "O"}:
            # CO
            return [n for n in unsat_elems if graph.nodes[n]["elem"] == "C"]
        
        if len(graph) == 3 and [graph.nodes[n]["elem"] for n in graph.nodes()].count("O") == 2 and [graph.nodes[n]["elem"] for n in graph.nodes()].count("C") == 1:
            # CO2
            return [n for n in unsat_elems if graph.nodes[n]["elem"] == "C"]
        return unsat_elems

    if len(graph) == 2 and all(graph.nodes[node]["elem"] == "H" for node in graph.nodes()):
        # H2
        return list(graph.nodes())

    # fully-saturated molecule
    sat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] != "H"]
    
    if any(graph.nodes[node]["elem"] == "O" for node in graph.nodes()):
        return [node for node in sat_elems if graph.nodes[node]["elem"] == "O"]
    return sat_elems


def generate_inp_vars(
    adsorbate: Atoms,
    surface: Atoms,
    ads_height: float,
    max_structures: int,
    molec_ctrs: list,
    sites: list,
) -> dict[str, Any]:
    """
    Generates input for DockOnSurf.

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
    return {
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

def adapt_surface(molec_ase: Atoms, 
                  surface: Surface, 
                  tolerance: float = 2.0) -> Atoms:
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

def get_active_sites(surface: Surface) -> list[dict]:
    """
    Get surface active sites with ACAT.
    """
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

def place_adsorbate(
    intermediate: Intermediate, 
    surface: Surface, 
    num_configs: int,
    surface_sites: list[int] | None = None,
    adsorbate_atom: int | None = None,
    min_ads_height: float = 2.0,
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
    surface_sites : list[int] | None
        List of surface site indices to consider for adsorption. If None, active sites found by 
        DockOnSurf will be used.
    adsorbate_atoms : list[int] | None
        List of adsorbate atom indices to consider for adsorption. If None, input adsorbate
        configuration will be used.
    min_ads_height : float
        Minimum adsorption height in Angstrom. Default is 2.0 Angstrom.

    Returns
    -------
    total_config_list : list[Atoms]
        List of Atoms objects with the initial adsorption structures.
    """
    adsorptions = []
    n_slab = len(surface.slab)
    n_adsorbate = len(intermediate.molecule)
    slab = adapt_surface(intermediate.molecule, surface)

    if surface_sites:
        if all(x in surface.surface_atoms for x in surface_sites):
            active_sites = [surface_sites]
        else:
            raise ValueError("Some specified atoms indices do not belong to the surface.")
    else:
        active_sites = [site["indices"] for site in get_active_sites(surface)]
    
    configs = intermediate.gen_gas_configs()
    if adsorbate_atom:
        if adsorbate_atom < 0 or adsorbate_atom >= n_adsorbate:
            raise ValueError("adsorbate_atom index is out of bounds.")
        anchor_atoms = [adsorbate_atom]
    else:
        anchor_atoms = [connectivity_analysis(atoms_to_graph(x)) for x in configs]

    try:
        if n_adsorbate > 1:
            if n_adsorbate <= 10:
                ads_height = 1.8 if intermediate.formula != "H2" else 1.5
            else:
                ads_height = min_ads_height
            for active_site in active_sites:
                for config in configs:
                    config_list = []
                    while config_list == []:
                        x = generate_inp_vars(
                            adsorbate=config,
                            surface=slab,
                            ads_height=ads_height,
                            max_structures=3 if n_adsorbate >10 else 1,
                            molec_ctrs=anchor_atoms,
                            sites=active_site,
                        )
                        config_list = dos.dockonsurf(x)
                        ads_height += 0.2 if n_adsorbate > 10 else 0.1
                        adsorptions.append(config_list)     
        else:
            for active_site in active_sites:
                atoms = slab.copy()
                atoms.append(intermediate.molecule[0])
                site_pos = active_site["position"] + [0, 0, CORDERO[slab.get_chemical_symbols()[0]]]
                atoms.positions[-1] = site_pos
                atoms.set_cell(surface.slab.get_cell())
                atoms.set_pbc(surface.slab.get_pbc())
                adsorptions.append(atoms)
    except Exception as e:
        mol_index = 0  if adsorbate_atom is None else adsorbate_atom
        positions = []
        if surface_sites:
            x_pos = [surface.slab.get_positions()[idx][0] for idx in surface_sites]
            y_pos = [surface.slab.get_positions()[idx][1] for idx in surface_sites]
            x, y = sum(x_pos) / len(x_pos), sum(y_pos) / len(y_pos)
            positions.append((x, y))
        else:
            num_configs = num_configs if num_configs != -1 else 5
            for configuration in range(num_configs):
                x = surface.slab.get_cell()[0, 0] / (num_configs+1) * configuration
                y = surface.slab.get_cell()[1, 1] / (num_configs+1) * configuration
                positions.append((x, y))
        for xy in positions:
            adsorption = surface.slab.copy()
            add_adsorbate(adsorption, intermediate.molecule, min_ads_height, position=xy, mol_index=mol_index)
            adsorptions.append(adsorption)
    if num_configs != -1:
        if isinstance(adsorptions[0], Atoms):
            adsorptions = adsorptions[:num_configs]
        else:
            new_adsorptions, idx = [], 0
            while len(new_adsorptions) < num_configs:
                items_at_index = [lst[idx] for lst in adsorptions if idx < len(lst)]
                if not items_at_index:
                    break
                new_adsorptions.extend(items_at_index)
                if len(new_adsorptions) > num_configs:
                    new_adsorptions = new_adsorptions[:num_configs]
                    break
                idx += 1
            adsorptions = new_adsorptions
    else:
        if isinstance(adsorptions[0], list):
            flattened_adsorptions = []
            for sublist in adsorptions:
                flattened_adsorptions.extend(sublist)
            adsorptions = flattened_adsorptions
    for ad in adsorptions:
        ad.set_array('atom_tags', [0] * n_slab + [1] * n_adsorbate, dtype=int)
        ad.set_constraint(FixAtoms(indices=surface.fixed_atoms))
    return adsorptions
