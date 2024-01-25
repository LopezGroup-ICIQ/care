import argparse
import itertools as it
import multiprocessing as mp
import numpy as np
import os
from pickle import dump, load
import time

from ase import Atoms
from ase.db import connect
from sklearn.preprocessing import OneHotEncoder

from care import DB_PATH, MODEL_PATH
from care.adsorption.adsorbate_placement import (ads_placement,
                                                 ads_placement_graph)
from care.constants import METAL_STRUCT_DICT, METALS, ADSORBATE_ELEMS
from care.crn.surface import Surface
from care.gnn.graph import atoms_to_data
from care.gnn.graph_tools import nx_to_pyg_adsorb


def main():
    argparser = argparse.ArgumentParser(
        description="Generate initial adsorption structures for the provided intermediates."
    )
    argparser.add_argument(
        "-i",
        type=str,
        dest="i",
        help="path to the folder containing the intermediates.pkl file",
    )
    argparser.add_argument(
        "-m",
        type=str,
        dest="m",
        help="Metal of the surface. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn",
    )
    argparser.add_argument(
        "-hkl",
        type=str,
        dest="hkl",
        help="Surface facet. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11",
    )
    argparser.add_argument(
        "-t",
        type=str,
        dest="t",
        default="s",
        help="Type of adsorbate placement. Available options: g (graph), s (dockonsurf)",
    )
    argparser.add_argument(
        "-ncores",
        type=int,
        dest="ncores",
        default=os.cpu_count(),
        help="Number of cores to use for parallelization",
    )
    args = argparser.parse_args()

    # Loading surface from database
    data_db = connect(os.path.abspath(DB_PATH))
    metal_struct = METAL_STRUCT_DICT[args.m]
    full_facet = f"{metal_struct}({args.hkl})"
    surface = data_db.get_atoms(calc_type="surface", metal=args.m, facet=full_facet)
    surface = Surface(surface, args.hkl)
    output_dir = f"{args.i}/{args.m}{args.hkl}"
    os.makedirs(output_dir, exist_ok=True)

    # Loading GNN graph conversion params
    with open(MODEL_PATH + "/input.txt", "r") as f:
        configuration_dict = eval(f.read())
    graph_params = configuration_dict["graph"]

    with open(args.i + "/intermediates.pkl", "rb") as f:
        intermediates = load(f)

    ohe_elements = OneHotEncoder().fit(
        np.array(METALS + ADSORBATE_ELEMS).reshape(-1, 1)
    )

    adsorbed_dict = {}
    for key, intermediate in intermediates.items():
        if intermediate.is_surface:  # empty surface
            intermediate.ads_configs = {
                "surf": {"conf": intermediate.molecule, "mu": 0.0, "s": 0.0}
            }
        elif intermediate.phase == "gas":  # gas phase molecule
            intermediate.ads_configs = {
                "gas": {"conf": intermediate.molecule, "mu": 0.0, "s": 0.0}
            }
        else:  # adsorbed intermediate
            adsorbed_dict[key] = intermediate

    if args.t == "s":
        with mp.Pool(processes=args.ncores) as p:
            result_list = p.starmap(
                ads_placement,
                iterable=zip(list(adsorbed_dict.values()), it.repeat(surface)),
            )
        adsorption_structs = {key: value for key, value in result_list}

    elif args.t == "g":
        print('Using graph-based adsorbate placement.')
        t0 = time.time()
        with mp.Pool(processes=args.ncores // 2) as p:
            result_list = p.starmap(
                ads_placement_graph,
                iterable=zip(list(adsorbed_dict.values()), it.repeat(surface)),
            )
        print(f'Adsorbate placement took {time.time() - t0} seconds.')
        adsorption_structs = {key: value for key, value in result_list}

    print('Retrieving adsorption structures...')
    t1 = time.time()
    for key, intermediate in adsorption_structs.items():
        ads_config_dict = {}
        counter = 0
        for config in adsorption_structs[key]:
            ads_config_dict[f"{counter}"] = {}
            ads_config_dict[f"{counter}"]["config"] = config
            ads_config_dict[f"{counter}"]["pyg"] = (
                atoms_to_data(config, graph_params)
                if isinstance(config, Atoms)
                else nx_to_pyg_adsorb(config, ohe_elements, graph_params)
            )
            ads_config_dict[f"{counter}"]["mu"] = 0
            ads_config_dict[f"{counter}"]["s"] = 0
            counter += 1
        intermediates[key].ads_configs = ads_config_dict
    print('Adsorption structures retrieved.')
    print(f'Retrieval took {time.time() - t1} seconds.')

    if args.t == "s":
        with open(f"{output_dir}/ads_intermediates.pkl", "wb") as f:
            dump(intermediates, f)
        print(
            "Adsorbate placement completed. The results are saved in the folder: {}".format(
                output_dir
            )
        )
    else:
        with open(f"{output_dir}/ads_intermediates.pkl", "wb") as f:
            dump(intermediates, f)
        print(
            "Adsorbate placement completed. The results are saved in the folder: {}".format(
                output_dir
            )
        )

    with open(f"{output_dir}/surface.pkl", "wb") as f:
        dump(surface, f)


if __name__ == "__main__":
    main()
