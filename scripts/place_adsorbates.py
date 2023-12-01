import argparse
from pickle import load, dump
import os
import multiprocessing as mp
import itertools as it

from ase.db import connect
import os

from care import DB_PATH
from care.netgen.networks.surface import Surface
from care.netgen.adsorbate_placement import ads_placement, ads_placement_graph

from care.netgen.data.constants_and_data import METAL_STRUCT_DICT


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate initial adsorption structures for the provided intermediates.")
    argparser.add_argument('-t', type=str, dest='t', help="Type of adsorbate placement. Available options: graph, dockonsurf")
    argparser.add_argument('-i', type=str, dest='i', help="path to the folder containing the intermediates.pkl file")
    argparser.add_argument('-m', type=str, dest='m', help="Metal of the surface. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn")
    argparser.add_argument('-hkl', type=str, dest='hkl', help="Surface facet. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11")
    argparser.add_argument('-ncores', type=int, dest='ncores', default=os.cpu_count(), help="Number of cores to use for parallelization")
    args = argparser.parse_args()

    # Loading surface from database
    data_db = connect(os.path.abspath(DB_PATH))
    metal_struct = METAL_STRUCT_DICT[args.m]
    full_facet = f"{metal_struct}({args.hkl})"
    surface = data_db.get_atoms(calc_type='surface',metal=args.m, facet=full_facet)
    surface = Surface(surface, args.hkl)
    output_dir = f'{args.i}/{args.m}{args.hkl}'
    os.makedirs(output_dir, exist_ok=True)

    with open(args.i+'/intermediates.pkl', 'rb') as f:
        intermediates = load(f)

    adsorbed_dict = {}
    for key, intermediate in intermediates.items():      
        if intermediate.is_surface:  # empty surface
            intermediate.ads_configs = {'surf': {'conf': intermediate.molecule, 'mu': 0.0, 's': 0.0}}
        elif intermediate.phase == 'gas':  # gas phase molecule
            intermediate.ads_configs = {'gas': {'conf': intermediate.molecule, 'mu': 0.0, 's': 0.0}}
        else:  # adsorbed intermediate
            adsorbed_dict[key] = intermediate

    if args.t == 'dockonsurf':
        with mp.Pool(processes=args.ncores) as p:
            result_list = p.starmap(ads_placement, iterable=zip(list(adsorbed_dict.values()), it.repeat(surface)))
        adsorption_structs = {key: value for key, value in result_list}
    
    elif args.t == 'graph':
        with mp.Pool(processes=args.ncores) as p:
            result_list = p.starmap(ads_placement_graph, iterable=zip(list(adsorbed_dict.values()), it.repeat(surface)))
        adsorption_structs = {key: value for key, value in result_list}

    for key, intermediate in adsorption_structs.items():
        ads_config_dict = {}
        counter = 0
        for config in adsorption_structs[key]:
            ads_config_dict[f'{counter}'] = {}
            ads_config_dict[f'{counter}']['conf'] = config
            ads_config_dict[f'{counter}']['mu'] = 0
            ads_config_dict[f'{counter}']['s'] = 0
            counter += 1  
        intermediates[key].ads_configs = ads_config_dict
    
    with open(f'{output_dir}/ads_intermediates.pkl', 'wb') as f:
        dump(intermediates, f)
    print("Adsorbate placement completed. The results are saved in the folder: {}".format(output_dir))

    with open(f'{output_dir}/surface.pkl', 'wb') as f:
        dump(surface, f)
