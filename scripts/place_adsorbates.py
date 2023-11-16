import argparse
from pickle import load, dump
import os
import multiprocessing as mp
import itertools as it

from ase.db import connect

from care import DB_PATH
from care.rnet.networks.surface import Surface
from care.rnet.networks.utils import metal_structure_dict
from care.rnet.adsorbate_placement import ads_placement

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate initial adsorption structures for the provided intermediates.")
    argparser.add_argument('-i', type=str, dest='i', help="path to the folder containing the intermediates.pkl file")
    argparser.add_argument('-m', type=str, dest='m', help="Metal of the surface. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn")
    argparser.add_argument('-hkl', type=str, dest='hkl', help="Surface facet. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11")
    args = argparser.parse_args()

    # Loading surface from database
    data_db = connect(os.path.abspath(DB_PATH))
    metal_struct = metal_structure_dict[args.m]
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
            intermediates[key].ads_configs = {'surf': {'ase': intermediate.molecule, 'energy': 0.0, 'std': 0.0}}
        elif intermediate.phase == 'gas':  # gas phase molecule
            intermediates[key].ads_configs = {'gas': {'ase': intermediate.molecule, 'energy': 0.0, 'std': 0.0}}
        else:  # adsorbed intermediate
            adsorbed_dict[key] = intermediate

    with mp.Pool(processes=os.cpu_count()//2) as p:
        result_list = p.starmap(ads_placement, iterable=zip(list(adsorbed_dict.values()), it.repeat(surface)))
    adsorption_structs = {key: value for key, value in result_list}
    
    with open(f'{output_dir}/adsorbed_configs.pkl', 'wb') as f:
        dump(adsorption_structs, f)
    print("Adsorbate placement completed. The results are saved in the folder: {}".format(output_dir))

    
