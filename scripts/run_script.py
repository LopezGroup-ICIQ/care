import os
import pickle
import time
import argparse
import multiprocessing as mp
import itertools as it

from ase.db import connect
from torch import load

from GAMERNet import DB_PATH, MODEL_PATH
from GAMERNet.rnet.gen_rxn_net import generate_rxn_net
from GAMERNet.gnn_eads.nets import UQTestNet
from GAMERNet.rnet.networks.surface import Surface
from GAMERNet.rnet.networks.reaction_network import ReactionNetwork
from GAMERNet.rnet.adsorbate_placement import ads_placement
from GAMERNet.rnet.config_energy_eval import intermediate_energy_evaluator, get_fragment_energy

metal_structure_dict = {
    "Ag": "fcc",
    "Au": "fcc",
    "Cd": "hcp",
    "Co": "hcp",
    "Cu": "fcc",
    "Fe": "bcc",
    "Ir": "fcc",
    "Ni": "fcc",
    "Os": "hcp",
    "Pd": "fcc",
    "Pt": "fcc",
    "Rh": "fcc",
    "Ru": "hcp",
    "Zn": "hcp"
}

def process_adsorbed_intermediate(key: str, rxn_net: ReactionNetwork, surface: Surface):
    """
    Generates the adsorption configurations for a given adsorbed intermediate and surface.

    Parameters
    ----------
    key : str
        Code of the adsorbed intermediate.
    rxn_net : ReactionNetwork
        Reaction network.
    surface : Surface
        Surface.

    Returns
    -------
    key : str
        Code of the adsorbed intermediate.
    gen_ads_config : list[Atoms]
        List of adsorption configurations.
    """
    intermediate = rxn_net.intermediates[key]
    gen_ads_config = ads_placement(intermediate, surface)
    return key, gen_ads_config


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate reaction network blueprint for processes involving C, H, O on surfaces.")
    argparser.add_argument('-ncc', type=int, dest='ncc',
                            help="Network Carbon Cutoff (ncc). It defines the size of the reaction network based on the maximum number of carbon atoms in the intermediates.")
    argparser.add_argument('-m', type=str, dest='m',
                            help="Metal of the surface. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn")
    argparser.add_argument('-hkl', type=str, dest='hkl',
                            help="Surface facet. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11")
    argparser.add_argument('-o', type=str, 
                           help="Output directory for the generated results")
    args = argparser.parse_args()
    

    print('\nInitializing CARE...')
    print('Network Carbon Cutoff (ncc): {}'.format(args.ncc))
    print('Metal(facet): {}{}'.format(args.m, args.hkl))

    os.makedirs(args.o, exist_ok=True)
    
    # Loading surface from database
    print('Loading surface from database...')
    data_db = connect(os.path.abspath(DB_PATH))
    metal_struct = metal_structure_dict[args.m]
    full_facet = f"{metal_struct}({args.hkl})"
    surface = data_db.get_atoms(calc_type='surface',metal=args.m, facet=full_facet)
    surface = Surface(surface, args.hkl)

    print('Loading GAME-Net UQ (GNN model)...')
    # Loading GAME-Net UQ 
    one_hot_encoder_elements = load(MODEL_PATH + "/one_hot_encoder_elements.pth")
    node_features_list = one_hot_encoder_elements.categories_[0].tolist()
    model_elements = one_hot_encoder_elements.categories_[0].tolist()
    node_features_list.extend(["Valence", "Gcn", "Magnetization"])
    scaling_params = {"scaling":"std", "mean": -52.052330, "std": 29.274139}
    model = UQTestNet(features_list=node_features_list, scaling_params=scaling_params,
                      dim=176, N_linear=0, N_conv=3, 
                      bias=False, pool_heads=1)
    model.load_state_dict(load(MODEL_PATH + "/GNN.pth"))

    # Loading geometry->graph conversion parameters
    with open(MODEL_PATH+'/input.txt', 'r') as f:
            configuration_dict = eval(f.read())
    graph_params = configuration_dict["graph"]
    # graph_params['structure']['scaling_factor'] = 1.6  # solve this later

    print('CARE initialized successfully!\n')

    time0 = time.time()
    # Generating the reaction network
    print('Generating reaction network...')
    rxn_net = generate_rxn_net(surface.slab, args.ncc)
    print('Time taken to generate the reaction network: {:.2f} s\n'.format(time.time() - time0))

    print('Generating adsorption configurations...')
    print('It can take some time, please be patient...')
    # Evaluate energy of intermediates
    t00 = time.time()
    # List to keep keys of adsorbed intermediates
    adsorbed_keys = []
    for key, intermediate in rxn_net.intermediates.items():        
        if intermediate.is_surface == True:  # empty surface
            rxn_net.intermediates[key].ads_configs = {'surf': {'ase': intermediate.molecule, 'energy': 0.0, 'std': 0.0}}
        
        elif intermediate.phase == 'gas':  # gas phase molecule
            rxn_net.intermediates[key].ads_configs = {'gas': {'ase': intermediate.molecule, 'energy': get_fragment_energy(intermediate.molecule), 'std': 0.0}}
        
        else:  # adsorbed intermediate
            adsorbed_keys.append(key)
    # Parallelize ads_placement function for adsorbed intermediates
    with mp.get_context('spawn').Pool(processes=os.cpu_count()) as p:
        result_list = p.starmap(process_adsorbed_intermediate, iterable=zip(adsorbed_keys, it.repeat(rxn_net), it.repeat(surface)))
    print('Total number of adsorption configurations: {}\n'.format(len(result_list)))
    print('Time taken to generate the adsorption configurations for all intermediates: {:.2f} s'.format(time.time() - t00))

    # Evaluate energy of intermediates
    print('Evaluating the energies of the adsorption configurations...')
    t00 = time.time()
    conf_per_act_site = 3
    for tuple in result_list:
        key = tuple[0]
        gen_ads_config = tuple[1]
        data_dict_configs = intermediate_energy_evaluator(gen_ads_config, conf_per_act_site, surface, model, graph_params, model_elements)
        rxn_net.intermediates[key].ads_configs = data_dict_configs
    
    print('Number of evaluated configurations per active site: {}'.format(conf_per_act_site))
    print('Total number of adsorption configurations evaluated: {}\n'.format(len(result_list)*conf_per_act_site))
    print('Time taken to evaluate the energy of all the adsorption configurations: {:.2f} s'.format(time.time() - t00))
    # Export reaction network as pickle file

    print('Exporting the reaction network as a pickle file...')
    rxn_net_dict = rxn_net.to_dict()
    rxn_net_dict['ncc'] = args.ncc
    rxn_net_dict['surface'] = surface
    with open(f"{args.o}/rxn_net.pkl", "wb") as outfile:
        pickle.dump(rxn_net_dict, outfile)
        print(f"The reaction network pickle file has been generated\n")

