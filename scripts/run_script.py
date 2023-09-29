import os
import pickle
import time
import argparse

from ase.db import connect
from torch import load

from GAMERNet import DB_PATH, MODEL_PATH
from GAMERNet.rnet.gen_rxn_net import generate_rxn_net
from GAMERNet.rnet.dock_ads_surf.gen_ads_surf import run_docksurf
from GAMERNet.gnn_eads.nets import UQTestNet
from GAMERNet.rnet.networks.surface import Surface

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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate reaction network with R-Nets for reaction involving C, H, O on transition metal surfaces.")
    argparser.add_argument('-ncc', type=int, dest='ncc',
                            help="Network Carbon Cutoff (ncc). It defines the maximum number of carbon atoms allowed during the reaction network generation.")
    argparser.add_argument('-m', type=str, dest='m',
                            help="Metal of interest. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn")
    argparser.add_argument('-hkl', type=str, dest='hkl',
                            help="Surface facet of the metal. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11")
    argparser.add_argument('-o', type=str, 
                           help="Output directory for the generated results")
    args = argparser.parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    time0 = time.time()
    
    # Loading surface from database
    surf_db = connect(os.path.abspath(DB_PATH))
    metal_struct = metal_structure_dict[args.m]
    full_facet = f"{metal_struct}({args.hkl})"
    surface = surf_db.get_atoms(metal=args.m, facet=full_facet)
    surface = Surface(surface, args.hkl)

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

    # Generating the reaction network
    print('\nGenerating reaction network...')
    rxn_net = generate_rxn_net(surface.slab, args.ncc)
    print('\nReaction network generated')
    print(rxn_net)
    print('\nTime taken to generate the reaction network: {:.2f} s'.format(time.time() - time0))
    # Converting the reaction network as a dictionary
    rxn_net_dict = rxn_net.to_dict()
    rxn_net_dict['ncc'] = args.ncc
    rxn_net_dict['surface'] = surface

    # Exporting the reaction network as a pickle file
    with open(f"{args.o}/rxn_net.pkl", "wb") as outfile:
            pickle.dump(rxn_net_dict, outfile)
            print(
                f"The reaction network pickle file has been generated")


    list_ase_inter = list(rxn_net.intermediates.values())
    print('\nlist_ase_inter: ', list_ase_inter)
    quit()
    for intermediate in rxn_net.intermediates.values():
        print('\nIntermediate code(formula): {}({})'.format(intermediate.code, intermediate.molecule.get_chemical_formula()))
        # If the molecule has only one atom, pass (need to see how to overcome this)
        if len(intermediate.molecule) == 1 or intermediate.code == '000000':
            continue
        best_eads = run_docksurf(intermediate, surface, model, graph_params, model_elements, args.o)
        print('best_eads: ', best_eads)

    # # Loading the reaction network from a pickle file
    # with open("results/rxn_net.pkl", "rb") as infile:
    #     rxn_net_dict = pickle.load(infile)

    # # Converting the reaction network dictionary to a reaction network object

    # rxn_net = ReactionNetwork.from_dict(rxn_net_dict)
