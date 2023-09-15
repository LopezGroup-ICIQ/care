import os
import pickle
import time
import argparse

from ase.db import connect
from torch import load

from GAMERNet import DB_PATH, MODEL_PATH
from GAMERNet.rnet.gen_inter_from_prod import gen_inter
from GAMERNet.rnet.gen_rxn_net import generate_rxn_net
from GAMERNet.rnet.dock_ads_surf.gen_ads_surf import run_docksurf
from GAMERNet.gnn_eads.nets import UQTestNet

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
    argparser.add_argument('--mol_cutoff', type=str, dest='mol_cutoff',
                            help="The biggest molecule present in the generated reaction mechanism.")
    argparser.add_argument('--metal', type=str, dest='metal',
                            help="Metal of interest. Available options: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn")
    argparser.add_argument('-hkl', type=str, dest='hkl',
                            help="Surface facet of the metal. Available options: fcc/bcc 111, 100, 110; hcp 0001, 10m10, 10m11")
    argparser.add_argument('-o', type=str, 
                           help="Output directory for the generated results")
    args = argparser.parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    time0 = time.time()
    
    backbone_carbon_class = args.mol_cutoff
    metal = args.metal
    surface_facet = args.hkl

    # Loading metal surface from ASE database
    metal_surf_db_file = DB_PATH
    metal_db_path = os.path.abspath(metal_surf_db_file)
    surf_db = connect(metal_db_path)

    metal_struct = metal_structure_dict[metal]
    full_facet = f"{metal_struct}({surface_facet})"
    slab_ase_obj = surf_db.get_atoms(metal=metal, facet=full_facet)
    slab_ase_obj.info["surface_orientation"] = full_facet

    # Load GAME-Net UQ 
    one_hot_encoder_elements = load(MODEL_PATH + "/one_hot_encoder_elements.pth")
    node_features_list = one_hot_encoder_elements.categories_[0].tolist()
    model_elements = one_hot_encoder_elements.categories_[0].tolist()
    node_features_list.append("Valence")
    node_features_list.append("Gcn")
    node_features_list.append("Magnetization")
    model = UQTestNet(features_list=node_features_list, 
                      scaling_params={"scaling":"std", "mean": -52.052330, "std": 29.274139},
                      dim=176, 
                      N_linear=0, 
                      N_conv=3, 
                      bias=False, 
                      pool_heads=1)
    model.load_state_dict(load(MODEL_PATH + "/GNN.pth"))

    # load dict from input.txt
    with open(MODEL_PATH+'/input.txt', 'r') as f:
            configuration_dict = eval(f.read())
    graph_params = configuration_dict["graph"]
    graph_params['structure']['scaling_factor'] = 1.6  # solve this later

    # Generating all the possible intermediates
    print('Generating intermediates...')
    intermediate_dict, map_dict = gen_inter(backbone_carbon_class)
    print('Time to generate intermediates: ', time.time() - time0)
    # Saving the map dictionary as a pickle file
    with open(f"{args.o}/map_dict.pkl", "wb") as outfile:
        pickle.dump(map_dict, outfile)
        print(
            f"The intermediate map dictionary pickle file has been generated")
    # Saving the intermediate dictionary as a pickle file
    with open(f"{args.o}/intermediate_dict.pkl", "wb") as outfile:
        pickle.dump(intermediate_dict, outfile)
        print(
            f"The intermediate dictionary pickle file has been generated")
    # Generating the reaction network
    print('\nGenerating reaction network...')
    rxn_net = generate_rxn_net(slab_ase_obj, intermediate_dict, map_dict)
    print('The reaction network has been generated')
    # Converting the reaction network as a dictionary
    rxn_net_dict = rxn_net.to_dict()

    # Exporting the reaction network as a pickle file
    with open(f"{args.o}/rxn_net.pkl", "wb") as outfile:
            pickle.dump(rxn_net_dict, outfile)
            print(
                f"The reaction network pickle file has been generated")


    list_ase_inter = list(rxn_net.intermediates.values())
    print('\nlist_ase_inter: ', list_ase_inter)

    for intermediate in rxn_net.intermediates.values():
        print('\nIntermediate code: ', intermediate.code)
        print('Intermediate molecule: ', intermediate.molecule)
        # If the molecule has only one atom, pass (need to see how to overcome this)
        if len(intermediate.molecule) == 1 or intermediate.code == '000000':
            continue
        best_eads = run_docksurf(intermediate, slab_ase_obj, surface_facet, model, graph_params, model_elements, args.o)
        print('best_eads: ', best_eads)

    # # Loading the reaction network from a pickle file
    # with open("results/rxn_net.pkl", "rb") as infile:
    #     rxn_net_dict = pickle.load(infile)

    # # Converting the reaction network dictionary to a reaction network object

    # rxn_net = ReactionNetwork.from_dict(rxn_net_dict)
