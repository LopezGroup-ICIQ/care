import os
import argparse
from torch import load
import pickle as pkl

from care import MODEL_PATH
from care.gnns.gnn_eads.nets import UQTestNet
from care.netgen.config_energy_eval import energy_eval_config, get_fragment_energy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Energy evaluation of the adsorbate structures for a given case.")
    argparser.add_argument('-i', type=str, dest='i', help="path to the folder containing the ads_intermediates.pkl file")
    argparser.add_argument('-ncores', type=int, dest='ncores', default=os.cpu_count(), help="Number of cores to use for parallelization")
    args = argparser.parse_args()
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

    with open(args.i+'/ads_intermediates.pkl', 'rb') as f:
        ads_intermediates = pkl.load(f)

    with open(args.i+'/surface.pkl', 'rb') as f:
        surface = pkl.load(f)
    
    for label, intermediate in ads_intermediates.items():
        print(intermediate.molecule.get_chemical_formula())
        if intermediate.phase == 'gas':
            # Evaluating the energy of the gas phase molecule
            e_gas = get_fragment_energy(intermediate.molecule)
            # Updating the ads_config_dict
            intermediate.ads_configs['gas']['mu'] = e_gas
        else:
            for ads_config_dict in intermediate.ads_configs.values():
                print('Evaulating intermediate: {}'.format(intermediate.code))
                # Evaluating the energy of the adsorption configurations and updating the ads_config_dict
                energy_eval_config(ads_config_dict, surface, model, graph_params, model_elements)

    with open(args.i+'/ads_intermediates.pkl', 'wb') as f:
        pkl.dump(ads_intermediates, f)