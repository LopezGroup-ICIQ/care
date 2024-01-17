import os
import argparse
from torch import load
import pickle as pkl

from care import MODEL_PATH
from care.gnn.nets import GameNetUQ


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Energy evaluation of the adsorbate structures for a given case.")
    argparser.add_argument('-i', type=str, dest='i', help="path to the folder containing the ads_intermediates.pkl file")
    argparser.add_argument('-ncores', type=int, dest='ncores', default=os.cpu_count(), help="Number of cores to use for parallelization")
    args = argparser.parse_args()
    print('Loading GAME-Net UQ (GNN model)...')
    # Loading GAME-Net UQ 
    one_hot_encoder_elements = load(MODEL_PATH + "/one_hot_encoder_elements.pth")
    # model_hyperparams = load(MODEL_PATH + "/input.txt")  #TODO: add this to the model
    node_features_list = one_hot_encoder_elements.categories_[0].tolist()
    node_features_list.extend(["Gcn"])
    scaling_params = {"scaling":"std", "mean": -44.170277, "std": 27.821667}
    model = GameNetUQ(20, 192)
    model.load_state_dict(load(MODEL_PATH + "/GNN.pth"))
    model.scaling_params = scaling_params

    # Loading geometry->graph conversion parameters
    with open(MODEL_PATH+'/input.txt', 'r') as f:
            configuration_dict = eval(f.read())
    graph_params = configuration_dict["graph"]

    with open(args.i+'/ads_intermediates_s.pkl', 'rb') as f:
        ads_intermediates = pkl.load(f)

    with open(args.i+'/surface.pkl', 'rb') as f:
        surface = pkl.load(f)
    
    for label, intermediate in ads_intermediates.items():  # Intermediate loop      
        if intermediate.phase == 'gas':
            e_gas = intermediate.ref_energy()
            intermediate.ads_configs['gas']['mu'] = e_gas
        else:
            print('Evaulating intermediate: {}'.format(intermediate.code, intermediate.formula))
            for i, config in enumerate(intermediate.ads_configs.values()):  # Configuration loop
                y = model(config['pyg'])  # unitless
                mu = y.mean * model.scaling_params['std'] + model.scaling_params['mean'] # eV
                s = y.scale * model.scaling_params['std'] # eV
                config['mu'] = mu.item()
                config['s'] = s.item()
                print("{}   Mu: {:.2f} eV   Std: {:.2f} eV".format(i+1, config['mu'], config['s']))

    with open(args.i+'/ads_intermediates.pkl', 'wb') as f:
        pkl.dump(ads_intermediates, f)