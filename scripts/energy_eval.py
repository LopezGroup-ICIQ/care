import os
import argparse
from torch import no_grad
import pickle as pkl

from care import MODEL_PATH
from care.gnn.functions import load_model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate energy of the adsorbed species with GAME-Net UQ.")
    argparser.add_argument('-i', type=str, dest='i', help="Path to the folder containing the ads_intermediates.pkl file")
    argparser.add_argument('-ncores', type=int, dest='ncores', default=os.cpu_count(), help="Number of cores to use for parallelization")
    args = argparser.parse_args()

    model = load_model(MODEL_PATH)

    with open(args.i+'/ads_intermediates_s.pkl', 'rb') as f:
        ads_intermediates = pkl.load(f)
    with open(args.i+'/surface.pkl', 'rb') as f:
        surface = pkl.load(f)
    
    for intermediate in ads_intermediates.values():  # Intermediate loop      
        if intermediate.phase == 'gas':
            intermediate.ads_configs['gas']['mu'] = intermediate.ref_energy()
        else:
            with no_grad():
                print('Evaulating intermediate: {}'.format(intermediate.code, intermediate.formula))
                for i, config in enumerate(intermediate.ads_configs.values()):  # Configuration loop
                    y = model(config['pyg'])  # unitless
                    config['mu'] = (y.mean * model.y_scale_params['std'] + model.y_scale_params['mean']).item()
                    config['s'] = (y.scale * model.y_scale_params['std']).item()
                    print("{}   mu: {:.2f} eV   std: {:.2f} eV".format(i+1, config['mu'], config['s']))

    with open(args.i+'/ads_intermediates.pkl', 'wb') as f:
        pkl.dump(ads_intermediates, f)