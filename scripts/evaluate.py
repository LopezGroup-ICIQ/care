import os
import argparse
from torch import no_grad
import pickle as pkl

from care import MODEL_PATH
from care.gnn import load_model
from care import ReactionNetwork


def main():
    argparser = argparse.ArgumentParser(description="Evaluate energy of the adsorbed species with GAME-Net UQ.")
    argparser.add_argument('-i', type=str, dest='i', help="Path to the folder containing the ads_intermediates.pkl file")
    argparser.add_argument('-ncores', type=int, dest='ncores', default=os.cpu_count(), help="Number of cores to use for parallelization")
    args = argparser.parse_args()

    model = load_model(MODEL_PATH)

    with open(args.i+'/ads_intermediates.pkl', 'rb') as f:
        ads_intermediates = pkl.load(f)
    with open(args.i+'/surface.pkl', 'rb') as f:
        surface = pkl.load(f)
    with open(args.i+'/../reactions.pkl', 'rb') as f:
        reactions = pkl.load(f)
    
    # Intermediate energies
    for intermediate in ads_intermediates.values():  # Intermediate      
        if intermediate.phase == 'gas':
            intermediate.ads_configs['gas']['mu'] = intermediate.ref_energy()
        else:
            with no_grad():
                for i, config in enumerate(intermediate.ads_configs.values()):  # Configuration
                    y = model(config['pyg'])  # unitless
                    config['mu'] = (y.mean * model.y_scale_params['std'] + model.y_scale_params['mean']).item()  # eV
                    config['s'] = (y.scale * model.y_scale_params['std']).item()  # eV

    # TS energy, reaction energy and barrier
    crn = ReactionNetwork(ads_intermediates, reactions)
    with no_grad():
        for reaction in crn.reactions:
            crn.calc_reaction_energy(reaction)
            if '-' in reaction.r_type:
                crn.ts_graph(reaction)
                y = model(reaction.ts_graph)
                reaction.e_ts = y.mean.item()*model.y_scale_params['std'] + model.y_scale_params['mean'], y.scale.item()*model.y_scale_params['std']
            crn.calc_reaction_barrier(reaction)
            print(reaction, reaction.r_type)
            print("Eact [eV]: N({:.2f}, {:.2f})    Erxn [eV]: N({:.2f}, {:.2f})".format(reaction.e_act[0], 
                                                                                        reaction.e_act[1], 
                                                                                        reaction.e_rxn[0],
                                                                                        reaction.e_rxn[1]))

    # Save the entire crn object
    with open(args.i+'/crn.pkl', 'wb') as f:
        pkl.dump(crn, f)


if __name__ == "__main__":
    main()

        