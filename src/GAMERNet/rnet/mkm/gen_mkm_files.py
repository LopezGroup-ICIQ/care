import sys
sys.path.insert(0, "../")
import os
import utilities.additional_funcs as af
import pickle


def gen_mkm_files(one_net, gb_rxn, SYSTEM, path):
    ts_list=[]
    for ts in one_net.t_states:
        ts_list.append(ts.code)

    mkm_folder = f'{SYSTEM}/data/mkm'
    mkm_path = os.path.join(path, mkm_folder)
    
    os.makedirs(mkm_path, exist_ok=True)
    af.mkm_g_file_TS(one_net, filename=f'{mkm_path}/g_{SYSTEM}.mkm')
    t_state_array = af.mkm_rxn_file(one_net, gb_rxn, filename=f'{mkm_path}/rm_{SYSTEM}.mkm')
    af.print_intermediates(one_net, filename=f'{mkm_path}/intermediates_{SYSTEM}.dat')
    af.print_t_states(one_net, filename=f'{mkm_path}/ts_{SYSTEM}.dat')

    with open(f'{mkm_path}/org_rxn_mkm_{SYSTEM}.obj', 'wb') as outfile:
        pickle.dump(one_net, outfile)