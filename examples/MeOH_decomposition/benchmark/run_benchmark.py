import os
import pandas as pd
import pickle
import time

from care import gen_blueprint, ElementaryReaction
from care.evaluators.gamenet_uq.graph import atoms_to_data
from care.evaluators.gamenet_uq.graph_filters import fragment_filter, ase_adsorption_filter
from care.evaluators.mace import MACEReactionEvaluator
from care.evaluators.ocp import OCPReactionEvaluator

ADSORBATE_ELEMENTS = ['C', 'H', 'O', 'N', 'S']

graph_params = {"structure":{"tolerance": 0.25, "scaling_factor": 1.5, "second_order": True}, "features": {"gcn": True}}


MODELS = {'gamenetuq': {'model': 'gamenetuq',
                        'name': 'None', # Model name
                        'size': 'None', # Model size (small/medium/large)
                        'device': 'None', # Device (cpu or cuda)
                        'fmax': 'None',  # Convergence criterion: if fmax lower than fmax, stop relaxation
                        'max_steps': 'None',  # Max number of ionic steps
                        'dtypes': 'None', # Data type (float32 or float64)
                        'num_configs': 3,  # Number of adsorption configurations per adsorbate-surface pair to screen
                        'dispersion': 'None',  # include dispersion correction
                        'del_traj': 'None',},  # Delete trajectory files
          'equiformerv2_31M': {'model': 'ocp',
                               'name': 'EquiformerV2-31M-S2EF-OC20-All+MD', # Model name
                               'size': 'None', # Model size (small/medium/large)
                               'device': 'cpu', # Device (cpu or cuda)
                               'fmax': 0.05,  # Convergence criterion: if fmax lower than fmax, stop relaxation
                               'max_steps': 250,  # Max number of ionic steps
                               'dtypes': 'None', # Data type (float32 or float64)
                               'num_configs': 3,  # Number of adsorption configurations per adsorbate-surface pair to screen
                               'dispersion': 'None',  # Include dispersion correction
                               'del_traj': 'True',},  # Delete trajectory files
          'macemp0_large': {'model': 'mace',
                               'name': 'None', # Model name
                               'size': 'large', # Model size (small/medium/large)
                               'device': 'cpu', # Device (cpu or cuda)
                               'fmax': 0.05,  # Convergence criterion: if fmax lower than fmax, stop relaxation
                               'max_steps': 250,  # Max number of ionic steps
                               'dtypes': 'float32', # Data type (float32 or float64)
                               'num_configs': 3,  # Number of adsorption configurations per adsorbate-surface pair to screen
                               'dispersion': 'True',  # Include dispersion correction
                               'del_traj': 'True',},  # Delete trajectory files
            }
METALS = {
    'Ag': ['111', '100', '110'],
    'Au': ['111', '100', '110'],
    'Cu': ['111', '100', '110'],
    'Ir': ['111', '100', '110'],
    'Pd': ['111', '100', '110'],
    'Pt': ['111', '100', '110'],
    'Rh': ['111', '100', '110'],
    'Ru': ['0001', '10m10', '10m11'],
}

blueprint_directory = "blueprints"
crn_directory = "crns"
mkm_directory = "mkms"

# Experimental data for the MeOH decomposition
csv_file = 'CatTestHub_Methanol_Dehydrogenation_filtered.csv'
df = pd.read_csv(csv_file)

# Operating conditions
T = 473
P = 102.73 * 1000
U = None
pH = None
oc = {'T': T, 'P': P, 'U': U, 'pH': pH}

def gen_eval_input(metal: str, facet: str, model: str,) -> str:
    """
    Generate the input TOML file for the energy evaluation of the CRNs.
    Args:
        metal: Metal.
        facet: Surface facet.
        model: Model name.
    Returns:
        INPUT_TOML: Input TOML file.
    """

    INPUT_TOML = f"""[surface]
metal = '{metal}'
hkl = '{facet}'

[evaluator]
'model'= '{MODELS[model]['model']}'
'name'= '{MODELS[model]['name']}'
'size'= '{MODELS[model]['size']}'
'device'= '{MODELS[model]['device']}'
'fmax'= {MODELS[model]['fmax']}
'max_steps'= {MODELS[model]['max_steps']}
'dtypes'= '{MODELS[model]['dtypes']}'
'num_configs'= {MODELS[model]['num_configs']}
'dispersion'= '{MODELS[model]['dispersion']}'
"""
    return INPUT_TOML

def apply_bep(reaction: ElementaryReaction):
    """
    Apply Brønsted-Evans-Polanyi (BEP) relation to the reaction energy.
    Updates the reaction object with the new activation energy.
    Args:
        reaction: ElementaryReaction object.
    Returns:
        None
    """

    oh_code = 'TUJKJAMUKRIRHC-UHFFFAOYSA-N*'

    alpha, beta = 0, 0
    bb = False
    if reaction.r_type in ['O-H', 'H-O']:
        alpha, beta = 0.24, 0.67
        bb = True
    elif reaction.r_type in ['C-H', 'H-C']:
        alpha, beta = 0.71, 0.79
        bb = True
    elif reaction.r_type in ['O-C', 'C-O']:
        C_OH_bool = False
        for prod in reaction.products:
            if prod.code == oh_code:
                C_OH_bool = True
                break
        if C_OH_bool:  # C-OH bond breakings
            alpha, beta = 0.58, 1.27
            bb = True
        else:  # C-O bond breaking
            alpha, beta = 0.66, 1.41
            bb = True

    elif reaction.r_type in ['C-C']:
        alpha, beta = 0.56, 1.30
        bb = True
    else:
        print('Reaction type to not apply BEP')
    if bb:
        reaction.e_act = (max(alpha * reaction.e_rxn[0] + beta, reaction.e_rxn[0], 0.0), 0.0)
        print('BEP applied')
    else:
        reaction.e_act = (max(reaction.e_rxn[0], 0.0), 0.0)
        print('BEP not applied')

n_cpus_blueprint = os.cpu_count()
n_cpus_eval = os.cpu_count() / 2 

def main():

    # 0. Directory management
    os.makedirs(blueprint_directory, exist_ok=True)
    os.makedirs(crn_directory, exist_ok=True)
    os.makedirs(mkm_directory, exist_ok=True)

    for metal, facets in METALS.items():
        for facet in facets:
            for model in MODELS.keys():
                os.makedirs(f"{crn_directory}/{metal}/{facet}/{model}", exist_ok=True)
                os.makedirs(f"{mkm_directory}/{metal}/{facet}/{model}", exist_ok=True)

    # 1. CRN blueprint generation for MeOH decomposition (C1O2)
    print('CRN blueprint generation....')
    ncc_meoh_decomp = 1
    noc_meoh_decomp = 2
    add_rxns_meoh_decomp = False
    electro_meoh_decomp = False

    crn_bp = gen_blueprint(ncc=ncc_meoh_decomp, noc=noc_meoh_decomp, 
                        cyclic=False, 
                        additional_rxns=add_rxns_meoh_decomp, 
                        electro=electro_meoh_decomp, 
                        num_cpu=1, 
                        show_progress=True)

    if add_rxns_meoh_decomp:
        str_addrxns = "addrxns"
    else:
        str_addrxns = "noaddrxns"

    if electro_meoh_decomp:
        str_electro = "electro"
    else:
        str_electro = "thermal"

    crn_bp_path = f'{blueprint_directory}/C{ncc_meoh_decomp}O{noc_meoh_decomp}_{str_electro}_{str_addrxns}'

    with open(f'{crn_bp_path}.pkl', 'wb') as f:
        pickle.dump(crn_bp, f)

    # 2. Input generation for the energy evaluation of the CRNs
    for metal in METALS.keys():
        for facet in METALS[metal]:
            for model in MODELS.keys():
                INPUT_TOML = gen_eval_input(metal, facet, model, crn_bp_path, crn_directory, mkm_directory)
                # Droping None values
                INPUT_TOML = '\n'.join([line for line in INPUT_TOML.split('\n') if 'None' not in line])
                input_path = f"{crn_directory}/{metal}/{facet}/{model}/input.toml"
                with open(f"{input_path}", "w") as f:
                    f.write(INPUT_TOML)

    # 3. Energy evaluation of the CRNs
    print('Energy evaluation of the CRNs....')
    for model in MODELS.keys():
        for metal in sorted(METALS.keys()):
            for facet in METALS[metal]:

                OUTPUT_NAME = f"{crn_directory}/{metal}/{facet}/{model}/eval_crn"
                input_path = f"{crn_directory}/{metal}/{facet}/{model}/input.toml"

                time0 = time.time()
                input_wd = os.path.join(os.getcwd(), input_path)
                print('input_wd: ', input_wd)
                crn_bp_wd = os.path.join(os.getcwd(), f'{crn_bp_path}.pkl')
                print('crn_bp_wd: ', crn_bp_wd)
                print('OUTPUT_NAME: ', OUTPUT_NAME) 
                os.system(f'eval_crn -i {input_wd} -bp {crn_bp_wd} -o {OUTPUT_NAME} -ncpu {n_cpus_eval}')
                time1 = time.time()
                print('Time:', time1 - time0)
                with open(f'{crn_directory}/{metal}/{facet}/{model}/time.txt', 'w') as f:
                    f.write(f'Time [s]: {time1 - time0}\n')
                    f.write(f'Num CPU cores: {n_cpus_eval}\n')
                    f.write(f'CPU info:\n')

                    os.system('lscpu >> ' + f.name)
                    f.write(f'Memory info:\n')
                    os.system('free -h >> ' + f.name)
                    f.write(f'Disk info:\n')
                    os.system('df -h >> ' + f.name)
    
    # 4. Cleaning wrong adsorbate relaxations for MACE and EquiformerV2-31M
    print('Cleaning wrong adsorbate relaxations....')
    wrong_dict_labels = {'macemp0_large': {}, 'equiformerv2_31M': {}}
    for model in wrong_dict_labels.keys():
        for metal in sorted(METALS.keys()):
            wrong_dict_labels[model][metal] = {}
            for facet in METALS[metal]:
                wrong_dict_labels[model][metal][facet] = []
                with open(f"{crn_directory}/{metal}/{facet}/{model}/eval_crn.pkl", "rb") as f:
                    crn = pickle.load(f)
                for code, inter in crn.intermediates.items():
                    for id, config in inter.ads_configs.items():
                        try:
                            pyg = atoms_to_data(config['ase'], graph_params)
                        except:
                            wrong_dict_labels[model][metal][facet].append((code, inter.formula, id))
                            continue
                        # print(pyg)
                        if not (fragment_filter(pyg, ADSORBATE_ELEMENTS) and ase_adsorption_filter(config['ase'], ADSORBATE_ELEMENTS)):
                            wrong_dict_labels[model][metal][facet].append((code, inter.formula, id))
            print(f'{model} {metal} {facet} {wrong_dict_labels[model][metal][facet]}')

    for model in wrong_dict_labels.keys():
        for metal in METALS:
            for facet in METALS[metal]:
                with open(f"{crn_directory}/{metal}/{facet}/{model}/eval_crn.pkl", "rb") as f:
                    crn = pickle.load(f)
                for code, _, id in wrong_dict_labels[model][metal][facet]:
                    print(f'{model} {metal} {facet} {code} {id}')
                    del crn.intermediates[code].ads_configs[id]
                with open(f'{crn_directory}/{metal}/{facet}/{model}/eval_crn_clean.pkl', 'wb') as f:
                    pickle.dump(crn, f)
    print('Cleaning done!')
    

    # 5. Applying BEP to the CRNs for MACE and EquiformerV2-31M
    print('Applying BEPs for MLP models....')
    for model in MODELS.keys():
        if model == 'gamenetuq':
            continue
        evaluator = MACEReactionEvaluator(crn.intermediates) if model == 'macemp0_large' else OCPReactionEvaluator(crn.intermediates)
        for metal in sorted(METALS.keys()):
            for facet in METALS[metal]:
                with open(f"{crn_directory}/{metal}/{facet}/{model}/eval_crn_clean.pkl", "rb") as f:
                    crn = pickle.load(f)
                
                for reaction in crn.reactions:
                    evaluator.eval(reaction)
                    apply_bep(reaction)

                with open(f"{crn_directory}/{metal}/{facet}/{model}/eval_crn_clean.pkl", "wb") as f:
                    pickle.dump(crn, f)
    print('BEPs applied!')

    # 6. Generating MKMs for the CRNs
    print('Generating MKMs....')
    for model in MODELS.keys():
        crn_filename = 'eval_crn' if model == 'gamenetuq' else 'eval_crn_clean'
        for metal in sorted(METALS.keys()):
            for facet in METALS[metal]:
                with open(f"{crn_directory}/{metal}/{facet}/{model}/{crn_filename}.pkl", "rb") as f:
                    crn = pickle.load(f)
                row = df.loc[(df['Metal'] == metal)].iterrows()
                for r in row:
                    row = r[1]
                    inert_gas = row['Inlet Inert Used']
                    inert_fraction = row["Inert, Inlet Mole fraction [%] "]

                    CH4O_inlet = row['Methanol, Inlet Mole fraction [%]']

                y0 = {
                    'CH4O': round(CH4O_inlet / 100, 4),
                    inert_gas: round(inert_fraction / 100, 4),
                }

                results = crn.run_microkinetic(
                iv=y0,
                oc={"T": T, "P": P, "U": U, "pH": pH},
                uq=False,
                nruns=10,
                thermo=False,
                solver='Julia',
                barrier_threshold=False,
                ss_tol=1e-10,
                tfin=1e10,
                eapp=False,
                )
                with open(f"{mkm_directory}/{metal}/{facet}/{model}/mkm.pkl", "wb") as f:
                    pickle.dump(results, f)
                with open(f"{crn_directory}/{metal}/{facet}/{model}/{crn_filename}.pkl", "wb") as f:
                    pickle.dump(crn, f)
    print('MKMs generated!')
    print('All done!')



if __name__ == "__main__":
    main()
