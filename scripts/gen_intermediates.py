import multiprocessing as mp
import argparse
from pickle import dump
import os

from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

from care.rnet.intermediates_funcs import gen_alkanes_smiles, add_oxygens_to_molecule, gen_ether_smiles, gen_epoxides_smiles, rdkit_2_graph, id_group_dict, process_molecule
from care.rnet.networks.utils import gen_inter_objs
from care.rnet.utilities.functions import MolPack



def generate_intermediates(ncc: int, noc: int,) -> dict[str, dict[int, list[MolPack]]]:
    """
    Generates all the intermediates of the reaction network.

    Parameters
    ----------
    ncc : int
        Maximum number of carbon atoms in the intermediates.
    noc : int
        Maximum number of oxygen atoms in the intermediates.

    Returns
    -------
    dict[str, dict[int, list[MolPack]]]
        Dictionary containing the intermediates of the reaction network.
        each key is the smiles of a saturated CHO molecule, and each value is a dictionary   
        containing the intermediates of the reaction network for that molecule.
        Each key of the subdictionary is the number of hydrogen removed from the molecule, 
        and each value is a list of MolPack objects, each one representing a specific isomer.
    """
    
    # 1) Generate all closed-shell satuarated CHO molecules
    alkanes_smiles = gen_alkanes_smiles(ncc)    
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in list(alkanes_smiles)]
    
    cho_smiles = [add_oxygens_to_molecule(mol, noc) for mol in mol_alkanes] 
    cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]  # flatten list of lists
    #epoxides_smiles = gen_epoxides_smiles(mol_alkanes, noc)
    ethers_smiles = gen_ether_smiles(mol_alkanes, noc)
    #cho_smiles += epoxides_smiles
    cho_smiles += ethers_smiles
    cho_smiles += ['CO','C(O)O','O', 'OO', '[H][H]'] 
    mol_cho = [Chem.MolFromSmiles(smiles) for smiles in cho_smiles]

    intermediates_formula = [rdMolDescriptors.CalcMolFormula(intermediate) for intermediate in mol_alkanes + mol_cho]
    intermediates_graph = [rdkit_2_graph(intermediate) for intermediate in mol_alkanes + mol_cho]
    intermediates_smiles = [Chem.MolToSmiles(intermediate) for intermediate in mol_alkanes + mol_cho]
    
    if len(intermediates_smiles) == len(intermediates_formula) == len(intermediates_graph) == len(mol_alkanes + mol_cho):
        inter_precursor_dict = {
            smiles: {
                'Smiles': smiles,
                'Formula': formula, 
                'Graph': graph, 
                'RDKit': rdkit_obj
            } 
            for smiles, formula, graph, rdkit_obj in zip(intermediates_smiles, intermediates_formula, intermediates_graph, mol_alkanes + mol_cho)
        }
    else:
        print("Error: Lists are not of equal length.")
        
    isomeric_groups = id_group_dict(inter_precursor_dict)  # Define specific labels for isomers
    # isomeric_groups = optimized_id_group_dict(inter_precursor_dict)  # Define specific labels for isomers    

    # 2) Generate all possible open-shell intermediates by H abstraction
    repeat_molec = set()
    args_list = []
    for smiles, iso_smiles in isomeric_groups.items():
        for molec_grp in iso_smiles:
            if molec_grp not in repeat_molec:
                repeat_molec.add(molec_grp)
                args_list.append([smiles, molec_grp, inter_precursor_dict, isomeric_groups])
    
    num_cores = mp.cpu_count()
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_molecule, args_list)
    results = {item[0]: item[1] for item in results}
    inter_objs_dict = gen_inter_objs(results)
    intermediates_dict = {}
    for item in inter_objs_dict.keys():
        select_net = inter_objs_dict[item]
        intermediates_dict.update(select_net['intermediates'])
    return intermediates_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate intermediates of the reaction network.')
    parser.add_argument('-ncc', type=int, help='Maximum number of carbon atoms in the intermediates.', dest='ncc')
    parser.add_argument('-noc', type=int, help='Maximum number of oxygen atoms in the intermediates.', dest='noc', default=-1)
    args = parser.parse_args()

    output_dir = f'C{args.ncc}_O{args.noc}'
    os.makedirs(output_dir, exist_ok=True)

    intermediates = generate_intermediates(args.ncc, args.noc)
    with open(f'{output_dir}/intermediates.pkl', 'wb') as f:
        dump(intermediates, f)

    print(f'Generated {len(intermediates)} intermediates of the C{args.ncc}O{args.noc} reaction network and saved in {output_dir}/intermediates.pkl')
    