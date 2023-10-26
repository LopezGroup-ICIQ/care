from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
import multiprocessing as mp
import resource
from torch_geometric.loader import DataLoader
from torch.nn import Module
from GAMERNet.gnn_eads.create_pyg_dataset import atoms_to_data
from GAMERNet.rnet.networks.surface import Surface
import itertools as it

def process_chunk(atoms, graph_params, model_elems, calc_type):
    return [atoms_to_data(atoms, graph_params, model_elems, calc_type)]

def atoms_to_data_parallel(atoms_list, graph_params, model_elems, calc_type='adsorption'):
    # Split atoms_list into chunks
    num_cores = mp.cpu_count()
    # chunks = [(atoms_list[i::num_cores], graph_params, model_elems, calc_type) for i in range(num_cores)]

    # Use multiprocessing.Pool to parallelize the conversion
    with mp.Pool(num_cores//2) as pool:
        results = pool.starmap(process_chunk, zip(atoms_list, it.repeat(graph_params), it.repeat(model_elems), it.repeat(calc_type)))

    # Flatten the list of results and return
    return [data for sublist in results for data in sublist]

def intermediate_energy_evaluator(total_config_list: list[Atoms],
                     n_configs: int,
                     surface: Surface, 
                     model: Module, 
                     graph_params:dict, 
                     model_elems:list, ) -> dict:
    """Evaluates the energy of the adsorption configurations.

    Parameters
    ----------
    total_config_list : list[Atoms]
        List containing the adsorption configurations.
    n_configs : int
        Number of configurations to evaluate.
    surface : Surface
        Surface.
    model : Module
        Model.
    graph_params : dict
        Graph parameters.
    model_elems : list
        List of elements.

    Returns
    -------
    dict
        Dictionary containing the the key information of the n lowest configurations.
    """ 
    
    # Preparing the components for the energy evaluation
    # Removing the metal atoms with selective dynamics == False
    fixed_atms_idxs = surface.slab.todict().get('constraints', None)[0].get_indices()
    fixed_atms = surface.slab[np.isin(range(len(surface.slab)), fixed_atms_idxs)]
    for idx, atoms_obj in enumerate(total_config_list):
        # Removing the atoms which indices are in fixed_atms
        atoms_obj = atoms_obj[~np.isin(range(len(atoms_obj)), fixed_atms_idxs)]
        total_config_list[idx] = atoms_obj
    
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096*2, rlimit[1]))
    
    if len(total_config_list) >= 100:
        ads_graph_list = atoms_to_data_parallel(total_config_list, graph_params, model_elems)
    else:
        ads_graph_list = [atoms_to_data(atom, graph_params, model_elems) for atom in total_config_list]

    # Setting back the default limit
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    loader = DataLoader(ads_graph_list, batch_size=len(ads_graph_list), shuffle=False)
    for batch in loader:
        energy_list = model(batch)  # unitless (scaled values)
        mean_tensor = energy_list.mean * model.scaling_params['std'] + model.scaling_params['mean'] # eV
        std_tensor = energy_list.scale * model.scaling_params['std'] # eV
    
    # Transforming the tensor to numpy array
    mean_tensor = mean_tensor.detach().numpy()
    std_tensor = std_tensor.detach().numpy()
    
    # Sorting the array
    idxs = np.argsort(mean_tensor)
    
    # Getting the n_configs lowest energy adsorption configurations
    best_configs = [[total_config_list[idx], idx] for idx in idxs[:n_configs]]
    
    # Adding the fixed atoms to the best configurations
    ads_config_dict = {}
    counter = 0
    for best_config_idx in best_configs:
        ads_config_dict[f'config_{counter}'] = {}

        best_config = best_config_idx[0]
        idx = best_config_idx[1]

        best_config.extend(fixed_atms)
        last_idxs_best_config = len(best_config) - len(fixed_atms)
        last_idxs_list = list(range(last_idxs_best_config, len(best_config)))
        fixed_atms_constr = FixAtoms(indices=last_idxs_list)
        best_config.set_constraint(fixed_atms_constr)

        ads_config_dict[f'config_{counter}']['ase'] = best_config
        ads_config_dict[f'config_{counter}']['energy'] = mean_tensor[idx]
        ads_config_dict[f'config_{counter}']['std'] = std_tensor[idx]
        
        counter += 1  
    return ads_config_dict


def get_fragment_energy(atoms: Atoms) -> float:
    """Calculate fragment energy from closed shell structures.
    This function allows to calculate the energy of both open- and closed-shell structures, 
    keeping the same reference.
    Args:
        structure (list[int]): list of atom numbers in the order C, H, O, N, S
    Returns:
        e_fragment (float): fragment energy in eV
    """ 
    # Count elemens in the structure
    n_C = atoms.get_chemical_symbols().count('C')
    n_H = atoms.get_chemical_symbols().count('H')
    n_O = atoms.get_chemical_symbols().count('O')
    n_N = atoms.get_chemical_symbols().count('N')
    n_S = atoms.get_chemical_symbols().count('S')

    # Reference DFT energy for C, H, O, N, S
    e_H2O = -14.21877278  # O
    e_H2 = -6.76639487    # H
    e_NH3 = -19.54236910  # N
    e_H2S = -11.20113092  # S
    e_CO2 = -22.96215586  # C
    return n_C * e_CO2 + (n_O - 2*n_C) * e_H2O + (4*n_C + n_H - 2*n_O - 3*n_N - 2*n_S) * e_H2 * 0.5 + (n_N * e_NH3) + (n_S * e_H2S)