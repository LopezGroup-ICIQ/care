from ase import Atoms
from ase.constraints import FixAtoms
import time
import numpy as np
import multiprocessing
import resource
from torch_geometric.loader import DataLoader
from torch.nn import Module
from GAMERNet.gnn_eads.create_pyg_dataset import atoms_to_data
from GAMERNet.rnet.networks.surface import Surface

def process_chunk(args):
    chunk, graph_params, model_elems, calc_type = args
    return [atoms_to_data(structure, graph_params, model_elems, calc_type) for structure in chunk]

def atoms_to_data_parallel(atoms_list, graph_params, model_elems, calc_type='adsorption'):
    # Split atoms_list into chunks
    num_cores = multiprocessing.cpu_count()
    chunks = [(atoms_list[i::num_cores], graph_params, model_elems, calc_type) for i in range(num_cores)]

    # Use multiprocessing.Pool to parallelize the conversion
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)

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

    t_0 = time.time()
    
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
    
    ads_graph_list = atoms_to_data_parallel(total_config_list, graph_params, model_elems)

    # Setting back the default limit
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    loader = DataLoader(ads_graph_list, batch_size=len(ads_graph_list), shuffle=False)
    print('Time to convert adsorption configurations to graphs: {:.2f} s'.format(time.time()-t_0))
    t_00 = time.time()
    for batch in loader:
        energy_list = model(batch)  # unitless (scaled values)
        mean_tensor = energy_list.mean * model.scaling_params['std'] + model.scaling_params['mean'] # eV
        std_tensor = energy_list.scale * model.scaling_params['std'] # eV
    print('Time to evaluate adsorption configurations: {:.2f} s'.format(time.time()-t_00))
    
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