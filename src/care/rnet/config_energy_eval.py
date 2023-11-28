from ase import Atoms
import numpy as np
from torch_geometric.loader import DataLoader
from torch.nn import Module
from care.gnn_eads.create_pyg_dataset import atoms_to_data
from care.rnet.networks.surface import Surface


def energy_eval_config(config_dict: dict[str, Atoms | float | float],
                     surface: Surface, 
                     model: Module, 
                     graph_params: dict, 
                     model_elems: list) -> dict:
    """Evaluates the energy of the adsorption configurations.

    Parameters
    ----------
    config_dict : Atoms
        Adsorption configuration to evaluate
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
    config = config_dict['ase']

    slab_copy = surface.slab.copy()
    # Preparing the components for the energy evaluation
    # Removing the metal atoms with selective dynamics == False
    fixed_atms_idxs = slab_copy.todict().get('constraints', None)[0].get_indices()

    # Removing the atoms which indices are in fixed_atms
    config = config[~np.isin(range(len(config)), fixed_atms_idxs)]

    ads_pyg_data = atoms_to_data(config, graph_params, model_elems)

    loader = DataLoader([ads_pyg_data], batch_size=len(ads_pyg_data), shuffle=False)
    for batch in loader:
        energy_list = model(batch)  # unitless (scaled values)
        mean_tensor = energy_list.mean * model.scaling_params['std'] + model.scaling_params['mean'] # eV
        std_tensor = energy_list.scale * model.scaling_params['std'] # eV
        
    # Transforming the tensor to numpy array
    mean_tensor = mean_tensor.detach().numpy()
    std_tensor = std_tensor.detach().numpy()

    # Updating the energy and std of the adsorption configuration
    config_dict['mu'] = mean_tensor.item()
    config_dict['s'] = std_tensor.item()

    print(f"Configuration: {config_dict['ase'].get_chemical_formula()}")
    print(f"Energy of the adsorption configuration: {config_dict['mu']} eV")
    print(f"Standard deviation of the adsorption configuration: {config_dict['s']} eV")


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