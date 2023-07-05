""" Paths to the folders containing the datasets.
NB: datasets with initial capital letter have different structures than the others."""

from pathlib import Path


def create_paths(dataset_names: list[str], 
                root: str, 
                id: str) -> dict:
    """Generate Paths for accessing Data of the FG_dataset.

    Args:
        dataset_names (list[str]): List of chemical families 
        root (str): path to Data folder
        id (str): Graph representation identifier

    Returns:
        path_dict (dict)
    """
    path_dict = {}
    for family in dataset_names:                    
        int_dict = {}
        int_dict['root'] = root / Path(family)
        int_dict['geom'] = root / Path(family + "/structures")
        int_dict['ener'] = root / Path(family + "/energies.dat")
        int_dict['dataset'] = root / Path(family + "/pre_{}".format(id))
        int_dict['dataset_p'] = root / Path(family + "/post_{}".format(id))
        path_dict[family] = int_dict
    path_dict["metal_surfaces"] = {'root': root / Path('metal_surfaces'), 
                                   'geom': root / Path('metal_surfaces/structures'), 
                                   'ener': root / Path('metal_surfaces/energies.dat')}
    return path_dict