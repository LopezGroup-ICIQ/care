"""Convert DFT samples of the FG dataset to graph formats"""

from gnn_eads.functions import get_tuples, export_tuples, geometry_to_graph_analysis


def create_graph_datasets(graph_settings: dict, 
                          paths_dict: dict):
    """Convert DFT FG-dataset to raw graph dataset. 
    For each chemical family in the FG-dataset, a text file "pre_xx_bool_xxx.dat" is created. "xx" refers to 
    the voronoi tol, "bool" defines whether the 2-hop metal atoms neighbours are included in the graphs (True/False),
    and "xxx" is the scaling factor applied to the metals atomic radii.
    Example: "pre_05_False_15.dat" : Voronoi tolerance of 0.5 Angstrom, 
    No 2-hop metal neighbours included and a scaling factor of 1.5 for the metal atomic radii.
    Args:
        graph_settings (dict): {"voronoi_tol":0.5, "scaling_factor":1.5, "second_order_nn": False}
            voronoi_tolerance (float): Tolerance of the tessellation algorithm for edge creation
            second_order_nn (bool): Whether to include the 2-hop metal atoms neighbours. 
            scaling_factor (float): Scaling factor applied to metal atomic radii
        paths_dict (dict): Data paths
    Returns:
       bad_samples, tot_samples (tuple[int]): total number of wrong adsorption graphs and samples in the 
                                              whole FG-dataset.
    """
    bad_samples = 0
    tot_samples = 0
    voronoi_tolerance = graph_settings["voronoi_tol"]
    second_order_nn = graph_settings["second_order_nn"]
    scaling_factor = graph_settings["scaling_factor"]
    print("GEOMETRY -> GRAPH CONVERSION")
    print("Voronoi tolerance = {} Angstrom".format(voronoi_tolerance))
    print("2-hop metal neighbours = {}".format(second_order_nn))
    print("Scaling factor = {}".format(scaling_factor))
    chemical_families = list(paths_dict.keys())
    chemical_families.remove('metal_surfaces')
    for chemical_family in chemical_families:
        tuple = get_tuples(chemical_family, 
                           voronoi_tolerance,
                           second_order_nn,
                           scaling_factor,
                           paths_dict)
        export_tuples(paths_dict[chemical_family]['dataset'],
                      tuple)  # Generate pre_xx_bool_xxx.dat 
        x = geometry_to_graph_analysis(chemical_family, paths_dict)
        bad_samples += x[0]
        tot_samples += x[2]
    print("Bad conversions: {}/{} ({:.2f}%)".format(bad_samples, tot_samples, bad_samples*100/tot_samples))
    return bad_samples, tot_samples