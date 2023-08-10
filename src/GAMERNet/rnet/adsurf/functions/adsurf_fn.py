import os
import ase
import networkx as nx
from ase import io  



DOS_PATH = os.path.abspath("../dockonsurf/dockonsurf.py")

def connectivity_analysis(graph: nx.Graph) -> list:
    """This function will return a list with the number of connections for each atom in the molecule.
    Parameters
    ----------
    graph : nx.Graph
        Graph representation of the molecule.
    Returns
    -------
    list
        List with the number of connections for each atom in the molecule.
    """
    max_conns = {'C': 4, 'O': 2, 'N': 3, 'P': 5, 'S': 4}

    unsat_elems = [node for node in graph.nodes() if graph.degree(node) < max_conns.get(graph.nodes[node]["elem"], 0)]
    if not unsat_elems:
        sat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] != 'H']
        return list(set(sat_elems))
    # Specifying the carbon monoxide case
    elif len(graph.nodes()) == 2 and ((graph.nodes[0]["elem"] == 'C' and graph.nodes[1]["elem"] == 'O') or (graph.nodes[0]["elem"] == 'O' and graph.nodes[1]["elem"] == 'C')):
        # Extracting only the Carbon atom index
        unsat_elems = [node for node in graph.nodes() if graph.nodes[node]["elem"] == 'C']
        return list(set(unsat_elems))
    else:
        return list(set(unsat_elems))



def gen_docksurf_file(tmp_subdir: str, molecule_id: str, mol_obj: ase.Atoms, conn_idxs: list, slab_poscar_file: str, metal_lattice: list, activ_site: str, active_idxs: list, ads_height:float) -> None:
    """This function will generate the dockonsurf input file for the adsorbate-slab of interest.
    Parameters
    ----------
    molecule_id : str
        Name of the molecule.
    mol_obj : ase.Atoms
        ASE object of the molecule.
    conn_idxs : list
        List with the number of connections for each atom in the molecule.
    metal : str
        Name of the metal.
    metal_path : str
        Path to the metal directory.
    metal_lattice : list
        Lattice of the metal.
    run_path : str
        Path to the run directory.
    activ_site : str
        Name of the active site.
    active_idxs : list
        List with the indexes of the active sites.
    ads_height : float
        Height of the adsorbate used by DockOnSurf to screen potential configurations.
    Returns
    -------
    None
    """
    mol_obj.set_cell(metal_lattice)

    files_path = os.path.abspath('./adsurf/data')
    docksurf_template = './adsurf/data/dockonsurf.inp'
    active_idxs = list(set(active_idxs))
    if len(active_idxs) == 1:
        active_idxs = f'{active_idxs[0]}'
    else:
        active_idxs = '(' + ' '.join([str(i) for i in active_idxs]) + ')'

    if len(conn_idxs) == 1:
        str_conn_idxs = f'{conn_idxs[0]}'
    else:
        str_conn_idxs = '(' + ' '.join([str(i) for i in conn_idxs]) + ')'
    
    # Generating the poscar directory with the dockonsurf poscars
    poscar_directory = f"{tmp_subdir}/poscar_docksurf"
    os.makedirs(poscar_directory, exist_ok=True)
    
    # Generating the directories for the dockonsurf calculations
    # These folders contain the inputs and logs of the dockonsurf run and all the outputs generated on the dockonsurf run (screening)
    docksurf_path = f"{tmp_subdir}/dos_outputs/{activ_site}"
    os.makedirs(docksurf_path, exist_ok=True)
    io.write(f'{poscar_directory}/POSCAR', mol_obj, format='vasp')

    # Reading the template dockonsurf input and modifying it with information to run the analyses of interest
    with open(docksurf_template, 'r') as read:
        data = read.readlines()

    data[4] = f"project_name = {molecule_id}\n"
    data[15] = f"screen_inp_file = {files_path}/INCAR {files_path}/KPOINTS\n"
    data[16] = f"surf_file = {os.path.abspath(slab_poscar_file)}\n"
    data[17] = f"use_molec_file = {os.path.abspath(poscar_directory)}/POSCAR\n"
    data[19] = f"sites = {active_idxs}\n"
    data[22] = f"molec_ctrs = {str_conn_idxs}\n"
    # Check these two parameters
    data[23] = "min_coll_height = 1.5\n" # Default value
    data[26] = "adsorption_height = {}\n".format(ads_height) # Default value
    # Saving the modified dockonsurf input for each system  
    with open(f"{docksurf_path}/dockonsurf_{molecule_id}.inp", 'w') as write:
        write.writelines(data)
    return


def bottom_poscar(axis: int, run_directory: str) -> None:
    """Drags the structure to the bottom along the selected axis.

    Parameters
    ----------
    axis : int
        Axis along which the structure will be dragged to the bottom.
        0 = x-axis,
        1 = y-axis,
        2 = z-axis
    run_directory : str
        Path to the directory (folder) where the POSCAR files are located.
    Returns
    -------
    None
    """
    for root, dirs, files in os.walk(run_directory):
        init_path = os.getcwd()
        for file in files:
            if file == "POSCAR":
                os.chdir(root)
                pos=io.read(file)
                pos.wrap()
                coord=pos.get_positions()[:,axis]
                bottom = min(coord)
                pos.positions[:,axis]=coord - bottom
                pos.write("POSCAR")
                os.chdir(init_path)