import os
import pickle
import dict.dict_functions as dtfn
import io_fns.io_functions as iofn
import utilities.utilities as utfn
import utilities.functions as fn

def gen_init_network(SYSTEM: str):
    
    print("Generating initial network...")
    INPUT_DIR = "/inputs"
    OBJ_PATH= f"./{SYSTEM}/objects/"
    TMP_DIR = f"./{SYSTEM}/inputs/.tmp_db_data"
    # Recusively travels through the actual work directory to find the "input" folder
    for root, directs, files in os.walk("."):
        if root.endswith(INPUT_DIR) and SYSTEM in root:
            directory = root

    # Generating the initial dictionary
    init_dict = dtfn.formula_graph_dict(directory)

    # Checking if there are any repeated species
    check_dict = dtfn.check_graph_isomorph_dict(init_dict, directory)

    # Adding H nodes to the unsaturated systems
    saturated_dict = dtfn.add_H_nodes(check_dict)

    utfn.get_xyz_pubchem(saturated_dict, TMP_DIR)
    dtfn.compare_graphs_pubchem(saturated_dict, directory, TMP_DIR)

    updt_files = iofn.gen_file_list(directory)
    updt_dict = dtfn.formula_graph_dict(directory)

    # Classifying the molecules in family groups
    id_group_dict = dtfn.id_group_dict(updt_dict)

    # Data dictionary generated containing the pyRDTP.molecule.Molecule objects
    data_dict = dtfn.molobj_dict(directory, updt_files)

    pack_dict = {}
    map_dict = {}
    repeat_molec = []
    print("Generating packs and maps...")
    for name, molec in id_group_dict.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)

                pack = fn.generate_pack(data_dict[molec_grp], data_dict[molec_grp].elem_inf()['H'], id_group_dict[name].index(molec_grp) + 1,)
                map_tmp = fn.generate_map(pack, 'H')
                
                pack_dict[molec_grp] = pack
                map_dict[molec_grp] = map_tmp

    pack_dict = dtfn.update_pack_dict(init_dict, saturated_dict, pack_dict)
    
    pack_dict = dtfn.remove_labels(pack_dict)
    map_dict = dtfn.remove_labels(map_dict)
    
    # Saving the pack and map dictionaries
    dtfn.mv_inp_files(directory)

    # Checks if the directory "objects" exists and if not, it creates it
    os.makedirs(OBJ_PATH, exist_ok=True)

    # Output files (.obj) containing the lot and map information are generated
    with open(f"{OBJ_PATH}/pack_dict_{SYSTEM}.obj", "wb") as outfile:
        pickle.dump(pack_dict, outfile)
        print(
            f"The lot object file has been generated")

    with open(f"{OBJ_PATH}/map_{SYSTEM}.obj", "wb") as outfile:
        pickle.dump(map_dict, outfile)
        print(
            f"The map object file has been generated")
    
    
    return pack_dict, map_dict