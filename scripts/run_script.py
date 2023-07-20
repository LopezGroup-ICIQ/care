import os
import pickle

from ase.db import connect

from GAMERNet import DB_PATH
from GAMERNet.rnet.gen_inter_from_prod import gen_inter
from GAMERNet.rnet.gen_rxn_net import generate_rxn_net

metal_structure_dict = {
    "Ag": "fcc",
    "Au": "fcc",
    "Cd": "hcp",
    "Co": "hcp",
    "Cu": "fcc",
    "Fe": "bcc",
    "Ir": "fcc",
    "Ni": "fcc",
    "Os": "hcp",
    "Pd": "fcc",
    "Pt": "fcc",
    "Rh": "fcc",
    "Ru": "hcp",
    "Zn": "hcp"
}

# Generating the results directory
res_path = "results"

os.makedirs(res_path, exist_ok=True)

# Input of the desired final product
input_molecule_list = ['propylene']

# Input of the metal surface
metal = 'Cu'
surface_facet = '100'

# Loading metal surface from ASE database
metal_surf_db_file = DB_PATH
metal_db_path = os.path.abspath(metal_surf_db_file)

surf_db = connect(metal_db_path)

metal_struct = metal_structure_dict[metal]
full_facet = f"{metal_struct}({surface_facet})"
slab_ase_obj = surf_db.get_atoms(metal=metal, facet=full_facet)

# Generating all the possible intermediates
intermediate_dict, map_dict = gen_inter(input_molecule_list)

# Saving the map dictionary as a pickle file
with open(f"{res_path}/map_dict.pkl", "wb") as outfile:
    pickle.dump(map_dict, outfile)
    print(
        f"The intermediate map dictionary pickle file has been generated")
# Saving the intermediate dictionary as a pickle file
with open(f"{res_path}/intermediate_dict.pkl", "wb") as outfile:
    pickle.dump(intermediate_dict, outfile)
    print(
        f"The intermediate dictionary pickle file has been generated")

# Generating the reaction network
rxn_net = generate_rxn_net(slab_ase_obj, intermediate_dict, map_dict)
print('rxn_net: ', len(rxn_net.intermediates), len(rxn_net.t_states))

# Converting the reaction network as a dictionary
rxn_net_dict = rxn_net.to_dict()

# Exporting the reaction network as a pickle file
with open(f"{res_path}/rxn_net.pkl", "wb") as outfile:
        pickle.dump(rxn_net_dict, outfile)
        print(
            f"The reaction network pickle file has been generated")
