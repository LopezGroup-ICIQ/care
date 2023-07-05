import os
import pickle
import pprint as pp

from ase import Atoms
from ase.db import connect

from GAMERNet import DB_PATH, MODEL_PATH
from GAMERNet.gnn_eads.new_web.web_script import gnn_eads_predict
from GAMERNet.gnn_eads.src.gnn_eads.nets import PreTrainedModel
from GAMERNet.rnet.gen_intermediates import gen_intermediates
from GAMERNet.rnet.organic_network import organic_network
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

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

# Inputs
input_molecule_list = ['water', 'methane', 'methanol', 'formic acid', 'ethane', 'ethanol', 'ethylene glycol', 'propane',
                       '1-propanol', '2-propanol', '1,2-propylene glycol', '1,3-propylene glycol', 'glycerol']

metal = 'Cu'
surface_facet = '100'

# Loading metal surface from ASE database
metal_surf_db_file = DB_PATH
metal_db_path = os.path.abspath(metal_surf_db_file)

surf_db = connect(metal_db_path)

metal_struct = metal_structure_dict[metal]
full_facet = f"{metal_struct}({surface_facet})"
slab_ase_obj = surf_db.get_atoms(metal=metal, facet=full_facet)

###############################################
# Generating the intermediates
intermediate_dict, map_dict = gen_intermediates(input_molecule_list)

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
###############################################

###############################################
# Generating the organic network
rxn_net = organic_network(slab_ase_obj, intermediate_dict, map_dict)



pp.pprint(intermediate_dict)
# print('N of intermediates:',len(rxn_net.intermediates))
# # pp.pprint(rxn_net.intermediates)
# print('N of TS', len(rxn_net.t_states))
# # pp.pprint(rxn_net.t_states)

# # Converting the organic network as a dictionary
# rxn_net_dict = rxn_net.to_dict()

# # Exporting the organic network as a pickle file
# with open(f"{res_path}/rxn_net.pkl", "wb") as outfile:
#         pickle.dump(rxn_net_dict, outfile)
# ###############################################

# ###############################################
# # Converting the pyRDTP molecule objects to ASE atoms objects
# ase_dict = {}
# for inter in range(len(rxn_net_dict['intermediates'])):
#     compound = rxn_net_dict['intermediates'][inter]['molecule']
#     code = rxn_net_dict['intermediates'][inter]['code']

#     compound_atoms = [atom.element for atom in compound.atoms]
#     compound_coords = compound.coords
#     compound_ase_obj = Atoms(compound_atoms, positions=compound_coords)
#     ase_dict[code] = compound_ase_obj

# # Load GNN model on CPU
# model = PreTrainedModel(MODEL_PATH)

# print('--------------------------------------')
# import time
# start = time.time()
# for code, ase_obj in ase_dict.items():
#     print(f"Predicting system {code}: {ase_obj.symbols}")
#     try:
#         energy = gnn_eads_predict(ase_obj, slab_ase_obj, code, surface_facet, model)
#         print(f"Predicted ensemble energy of {code}: {ase_obj.symbols} is {energy:.2f} eV")
#     except:
#         print(f"Could not predict energy of {code}: {ase_obj.symbols}")
#     print('--------------------------------------')
# end = time.time()
# print(f"Time taken: {end-start:.2f} s")