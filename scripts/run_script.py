import os
import pickle
import pprint as pp

from ase.db import connect

from GAMERNet import DB_PATH, MODEL_PATH
from GAMERNet.gnn_eads.new_web.web_script import gnn_eads_predict
from GAMERNet.gnn_eads.src.gnn_eads.nets import PreTrainedModel
from GAMERNet.rnet.gen_inter_from_prod import gen_inter
from GAMERNet.rnet.organic_network import generate_rxn_net
from GAMERNet.rnet.utilities import paths as pt
from GAMERNet.rnet.utilities import additional_funcs as af

import numpy as np
from graph_tool import Graph
import plotly.graph_objects as go
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from graph_tool.topology import all_paths, all_shortest_paths
from matplotlib import cm

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
input_molecule_list = ['ethane']

metal = 'Cu'
surface_facet = '100'

# Loading metal surface from ASE database
metal_surf_db_file = DB_PATH
metal_db_path = os.path.abspath(metal_surf_db_file)

surf_db = connect(metal_db_path)

metal_struct = metal_structure_dict[metal]
full_facet = f"{metal_struct}({surface_facet})"
slab_ase_obj = surf_db.get_atoms(metal=metal, facet=full_facet)

##############################################
# Generating the intermediates
intermediate_dict, map_dict = gen_inter(input_molecule_list)
pp.pprint(intermediate_dict) 

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

# # Loading the intermediate dictionary from a pickle file
# with open(f"{res_path}/intermediate_dict.pkl", "rb") as infile:
#     intermediate_dict = pickle.load(infile)
#     print(
#         f"The intermediate dictionary pickle file has been loaded")

# # Loading the map dictionary from a pickle file
# with open(f"{res_path}/map_dict.pkl", "rb") as infile:
#     map_dict = pickle.load(infile)
#     print(
#         f"The intermediate map dictionary pickle file has been loaded")

# pp.pprint(intermediate_dict)
###############################################
# Generating the organic network
rxn_net = generate_rxn_net(slab_ase_obj, intermediate_dict, map_dict)

# with open(f'{res_path}/ase_inter.txt', 'w') as file:
#     for inter in rxn_net.intermediates:
#         file.write(inter + '\n')

print(len(rxn_net.intermediates))
print(len(rxn_net.t_states))
quit()

digraph = rxn_net.gen_graph()

# terminal_nodes = []
# for node in digraph.nodes:
#     if digraph.out_degree(node) == 3:
#         terminal_nodes.append(node)

terminal_nodes = ['383101']
# Retrieve attributes of terminal nodes
terminal_node_attributes = digraph.nodes.data()
terminal_node_attributes = {node: attrs for node, attrs in terminal_node_attributes if node in terminal_nodes}

# Print terminal nodes and their attributes
for node, attributes in terminal_node_attributes.items():
    print("Terminal Node:", node)
    print("Attributes:", attributes)
    # Cheking the adjacent nodes
    print("Adjacent Nodes:", list(digraph.adj[node]))
    print('-----------------------------')
quit()

# Converting the organic network as a dictionary
rxn_net_dict = rxn_net.to_dict()

# Exporting the organic network as a pickle file
with open(f"{res_path}/rxn_net.pkl", "wb") as outfile:
        pickle.dump(rxn_net_dict, outfile)
################################################
digraph = rxn_net.gen_graph()

###############################################
# Generate Directed graphs
new_graph = pt.gen_dir_net(rxn_net)
rm_list = []

for edge in new_graph.edges():
    non_values = set(['000000', '010101', '011101', 'e-', '001101', 'H+'])#, '101101'])
    if non_values.intersection(set(edge)):
        rm_list.append(edge)
new_graph.remove_edges_from(rm_list) 

pairs = pt.generate_pairs(rxn_net)

pairs = []
for edge in new_graph.edges():
    pairs.append(edge)
pairs = np.asarray(pairs)

opt_graph = Graph(directed=True)
opt_graph_edg = opt_graph.add_edge_list(pairs, hashed=True)
##############################################

###############################################
# Generate boxes
def path_finder(input_molecule_list, rxn_net, new_graph, opt_graph, opt_graph_edg):

    CO = rxn_net.search_inter_by_elements({'C': 1, 'H': 0 ,'O': 2})[0]
    CO_idx = af.search_species(opt_graph_edg, CO.code)

    ts_hasher = {item.code: item for item in rxn_net.t_states}

    total_molecule_eners = []
    for molecule in input_molecule_list:
        if molecule == 'water':
            pass
        else:
            molecule_loc = intermediate_dict[molecule][0][0]
            molecule = molecule_loc.mol
            molecule_idx = af.search_species(opt_graph_edg, molecule_loc.code)
            molecule_paths = list(all_paths(opt_graph, molecule_idx, CO_idx,cutoff=20))
            molecule_eners = [0] * len(molecule_paths)
            for idx, item in enumerate(molecule_paths):
                molecule_eners[idx] = pt.trans_and_ener(new_graph, opt_graph_edg, np.asarray(item), ts_hasher, inter_hasher=rxn_net.intermediates, bader=True)
            molecule_eners = sorted(molecule_eners, key=lambda x: x[0])
            total_molecule_eners.extend(molecule_eners)
    
    return total_molecule_eners

total_molecule_eners = path_finder(input_molecule_list, rxn_net, new_graph, opt_graph, opt_graph_edg)

turu = nx.DiGraph()
for item in total_molecule_eners:
    reversed_item = item[2::]
    reversed_item = reversed_item[::-1]
    for index, path_element in enumerate(reversed_item):
        if path_element in rxn_net.intermediates:
            turu.add_node(path_element)
        else:
            turu.add_edge(reversed_item[index - 1], reversed_item[index + 1])

# Ethanol
min_h = 0
max_h = 5
max_o = 2

surf_hydrogen = rxn_net.intermediates['010101'].bader_energy
surf_oh = rxn_net.intermediates['011101'].bader_energy
surf_clean = rxn_net.surface[0].bader_energy

edge_width = 3
inters = []
radicals = []

for node in turu.nodes:
    inters.append(rxn_net.intermediates[node])

for edge in turu.edges:
    try:
        t_state = rxn_net.search_ts([edge[0]], final=[edge[1]])
    except KeyError:
        continue
    try:
        custom_energy = af.calc_ts_hydrogens(t_state[0],
                                          surf_hydrogen,
                                          surf_clean,
                                          min_hydrogens=min_h,
                                          max_hydrogens=max_h,
                                          max_oxygens=max_o)
    except IndexError:
        continue


edge_width = 3
inters = []
radicals = []
deleters = []

vals = [inter.energy for inter in rxn_net.intermediates.values()]
min_val = min(vals)
max_val = max(vals)
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val+1)

color_dict = {}
edge_dict = {}
for node in turu.nodes:
    if node not in rxn_net.intermediates:
        continue
    if node in deleters:
        colormap = cm.binary
    else:
        colormap = cm.inferno_r
    inters.append(rxn_net.intermediates[node])
    inter = rxn_net.intermediates[node]

    formula = af.code_mol_graph(inter.graph, ['C'])
    formula = formula.replace('(', '')
    formula = formula.replace(')', '')
    formula = formula.replace('2', '₂')
    formula = formula.replace('3', '₃')
    formula = formula.split('-')
    formula = '-'.join(formula)
    # if af.radical_calc(inter):
    #     formula = af.underline_label(formula)
    #     radicals.append(inter)
    custom_energy = inter.bader_energy + af.calc_hydrogens_energy(inter,
                                                               surf_hydrogen,
                                                               surf_clean,
                                                               min_hydrogens=min_h,
                                                               max_hydrogens=max_h,
                                                               max_oxygens=max_o)

    colors, codes = af.generate_colors(inter, colormap, norm, bader=True, custom_energy=custom_energy)
    label = af.generate_label(formula, colors, codes, html_template=af.BOX_TMP_0)
    color_dict[node] = {'label': label, 'shape': 'plaintext', 'fontname': 'Arial'}

for edge in turu.edges:

    try:
        t_state = rxn_net.search_ts([edge[0]], final=[edge[1]])

    except KeyError:
        continue

    if edge[0] in deleters or edge[1] in deleters:
        colormap = cm.inferno_r
    else:
        colormap = cm.inferno_r
    
    #custom_energy -= renorm
    ts_code=t_state[0].code

    ts_code=ts_code.split('i')[1:]
    ts_code.sort()
    ts_code=''.join(str(e) for e in ts_code)
    edge_color = colormap(norm(t_state[0].bader_energy + custom_energy))
    edge_color = mpl.colors.to_hex(edge_color)
    reverse = False
    if t_state[0].r_type == 'C-H':
        reverse = True
    edge_dict[edge] = {'color': '#000000', 'penwidth': 1, 'dir': 'forward'}

empty = af.clear_graph(turu)
nx.set_node_attributes(empty, color_dict )
nx.set_edge_attributes(empty, edge_dict)
for node in list(empty.nodes()):
    if node not in color_dict:
        empty.remove_node(node)
for edge in list(empty.edges()):
    if edge not in edge_dict:
        empty.remove_edge(edge[0],edge[1])
empty.graph['graph']={'rankdir':'LR', 'ranksep':'0.30', 'nodesep': '0.1', 'margin': '0.2', 'fontname': 'Arial', 'center': 'true'}
map_pydot = nx.nx_pydot.to_pydot(empty)
# map_pydot = nx.nx_agraph.pygraphviz_layout(empty)
# nx.nx_agraph.view_pygraphviz(empty)
map_pydot.write_svg(f'{res_path}/plot_propylene_ase.svg')



















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