import copy
import time
import networkx as nx
from itertools import combinations

from GAMERNet.rnet.utilities import functions as fn
import GAMERNet.rnet.graphs.graph_fn as gfn
from GAMERNet.rnet.dict.dict_fn import generate_vars, id_group_dict, update_inter_dict
from multiprocessing import Pool, cpu_count

def generate_subgraphs(args):
    breaking_bond, i, sat_molecule_nx_graph_oxy = args
    subgraph_list = []
    for comb in combinations(breaking_bond, i+1):
        n_subgraphs = len(comb) + 1
        copy_graph = copy.deepcopy(sat_molecule_nx_graph_oxy)
        for bond in comb:
            copy_graph.remove_edge(bond[0], bond[1])
        for j in range(n_subgraphs):
            subgraph = copy_graph.subgraph(list(nx.node_connected_component(copy_graph, list(nx.nodes(copy_graph))[j])))
            # Resetting the indexes of the nodes for the subgraph
            subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0, ordering='default', label_attribute=None)
            # Isomorphisim check in order to remove duplicate graphs
            is_duplicate = False
            for unique_graph in subgraph_list:
                if nx.is_isomorphic(subgraph, unique_graph,  node_match=lambda x, y: x["elem"] == y["elem"]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                subgraph_list.append(subgraph)
    return subgraph_list

def process_graph(graph):
    sat_graph, _ = gfn.sat_H_graph(graph)
    return sat_graph

def is_graph_isomorphic(args):
    graph, unique_graph = args
    return nx.is_isomorphic(graph, unique_graph,  node_match=lambda x, y: x["elem"] == y["elem"])

def gen_inter(input_molecule_list: list[str]) -> tuple[dict, dict]:
    """
    Generates all the possible intermediates from the input molecule list.

    Parameters
    ----------
    input_molecule_list : list[str]
        List of the input molecules.

    Returns
    -------
    tuple[dict, dict]
        Tuple of the intermediate dictionary and the map dictionary.
    """
    
    product_list = copy.deepcopy(input_molecule_list)
    # Since we need to start from CO2, we need to add formic acid as an input molecule
    product_list.append('formic acid')
    
    input_dict = {}
    total_graph_list = []
    for input_molecule in product_list:
        molecular_formula, smiles, molecule_ase_obj, molecule_nx_graph = generate_vars(input_molecule)
        input_dict[input_molecule] = {
                'Formula': molecular_formula,
                'Smiles': smiles,
                'Atoms object': molecule_ase_obj,
                'nx.Graph': molecule_nx_graph,
            }

        oxy_molec_graph = gfn.add_O_2_molec(molecule_nx_graph)

        # sat_oxy_res = sat_H_graph(oxy_molec_graph)
        sat_molecule_oxy = gfn.sat_H_graph(oxy_molec_graph)[0]
        sat_molecule_nx_graph_oxy = gfn.find_new_struct(sat_molecule_oxy)[0]

        # Adding new edge attribute to the graph where the bond type is defined and stored
        for edge in sat_molecule_nx_graph_oxy.edges():

            node_0 = sat_molecule_nx_graph_oxy.nodes[edge[0]]['elem']
            node_1 = sat_molecule_nx_graph_oxy.nodes[edge[1]]['elem']

            sat_molecule_nx_graph_oxy.edges[edge]['bond_type'] = f'{node_0}-{node_1}'

        breaking_bond = []
        for u, v, attr in sat_molecule_nx_graph_oxy.edges(data=True):
            if attr['bond_type'] == 'C-C':
                breaking_bond.append((u, v))
            elif attr['bond_type'] == 'C-O' or attr['bond_type'] == 'O-C':
                breaking_bond.append((u, v))

        # subgraph_list = []
        # for i in range(len(breaking_bond)):
        #     for comb in combinations(breaking_bond, i+1):
        #         n_subgraphs = len(comb) + 1
        #         copy_graph = copy.deepcopy(sat_molecule_nx_graph_oxy)
        #         for bond in comb:
        #             copy_graph.remove_edge(bond[0], bond[1])
        #         for j in range(n_subgraphs):
        #             subgraph = copy_graph.subgraph(list(nx.node_connected_component(copy_graph, list(nx.nodes(copy_graph))[j])))
        #             # Resetting the indexes of the nodes for the subgraph
        #             subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0, ordering='default', label_attribute=None)
        #             subgraph_list.append(subgraph)
        
        # using multiprocessing Pool to generate subgraphs
        with Pool(cpu_count()) as p:
            subgraph_lists = p.map(generate_subgraphs, [(breaking_bond, i, sat_molecule_nx_graph_oxy) for i in range(len(breaking_bond))])

        # flatten the list of lists to a single list    
        subgraph_list = [subgraph for sublist in subgraph_lists for subgraph in sublist]
        with Pool(cpu_count()//2) as p:
            new_subgraph_list = p.map(process_graph, subgraph_list)

        # Now you can replace the original subgraph_list and also append these graphs to total_graph_list.
        subgraph_list = new_subgraph_list
        total_graph_list.extend(new_subgraph_list)
        
        # for graph in subgraph_list:
        #     sat_graph, _ = gfn.sat_H_graph(graph)
        #     subgraph_list[subgraph_list.index(graph)] = sat_graph
        #     total_graph_list.append(sat_graph)
        
        # # Adding the saturated oxygenated molecule to the list
        total_graph_list.append(sat_molecule_nx_graph_oxy)

    # t0 = time.time()
    unique_graphs = []
    # for graph in total_graph_list:
    #     is_duplicate = False
    #     for unique_graph in unique_graphs:
    #         if nx.is_isomorphic(graph, unique_graph,  node_match=lambda x, y: x["elem"] == y["elem"]):
    #             is_duplicate = True
    #             break
    #     if not is_duplicate:
    #         unique_graphs.append(graph)

    for graph in total_graph_list:
        with Pool(cpu_count()) as p:
            is_duplicate = any(p.map(is_graph_isomorphic, [(graph, unique_graph) for unique_graph in unique_graphs]))
        if not is_duplicate:
            unique_graphs.append(graph)

    molec_formula = []
    for graph in unique_graphs:
        f = gfn.molecular_formula_from_graph(graph)
        molec_formula.append(f)

    graph_formula_tuple = list(zip(unique_graphs, molec_formula))

    inter_dict = {}
    for graph, formula in graph_formula_tuple:
        mol_graph, upd_formula, mol_ase_obj, iupac_name = gfn.compare_strucures_from_pubchem(formula, graph)
        inter_dict[iupac_name] = {
                'Name': iupac_name,
                'Formula': upd_formula,
                'Atoms': mol_ase_obj,
                'Graph': mol_graph,
            }
        
    id_group = id_group_dict(inter_dict)

    tmp_inter_dict = {}
    repeat_molec = []
    for name, molec in id_group.items():
        for molec_grp in molec:
            if molec_grp not in repeat_molec:
                repeat_molec.append(molec_grp)
                mol_ase_obj = inter_dict[molec_grp]['Atoms']
                intermediate = fn.generate_pack(mol_ase_obj, sum(1 for atom in mol_ase_obj if atom.symbol == 'H'), id_group[name].index(molec_grp) + 1)
                tmp_inter_dict[molec_grp] = intermediate

    updated_inter_dict = update_inter_dict(tmp_inter_dict, input_dict)

    map_dict = {}
    for molecule in updated_inter_dict.keys():
        intermediate = updated_inter_dict[molecule]
        map_tmp = fn.generate_map(intermediate, 'H')
        map_dict[molecule] = map_tmp
    
    return updated_inter_dict, map_dict