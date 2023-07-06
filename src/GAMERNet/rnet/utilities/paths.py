import networkx as nx
import numpy as np
import multiprocessing as mp

def generate_pairs(network, skip=None):
    if skip is not None:
        skip_set = set(skip)
    else:
        skip_set = set([])
    pairs = []
    for t_state in network.t_states:
        components = t_state.bb_order()
        if t_state.r_type == 'ads':
            continue
        elif t_state.r_type not in ['C-C', 'C-H', 'O-H']:
            components = components[::-1]
        checker = True
        for group in components:
            pair_lst = []
            comp_ener = sum([comp.energy for comp in group])
            ed_ener = t_state.energy - comp_ener
            if checker:
                for comp in group:
                    pair_tmp = (comp.code, t_state.code)
                    if comp not in skip_set:
                        pair_lst.append(pair_tmp)
                checker = False
            else:
                for comp in group:
                    pair_tmp = (t_state.code, comp.code)
                    if comp not in skip_set:
                        pair_lst.append(pair_tmp)
        pairs += pair_lst
    return np.asarray(pairs)

def gen_dir_net(network):
    """Generate a directed network from an existing obj:`networks.OrganicNetwork`

    Args:
        network (obj:`networks.OrganicNetwork`): Network that will be used.

    Returns:
        obj:`nx.DiGraph` with the new graph of the network.
    """

    new_graph = nx.DiGraph()
    for inter in network.intermediates.values():
        new_graph.add_node(inter.code, energy=inter.energy,
                           category='intermediate')

    for t_state in network.t_states:
        new_graph.add_node(t_state.code, energy=t_state.energy,
                           category='ts', r_type=t_state.r_type)
        components = t_state.bb_order()
        if t_state.r_type == 'ads':
            continue
        elif t_state.r_type == 'C-OH' and t_state.is_electro:
            continue
        elif t_state.r_type not in ['C-C', 'C-H', 'O-H']:
            components = components[::-1]
        checker = True
        for group in components:
            comp_ener = sum([comp.energy for comp in group])
            ed_ener = t_state.energy - comp_ener
            if checker:
                for comp in group:
                    new_graph.add_edge(comp.code, t_state.code, energy=ed_ener,
                                       break_type=None)
                checker = False
            else:
                for comp in group:
                    new_graph.add_edge(t_state.code, comp.code, energy=ed_ener,
                                       break_type=None)
    return new_graph

def gen_dir_graph_graph_tool(network):
    pass

def search_species(vertex_map, code):
    for index, species in enumerate(vertex_map):
        if code == species:
            return str(index)

def translate_path(vertex_map, path):
    trans_path = np.chararray(path.shape, unicode=True, itemsize=32)
    for index, item in enumerate(path):
        trans_path[index] = vertex_map[item]
    return trans_path

def calc_max_path_energy(graph, path, ts_hasher, reverse=True, bader=False):
    accumulator = []
    if reverse:
        inv_path = path[::-1]
    else:
        inv_path = path
    for index in range(1, len(inv_path) - 1, 2):
        #t_state_ener = graph.nodes[inv_path[index]]['energy']
        try:
            reverse_tmp = reverse
            if ts_hasher[inv_path[index]].r_type not in ['O-H', 'C-H', 'C-C']:
                reverse_tmp = not reverse_tmp
            ts_ener = ts_hasher[inv_path[index]].calc_activation_energy(reverse=reverse_tmp, bader=bader)
            #inter_ener = graph.nodes[inv_path[index - 1]]['energy']
        except KeyError:
            print(inv_path[index - 1])
        accumulator.append(ts_ener)
    return max(accumulator)

def calc_min_max_path_energy(path, ts_hasher, inter_hasher, reverse=True, bader=False):
    if reverse:
        inv_path = path[::-1]
    else:
        inv_path = path
    min_ener = np.inf
    max_ener = -np.inf
    pairs = []
    first_loop = True
    for index in inv_path:
        try:
            reverse_tmp = reverse
            if ts_hasher[index].r_type not in ['O-H', 'C-H', 'C-C']:
                reverse_tmp = not reverse_tmp
            if bader:
                energy = ts_hasher[index].calc_activation_energy(reverse=reverse_tmp,
                                                                 bader=True)
                energy += inter_pivot
            else:
                energy = ts_hasher[index].calc_activation_energy(reverse=reverse_tmp,
                                                                 bader=False)
                energy += ts_hasher[index].bader_energy
        except KeyError:
            if bader:
                energy = inter_hasher[index].energy
                inter_pivot = energy
            else:
                energy = inter_hasher[index].bader_energy
        if energy < min_ener:
            if first_loop:
                first_loop = False
                min_ener = energy
                max_ener = energy
                continue
            pairs.append(max_ener - min_ener)
            min_ener = energy
            max_ener = energy
        if energy > max_ener:
            max_ener = energy
    else:
        pairs.append(max_ener - min_ener)
    return pairs

def trans_and_ener(net_graph, vertex_map, path, ts_hasher, inter_hasher=False, reverse=True, maximum=True, bader=False):
    if inter_hasher:
        index_start = 2
    else:
        index_start = 1
    out_lst = np.zeros(len(path) + index_start, dtype=object)
    trans_lst = translate_path(vertex_map, path)
    for index, value in enumerate(trans_lst):
        out_lst[index + index_start] = value
    out_lst[0] = calc_max_path_energy(net_graph, trans_lst, ts_hasher, reverse, bader=bader)
    if inter_hasher:
        out_lst[1] = calc_min_max_path_energy(trans_lst, ts_hasher, inter_hasher, reverse, bader=bader)
        if maximum:
            try:
                out_lst[1] = max(out_lst[1])
            except ValueError:
                out_lst[1] = np.inf
    return out_lst

def trans_and_ener_only_ener(net_graph, vertex_map, path, ts_hasher, reverse=True):
    # out_lst = np.zeros([2], dtype=object)
    trans_lst = translate_path(vertex_map, path)
    # for index, value in enumerate(trans_lst):
        # out_lst[index + 1] = value
    out_lst = calc_max_path_energy(net_graph, trans_lst, ts_hasher, reverse)
    return out_lst

def crawl_paths(graph, network, paths):
    energies = []
    branches = []
    for single_path in paths:
        ramifications = []
        ea_max = calc_max_path_energy(graph, single_path)
        source_node = single_path[-1]
        previous_node = source_node
        for code in single_path:
            if 'r_type' in graph.nodes[code]:
                if graph.nodes[code]['r_type'] == 'C-C':
                    comps = list(network.search_ts_by_code(code).bb_order()[1])
                    comps = [item.code for item in comps]
                    for code in comps:
                        if code != previous_node:
                            target_node = code
                    new_paths = list(nx.all_shortest_paths(graph, target_node, source_node))
                    new_ram = crawl_paths(graph, network, new_paths)
                    if new_ram['ea_max'] > ea_max:
                        ea_max = new_ram['ea_max']
                    ramifications.append(new_ram)
        energies.append(ea_max)
        branches.append(ramifications)
    out_index = np.argmin(energies)
    out_dict = {'ea_max': energies[out_index], 'branches': branches[out_index], 'path': paths[out_index]}
    return out_dict

def crawl_prep_iter(graph, network, path_list):
    def tmp_func(index):
        return crawl_paths(graph, network, [path_list[index]])
    return tmp_func


def crawl_all_paths_parallel(crawl_func, path_list, cores=4):
    pool = mp.Pool(cores)
    results = pool.map(crawl_func, range(0, len(path_list)))
    return results
