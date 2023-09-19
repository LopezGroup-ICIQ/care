import re
from os.path import abspath

import networkx as nx
import networkx.algorithms.isomorphism as iso
from ase.io import write
import matplotlib.pyplot as plt
from matplotlib import cm

from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction

class ReactionNetwork:
    """
    Reaction network class for representing a network of surface reactions 
    starting from the Intermediate and ElementaryReaction objects.

    Attributes:
        intermediates (dict of obj:`Intermediate`): Intermediates that belong
            to the network
        t_states (list of obj:`ElementaryReaction`): List containing the
            transition states associated to the network.
        surface (obj:`pyRDTP.molecule.Bulk`): Surface of the network.
        graph (obj:`nx.DiGraph`): Graph of the network.
    """
    def __init__(self, 
                 intermediates: dict[str, Intermediate]=None, 
                 t_states: list[ElementaryReaction]=None, 
                 gasses: dict[str, Intermediate]=None):
        
        if intermediates is None:
            self.intermediates = {}
        else:
            self.intermediates = intermediates
        if t_states is None:
            self.t_states = []
        else:
            self.t_states = t_states
        if gasses is None:
            self.gasses = {}
        else:
            self.gasses = dict(gasses)
        self.excluded = None
        self._graph = None
        self._surface = None
        self.num_intermediates = len(self.intermediates)
        self.closed_shells = [inter for inter in self.intermediates.values() if inter.closed_shell]
        self.num_closed_shell_mols = len(self.closed_shells)
        self.num_reactions = len(self.t_states)

    def __getitem__(self, other):
        if other in self.intermediates:
            return self.intermediates[other]
        out_value = self.search_ts_by_code(other)
        if out_value:
            return out_value
        raise KeyError
    
    def __str__(self):
        string = "ReactionNetwork({} intermediates, {} closed-shell molecules, {} reactions)\n".format(self.num_intermediates, 
                                                                                                                self.num_closed_shell_mols, 
                                                                                                                self.num_reactions)
        string += "Surface: {} {}\n".format(self.get_surface()[0].molecule.get_chemical_formula(), 
                                              self.get_surface()[0].molecule.info["surface_orientation"])
        # string += "Carbon cutoff: N/A\n"
        return string

    @classmethod
    def from_dict(cls, net_dict: dict):
        """Generate a reaction network using a dictionary containing the
        intermediates and the transition states

        Args:
            net_dict (dict): Dictionary with two different keys "intermediates"
                and "ts" containing the obj:`Intermediate` and
                obj:`ElementaryReaction` respectively.

        Returns:
            obj:`OrganicNetwork` configured with the intermediates and
            transition states of the dictionary.
        """
        new_net = cls()

        int_list_gen = {}
        ts_list_gen = []
        for inter in net_dict['intermediates']:
            curr_inter = Intermediate(**inter)
            int_list_gen[curr_inter.code] = curr_inter

        for ts in net_dict['ts']:

            ts_comp_list = []

            for i in ts.keys():
                if i == 'components':
                    comp_data_list = ts[i]
                    ts_couple = []
                    for int_data in comp_data_list:
                        tupl_int = []
                        for ts_comp in int_data:
                            comp = Intermediate(**ts_comp)
                            tupl_int.append(comp)
                        ts_couple.append(frozenset(tuple(tupl_int)))
                    ts_comp_list.append(ts_couple)

            ts.pop('components')
            curr_ts = ElementaryReaction(**ts, components=ts_comp_list[0])
            ts_list_gen.append(curr_ts)

        new_net.intermediates = int_list_gen
        new_net.t_states = ts_list_gen
        new_net.num_intermediates = len(new_net.intermediates)        
        new_net.closed_shells = [inter for inter in new_net.intermediates.values() if inter.closed_shell]
        new_net.num_closed_shell_mols = len(new_net.closed_shells)
        new_net.num_reactions = len(new_net.t_states)
        return new_net

    def to_dict(self):
        intermediate_list = []
        ts_list = []

        for intermediate in self.intermediates.values():
            curr_inter = {}
            curr_inter['code'] = intermediate.code
            curr_inter['molecule']=intermediate.molecule
            curr_inter['adsorbate']=intermediate.adsorbate
            curr_inter['graph']=intermediate.graph
            curr_inter['energy']=intermediate.energy
            curr_inter['std_energy']=intermediate.std_energy
            curr_inter['entropy']=intermediate.entropy
            curr_inter['formula']=intermediate.formula
            curr_inter['electrons']=intermediate.electrons
            curr_inter['is_surface']=intermediate.is_surface
            curr_inter['phase']=intermediate.phase
            intermediate_list.append(curr_inter)

        for ts in self.t_states:
            curr_ts = {}
            curr_ts["code"]=ts.code
            ts_i_list = []

            for j in ts.components:
                tupl_comp_list = []
                for i in j:
                    curr_ts_i_dict = {}
                    curr_ts_i_dict['code'] = i.code
                    curr_ts_i_dict['molecule']=i.molecule
                    curr_ts_i_dict['adsorbate']=i.adsorbate
                    curr_ts_i_dict['graph']=i.graph
                    curr_ts_i_dict['energy']=i.energy
                    curr_ts_i_dict['entropy']=i.entropy
                    curr_ts_i_dict['formula']=i.formula
                    curr_ts_i_dict['electrons']=i.electrons
                    curr_ts_i_dict['is_surface']=i.is_surface
                    curr_ts_i_dict['phase']=i.phase
                    tupl_comp_list.append(curr_ts_i_dict)
                ts_i_list.append(tupl_comp_list)
            curr_ts["components"]=ts_i_list
            curr_ts["energy"]=ts.energy
            curr_ts["r_type"]=ts.r_type
            curr_ts["is_electro"]=ts.is_electro
            ts_list.append(curr_ts)

        export_dict = {'intermediates':intermediate_list, 'ts':ts_list}
        return export_dict

    def add_intermediates(self, inter_dict):
        """Add intermediates to the OrganicNetwork.

        Args:
            inter_dict (dict of obj:`Intermediate`): Intermediates that will be
            added the network.
        """
        self.intermediates.update(inter_dict)

    def add_gasses(self, gasses_dict):
        """Add Gasses to the OrganicNetwork.

        Args:
            gasses_dict (dict of obj:`Intermediate`): Dictionary
                containing the gasses that will be added to the network
        """
        self.gasses.update(gasses_dict)

    def add_ts(self, ts_lst):
        """Add transition states to the network.

        Args:
            ts_lst (list of obj:`ElementaryReaction`): List containing the
                transition states that will be added to network.
        """
        self.t_states += ts_lst

    def add_dict(self, net_dict):
        """Add dictionary containing two different keys: intermediates and ts.
        The items of the dictionary will be added to the network.

        Args:
           net_dict (dictionary): Intermediates and ElementaryReaction that will 
              be added to the dictionary.
        """
        self.intermediates.update(net_dict['intermediates'])
        self.t_states += net_dict['ts']

    def search_graph(self, mol_graph, cate=None):
        """Search for a intermediate with a isomorphic graph.

        Args:
            mol_graph (obj:`nx.DiGraph`): Digraph that will be used as query.
        """
        if cate is None:
            cate = iso.categorical_node_match(['elem', 'elem'], ['H', 'O'])
        coinc_lst = []
        for inter in self.intermediates.values():
            if len(mol_graph) != len(inter.graph):
                continue
            if nx.is_isomorphic(mol_graph, inter.graph, node_match=cate):
                coinc_lst.append(inter)
        return coinc_lst

    def gen_graph(self, paired_inter: bool=False):
        """Generate a graph using the intermediates and the transition states
        contained in this object.

        Returns:
            obj:`nx.DiGraph` of the network.
        """

        #norm_vals = self.get_min_max()
        new_graph = nx.DiGraph()
        for inter in self.intermediates.values():
            new_graph.add_node(inter.code, #energy=inter.energy,
                               category='intermediate')
            # Setting the category as a node attribute
            if inter.is_surface:
                nx.set_node_attributes(new_graph, {'category': 'surface'})
            else:
                nx.set_node_attributes(new_graph,  {'category': 'intermediate'})
            

        for t_state in self.t_states:
            new_graph.add_node(t_state.code, #energy=t_state.energy,
                               category='ts')
            nx.set_node_attributes(new_graph, {'category': 'ts'})
            for group in t_state.components:
                #comp_ener = sum([comp.energy for comp in group])
                #ed_ener = t_state.energy - comp_ener
                #weigth = 1 - (ed_ener - norm_vals[0]) / (norm_vals[1]
                #                                         - norm_vals[0])
                # if weigth < 0:
                #     weigth = 0
                if paired_inter:
                    new_graph.add_edge([comp.code for comp in group],
                                       t_state.code, 
                                       #weight=weigth,
                                       #energy=ed_ener, 
                                       break_type=None)
                    new_graph.add_edge(t_state.code, [comp.code for comp in
                                                      group], 
                                                      #weight=weigth,
                                       #energy=ed_ener,
                                       break_type=None)
                else:
                    for comp in group:
                        new_graph.add_edge(comp.code, t_state.code, 
                                           #weight=weigth,
                                           #energy=ed_ener, 
                                           break_type=None)
                        new_graph.add_edge(t_state.code, comp.code, 
                                           #weight=weigth,
                                           #energy=ed_ener, 
                                           break_type=None)
        return new_graph
    
    def gen_graph2(self, path: str=None):
        """Generate a graph using the intermediates and the transition states
        contained in this object.

        Returns:
            obj:`nx.DiGraph` of the network.
        """
        new_graph = nx.DiGraph()
        for inter in self.intermediates.values():  
            if inter.is_surface or inter.code == '010101':
                pass
            else:
                fig_path = '{}/{}.png'.format(path, inter.code)
                write(fig_path, inter.molecule, show_unit_cell=0)                
                new_graph.add_node(inter.code+"({})".format(inter.molecule.get_chemical_formula()), 
                                   category='intermediate', 
                                   gas_atoms=inter.molecule, 
                                   closed_shell=inter.closed_shell, 
                                   formula=inter.molecule.get_chemical_formula(), 
                                   smiles=inter.smiles,
                                   fig=abspath(fig_path))
            
        
        for t_state in self.t_states:  # edge
            rs, ps = [], []
            for intermediate in t_state.components[0]:
                if intermediate.is_surface or intermediate.code == '010101': # surface(*) or H*
                    pass
                else:
                    rs.append(intermediate.code+"({})".format(intermediate.molecule.get_chemical_formula()))
            for intermediate in t_state.components[1]:
                if intermediate.is_surface or intermediate.code == '010101':
                    pass
                else:
                    ps.append(intermediate.code+"({})".format(intermediate.molecule.get_chemical_formula()))
            for r in rs:
                for p in ps:
                    new_graph.add_edge(r, p, energy=t_state.energy)

        return new_graph

    def get_min_max(self):
        """Returns the minimum and the maximum energy of the intermediates and
        the transition states.

        Returns:
            list of two floats containing the min and max value.
        """
        # def get_min_max(self):
        eners = [node.energy for node in self.nodes if node.energy is not None]
    # return [min(eners), max(eners)]
        # eners = [inter.energy for inter in self.intermediates.values()]
        # eners += [t_state.energy for t_state in self.t_states]
        return [min(eners), max(eners)]

    def get_surface(self):
        """Search for surfaces among the intermediates and return the matches.

        Returns:
            list of obj:`Intermediate` containing all the surfaces of the
            network.
        """
        return [inter for inter in self.intermediates.values() if
                inter.is_surface]

    def write_dotgraph(self, filename):
        """Draw a graphviz dotgraph that represents the network.

        Args:
            filename (str): Location where the network will be stored.
        """
        tmp_graph = nx.Graph()
        for node in self.graph.nodes():
            tmp_graph.add_node(node)
            try:
                if self.graph.nodes[node]['category'] == 'ts':
                    nx.set_node_attributes(tmp_graph, {node: {'shape': 'box'}})
                else:
                    continue
            except KeyError:
                continue
        for edge in self.graph.edges():
            tmp_graph.add_edge(*edge)
        plot = nx.drawing.nx_pydot.to_pydot(tmp_graph)
        plot.set_nodesep(0.3)
        plot.set_rankdir('LR')
        plot.write_png(filename)

    def write_dotgraph2(self, fig_path:str, filename: str):
        graph = self.gen_graph2(fig_path)
        # try layout kamada_kawai_layout
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, pos, 'pos')
        plot = nx.drawing.nx_pydot.to_pydot(graph)
        # text in the node is the node attribute formula
        for node in plot.get_nodes():
            # all numbers in the node label are subscripted
            formula = node.get_attributes()['formula']
            closed_shell = node.get_attributes()['closed_shell']
            # print(formula, closed_shell)
            for num in re.findall(r'\d+', formula):
                SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                formula = formula.replace(num, num.translate(SUB))
            figure = node.get_attributes()['fig']
            node.set_fontname("Arial")
            # Add figure as html-like label without table borders
            node.set_label(f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
            <TR>
            <TD><IMG SRC="{figure}"/></TD>
            </TR>
            <TR>
            <TD>{formula}</TD>
            </TR>
            </TABLE>>""")
            node.set_style("filled")
            node.set_fillcolor("wheat")
            # set node shape as function of closed_shell attribute
            node.set_shape("ellipse")          
            node.set_width("1.5")
            node.set_height("1.5")
            node.set_fixedsize("true")

        # make graph undirected
        for edge in plot.get_edges():
            edge.set_dir("none")
            edge.set_arrowhead("none")
            edge.set_arrowtail("none")
            edge.set_arrowsize("1.0")
            edge.set_penwidth("0.5")
            edge.set_color("azure4")
        plot.set_nodesep(0.3)
        plot.set_rankdir('LR')
        plot.set_margin(0.2)
        plot.set_pad(0.2)
        plot.write_png(filename)

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.gen_graph()
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    def calc_ts_energy(self, ener_func: callable):
        """Use a custom function to calculate the energy of the transition
        states and the energy of the edges.

        Args:
            ener_func (function): Function that takes a transition state node
                as input value and returns a float with the calculated energy
                for this transition state.
        """
        energy_dict = {}
        for t_state in self.t_states:
            energy = ener_func(self._graph[t_state.code])
            energy_dict[t_state] = {'energy': energy}
        return energy_dict

    @property
    def surface(self):
        if self._surface is None:
            self._surface = self.get_surface()
        return self._surface

    @surface.setter
    def surface(self, other):
        self._surface = other

    def draw_graph(self):
        """Create a networkx graph representing the network.

        Returns:
            obj:`nx.DiGraph` with all the information of the network.
        """
        # norm_vals = self.get_min_max()
        colormap = cm.inferno_r
        # norm = mpl.colors.Normalize(*norm_vals)
        node_inf = {'inter': {'node_lst': [], 'color': [], 'size': []},
                    'ts': {'node_lst': [], 'color': [], 'size': []}}
        edge_cl = []
        for node in self.graph.nodes():
            sel_node = self.graph.nodes[node]
            try:
                # color = colormap(norm(sel_node['energy']))
                if sel_node['category'] == 'intermediate':
                    node_inf['inter']['node_lst'].append(node)
                    node_inf['inter']['color'].append('blue')
                    # node_inf['inter']['color'].append(mpl.colors.to_hex(color))
                    node_inf['inter']['size'].append(20)
                elif sel_node['category'] == 'ts':
                    if 'electro' in sel_node:
                        if sel_node['electro']:
                            node_inf['ts']['node_lst'].append(node)
                            node_inf['ts']['color'].append('red')
                            node_inf['ts']['size'].append(5)
                    else:
                        node_inf['ts']['node_lst'].append(node)
                        node_inf['ts']['color'].append('green')
                        # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                        node_inf['ts']['size'].append(5)
                elif sel_node['electro']:
                    node_inf['ts']['node_lst'].append(node)
                    node_inf['ts']['color'].append('green')
                    # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                    node_inf['ts']['size'].append(10)
            except KeyError:
                node_inf['ts']['node_lst'].append(node)
                node_inf['ts']['color'].append('green')
                node_inf['ts']['size'].append(10)

        # for edge in self.graph.edges():
        #     sel_edge = self.graph.edges[edge]
            # color = colormap(norm(sel_edge['energy']))
            # color = mpl.colors.to_rgba(color, 0.2)
            # edge_cl.append(color)

        fig = plt.Figure()
        axes = fig.gca()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        fig.patch.set_visible(False)
        axes.axis('off')

        pos = nx.drawing.layout.spring_layout(self.graph, iterations=200)


        nx.drawing.draw_networkx_nodes(self.graph, pos=pos, ax=axes,
                                       nodelist=node_inf['ts']['node_lst'],
                                       node_color=node_inf['ts']['color'],
                                       node_size=node_inf['ts']['size'],)
                                        
        nx.drawing.draw_networkx_nodes(self.graph, pos=pos, ax=axes,
                                       nodelist=node_inf['inter']['node_lst'],
                                       node_color=node_inf['inter']['color'],
                                       node_size=node_inf['inter']['size'],)
                                    #    node_shape='v')
        nx.drawing.draw_networkx_edges(self.graph, pos=pos, ax=axes,
                                    #    edge_color=edge_cl, 
                                       width=0.3,
                                       arrowsize=0.1)
        fig.tight_layout()

        return fig

    def search_connections(self):
        """Add to the intermediates the transition states where they are
        involved.
        """
        for t_state in self.t_states:
            comp = t_state.bb_order()
            r_type = t_state.r_type
            for index, inter in enumerate(comp):
                for react in list(inter):
                    brk = react.t_states[index]
                    if r_type not in brk:
                        brk[r_type] = []
                    if t_state in brk[r_type]:
                        continue
                    brk[r_type].append(t_state)

    def search_ts(self, init, final=None):
        """Given a list of codes or intermediates, search the related
        related transition states.

        Args:
            init (list of str or obj:`Intermediate`): List containing the
                reactants or the products of the wanted transition state.
            final (list of str or obj:`Intermediate`, optional): List
                containing the component at the another side of the reaction.
                Defaults to a simple list.

        Returns:
            tuple of obj:`ElementaryReaction` containing all the matches.
        """
        if final is None:
            final = []
        ts_lst = []
        new_init = []
        new_final = []
        for old, new in zip((init, final), (new_init, new_final)):
            for item in old:
                if isinstance(item, (str, int)):
                    new.append(self.intermediates[str(item)])
                else:
                    new.append(item)
        new_init = frozenset(new_init)
        new_final = frozenset(new_final)
        for t_state in self.t_states:
            comps = t_state.components
            if (new_init.issubset(comps[0]) and new_final.issubset(comps[1]) or
                new_init.issubset(comps[1]) and new_final.issubset(comps[0])):
                ts_lst.append(t_state)
        return tuple(ts_lst)
    
    def search_ts_1(self, init, final=None):
        """Given a list of codes or intermediates, search the related
        related transition states.

        Args:
            init (list of str or obj:`Intermediate`): List containing the
                reactants or the products of the wanted transition state.
            final (list of str or obj:`Intermediate`, optional): List
                containing the component at the another side of the reaction.
                Defaults to a simple list.

        Returns:
            tuple of obj:`ElementaryReaction` containing all the matches.
        """
        if final is None:
            final = []
        ts_lst = []
        new_init = []
        new_final = []
        for old, new in zip((init, final), (new_init, new_final)):
            for item in old:
                if isinstance(item, (str, int)):
                    new.append(self.intermediates[str(item)])
                else:
                    new.append(item)
        new_init = frozenset(new_init)
        new_final = frozenset(new_final)
        for t_state in self.t_states:
            comps = t_state.components
            if (new_init.issubset(comps[0]) and new_final.issubset(comps[1]) or
                new_init.issubset(comps[1]) and new_final.issubset(comps[0])):
                ts_lst.append(t_state)
        return t_state

    def search_ts_by_code(self, code: str):
        """Given an arbitrary code, returns the TS with the matching code.

        Args:
            code (str): Code of the transition state.

        Returns:
            obj:`ElementaryReaction` with the matching code
        """
        for t_state in self.t_states:
            if t_state.code == code:
                return t_state
        return None

    def search_inter_by_elements(self, element_dict):
        """Given a dictionary with the elements as keys and the number of each
        element as value, returns all the intermediates that contains these
        elements.

        Args:
            element_dict (dict): Dictionary with the symbol of the elements as
            keys and the number of each element as value.

        Returns:
            tuple of obj:`Intermediate` containing all the matching intermediates.
        """
        matches = []
        for inter in self.intermediates.values():
            elem_tmp = {'C': sum([1 for elem in inter.molecule if elem.symbol == 'C']),
                        'H': sum([1 for elem in inter.molecule if elem.symbol == 'H']),
                        'O': sum([1 for elem in inter.molecule if elem.symbol == 'O'])}
            if elem_tmp == element_dict:
                matches.append(inter)
        return tuple(matches)