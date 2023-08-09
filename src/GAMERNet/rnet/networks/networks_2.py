"""Module that implements the needed classes to create a reaction network.
"""
import networkx as nx
import networkx.algorithms.isomorphism as iso
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from pyRDTP.operations import graph
# from pyRDTP import data


class Intermediate:
    """Intermediate class that defines the intermediate species of the network.

    Attributes:
        code (str): Code of the intermediate.
        molecule (obj:`pyRDTP.molecule.Molecule`): Associated molecule.
        graph (obj:`nx.graph`): Associated molecule graph.
        energy (float): DFT energy of the intermediate.
        entropy (float): Entropy of the intermediate
        formula (str): Formula of the intermediate.
    """
    def __init__(self, code=None, molecule=None, graph=None, energy=None, entropy=None,
                 formula=None, electrons=None, is_surface=False, phase=None):
        self.code = code
        self.molecule = molecule
        self._graph = graph
        self.energy = energy
        self.entropy = entropy
        self.formula = formula
        self.electrons = electrons
        self.is_surface = is_surface
        self.bader = None
        self.voltage = None
        if self.is_surface:
            self.phase = 'surface'
        else:
            self.phase = phase
        self.t_states = [{}, {}]

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.code == other
        if isinstance(other, Intermediate):
            return self.code == other.code
        raise NotImplementedError

    def __repr__(self):
        string = (self.code + '({})'.format(self.molecule.formula))
        return string

    def draft(self):
        """Draft of the intermediate generated using the associated graph.

        Returns:
            obj:`matplotlib.pyplot.Figure` with the image of the draft.
        """
        color_map = []
        node_size = []
        for node in self.graph.nodes():
            color_map.append(data.colors.rgb_colors[node.element])
            node_size.append(data.radius.CORDERO[node.element] * 5000)
        return nx.draw(self.graph, node_color=color_map,
                       node_size=node_size, width=15)

    @property
    def bader_energy(self):
        if self.bader is None or self.voltage is None:
            return self.energy
        # -1 is the electron charge
        return self.energy - ((self.bader + self.electrons) * (-1.) * self.voltage)

    @classmethod
    def from_molecule(cls, molecule, code=None, energy=None, entropy=None, is_surface=False, phase=None):
        """Create an Intermediate using a molecule obj.

        Args:
            molecule (obj:`pyRDTP.molecule.Molecule`): Molecule from which the
                intermediate will be created.
            code (str, optional): Code of the intermediate. Defaults to None.
            energy (float, optional): Energy of the intermediate. Defaults to
                None.
            is_surface (bool, optional): Defines if the intermediate is the
                surface.

        Returns:
            obj:`Intermediate` with the given values.
        """
        new_mol = molecule.copy()
        new_mol.connection_clear()
        new_mol.connectivity_search_voronoi()
        new_graph = graph.generate(new_mol)
        new_formula = new_mol.formula
        new_inter = cls(code=code, molecule=new_mol, graph=new_graph,
                        formula=new_formula, energy=energy, entropy=entropy,
                        is_surface=is_surface, phase=phase)
        return new_inter

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.gen_graph()
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    def gen_graph(self):
        """Generate a graph of the molecule.

        Returns:
            obj:`nx.DiGraph` Of the associated molecule.
        """
        return graph.generate(self.molecule)


class TransitionState:
    """Definition of the transition state.

    Attributes:
        code (str): Code associated with the transition state.
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        energy (float): Energy of the transition state.
        r_type (str): Type of reaction of the transition state.
    """
    def __init__(self, code=None, components=None, energy=None,
                 r_type=None, is_electro=False):
        self._code = code
        self._components = None
        self.components = components
        self.energy = energy
        self._bader_energy = None
        self.r_type = r_type
        self.is_electro = is_electro

    def __repr__(self):
        out_str = ''
        for comp in self.components:
            for inter in comp:
                try:
                    out_str += inter.molecule.formula + '+'
                except:
                    out_str += inter.code + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        return out_str[:-3]

    def __eq__(self, other):
        if isinstance(other, TransitionState):
            return self.bb_order() == other.bb_order()
        return False

    def __hash__(self):
        return hash((self.components))

    @property
    def bader_energy(self):
        if self._bader_energy is None:
            return self.energy
        return self._bader_energy

    @bader_energy.setter
    def bader_energy(self, other):
        self._bader_energy = other

    def bb_order(self):
        """Order the components of the transition state in the direction of the
        bond breaking reaction.

        Returns:
            list of frozensets containing the reactants before the bond
            breaking in the firs position and the products before the breakage.
        """
        new_list = list(self.components)
        flatten = [list(comp) for comp in self.components]
        max_numb = 0
        for index, item in enumerate(flatten):
            for species in item:
                if species.is_surface:
                    new_list.insert(0, new_list.pop(index))
                    break
                tmp_numb = len(species.molecule)
                if tmp_numb > max_numb:
                    max_numb = tmp_numb
                    index_numb = index
            else:
                continue
            break
        else:
            new_list.insert(0, new_list.pop(index_numb))
        return new_list

    def full_order(self):
        ordered = self.bb_order()
        react, prod = ordered
        react = list(react)
        prod = list(prod)
        order = []
        for item in [react, prod]:
            try:
                mols = [item[0],
                        item[1]]
            except IndexError:
                mols = [item[0], item[0]]
                
            if mols[0].is_surface:
                order.append(mols[::-1])
            elif not mols[1].is_surface and (mols[0].molecule.atom_numb <
                                             mols[1].molecule.atom_numb):
                order.append(mols[::-1])
            else:
                order.append(mols)
        return order

    def full_label(self):
        order = self.full_order()
        full_label = ''
        for item in order:
            for inter in item:
                if inter.phase == ['cat']:
                    full_label += 'i'
                elif inter.code == 'e-':
                    full_label += 'xxxxxxx'
                    continue
                else:
                    full_label += 'g'
                full_label += str(inter.code)
        return full_label

    def calc_activation_energy(self, reverse=False, bader=False):
        components = [list(item) for item in self.bb_order()]
        if reverse:
            components = components[1]
        else:
            components = components[0]

        if len(components) == 1:
            components = components * 2

        if bader:
            inter_ener = sum([inter.bader_energy for inter in components])
            out_ener = self.bader_energy - inter_ener
        else:
            inter_ener = sum([inter.energy for inter in components])
            out_ener = self.energy - inter_ener
        return out_ener

    def calc_delta_energy(self, reverse=False, bader=False):
        components = [list(item) for item in self.bb_order()]
        start_comp, end_comp = components

        if len(end_comp) == 1:
            end_comp = end_comp * 2

        if reverse:
            start_comp, end_comp = end_comp, start_comp

        if bader:
            start_ener = sum([inter.bader_energy for inter in start_comp])
            end_ener = sum([inter.bader_energy for inter in end_comp])
        else:
            start_ener = sum([inter.energy for inter in start_comp])
            end_ener = sum([inter.energy for inter in end_comp])
        out_ener = end_ener - start_ener
        return out_ener

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, other):
        if other is None:
            self._components = []
        else:
            _ = []
            for item in other:
                _.append(frozenset(item))
            self._components = tuple(_)

    @property
    def code(self):
        if self._code is None:
            self._code = self.get_code(option='g')
        return self._code

    @code.setter
    def code(self, other):
        self._code = other

    def add_components(self, pair):
        """Add components to the transition state.

        Args:
            pair (list of Intermediate): Intermediates that will be added to
                the components.
        """
        new_pair = frozenset(pair)
        self.components.append(new_pair)

    def get_code(self, option='g'):
        """Automatically generate a code for the transition state using the
        code of the intermediates.

        Args:
            option (str, optional): 'g' or 'i'. If 'g' the position of the
                intermediates in the code is inverted. Defaults to None.
        """
        end_str = ''
        in_str = 'i'
        out_str = 'f'
        act_comp = self.bb_order()
        if option in ['inverse', 'i']:
            act_comp = self.components[::-1]
        elif option in ['general', 'g']:
            out_str = 'i'

        for species in act_comp[0]:
            end_str += in_str + str(species.code)
        for species in act_comp[1]:
            end_str += out_str + str(species.code)
        return end_str

    def draft(self):
        """Draw a draft of the transition state using the drafts of the
        components.

        Returns:
            obj:`matplotlib.pyplot.Figure` containing the draft.
        """
        counter = 1
        for item in self.components:
            for component in item:
                plt.subplot(2, 2, counter)
                component.draft()
                counter += 1
        return plt.show()

'''
class OrganicNetwork:
    """Implements the organic network.

    Attributes:
        intermediates (dict of obj:`Intermediate`): Intermediates that belong
            to the network
        t_states (list of obj:`TransitionState`): List containing the
            transition states associated to the network.
        surface (obj:`pyRDTP.molecule.Bulk`): Surface of the network.
        graph (obj:`nx.DiGraph`): Graph of the network.
    """
    def __init__(self, intermediates=None, t_states=None, gasses=None):
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

    def __getitem__(self, other):
        if other in self.intermediates:
            return self.intermediates[other]
        out_value = self.search_ts_by_code(other)
        if out_value:
            return out_value
        raise KeyError

    @classmethod
    def from_dict(cls, net_dict):
        """Generate a reaction network using a dictionary containing the
        intermediates and the transition states

        Args:
            net_dict (dict): Dictionary with two different keys "intermediates"
                and "ts" containing the obj:`Intermediate` and
                obj:`TransitionState`

        Returns:
            obj:`OrganicNetwork` configured with the intermediates and
            transition states of the dictionary.
        """
        new_net = cls()
        new_net.intermediates = net_dict['intermediates']
        new_net.t_states = net_dict['ts']
        return new_net

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
            ts_lst (list of obj:`TransitionState`): List containing the
                transition states that will be added to network.
        """
        self.t_states += ts_lst

    def add_dict(self, net_dict):
        """Add dictionary containing two different keys: intermediates and ts.
        The items of the dictionary will be added to the network.

        Args:
           net_dict (dictionary): Intermediates and TransitionStates that will 
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

    def gen_graph(self, paired_inter=False):
        """Generate a graph using the intermediates and the transition states
        contained in this object.

        Returns:
            obj:`nx.DiGraph` of the network.
        """
        norm_vals = self.get_min_max()
        new_graph = nx.DiGraph()
        for inter in self.intermediates.values():
            new_graph.add_node(inter.code, energy=inter.energy,
                               category='intermediate')

        for t_state in self.t_states:
            new_graph.add_node(t_state.code, energy=t_state.energy,
                               category='ts')
            for group in t_state.components:
                comp_ener = sum([comp.energy for comp in group])
                ed_ener = t_state.energy - comp_ener
                weigth = 1 - (ed_ener - norm_vals[0]) / (norm_vals[1]
                                                         - norm_vals[0])
                if weigth < 0:
                    weigth = 0
                if paired_inter:
                    new_graph.add_edge([comp.code for comp in group],
                                       t_state.code, weight=weigth,
                                       energy=ed_ener, break_type=None)
                    new_graph.add_edge(t_state.code, [comp.code for comp in
                                                      group], weight=weigth,
                                       energy=ed_ener,
                                       break_type=None)
                else:
                    for comp in group:
                        new_graph.add_edge(comp.code, t_state.code, weight=weigth,
                                           energy=ed_ener, break_type=None)
                        new_graph.add_edge(t_state.code, comp.code, weight=weigth,
                                           energy=ed_ener, break_type=None)
        return new_graph

    def get_min_max(self):
        """Returns the minimum and the maximum energy of the intermediates and
        the transition states.

        Returns:
            list of two floats containing the min and max value.
        """
        eners = [inter.energy for inter in self.intermediates.values()]
        eners += [t_state.energy for t_state in self.t_states]
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

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.gen_graph()
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    def calc_ts_energy(self, ener_func):
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
        norm_vals = self.get_min_max()
        colormap = cm.inferno_r
        norm = mpl.colors.Normalize(*norm_vals)
        node_inf = {'inter': {'node_lst': [], 'color': [], 'size': []},
                    'ts': {'node_lst': [], 'color': [], 'size': []}}
        edge_cl = []
        for node in self.graph.nodes():
            sel_node = self.graph.nodes[node]
            try:
                color = colormap(norm(sel_node['energy']))
                if sel_node['category'] == 'intermediate':
                    node_inf['inter']['node_lst'].append(node)
                    node_inf['inter']['color'].append(mpl.colors.to_hex(color))
                    node_inf['inter']['size'].append(20)
                elif sel_node['category'] == 'ts':
                    if 'electro' in sel_node:
                        if sel_node['electro']:
                            node_inf['ts']['node_lst'].append(node)
                            node_inf['ts']['color'].append('red')
                            node_inf['ts']['size'].append(5)
                    else:
                        node_inf['ts']['node_lst'].append(node)
                        node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                        node_inf['ts']['size'].append(5)
                elif sel_node['electro']:
                    node_inf['ts']['node_lst'].append(node)
                    node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                    node_inf['ts']['size'].append(10)
            except KeyError:
                node_inf['ts']['node_lst'].append(node)
                node_inf['ts']['color'].append('green')
                node_inf['ts']['size'].append(10)

        for edge in self.graph.edges():
            sel_edge = self.graph.edges[edge]
            color = colormap(norm(sel_edge['energy']))
            color = mpl.colors.to_rgba(color, 0.2)
            edge_cl.append(color)

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
                                       node_size=node_inf['inter']['size'],
                                       node_shape='v')
        nx.drawing.draw_networkx_edges(self.graph, pos=pos, ax=axes,
                                       edge_color=edge_cl, width=0.3,
                                       arrowsize=0.1)
        fig.tight_layout()

        return fig

    def search_connections(self):
        """Add the to the intermediates the transition states where they are
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
            tuple of obj:`TransitionState` containing all the matches.
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
            tuple of obj:`TransitionState` containing all the matches.
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

    def search_ts_by_code(self, code):
        """Given an arbitrary code, returns the TS with the matching code.

        Args:
            code (str): Code of the transition state.

        Returns:
            obj:`TransitionState` with the matching code
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
            elem_tmp = inter.molecule.elem_inf()
            if elem_tmp == element_dict:
                matches.append(inter)
        return tuple(matches)
'''

class Reactant:
    """Reactant class that contains all the species that are not an intermediate
    but are involved in certain reactions.

    Attributes:
        nome (str): Name of the reactant.
        code (str): Code of the reactant, if any.
        enregy (float): Energy of the reactant, if any.
    """
    def __init__(self, name=None, code=None, energy=None):
        self.name = name
        self.code = code
        self.energy = energy

    def __str__(self):
        out_str = 'Reactant: '
        if self.name is not None:
            out_str += self.name
        if self.code is not None:
            out_str += '({})'.format(self.code)
        return out_str

    def __repr__(self):
        return '[{}]'.format(str(self))
