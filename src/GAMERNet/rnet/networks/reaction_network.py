import re
from os.path import abspath
from os import makedirs
from shutil import rmtree

import networkx as nx
import networkx.algorithms.isomorphism as iso
from ase.io import write
import matplotlib.pyplot as plt
from matplotlib import cm

from GAMERNet.rnet.networks.intermediate import Intermediate
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction
from GAMERNet.rnet.networks.surface import Surface

class ReactionNetwork:
    """
    Reaction network class for representing a network of surface reactions 
    starting from the Intermediate and ElementaryReaction objects.

    Attributes:
        intermediates (dict of obj:`Intermediate`): Dictionary containing the
            intermediates of the network.
        reactions (list of obj:`ElementaryReaction`): List containing the   
            elementary reactions of the network.
        gasses (dict of obj:`Intermediate`): Dictionary containing the gas species
            of the network.
        surface (obj:`Surface`): Surface of the network.
        ncc (int): Carbon cutoff of the network.
    """
    def __init__(self, 
                 intermediates: dict[str, Intermediate]=None, 
                 reactions: list[ElementaryReaction]=None, 
                 surface: Surface=None, 
                 ncc: int=None):
        
        if intermediates is None:
            self.intermediates = {}
        else:
            self.intermediates = intermediates
        if reactions is None:
            self.reactions = []
        else:
            self.reactions = reactions
        self.excluded = None
        self._graph = None
        self._surface = None
        self.ncc = ncc
        self.surface = surface
        self.closed_shells = [inter for inter in self.intermediates.values() if inter.closed_shell and inter.phase == 'gas']
        self.num_closed_shell_mols = len(self.closed_shells)
        self.num_intermediates = len(self.intermediates) - self.num_closed_shell_mols - 1  # -1 for the surface
        self.num_reactions = len(self.reactions)

    def __getitem__(self, other):
        if other in self.intermediates:
            return self.intermediates[other]
        out_value = self.search_ts_by_code(other)
        if out_value:
            return out_value
        raise KeyError
    
    def __str__(self):
        string = "ReactionNetwork({} intermediates, {} closed-shell molecules, {} reactions)\n".format(len(self.intermediates) - self.num_closed_shell_mols - 1, 
                                                                                                            self.num_closed_shell_mols, 
                                                                                                            len(self.reactions))
        string += "Surface: {}\n".format(self.surface)
        string += "Network Carbon cutoff: C{}\n".format(self.ncc)
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
            obj:`ReactionNetwork` configured with the intermediates and
            transition states of the dictionary.
        """
        new_net = cls()

        int_list_gen = {}
        ts_list_gen = []
        for inter in net_dict['intermediates']:
            curr_inter = Intermediate(**inter)
            int_list_gen[curr_inter.code] = curr_inter

        for reaction in net_dict['ts']:
            ts_comp_list = []
            for i in reaction.keys():
                if i == 'components':
                    comp_data_list = reaction[i]
                    ts_couple = []
                    for int_data in comp_data_list:
                        tupl_int = []
                        for ts_comp in int_data:
                            comp = Intermediate(**ts_comp)
                            tupl_int.append(comp)
                        ts_couple.append(frozenset(tuple(tupl_int)))
                    ts_comp_list.append(ts_couple)

            reaction.pop('components')
            curr_ts = ElementaryReaction(**reaction, components=ts_comp_list[0])
            ts_list_gen.append(curr_ts)

        new_net.intermediates = int_list_gen
        new_net.reactions = ts_list_gen
        new_net.closed_shells = [inter for inter in new_net.intermediates.values() if inter.closed_shell and inter.phase == 'gas']
        new_net.num_closed_shell_mols = len(new_net.closed_shells)  
        new_net.num_intermediates = len(new_net.intermediates) - new_net.num_closed_shell_mols - 1
        new_net.num_reactions = len(new_net.reactions)
        new_net.ncc = net_dict['ncc']
        new_net.surface = net_dict['surface']
        new_net.graph = new_net.gen_graph()
        return new_net

    def to_dict(self):
        intermediate_list = []
        reaction_list = []

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

        for reaction in self.reactions:
            curr_ts = {}
            curr_ts["code"]=reaction.code
            ts_i_list = []

            for j in reaction.components:
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
            curr_ts["energy"]=reaction.energy
            curr_ts["r_type"]=reaction.r_type
            curr_ts["is_electro"]=reaction.is_electro
            reaction_list.append(curr_ts)

        export_dict = {'intermediates': intermediate_list, 'ts': reaction_list}
        return export_dict

    def add_intermediates(self, inter_dict: dict[str, Intermediate]):
        """Add intermediates to the ReactionNetwork.

        Args:
            inter_dict (dict of obj:`Intermediate`): Intermediates that will be
            added the network.
        """
        self.intermediates.update(inter_dict)

    def add_reactions(self, rxn_lst: list[ElementaryReaction]):
        """Add transition states to the network.

        Args:
            ts_lst (list of obj:`ElementaryReaction`): List containing the
                transition states that will be added to network.
        """
        self.reactions += rxn_lst

    def add_dict(self, net_dict):
        """Add dictionary containing two different keys: intermediates and ts.
        The items of the dictionary will be added to the network.

        Args:
           net_dict (dictionary): Intermediates and ElementaryReaction that will 
              be added to the dictionary.
        """
        self.intermediates.update(net_dict['intermediates'])
        self.reactions += net_dict['ts']

    def search_graph(self, mol_graph, cate=None):
        """Search for an intermediate with a isomorphic graph.

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

    
    def gen_graph(self, del_surf:bool=False) -> nx.DiGraph:
        """Generate a graph using the intermediates and the transition states
        contained in this object.

        Returns:
            obj:`nx.DiGraph` of the network.
        """
        nx_graph = nx.DiGraph()
        makedirs('tmp', exist_ok=True)
        for inter in self.intermediates.values():
            fig_path = abspath("tmp/{}.png".format(inter.code))
            write(fig_path, inter.molecule, show_unit_cell=0)
            phase = inter.phase
            if phase == 'surf':
                nx_graph.add_node(inter.code,
                                    category=phase, 
                                    gas_atoms=inter.molecule, 
                                    code=inter.code, 
                                    formula=inter.molecule.get_chemical_formula()+"*", 
                                    fig_path=fig_path, 
                                    facet=self.surface.facet)
            elif phase == 'gas':
                nx_graph.add_node(inter.code,
                                    category=phase, 
                                    gas_atoms=inter.molecule, 
                                    code=inter.code, 
                                    formula=inter.molecule.get_chemical_formula()+"(g)", 
                                    fig_path=fig_path)
            else: # ads
                nx_graph.add_node(inter.code, 
                               category=phase, 
                               gas_atoms=inter.molecule, 
                               code=inter.code, 
                               formula=inter.molecule.get_chemical_formula()+"*", 
                               fig_path=fig_path)
                            
        for reaction in self.reactions:  
            if reaction.r_type == 'desorption':
                category = 'desorption'
            elif reaction.r_type == 'eley_rideal':
                category = 'eley_rideal'
            else:
                category = 'surface_reaction'
            nx_graph.add_node(reaction.code, category=category)
            for comp in reaction.components[0]: # reactants to reaction node
                nx_graph.add_edge(comp.code,
                                    reaction.code)
            for comp in reaction.components[1]: # reaction node to products
                nx_graph.add_edge(reaction.code, comp.code)

        if del_surf:
            nx_graph.remove_node('000000*')               
            
        return nx_graph

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

    
    def write_dotgraph(self, fig_path:str, filename: str, del_surf: bool=False):
        graph = self.gen_graph(del_surf=del_surf)
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, pos, 'pos')
        plot = nx.drawing.nx_pydot.to_pydot(graph)
        for node in plot.get_nodes():
            category = node.get_attributes()['category']
            if category in ('ads', 'gas', 'surf'):
                formula = node.get_attributes()['formula']
                for num in re.findall(r'\d+', formula):
                    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                    formula = formula.replace(num, num.translate(SUB))
                fig_path = node.get_attributes()['fig_path']
                node.set_fontname("Arial")
                # Add figure as html-like label without table borders
                node.set_label(f"""<
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR>
                <TD><IMG SRC="{fig_path}"/></TD>
                </TR>
                <TR>
                <TD>{formula}</TD>
                </TR>
                </TABLE>>""")
                node.set_style("filled")
                if category != 'gas':
                    node.set_fillcolor("wheat")
                else:
                    node.set_fillcolor("lightpink")
                node.set_shape("ellipse")          
                node.set_width("1.5")
                node.set_height("1.5")
                node.set_fixedsize("true")
            else:  # reaction node
                node.set_shape("square")
                node.set_style("filled")
                node.set_label("")
                node.set_width("0.5")
                node.set_height("0.5")
                if node.get_attributes()['category'] == 'desorption':
                    node.set_fillcolor("palegreen2")
                elif node.get_attributes()['category'] == 'eley_rideal':
                    node.set_fillcolor("mediumpurple1")
                else:
                    node.set_fillcolor("steelblue3")

        for edge in plot.get_edges():
            edge.set_color("azure4")
            # make edges bidirectional
            # edge.set_dir("both")

        a4_dims = (8.3, 11.7)  # In inches
        plot.set_size(f"{a4_dims[0],a4_dims[1]}!")
        plot.set_orientation("landscape")   
        plot.write_png('./'+filename)
        # remove not empty tmp folder
        rmtree('tmp')
        


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
        for reaction in self.reactions:
            energy = ener_func(self._graph[reaction.code])
            energy_dict[reaction] = {'energy': energy}
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
                if sel_node['category'] in ('gas', 'ads', 'surf'):
                    node_inf['inter']['node_lst'].append(node)
                    node_inf['inter']['color'].append('blue')
                    # node_inf['inter']['color'].append(mpl.colors.to_hex(color))
                    node_inf['inter']['size'].append(20)
                # elif sel_node['category']  'ts':
                else:
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
                # elif sel_node['electro']:
                #     node_inf['ts']['node_lst'].append(node)
                #     node_inf['ts']['color'].append('green')
                #     # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                #     node_inf['ts']['size'].append(10)
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

        pos = nx.drawing.layout.kamada_kawai_layout(self.graph)


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
        # add white background to the plot
        axes.set_facecolor('white')

        fig.tight_layout()

        return fig

    def search_connections(self):
        """Add to the intermediates the transition states where they are
        involved.
        """
        for reaction in self.reactions:
            comp = reaction.bb_order()
            r_type = reaction.r_type
            for index, inter in enumerate(comp):
                for react in list(inter):
                    brk = react.reactions[index]
                    if r_type not in brk:
                        brk[r_type] = []
                    if reaction in brk[r_type]:
                        continue
                    brk[r_type].append(reaction)

    def search_ts(self, init, final=None):
        """Given a list of codes or intermediates, search the related elementary reactions.

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
        for t_state in self.reactions:
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
        for reaction in self.reactions:
            comps = reaction.components
            if (new_init.issubset(comps[0]) and new_final.issubset(comps[1]) or
                new_init.issubset(comps[1]) and new_final.issubset(comps[0])):
                ts_lst.append(reaction)
        return reaction

    def search_ts_by_code(self, code: str):
        """Given an arbitrary code, returns the TS with the matching code.

        Args:
            code (str): Code of the transition state.

        Returns:
            obj:`ElementaryReaction` with the matching code
        """
        for reaction in self.reactions:
            if reaction.code == code:
                return reaction
        return None

    def search_inter_by_elements(self, element_dict):
        """Given a dictionary with the elements as keys and the number of each
        element as value, returns all the intermediates that contains these
        elements. The dictionary must contain the elements C, H and O.

        Args:
            element_dict (dict): Dictionary with the symbol of the elements as
            keys and the number of each element as value.

        Returns:
            tuple of obj:`Intermediate` containing all the matching intermediates.
        """
        matches = []
        for inter in self.intermediates.values():
            formula = inter.molecule.get_chemical_formula()
            elem_tmp = {'C': formula.count('C'),
                        'H': formula.count('H'),
                        'O': formula.count('O')}
            if elem_tmp == element_dict:
                matches.append(inter)
        return tuple(matches)
    
    def add_eley_rideal(self, gas_mol: str, ads_int1: str, ads_int2: str): 
        """
        Add an Eley-Rideal reaction to the network.

        A(g) + B* <-> C*

        Args:
            gas_mol (str): Code of the gas molecule.
            ads_int (str): Code of the adsorbed intermediate.
        """
        # check that gas_mol is closed shell and in gas phase
        if self.intermediates[gas_mol].phase != 'gas':
            raise ValueError('First argument must be gas phase')
        if self.intermediates[ads_int1].phase != 'ads':
            raise ValueError('Second argument must be adsorbed')
        if self.intermediates[ads_int2].phase != 'ads':
            raise ValueError('Third argument must be adsorbed')
        reaction = ElementaryReaction(components=[[self.intermediates[ads_int2]], [self.intermediates[gas_mol], self.intermediates[ads_int1]]],
                                      r_type='eley_rideal')
        self.add_reactions([reaction])

    def get_shortest_path(self, mol1_code: str, mol2_code: str) -> int:
        """
        Given a reactant and a product, return the minimum number of reactions to go from one to the other.
        
        Parameters
        ----------
        mol1_code : str
            Code of the reactant.
        mol2_code : str
            Code of the product.
        
        Returns
        -------
        int
            Minimum number of reactions to go from one to the other.
        """

        # NOT FINISHED
        if mol1_code not in self.intermediates.keys():
            raise ValueError('First argument must be in the network')
        if mol2_code not in self.intermediates.keys():
            raise ValueError('Second argument must be in the network')
        if not self.intermediates[mol1_code].closed_shell:
            raise ValueError('First argument must be closed shell')
        if not self.intermediates[mol2_code].closed_shell:
            raise ValueError('Second argument must be closed shell')
        
        
        graph = self.graph.to_undirected()
        # remove nodes that do not contain C* as they provide a shortcut
        graph.remove_node('000000*')
        graph.remove_node('010101*')
        graph.remove_node('011101*')
        graph.remove_node('001101*')
        # return shortest path and nodes traversed
        rxn_condition = lambda node: '<->' in node

        nC1 = self.intermediates[mol1_code].molecule.get_chemical_formula().count('C')
        nC2 = self.intermediates[mol2_code].molecule.get_chemical_formula().count('C')
        if nC1 == nC2:
            for intermediate in self.intermediates.values():
                if intermediate.molecule.get_chemical_formula().count('C') > max(nC1, nC2):
                    graph.remove_node(intermediate.code)
            sp = nx.shortest_path(graph, mol1_code, mol2_code)
            steps = [node for node in sp if '<->' in node]
        else:
            sp = nx.shortest_path(graph, mol1_code, mol2_code)
            steps = [node for node in sp if '<->' in node]

        # steps = [node for node in sp if '<->' in node]

        # write customized dijkstra algorithm
        # 
        return len(steps), steps

            
        
