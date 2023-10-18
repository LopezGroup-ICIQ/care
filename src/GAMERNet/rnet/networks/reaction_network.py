import re
from os.path import abspath
from os import makedirs
from shutil import rmtree
from typing import Union

import networkx as nx
import networkx.algorithms.isomorphism as iso
from ase.io import write
from ase.visualize import view
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle, Rectangle
from matplotlib import cm
from pydot import Node
from energydiagram import ED
import numpy as np
from os import makedirs
from ase.io import write
from os.path import abspath
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
                 intermediates: dict[str, Intermediate]={}, 
                 reactions: list[ElementaryReaction]=[], 
                 surface: Surface=None, 
                 ncc: int=None):
        
        self.intermediates = intermediates
        self.reactions = reactions
        self.surface = surface
        self.ncc = ncc
        self.excluded = None
        self._graph = None
        self._surface = None
        self.closed_shells = [inter for inter in self.intermediates.values() if inter.closed_shell and inter.phase == 'gas']
        self.num_closed_shell_mols = len(self.closed_shells)
        self.num_intermediates = len(self.intermediates) - 1  # -1 for the surface
        self.num_reactions = len(self.reactions)

    def __getitem__(self, other: Union[str, int]):
        if isinstance(other, str):
            return self.intermediates[other]
        elif isinstance(other, int):
            return self.reactions[other]
        else:
            raise TypeError("Index must be str or int")
    
    def __str__(self):
        string = "ReactionNetwork({} adsorbed intermediates, {} gas molecules, {} elementary reactions)\n".format(len(self.intermediates) - 1 - self.num_closed_shell_mols, 
                                                                                                            self.num_closed_shell_mols, 
                                                                                                            len(self.reactions))
        string += "Surface: {}\n".format(self.surface)
        string += "Network Carbon cutoff: C{}\n".format(self.ncc)
        return string
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.reactions)
    
    def __iter__(self):
        return iter(self.reactions)
    
    def __contains__(self, other: Union[str, Intermediate, ElementaryReaction]):
        if isinstance(other, str):
            return other in self.intermediates
        elif isinstance(other, Intermediate):
            return other in self.intermediates.values()
        elif isinstance(other, ElementaryReaction):
            return other in self.reactions
        else:
            raise TypeError("Index must be str, Intermediate or ElementaryReaction")
        
    def __eq__(self, other):
        if isinstance(other, ReactionNetwork):
            return self.intermediates == other.intermediates and self.reactions == other.reactions
        else:
            return False
        
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __add__(self, other):
        if isinstance(other, ReactionNetwork):
            new_net = ReactionNetwork()
            new_net.intermediates = {**self.intermediates, **other.intermediates}
            new_net.reactions = self.reactions + other.reactions
            new_net.closed_shells = [inter for inter in new_net.intermediates.values() if inter.closed_shell and inter.phase == 'gas']
            new_net.num_closed_shell_mols = len(new_net.closed_shells)  
            new_net.num_intermediates = len(new_net.intermediates) - 1
            new_net.num_reactions = len(new_net.reactions)
            new_net.ncc = self.ncc
            new_net.surface = self.surface
            new_net.graph = None
            return new_net
        else:
            raise TypeError("Can only add ReactionNetwork to ReactionNetwork")

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
        int_list_gen, rxn_list_gen = {}, []
        for inter in net_dict['intermediates']:
            curr_inter = Intermediate(**inter)
            int_list_gen[curr_inter.code] = curr_inter

        for reaction in net_dict['reactions']:
            rxn_comp_list = []
            for i in reaction.keys():
                if i == 'components':
                    comp_data_list = reaction[i]
                    rxn_couple = []
                    for int_data in comp_data_list:
                        tupl_int = []
                        for ts_comp in int_data:
                            comp = Intermediate(**ts_comp)
                            tupl_int.append(comp)
                        rxn_couple.append(frozenset(tuple(tupl_int)))
                    rxn_comp_list.append(rxn_couple)
            reaction.pop('components')
            curr_ts = ElementaryReaction(**reaction, components=rxn_comp_list[0])
            rxn_list_gen.append(curr_ts)

        new_net.intermediates = int_list_gen
        new_net.reactions = rxn_list_gen
        new_net.closed_shells = [inter for inter in new_net.intermediates.values() if inter.closed_shell and inter.phase == 'gas']
        new_net.num_closed_shell_mols = len(new_net.closed_shells)  
        new_net.num_intermediates = len(new_net.intermediates) - 1
        new_net.num_reactions = len(new_net.reactions)
        new_net.ncc = net_dict['ncc']
        new_net.surface = net_dict['surface']
        new_net.graph = None
        return new_net

    def to_dict(self):
        intermediate_list, reaction_list = [], []

        for intermediate in self.intermediates.values():
            curr_inter = {}
            curr_inter['code'] = intermediate.code
            curr_inter['molecule']=intermediate.molecule
            curr_inter['graph']=intermediate.graph
            curr_inter['ads_configs']=intermediate.ads_configs
            curr_inter['is_surface']=intermediate.is_surface
            curr_inter['phase']=intermediate.phase
            intermediate_list.append(curr_inter)

        for reaction in self.reactions:
            curr_reaction = {}
            curr_reaction["code"]=reaction.code
            reaction_i_list = []
            for j in reaction.components:
                tupl_comp_list = []
                for i in j:
                    curr_reaction_i_dict = {}
                    curr_reaction_i_dict['code'] = i.code
                    curr_reaction_i_dict['molecule']=i.molecule
                    curr_reaction_i_dict['graph']=i.graph
                    curr_reaction_i_dict['ads_configs']=i.ads_configs
                    curr_reaction_i_dict['is_surface']=i.is_surface
                    curr_reaction_i_dict['phase']=i.phase
                    tupl_comp_list.append(curr_reaction_i_dict)
                reaction_i_list.append(tupl_comp_list)
            curr_reaction["components"]=reaction_i_list
            curr_reaction["r_type"]=reaction.r_type
            curr_reaction["is_electro"]=reaction.is_electro
            reaction_list.append(curr_reaction)

        return {'intermediates': intermediate_list, 'reactions': reaction_list}

    def add_intermediates(self, inter_dict: dict[str, Intermediate]):
        """Add intermediates to the ReactionNetwork.

        Args:
            inter_dict (dict of obj:`Intermediate`): Intermediates that will be
            added to the network.
        """
        self.intermediates.update(inter_dict)

    def del_intermediates(self, inter_lst: list[str]):
        """Delete intermediates from the network and all elementary reactions involving them.

        Args:
            inter_lst (list of str): List containing the codes of the
                intermediates that will be deleted.
        """
        num_rxns_0, num_ints_0 = len(self.reactions), len(self.intermediates)
        inter_counter = 0
        rxn_counter = 0
        for inter in inter_lst:
            self.intermediates.pop(inter)
            for i, reaction in enumerate(self.reactions):
                if inter in reaction.reactants or inter in reaction.products:
                    self.reactions.pop(i)
                    rxn_counter += 1
            inter_counter += 1
        print(f"Deleted {inter_counter} intermediates and {rxn_counter} elementary reactions")
        print(f"Number of intermediates before: {num_ints_0}, after: {len(self.intermediates)}")
        print(f"Number of reactions before: {num_rxns_0}, after: {len(self.reactions)}")

    def add_reactions(self, rxn_lst: list[ElementaryReaction]):
        """
        Add elementary reactions to the network.

        Args:
            rxn_lst (list of obj:`ElementaryReaction`): List containing the
                elementary reactions.
        """
        self.reactions += rxn_lst


    def del_reactions(self, rxn_lst: list[str]):
        """
        Delete elementary reactions from the network. Additionally, delete
        all intermediates that are not participating in any reaction.

        Args:
            rxn_lst (list of str): List containing the codes of the transition
                states that will be deleted.
        """
        num_rxns_0 = len(self.reactions)
        inter_counter = 0
        rxn_counter = 0
        for i, _ in enumerate(rxn_lst):
            self.reactions.pop(i)
            rxn_counter += 1
        num_rxns_fin = len(self.reactions)
            
        # remove all intermediates that are not participating in any reaction
        for inter in list(self.intermediates.keys()):
            counter = 0
            for reaction in self.reactions:
                if inter in reaction.reactants or inter in reaction.products:
                    counter += 1
            if counter == 0:
                self.intermediates.pop(inter)
                inter_counter += 1
        print(f"Deleted {inter_counter} intermediates and {rxn_counter} elementary reactions")
        print(f"Number of reactions before: {num_rxns_0}, after: {num_rxns_fin}")

    def add_dict(self, net_dict: dict):
        """Add dictionary containing two different keys: intermediates and ts.
        The items of the dictionary will be added to the network.

        Args:
           net_dict (dictionary): Intermediates and ElementaryReaction that will 
              be added to the dictionary.
        """
        self.intermediates.update(net_dict['intermediates'])
        self.reactions += net_dict['reactions']

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

    
    def gen_graph(self, del_surf: bool=False, highlight: list=None, show_steps: bool=True) -> nx.DiGraph:
        """
        Generate a graph using the intermediates and the elementary reactions
        contained in this object.

        Args:
            del_surf (bool, optional): If True, the surface will not be
                included in the graph. Defaults to False.
            highlight (list, optional): List containing the codes of the
                intermediates that will be highlighted in the graph. Defaults
                to None.
            show_steps (bool, optional): If True, the elementary reactions will
                be shown as single nodes. Defaults to True.

        Returns:
            obj:`nx.DiGraph` of the network.
        """
        nx_graph = nx.DiGraph()
        makedirs('tmp', exist_ok=True)
        for inter in self.intermediates.values():
            fig_path = abspath("tmp/{}.png".format(inter.code))
            write(fig_path, inter.molecule, show_unit_cell=0)
            switch = None if highlight is None else True if re.sub(r'\(.*\)', '', inter.code) in highlight else False
            formula = inter.molecule.get_chemical_formula() + ("(g)" if inter.phase == 'gas' else "*")
            nx_graph.add_node(inter.code,
                              category=inter.phase, 
                              gas_atoms=inter.molecule, 
                              code=inter.code, 
                              formula=formula, 
                              fig_path=fig_path,  
                              switch=switch)
                            
        for reaction in self.reactions:
            switch = None if highlight is None else True if reaction.code in highlight else False  
            category = reaction.r_type if reaction.r_type in ('adsorption', 'desorption', 'eley_rideal') else 'surface_reaction'
            if show_steps:
                nx_graph.add_node(reaction.code, category=category, switch=switch)
                for comp in reaction.components[0]: # reactants to reaction node
                    nx_graph.add_edge(comp.code,
                                        reaction.code, switch=switch)
                for comp in reaction.components[1]: # reaction node to products
                    nx_graph.add_edge(reaction.code, comp.code, switch=switch)
            else: 
                reacts, prods = [], []
                for inter in reaction.reactants:
                    if not inter.is_surface and inter.molecule.get_chemical_formula().count('C') != 0:
                        reacts.append(inter)
                for inter in reaction.products:
                    if not inter.is_surface and inter.molecule.get_chemical_formula().count('C') != 0:
                        prods.append(inter)
                if len(reacts) == 0 or len(prods) == 0:
                    continue
                for react in reacts:
                    for prod in prods:
                        switch = None if highlight is None else True if react.code in highlight and prod.code in highlight else False
                        nx_graph.add_edge(react.code, prod.code, category=category, switch=switch, code=reaction.code)

        for node in list(nx_graph.nodes):
            if nx_graph.degree(node) == 0:
                nx_graph.remove_node(node)       
        # if del_surf and show_steps:
        #     nx_graph.remove_node('0-0-0-0-0-*')  

        # if highlight is not None and show_steps == False:
        #     # # make graph undirected and define edge direction based on the reaction code
        #     # nx_graph = nx_graph.to_undirected()
        #     for i, inter in enumerate(highlight):
        #         # check neighbours of the node and define edge direction to the only node whose code is in highlight
        #         for neigh in nx_graph.neighbors(inter):
        #             if neigh in highlight:
        #                 # check direction of the edge
        #                 if nx_graph.has_edge(inter, neigh):
        #                     pass
        #                 else:
        #                     # switch edge direction
        #                     nx_graph.remove_edge(neigh, inter)
        #                     nx_graph.add_edge(inter, neigh)
        #             if i == len(highlight) - 1:
        #                 break
        return nx_graph

    def get_min_max(self):
        """
        Return the minimum and the maximum energy of the intermediates and
        the elementary reactions.

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

    
    def write_dotgraph(self, 
                       fig_path:str, 
                       filename: str, 
                       del_surf: bool=False, 
                       highlight: list[str]=None, 
                       show_steps: bool=True):
        graph = self.gen_graph(del_surf=del_surf, highlight=highlight, show_steps=show_steps)
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
                if node.get_attributes()['category'] in ('adsorption', 'desorption'):
                    node.set_fillcolor("palegreen2")
                elif node.get_attributes()['category'] == 'eley_rideal':
                    node.set_fillcolor("mediumpurple1")
                else:
                    node.set_fillcolor("steelblue3")
            if highlight is not None and node.get_attributes()['switch'] == 'True':
                node.set_color("red")
                # increase node width
                node.set_penwidth("3")                

        for edge in plot.get_edges():
            if highlight is not None and edge.get_attributes()['switch'] == 'True':
                edge.set_color("red")
                edge.set_penwidth("4")
            else:
                edge.set_color("black")
                edge.set_penwidth("1")   
                    

        a4_dims = (8.3, 11.7)  # In inches
        plot.set_size(f"{a4_dims[1],a4_dims[0]}!")
        plot.set_orientation("landscape")  
        # define background color
        plot.set_bgcolor("white") 
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

    def calc_reaction_energy(self, ener_func: callable):
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
        """
        Search for each intermediate the elementary reactions in which it is
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
    
    def search_reaction_by_inter(self, inters: list[str]) -> list[ElementaryReaction]:
        """
        Return the elementary reactions that involve the given intermediates.

        Args:
            inters (list of str): List containing the codes of the intermediates.

        Returns:
            list of obj:`ElementaryReaction` containing all the matches.
        """
        condition = lambda step: all([inter in step.reactants or inter in step.products for inter in inters])
        rxn_list = [reaction for reaction in self.reactions if condition(reaction)]
        print(f"{len(rxn_list)} elementary reactions involving the intermediates {inters}")
        return rxn_list

    def search_reaction_by_code(self, code: str):
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

    def visualize_intermediate(self, inter_code: str):
        """Visualize the molecule of an intermediate.

        Args:
            inter_code (str): Code of the intermediate.
        """
        # view all the available ads_configs ase atoms object
        configs = [config['ase'] for _, config in self.intermediates[inter_code].ads_configs.items()]
        view(configs)

    def visualize_reaction(self, reaction_index: int, show_uncertainty: bool=False):
        """
        Visualize the reaction energy diagram.

        Args:
            reaction_index (int): Index of the reaction in the list of reactions.
            show_uncertainty (bool, optional): If True, the confidence interval
                of the energy of the transition state will be shown. Defaults   
                to False.
        Returns:
            obj:`ED` with the energy diagram of the reaction.
        """
        rxn = self.reactions[reaction_index].__repr__()
        components = rxn.split("<->")
        reactants, products = components[0].split("+"), components[1].split("+")
        for i, inter in enumerate(reactants):
            if '00000*' in inter:
                where_surface = 'reactants'
                surf_index = i
                break
        for i, inter in enumerate(products):
            if '00000*' in inter:
                where_surface = 'products'
                surf_index = i
                break
        reactants = [re.findall(r'\((.*?)\)', reactant) for reactant in reactants]
        products = [re.findall(r'\((.*?)\)', product) for product in products]
        reactants = [item for sublist in reactants for item in sublist]
        products = [item for sublist in products for item in sublist]
        for i, reactant in enumerate(reactants):
            if abs(self.reactions[reaction_index].stoic[0][i]) != 1:
                reactants[i] = str(abs(self.reactions[reaction_index].stoic[0][i])) + reactant
            if '(g' in reactant:
                reactants[i] += ')'
            if where_surface == 'reactants' and i == surf_index:
                reactants[i] = "*"
        for i, product in enumerate(products):
            if abs(self.reactions[reaction_index].stoic[1][i]) != 1:
                products[i] = str(abs(self.reactions[reaction_index].stoic[1][i])) + product
            if '(g' in product:
                products[i] += ')'
            if where_surface == 'products' and i == surf_index:
                products[i] = "*"
        rxn_string = " + ".join(reactants) + " -> " + " + ".join(products)
        diagram = ED()
        diagram.add_level(0, rxn_string.split(" -> ")[0])
        diagram.add_level(round(self.reactions[reaction_index].e_act[0], 2), 'TS', color='r')
        diagram.add_level(round(self.reactions[reaction_index].energy[0], 2), rxn_string.split(" -> ")[1])
        diagram.add_link(0,1)
        diagram.add_link(1,2)
        y = diagram.plot(ylabel="Energy / eV") 
        plt.title(rxn_string, fontname='Arial', fontweight='bold',
                  y=1.05) 
        artists = diagram.fig.get_default_bbox_extra_artists()
        size = artists[2].get_position()[0] - artists[3].get_position()[0]
        ap_reactants = (artists[3].get_position()[0], artists[3].get_position()[1]+0.15)
        ap_products = (artists[11].get_position()[0], artists[11].get_position()[1]+0.15)
        from matplotlib.patches import Rectangle        
        makedirs('tmp', exist_ok=True)
        counter=0
        for i, inter in enumerate(self.reactions[reaction_index].reactants):
            if inter.is_surface:
                pass
            else:        
                fig_path = abspath("tmp/reactant_{}.png".format(i))
                write(fig_path, inter.molecule, show_unit_cell=0)
                arr_img = plt.imread(fig_path)
                im = OffsetImage(arr_img)
                if where_surface == 'reactants':
                    ab = AnnotationBbox(im, (ap_reactants[0]+size/2, ap_reactants[1]+size*(0.5+counter)), frameon=False)
                    diagram.ax.add_artist(ab) 
                    counter += 1
                else:
                    ab = AnnotationBbox(im, (ap_reactants[0]+size/2, ap_reactants[1]+size*(0.5+i)), frameon=False)
                    diagram.ax.add_artist(ab)
        counter = 0
        for i, inter in enumerate(self.reactions[reaction_index].products):
            if inter.is_surface:
                pass        
            else:
                fig_path = abspath("tmp/product_{}.png".format(i))
                write(fig_path, inter.molecule, show_unit_cell=0)
                arr_img = plt.imread(fig_path)
                im = OffsetImage(arr_img)
                if where_surface == 'products':
                    ab = AnnotationBbox(im, (ap_products[0]+size/2, ap_products[1]+size*(0.5+counter)), frameon=False)
                    diagram.ax.add_artist(ab) 
                    counter += 1
                else:
                    ab = AnnotationBbox(im, (ap_products[0]+size/2, ap_products[1]+size*(0.5+i)), frameon=False)
                    diagram.ax.add_artist(ab)        
        if show_uncertainty:
            from matplotlib.patches import Rectangle
            width = artists[2].get_position()[0] - artists[3].get_position()[0]
            height_ts = 1.96*2*self.reactions[reaction_index].e_act[1]
            anchor_point_ts = (min(artists[6].get_position()[0], artists[7].get_position()[0]), 
                               round(self.reactions[reaction_index].e_act[0],2) - 0.5*height_ts)
            ts_box = Rectangle(anchor_point_ts, width, height_ts, fill=True, color='#FFD1DC', linewidth=1.5, zorder=-1)
            diagram.ax.add_patch(ts_box)       
        return diagram

    def find_all_paths(self, source, targets, graph, path=[], cutoff=None):
        """
        TODO: Add docstring
        """
        path = path + [source]
        paths = []

        # Check if the path length has exceeded the cutoff
        if cutoff is not None and len(path) > cutoff:
            return []

        # Add the path if either it ends in the target or it has not exceeded the cutoff length
        if source in targets or (cutoff is not None and len(path) <= cutoff):
            paths.append(path)

        if not graph.has_node(source):
            return []

        for node in graph.neighbors(source):
            if node not in path:
                newpaths = self.find_all_paths(node, targets, graph, path, cutoff)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths


    def find_all_paths_from_sources_to_targets(self, sources, targets, cutoff=None):
        """
        TODO: Add docstring
        """
        graph = self.gen_graph(del_surf=True, show_steps=False)
        
        for edge in list(graph.edges):
            graph.add_edge(edge[1], edge[0])

        all_paths = {}
        for source in sources:
            for target in targets:
                if source != target:
                    paths = self.find_all_paths(source, targets, graph, cutoff=cutoff)
                    paths = [path for path in paths if path[-1] == target]
                    all_paths[(source, target)] = paths
        return all_paths



    def find_paths_through_intermediate(self, source, target, intermediate=None, cutoff=None):
        """TODO:  add docstring 
        """
        graph = self.gen_graph(del_surf=True, show_steps=False)
        
        for edge in list(graph.edges):
            graph.add_edge(edge[1], edge[0])

        if intermediate:
            paths_to_intermediate = self.find_all_paths(source, intermediate, graph, cutoff=cutoff/2)
            # Only storing those paths that end in the intermediate
            paths_to_intermediate = [path for path in paths_to_intermediate if path[-1] == intermediate]
            
            paths_from_intermediate = self.find_all_paths(intermediate, target, graph, cutoff=cutoff/2)
            # Only storing those paths that end in the target
            paths_from_intermediate = [path for path in paths_from_intermediate if path[-1] == target]
            # Concatenate the paths to get complete paths from source to target via intermediate
            complete_paths = []
            for path1 in paths_to_intermediate:
                for path2 in paths_from_intermediate:
                    # Check if concatenating the paths exceeds the cutoff
                    if cutoff is None or len(path1) + len(path2) - 1 <= cutoff:
                        # Remove the duplicate intermediate node before appending
                        complete_path = path1 + path2[1:]
                        complete_paths.append(complete_path)

            return complete_paths
        else:
            paths_to_intermediate = self.find_all_paths(source, target, graph, cutoff=cutoff) 
            return paths_to_intermediate        

    
    def filter_intersecting_paths(self, all_paths):
        grouped_paths = {}
        
        for (source1, target1), paths1 in all_paths.items():
            for path1 in paths1:
                # Convert the list to a tuple to use it as a dictionary key
                path1_tuple = tuple(path1)
                end_node1 = path1[-1]  # End node of the path
                if path1_tuple not in grouped_paths:
                    grouped_paths[path1_tuple] = []
                
                for (source2, target2), paths2 in all_paths.items():
                    # Make sure the source is different and the target is the same before proceeding
                    if source1 != source2 and target1 == target2:
                        for path2 in paths2:
                            end_node2 = path2[-1]  # End node of the path
                            if set(path1[1:-1]) & set(path2[1:-1]) and end_node1 == end_node2:
                                grouped_paths[path1_tuple].append(path2)

        # If grouped_paths.values is a dict of empty lists, return original all_paths
        if not any(grouped_paths.values()):
            return all_paths
        else:                 
            return grouped_paths
        
    def get_num_global_reactions(self, reactants: list[str], products: list[str]) -> int:
        """
        Given gaseous reactants and a list of gas products, provide the 
        overall global reactions with stoichiometry.

        Args:
            reactants (list[str]): List of gaseous reactants.
            products (list[str]): List of gaseous products.

        Returns:        
        """
        for reactant in reactants:
            if self.intermediates[reactant].phase != 'gas':
                raise ValueError('All reactants must be gas phase')
        for product in products:
            if self.intermediates[product].phase != 'gas':
                raise ValueError('All products must be gas phase')
        reactants_formulas = [self.intermediates[reactant].molecule.get_chemical_formula() for reactant in reactants]
        products_formulas = [self.intermediates[product].molecule.get_chemical_formula() for product in products]
        print(f"Reactants: {reactants_formulas}")
        print(f"Products: {products_formulas}")
        species = reactants + products
        elements = []
        for specie in species:
            elements += self.intermediates[specie].molecule.get_chemical_symbols()
        elements = list(set(elements))
        nc, na = len(species), len(elements)
        matrix = np.zeros((nc, na))
        for i, specie in enumerate(species):
            for j, element in enumerate(elements):
                matrix[i, j] = self.intermediates[specie].molecule.get_chemical_symbols().count(element)
        rank = np.linalg.matrix_rank(matrix)
        print(f"Number of chemical elements: {na}")
        print(f"Number of chemical species: {nc}")
        print(f"Rank of the species-element matrix: {rank}")
        print(f"Number of global reactions: {nc - rank}")
        return nc - rank