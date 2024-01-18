import re
from copy import deepcopy
from os import makedirs
from os.path import abspath
from shutil import rmtree
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from ase import Atoms
from ase.io import write
from ase.visualize import view
from energydiagram import ED
from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import BoxStyle, Rectangle
from pydot import Node
from torch import where
from torch_geometric.data import Data

from care import ElementaryReaction, Intermediate, Surface
from care.constants import INTER_ELEMS, OC_KEYS, OC_UNITS
from care.gnn.graph_filters import extract_adsorbate
from care.gnn.graph_tools import pyg_to_nx


class ReactionNetwork:
    """
    Reaction network class for representing a network of surface reactions
    starting from the Intermediate and ElementaryReaction objects.

    Attributes:
        intermediates (dict of obj:`Intermediate`): Dictionary containing the
            intermediates of the network.
        reactions (list of obj:`ElementaryReaction`): List containing the
            elementary reactions of the network.
        surface (obj:`Surface`): Surface of the network.
        ncc (int): Carbon cutoff of the network.
        noc (int): Oxygen cutoff of the network.
        oc (dict of str: float): Dictionary containing the operating conditions
    """

    def __init__(
        self,
        intermediates: dict[str, Intermediate] = {},
        reactions: list[ElementaryReaction] = [],
        surface: Surface = None,
        ncc: int = None,
        noc: int = None,
        oc: dict[str, float] = None,
    ):
        self.intermediates = intermediates
        self.reactions = reactions
        self.surface = surface
        self.ncc = ncc
        self.noc = noc
        self.excluded = None
        self._graph = None
        self._surface = None
        self.closed_shells = [
            inter
            for inter in self.intermediates.values()
            if inter.closed_shell and inter.phase == "gas"
        ]
        self.num_closed_shell_mols = len(self.closed_shells)
        self.num_intermediates = len(self.intermediates) - 1  # -1 for the surface
        self.num_reactions = len(self.reactions)
        # check that keys in oc are valid
        if oc is not None:
            if all([key in OC_KEYS for key in oc.keys()]):
                self.oc = oc
            else:
                raise ValueError(f"Keys of oc must be in {OC_KEYS}")
        else:
            self.oc = {"T": 0, "P": 0, "U": 0, "pH": 0}

    def __getitem__(self, other: Union[str, int]):
        if isinstance(other, str):
            return self.intermediates[other]
        elif isinstance(other, int):
            return self.reactions[other]
        else:
            raise TypeError("Index must be str or int")

    def __str__(self):
        string = "ReactionNetwork({} surface species, {} gas molecules, {} elementary reactions)\n".format(
            len(self.intermediates) - 1 - self.num_closed_shell_mols,
            self.num_closed_shell_mols,
            len(self.reactions),
        )
        string += "Surface: {}\n".format(self.surface)
        string += "Temperature: {} {}\n".format(self.oc["T"], OC_UNITS["T"])
        string += "Pressure: {} {}\n".format(self.oc["P"], OC_UNITS["P"])
        string += "Potential: {} {}\n".format(self.oc["U"], OC_UNITS["U"])
        string += "Network Carbon cutoff: C{}\n".format(self.ncc)
        string += "Network Oxygen cutoff: O{}\n".format(self.noc)
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
            return (
                self.intermediates == other.intermediates
                and self.reactions == other.reactions
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, ReactionNetwork):
            new_net = ReactionNetwork()
            new_net.intermediates = {**self.intermediates, **other.intermediates}
            new_net.reactions = list(set(self.reactions + other.reactions))
            new_net.closed_shells = [
                inter
                for inter in new_net.intermediates.values()
                if inter.closed_shell and inter.phase == "gas"
            ]
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
        for inter in net_dict["intermediates"]:
            curr_inter = Intermediate(**inter)
            int_list_gen[curr_inter.code] = curr_inter

        for reaction in net_dict["reactions"]:
            rxn_comp_list = []
            for i in reaction.keys():
                if i == "components":
                    comp_data_list = reaction[i]
                    rxn_couple = []
                    for int_data in comp_data_list:
                        tupl_int = []
                        for ts_comp in int_data:
                            comp = int_list_gen[ts_comp["code"]]
                            tupl_int.append(comp)
                        rxn_couple.append(frozenset(tuple(tupl_int)))
                    rxn_comp_list.append(rxn_couple)
            reaction.pop("components")
            curr_ts = ElementaryReaction(**reaction, components=rxn_comp_list[0])
            rxn_list_gen.append(curr_ts)

        new_net.intermediates = int_list_gen
        new_net.reactions = rxn_list_gen
        new_net.closed_shells = [
            inter
            for inter in new_net.intermediates.values()
            if inter.closed_shell and inter.phase == "gas"
        ]
        new_net.num_closed_shell_mols = len(new_net.closed_shells)
        new_net.num_intermediates = len(new_net.intermediates) - 1
        new_net.num_reactions = len(new_net.reactions)
        new_net.ncc = net_dict["ncc"]
        new_net.noc = net_dict["noc"]
        new_net.surface = net_dict["surface"]
        new_net.graph = None
        return new_net

    def to_dict(self):
        intermediate_list, reaction_list = [], []

        for intermediate in self.intermediates.values():
            curr_inter = {}
            curr_inter["code"] = intermediate.code
            curr_inter["molecule"] = intermediate.molecule
            curr_inter["graph"] = intermediate.graph
            curr_inter["ads_configs"] = intermediate.ads_configs
            curr_inter["is_surface"] = intermediate.is_surface
            curr_inter["phase"] = intermediate.phase
            intermediate_list.append(curr_inter)

        for reaction in self.reactions:
            curr_reaction = {}
            curr_reaction["code"] = reaction.code
            reaction_i_list = []
            for j in reaction.components:
                tupl_comp_list = []
                for i in j:
                    curr_reaction_i_dict = {}
                    curr_reaction_i_dict["code"] = i.code
                    tupl_comp_list.append(curr_reaction_i_dict)
                reaction_i_list.append(tupl_comp_list)
            curr_reaction["components"] = reaction_i_list
            curr_reaction["r_type"] = reaction.r_type
            curr_reaction["is_electro"] = reaction.is_electro
            reaction_list.append(curr_reaction)

        return {
            "intermediates": intermediate_list,
            "reactions": reaction_list,
            "ncc": self.ncc,
            "noc": self.noc,
            "surface": self.surface,
            "oc": self.oc,
        }

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
        print(
            f"Deleted {inter_counter} intermediates and {rxn_counter} elementary reactions"
        )
        print(
            f"Number of intermediates before: {num_ints_0}, after: {len(self.intermediates)}"
        )
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
        print(
            f"Deleted {inter_counter} intermediates and {rxn_counter} elementary reactions"
        )
        print(f"Number of reactions before: {num_rxns_0}, after: {num_rxns_fin}")

    def add_dict(self, net_dict: dict):
        """Add dictionary containing two different keys: intermediates and ts.
        The items of the dictionary will be added to the network.

        Args:
           net_dict (dictionary): Intermediates and ElementaryReaction that will
              be added to the dictionary.
        """
        self.intermediates.update(net_dict["intermediates"])
        self.reactions += net_dict["reactions"]

    def search_graph(self, mol_graph, cate=None):
        """Search for an intermediate with a isomorphic graph.

        Args:
            mol_graph (obj:`nx.DiGraph`): Digraph that will be used as query.
        """
        if cate is None:
            cate = iso.categorical_node_match(["elem", "elem", "elem"], ["C", "H", "O"])

        for inter in self.intermediates.values():
            if len(mol_graph) != len(inter.graph):
                continue
            if nx.is_isomorphic(
                mol_graph.to_undirected(), inter.graph.to_undirected(), node_match=cate
            ):
                return inter

    def search_graph_closed_shell(self, mol_graph, cate=None):
        """Search for a gas-phase intermediate with a isomorphic graph.

        Args:
            mol_graph (obj:`nx.DiGraph`): Digraph that will be used as query.
        """
        if cate is None:
            cate = iso.categorical_node_match(["elem", "elem", "elem"], ["C", "H", "O"])
        for inter in self.intermediates.values():
            if inter.phase == "gas":
                if len(mol_graph) != len(inter.graph):
                    continue
                if nx.is_isomorphic(
                    mol_graph.to_undirected(),
                    inter.graph.to_undirected(),
                    node_match=cate,
                ):
                    return inter

    def gen_graph(
        self, del_surf: bool = True, highlight: list = None, show_steps: bool = True
    ) -> nx.DiGraph:
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
        if show_steps:
            nx_graph = nx.DiGraph()
        else:
            nx_graph = nx.Graph()
        makedirs("tmp", exist_ok=True)
        for inter in self.intermediates.values():
            fig_path = abspath("tmp/{}.png".format(inter.code))
            write(fig_path, inter.molecule, show_unit_cell=0)
            switch = (
                "None"
                if highlight is None
                else True
                if re.sub(r"\(.*\)", "", inter.code) in highlight
                else False
            )
            formula = inter.molecule.get_chemical_formula() + (
                "(g)" if inter.phase == "gas" else "*"
            )
            nx_graph.add_node(
                inter.code,
                category=inter.phase,
                #   gas_atoms=inter.molecule,
                code=inter.code,
                formula=formula,
                fig_path=fig_path,
                switch=switch,
                nC=inter.molecule.get_chemical_symbols().count("C"),
                nH=inter.molecule.get_chemical_symbols().count("H"),
                nO=inter.molecule.get_chemical_symbols().count("O"),
            )

        for reaction in self.reactions:
            switch = (
                "None"
                if highlight is None
                else True
                if reaction.code in highlight
                else False
            )
            category = (
                reaction.r_type
                if reaction.r_type in ("adsorption", "desorption", "eley_rideal")
                else "surface_reaction"
            )
            if show_steps:
                nx_graph.add_node(
                    reaction.code,
                    category=category,
                    switch=switch,
                    reactants=list(reaction.reactants),
                    products=list(reaction.products),
                    code=reaction.code,
                )
                for comp in reaction.components[0]:  # reactants to reaction node
                    nx_graph.add_edge(comp.code, reaction.code, switch=switch)
                for comp in reaction.components[1]:  # reaction node to products
                    nx_graph.add_edge(reaction.code, comp.code, switch=switch)
                comps = reaction.code.split("<->")
                reverse_code = comps[1] + "<->" + comps[0]
                nx_graph.add_node(
                    reverse_code,
                    category=category,
                    switch=switch,
                    reactants=list(reaction.products),
                    products=list(reaction.reactants),
                    code=reverse_code,
                )
                for comp in reaction.components[0]:  # reactants to reaction node
                    nx_graph.add_edge(reverse_code, comp.code, switch=switch)
                for comp in reaction.components[1]:  # reaction node to products
                    nx_graph.add_edge(comp.code, reverse_code, switch=switch)

            else:
                reacts, prods = [], []
                for inter in reaction.reactants:
                    if (
                        not inter.is_surface
                        and inter.molecule.get_chemical_formula().count("C") != 0
                    ):
                        reacts.append(inter)
                for inter in reaction.products:
                    if (
                        not inter.is_surface
                        and inter.molecule.get_chemical_formula().count("C") != 0
                    ):
                        prods.append(inter)
                if len(reacts) == 0 or len(prods) == 0:
                    continue
                # if there are more than one reactant or product, remove the one with less C atoms
                if len(reacts) > 1:
                    reacts.sort(
                        key=lambda x: x.molecule.get_chemical_formula().count("C")
                    )
                    reacts.pop(0)
                if len(prods) > 1:
                    prods.sort(
                        key=lambda x: x.molecule.get_chemical_formula().count("C")
                    )
                    prods.pop(0)
                for react in reacts:
                    for prod in prods:
                        switch = (
                            "None"
                            if highlight is None
                            else True
                            if react.code in highlight and prod.code in highlight
                            else False
                        )
                        nx_graph.add_edge(
                            react.code,
                            prod.code,
                            category=category,
                            switch=switch,
                            code=reaction.code,
                        )

        for node in list(nx_graph.nodes):
            if nx_graph.degree(node) == 0:
                nx_graph.remove_node(node)
        if del_surf and show_steps == True:
            nx_graph.remove_node("*")

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

    def write_dotgraph(
        self,
        fig_path: str,
        filename: str,
        del_surf: bool = False,
        highlight: list[str] = None,
        show_steps: bool = True,
    ):
        graph = self.gen_graph(
            del_surf=del_surf, highlight=highlight, show_steps=show_steps
        )
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, pos, "pos")
        plot = nx.drawing.nx_pydot.to_pydot(graph)
        for node in plot.get_nodes():
            category = node.get_attributes()["category"]
            if category in ("ads", "gas", "surf"):
                formula = node.get_attributes()["formula"]
                for num in re.findall(r"\d+", formula):
                    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                    formula = formula.replace(num, num.translate(SUB))
                fig_path = node.get_attributes()["fig_path"]
                node.set_fontname("Arial")
                # Add figure as html-like label without table borders
                node.set_label(
                    f"""<
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR>
                <TD><IMG SRC="{fig_path}"/></TD>
                </TR>
                <TR>
                <TD>{formula}</TD>
                </TR>
                </TABLE>>"""
                )
                node.set_style("filled")
                if category != "gas":
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
                if node.get_attributes()["category"] in ("adsorption", "desorption"):
                    node.set_fillcolor("palegreen2")
                elif node.get_attributes()["category"] == "eley_rideal":
                    node.set_fillcolor("mediumpurple1")
                else:
                    node.set_fillcolor("steelblue3")
            if highlight is not None and node.get_attributes()["switch"] == "True":
                node.set_color("red")
                # increase node width
                node.set_penwidth("3")

        for edge in plot.get_edges():
            if highlight is not None and edge.get_attributes()["switch"] == "True":
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
        plot.write_png("./" + filename)
        # remove not empty tmp folder
        rmtree("tmp")

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.gen_graph()
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    def draw_graph(self):
        """Create a networkx graph representing the network.

        Returns:
            obj:`nx.DiGraph` with all the information of the network.
        """
        # norm_vals = self.get_min_max()
        colormap = cm.inferno_r
        # norm = mpl.colors.Normalize(*norm_vals)
        node_inf = {
            "inter": {"node_lst": [], "color": [], "size": []},
            "ts": {"node_lst": [], "color": [], "size": []},
        }
        edge_cl = []
        for node in self.graph.nodes():
            sel_node = self.graph.nodes[node]
            try:
                # color = colormap(norm(sel_node['energy']))
                if sel_node["category"] in ("gas", "ads", "surf"):
                    node_inf["inter"]["node_lst"].append(node)
                    node_inf["inter"]["color"].append("blue")
                    # node_inf['inter']['color'].append(mpl.colors.to_hex(color))
                    node_inf["inter"]["size"].append(20)
                # elif sel_node['category']  'ts':
                else:
                    if "electro" in sel_node:
                        if sel_node["electro"]:
                            node_inf["ts"]["node_lst"].append(node)
                            node_inf["ts"]["color"].append("red")
                            node_inf["ts"]["size"].append(5)
                    else:
                        node_inf["ts"]["node_lst"].append(node)
                        node_inf["ts"]["color"].append("green")
                        # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                        node_inf["ts"]["size"].append(5)
                # elif sel_node['electro']:
                #     node_inf['ts']['node_lst'].append(node)
                #     node_inf['ts']['color'].append('green')
                #     # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
                #     node_inf['ts']['size'].append(10)
            except KeyError:
                node_inf["ts"]["node_lst"].append(node)
                node_inf["ts"]["color"].append("green")
                node_inf["ts"]["size"].append(10)

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
        axes.axis("off")

        pos = nx.drawing.layout.kamada_kawai_layout(self.graph)

        nx.drawing.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=axes,
            nodelist=node_inf["ts"]["node_lst"],
            node_color=node_inf["ts"]["color"],
            node_size=node_inf["ts"]["size"],
        )

        nx.drawing.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=axes,
            nodelist=node_inf["inter"]["node_lst"],
            node_color=node_inf["inter"]["color"],
            node_size=node_inf["inter"]["size"],
        )
        #    node_shape='v')
        nx.drawing.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=axes,
            #    edge_color=edge_cl,
            width=0.3,
            arrowsize=0.1,
        )
        # add white background to the plot
        axes.set_facecolor("white")
        fig.tight_layout()
        return fig

    def search_reaction(
        self,
        inters: Union[list[str], str] = None,
        code: str = None,
        r_types: Union[list[str], str] = None,
    ) -> list[ElementaryReaction]:
        """
        Search for elementary reactions that match the given parameters.

        Args:
            inters (list of str, optional): List containing the codes of the
                intermediates that will be involved in the reaction. Defaults
                to None.
            code (str, optional): Code of the transition state. Defaults to None.
            r_type (str, optional): Type of the reaction. Defaults to None.

        Returns:
            list of obj:`ElementaryReaction` containing all the matches.
        """
        if inters is None and code is None and r_types is None:
            raise ValueError("At least one parameter must be given")
        condition_dict, matches = {}, []
        if inters is not None:
            if isinstance(inters, str):
                inters = [inters]
            for inter in inters:
                if inter not in self.intermediates.keys():
                    raise ValueError(f"Intermediate {inter} not in the network")
            inters_condition = lambda step: all(
                [inter in step.reactants or inter in step.products for inter in inters]
            )
            condition_dict["inters"] = inters_condition
        if code is not None:
            code_condition = lambda step: step.code == code
            condition_dict["code"] = code_condition
        if r_types is not None:
            if isinstance(r_types, str):
                r_types = [r_types]
            for r_type in r_types:
                if r_type not in ElementaryReaction.r_types:
                    raise ValueError(
                        f"r_type must be one of {ElementaryReaction.r_types}"
                    )
            r_type_condition = lambda step: step.r_type in r_types
            condition_dict["r_types"] = r_type_condition

        for reaction in self.reactions:
            if all([condition_dict[i](reaction) for i in condition_dict.keys()]):
                matches.append(reaction)
        # print(f"{len(matches)} elementary reactions found")
        return matches

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
            elem_tmp = {element: inter[element] for element in INTER_ELEMS}
            if all(
                [
                    elem_tmp[element] == element_dict[element]
                    for element in elem_tmp.keys()
                ]
            ):
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
        if self.intermediates[gas_mol].phase != "gas":
            raise ValueError("First argument must be gas phase")
        if self.intermediates[ads_int1].phase != "ads":
            raise ValueError("Second argument must be adsorbed")
        if self.intermediates[ads_int2].phase != "ads":
            raise ValueError("Third argument must be adsorbed")
        reaction = ElementaryReaction(
            components=[
                [self.intermediates[ads_int2]],
                [self.intermediates[gas_mol], self.intermediates[ads_int1]],
            ],
            r_type="eley_rideal",
        )
        self.add_reactions([reaction])

    def visualize_intermediate(self, inter_code: str):
        """Visualize the molecule of an intermediate.

        Args:
            inter_code (str): Code of the intermediate.
        """
        # view all the available ads_configs ase atoms object
        configs = [
            config["ase"]
            for config in self.intermediates[inter_code].ads_configs.values()
        ]
        if len(configs) == 0 and type(configs[0]) == Atoms:
            view(self.intermediates[inter_code].molecule)
        else:
            view(configs)

    def visualize_reaction(self, idx: int, show_uncertainty: bool = False):
        """
        Visualize the reaction energy profile.

        Args:
            idx (int): Index of the reaction in the list of reactions.
            show_uncertainty (bool, optional): If True, the confidence interval
                of the energy of the transition state will be shown. Defaults
                to False.
        Returns:
            obj:`ED` with the energy diagram of the reaction.
        """
        rxn = self.reactions[idx].__repr__()
        # components = rxn.split("<->")
        # reactants, products = components[0].split("+"), components[1].split("+")
        # for i, inter in enumerate(reactants):
        #     if "0000000000*" in inter:
        #         where_surface = "reactants"
        #         surf_index = i
        #         break
        # for i, inter in enumerate(products):
        #     if "0000000000*" in inter:
        #         where_surface = "products"
        #         surf_index = i
        #         break
        # v_reactants = [
        #     re.findall(r"\[([a-zA-Z0-9])\]", reactant) for reactant in reactants
        # ]
        # v_products = [re.findall(r"\[([a-zA-Z0-9])\]", product) for product in products]
        # v_reactants = [item for sublist in v_reactants for item in sublist]
        # v_products = [item for sublist in v_products for item in sublist]
        # reactants = [re.findall(r"\((.*?)\)", reactant) for reactant in reactants]
        # products = [re.findall(r"\((.*?)\)", product) for product in products]
        # reactants = [item for sublist in reactants for item in sublist]
        # products = [item for sublist in products for item in sublist]
        # for i, reactant in enumerate(reactants):
        #     if v_reactants[i] != "1":
        #         reactants[i] = v_reactants[i] + reactant
        #     if "(g" in reactant:
        #         reactants[i] += ")"
        #     if where_surface == "reactants" and i == surf_index:
        #         reactants[i] = "*"
        # for i, product in enumerate(products):
        #     if v_products[i] != "1":
        #         products[i] = v_products[i] + product
        #     if "(g" in product:
        #         products[i] += ")"
        #     if where_surface == "products" and i == surf_index:
        #         products[i] = "*"
        # rxn_string = " + ".join(reactants) + " -> " + " + ".join(products)
        rxn_string = self.reactions[idx].repr_hr
        where_surface = 'reactants' if any(inter.is_surface for inter in self.reactions[idx].reactants) else 'products'
        diagram = ED()
        diagram.add_level(0, rxn_string.split(" <-> ")[0])
        diagram.add_level(
            round(self.reactions[idx].e_act[0], 2), "TS", color="r"
        )
        diagram.add_level(
            round(self.reactions[idx].e_rxn[0], 2),
            rxn_string.split(" <-> ")[1],
        )
        diagram.add_link(0, 1)
        diagram.add_link(1, 2)
        y = diagram.plot(ylabel="Energy / eV")
        plt.title(rxn_string, fontname="Arial", fontweight="bold", y=1.05)
        artists = diagram.fig.get_default_bbox_extra_artists()
        size = artists[2].get_position()[0] - artists[3].get_position()[0]
        ap_reactants = (
            artists[3].get_position()[0],
            artists[3].get_position()[1] + 0.15,
        )
        ap_products = (
            artists[11].get_position()[0],
            artists[11].get_position()[1] + 0.15,
        )
        from matplotlib.patches import Rectangle

        makedirs("tmp", exist_ok=True)
        counter = 0
        for i, inter in enumerate(self.reactions[idx].reactants):
            if inter.is_surface:
                pass
            else:
                fig_path = abspath("tmp/reactant_{}.png".format(i))
                write(fig_path, inter.molecule, show_unit_cell=0)
                arr_img = plt.imread(fig_path)
                im = OffsetImage(arr_img)
                if where_surface == "reactants":
                    ab = AnnotationBbox(
                        im,
                        (
                            ap_reactants[0] + size / 2,
                            ap_reactants[1] + size * (0.5 + counter),
                        ),
                        frameon=False,
                    )
                    diagram.ax.add_artist(ab)
                    counter += 1
                else:
                    ab = AnnotationBbox(
                        im,
                        (
                            ap_reactants[0] + size / 2,
                            ap_reactants[1] + size * (0.5 + i),
                        ),
                        frameon=False,
                    )
                    diagram.ax.add_artist(ab)
        counter = 0
        for i, inter in enumerate(self.reactions[idx].products):
            if inter.is_surface:
                pass
            else:
                fig_path = abspath("tmp/product_{}.png".format(i))
                write(fig_path, inter.molecule, show_unit_cell=0)
                arr_img = plt.imread(fig_path)
                im = OffsetImage(arr_img)
                if where_surface == "products":
                    ab = AnnotationBbox(
                        im,
                        (
                            ap_products[0] + size / 2,
                            ap_products[1] + size * (0.5 + counter),
                        ),
                        frameon=False,
                    )
                    diagram.ax.add_artist(ab)
                    counter += 1
                else:
                    ab = AnnotationBbox(
                        im,
                        (ap_products[0] + size / 2, ap_products[1] + size * (0.5 + i)),
                        frameon=False,
                    )
                    diagram.ax.add_artist(ab)
        if show_uncertainty:
            from matplotlib.patches import Rectangle

            width = artists[2].get_position()[0] - artists[3].get_position()[0]
            height_ts = 1.96 * 2 * self.reactions[idx].e_act[1]
            anchor_point_ts = (
                min(artists[6].get_position()[0], artists[7].get_position()[0]),
                round(self.reactions[idx].e_act[0], 2) - 0.5 * height_ts,
            )
            ts_box = Rectangle(
                anchor_point_ts,
                width,
                height_ts,
                fill=True,
                color="#FFD1DC",
                linewidth=1.5,
                zorder=-1,
            )
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

    def find_all_paths_from_sources_to_targets(
        self, sources, targets, intermediates=None, cutoff=None
    ):
        """
        TODO: Add docstring
        """
        graph = self.gen_graph(del_surf=False, show_steps=False).to_undirected()

        for edge in list(graph.edges):
            graph.add_edge(edge[1], edge[0])

        all_paths = {}
        for source in sources:
            for target in targets:
                if source != target:
                    paths = self.find_all_paths(source, targets, graph, cutoff=cutoff)
                    paths = [path for path in paths if path[-1] == target]
                    # all_paths[(source, target)] = [path for path in nx.all_simple_paths(graph, source, target, cutoff=cutoff)]
                    all_paths[(source, target)] = paths

        if intermediates:
            dict_copy = deepcopy(all_paths)
            # Check if there are paths that go through the intermediate
            for source_target in all_paths.values():
                for path in source_target:
                    # Check if the path goes through ALL the intermediates
                    if not all([inter in path for inter in intermediates]):
                        # Deleting the path from the copy if it does not go through the intermediate
                        dict_copy[(path[0], path[-1])].remove(path)
            all_paths = dict_copy
        print(
            "The shortest path goes through {} intermediates".format(
                min([len(path) for path in all_paths.values() for path in path])
            )
        )

        return all_paths

    # def find_all_paths_from_sources_to_targets(self, sources, targets, cutoff=None):
    #     """
    #     TODO: Add docstring
    #     """
    #     graph = self.gen_graph(del_surf=True, show_steps=False)

    #     for edge in list(graph.edges):
    #         graph.add_edge(edge[1], edge[0])

    #     all_paths = {}
    #     for source in sources:
    #         for target in targets:
    #             if source != target:
    #                 paths = self.find_all_paths(source, targets, graph, cutoff=cutoff)
    #                 paths = [path for path in paths if path[-1] == target]
    #                 all_paths[(source, target)] = paths
    #     return all_paths

    # def find_paths_through_intermediate(self, source, target, intermediate=None, cutoff=None):
    #     """TODO:  add docstring
    #     """
    #     graph = self.gen_graph(del_surf=True, show_steps=False)

    #     for edge in list(graph.edges):
    #         graph.add_edge(edge[1], edge[0])

    #     if intermediate:
    #         paths_to_intermediate = self.find_all_paths(source, intermediate, graph, cutoff=cutoff/2)
    #         # Only storing those paths that end in the intermediate
    #         paths_to_intermediate = [path for path in paths_to_intermediate if path[-1] == intermediate]

    #         paths_from_intermediate = self.find_all_paths(intermediate, target, graph, cutoff=cutoff/2)
    #         # Only storing those paths that end in the target
    #         paths_from_intermediate = [path for path in paths_from_intermediate if path[-1] == target]
    #         # Concatenate the paths to get complete paths from source to target via intermediate
    #         complete_paths = []
    #         for path1 in paths_to_intermediate:
    #             for path2 in paths_from_intermediate:
    #                 # Check if concatenating the paths exceeds the cutoff
    #                 if cutoff is None or len(path1) + len(path2) - 1 <= cutoff:
    #                     # Remove the duplicate intermediate node before appending
    #                     complete_path = path1 + path2[1:]
    #                     complete_paths.append(complete_path)

    #         return complete_paths
    #     else:
    #         paths_to_intermediate = self.find_all_paths(source, target, graph, cutoff=cutoff)
    #         return paths_to_intermediate

    # def filter_intersecting_paths(self, all_paths):
    #     grouped_paths = {}

    #     for (source1, target1), paths1 in all_paths.items():
    #         for path1 in paths1:
    #             # Convert the list to a tuple to use it as a dictionary key
    #             path1_tuple = tuple(path1)
    #             end_node1 = path1[-1]  # End node of the path
    #             if path1_tuple not in grouped_paths:
    #                 grouped_paths[path1_tuple] = []

    #             for (source2, target2), paths2 in all_paths.items():
    #                 # Make sure the source is different and the target is the same before proceeding
    #                 if source1 != source2 and target1 == target2:
    #                     for path2 in paths2:
    #                         end_node2 = path2[-1]  # End node of the path
    #                         if set(path1[1:-1]) & set(path2[1:-1]) and end_node1 == end_node2:
    #                             grouped_paths[path1_tuple].append(path2)

    #     # If grouped_paths.values is a dict of empty lists, return original all_paths
    #     if not any(grouped_paths.values()):
    #         return all_paths
    #     else:
    #         return grouped_paths

    def get_num_global_reactions(
        self, reactants: list[str], products: list[str]
    ) -> int:
        """
        Given gaseous reactants and a list of gas products, provide the
        overall global reactions with stoichiometry.

        Args:
            reactants (list[str]): List of gaseous reactants.
            products (list[str]): List of gaseous products.

        Returns:
        """
        for reactant in reactants:
            if self.intermediates[reactant].phase != "gas":
                raise ValueError("All reactants must be gas phase")
        for product in products:
            if self.intermediates[product].phase != "gas":
                raise ValueError("All products must be gas phase")
        reactants_formulas = [
            self.intermediates[reactant].molecule.get_chemical_formula()
            for reactant in reactants
        ]
        products_formulas = [
            self.intermediates[product].molecule.get_chemical_formula()
            for product in products
        ]
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
                matrix[i, j] = (
                    self.intermediates[specie]
                    .molecule.get_chemical_symbols()
                    .count(element)
                )
        rank = np.linalg.matrix_rank(matrix)
        print(f"Number of chemical elements: {na}")
        print(f"Number of chemical species: {nc}")
        print(f"Rank of the species-element matrix: {rank}")
        print(f"Number of global reactions: {nc - rank}")
        return nc - rank

    def get_stoichiometric_matrix(self) -> np.ndarray:
        """
        Return the stoichiometric matrix of the network.
        """
        v = np.zeros((len(self.intermediates), len(self.reactions)))
        for i, inter in enumerate(self.intermediates.values()):
            for j, reaction in enumerate(self.reactions):
                if inter.code in reaction.stoic.keys():
                    v[i, j] = reaction.stoic[inter.code]
        return v

    def ts_graph(self, step: ElementaryReaction) -> Data:
        """
        Given the bond-breaking reaction, detect the broken bond in the
        transition state and label the corresponding edge.
        Given A* + * -> B* + C* and the bond-breaking type X-Y, take the graph of A*,
        break all the potential X-Y bonds and perform isomorphism with B* + C*.
        When isomorphic, the broken edge is labelled.

        Args:
            graph (Data): adsorption graph of the intermediate which is fragmented in the reaction.
            reaction (ElementaryReaction): Bond-breaking reaction.

        Returns:
            Data: graph with the broken bond labeled.
        """

        if "-" not in step.r_type:
            raise ValueError("Input reaction must be a bond-breaking reaction.")
        bond = tuple(step.r_type.split("-"))

        # Select intermediate that is fragmented in the reaction (A*)
        # inters_dict = {inter.code: inter for inter in list(step.reactants)+list(step.products) if not inter.is_surface}
        inters = {
            inter.code: inter.graph.number_of_edges()
            for inter in list(step.reactants) + list(step.products)
            if not inter.is_surface
        }
        inter_code = max(inters, key=inters.get)
        idx = min(
            self.intermediates[inter_code].ads_configs,
            key=lambda x: self.intermediates[inter_code].ads_configs[x]["mu"],
        )
        ts_graph = deepcopy(self.intermediates[inter_code].ads_configs[idx]["pyg"])
        competitors = [
            inter
            for inter in list(step.reactants) + list(step.products)
            if not inter.is_surface and inter.code != inter_code
        ]

        # Build the nx graph of the competitors (B* + C*)
        if len(competitors) == 1:
            if abs(step.stoic[competitors[0].code]) == 2:  # A* -> 2B*
                nx_bc = [competitors[0].graph, competitors[0].graph]
                mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
                nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
                nx_bc = nx.compose(nx_bc[0], nx_bc[1])
            elif abs(step.stoic[competitors[0].code]) == 1:  # A* -> B* (ring opening)
                nx_bc = competitors[0].graph
            else:
                raise ValueError("Reaction stoichiometry not supported.")
        else:  # asymmetric fragmentation
            nx_bc = [competitors[0].graph, competitors[1].graph]
            mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
            nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
            nx_bc = nx.compose(nx_bc[0], nx_bc[1])

        # Lool for potential edges to break
        atom_symbol = lambda idx: ts_graph.node_feats[
            where(ts_graph.x[idx] == 1)[0].item()
        ]
        potential_edges = []
        for i in range(ts_graph.edge_index.shape[1]):
            edge_idxs = ts_graph.edge_index[:, i]
            atom1, atom2 = atom_symbol(edge_idxs[0]), atom_symbol(edge_idxs[1])
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)
        counter = 0

        # Find correct one via isomorphic comparison
        while True:
            data = deepcopy(ts_graph)
            u, v = data.edge_index[:, potential_edges[counter]]
            mask = ~(
                (data.edge_index[0] == u) & (data.edge_index[1] == v)
                | (data.edge_index[0] == v) & (data.edge_index[1] == u)
            )
            data.edge_index = data.edge_index[:, mask]
            data.edge_attr = data.edge_attr[mask]
            adsorbate = extract_adsorbate(data, ["C", "H", "O", "N", "S"])
            nx_graph = pyg_to_nx(adsorbate, data.node_feats)
            if nx.is_isomorphic(
                nx_bc, nx_graph, node_match=lambda x, y: x["elem"] == y["elem"]
            ):
                ts_graph.edge_attr[potential_edges[counter]] = 1
                idx = np.where(
                    (ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u)
                )[0].item()
                ts_graph.edge_attr[idx] = 1
                break
            else:
                counter += 1
        step.ts_graph = ts_graph

    def calc_reaction_energy(
        self, reaction: ElementaryReaction, mean_field: bool = True
    ) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
            mean_field (bool, optional): If True, the reaction energy will be
                calculated using the mean field approximation, with the
                smallest energy for each intermediate.
                Defaults to True.
        """
        if mean_field:
            mu_is, var_is, mu_fs, var_fs = 0, 0, 0, 0
            for reactant in reaction.reactants:
                if reactant.is_surface:
                    continue
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[reactant.code].ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in self.intermediates[reactant.code].ads_configs.values()
                ]
                e_min_config = min(energy_list)
                s_min_config = s_list[energy_list.index(e_min_config)]
                mu_is += (
                    abs(reaction.stoic[self.intermediates[reactant.code].code])
                    * e_min_config
                )
                var_is += (
                    reaction.stoic[self.intermediates[reactant.code].code] ** 2
                    * s_min_config**2
                )
            for product in reaction.products:
                if product.is_surface:
                    continue
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[product.code].ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in self.intermediates[product.code].ads_configs.values()
                ]
                e_min_config = min(energy_list)
                s_min_config = s_list[energy_list.index(e_min_config)]
                mu_fs += (
                    abs(reaction.stoic[self.intermediates[product.code].code])
                    * e_min_config
                )
                var_fs += (
                    reaction.stoic[self.intermediates[product.code].code] ** 2
                    * s_min_config**2
                )
            mu_rxn = mu_fs - mu_is
            std_rxn = (var_fs + var_is) ** 0.5
        else:
            pass
        reaction.e_is = mu_is, var_is**0.5
        reaction.e_fs = mu_fs, var_fs**0.5
        reaction.e_rxn = mu_rxn, std_rxn

    def calc_reaction_barrier(self, reaction: ElementaryReaction) -> None:
        """
        Get activation energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        if "-" not in reaction.r_type:
            reaction.e_act = max(0, reaction.e_rxn[0]), reaction.e_rxn[1]
        else:  # bond-breaking reaction
            reaction.e_act = (
                reaction.e_ts[0] - reaction.e_is[0],
                (reaction.e_ts[1] ** 2 + reaction.e_is[1] ** 2) ** 0.5,
            )

            if reaction.e_act[0] < 0:  # Negative predicted barrier
                reaction.e_act = 0, 0
            if (
                reaction.e_act[0] < reaction.e_rxn[0]
            ):  # Barrier lower than reaction energy
                reaction.e_act = reaction.e_rxn[0], reaction.e_rxn[1]
