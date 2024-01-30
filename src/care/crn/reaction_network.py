import re
from copy import deepcopy

from shutil import rmtree
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from ase import Atoms
from ase.visualize import view
from pydot import Node
from torch import where
from torch_geometric.data import Data

from care import ElementaryReaction, Intermediate, Surface
from care.constants import INTER_ELEMS, K_B, K_BU, OC_KEYS, OC_UNITS, H
from care.crn.reactors import DifferentialPFR, DynamicCSTR
from care.gnn.graph_filters import extract_adsorbate
from care.gnn.graph_tools import pyg_to_nx
from care.crn.visualize import visualize_reaction


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
        
        if oc is not None:
            if all([key in OC_KEYS for key in oc.keys()]):
                self.oc = oc
            else:
                raise ValueError(f"Keys of oc must be in {OC_KEYS}")
        else:
            self.oc = {"T": 0, "P": 0, "U": 0, "pH": 0}

    @property
    def temperature(self):
        return self.oc["T"]
    
    @temperature.setter
    def temperature(self, other: float):
        self.oc["T"] = other

    @property
    def pressure(self):
        return self.oc["P"]
    
    @pressure.setter
    def pressure(self, other: float):
        self.oc["P"] = other

    @property
    def overpotential(self):
        return self.oc["U"]
    
    @overpotential.setter
    def overpotential(self, other: float):
        self.oc["U"] = other

    @property
    def pH(self):
        return self.oc["pH"]
    
    @pH.setter
    def pH(self, other: float):
        self.oc["pH"] = other

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
        string += "Overpotential: {} {}\n".format(self.oc["U"], OC_UNITS["U"])
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

    def gen_graph(self) -> nx.DiGraph:
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

        Notes:
            The graph is not directed properly, it needs further post-processing,
            i.e. microkinetic modeling.
        """
        graph = nx.DiGraph()

        for inter in self.intermediates.values():
            graph.add_node(
                inter.code,
                category="intermediate",
                phase=inter.phase,
                formula=inter.formula,
                nC=inter["C"],
                nH=inter["H"],
                nO=inter["O"],
            )

        for reaction in self.reactions:
            r_type = (
                reaction.r_type
                if reaction.r_type in ("adsorption", "desorption", "eley_rideal")
                else "surface_reaction"
            )
            graph.add_node(
                reaction.code,
                category="reaction",
                r_type=r_type,
            )

            # Reaction edges (intermediate -> reaction -> intermediate)
            for comp in reaction.components[0]:  # reactants to reaction node
                graph.add_edge(comp.code, reaction.code)
            for comp in reaction.components[1]:  # reaction node to products
                graph.add_edge(reaction.code, comp.code)

        return graph

    def get_min_max(self):
        """
        Return the minimum and the maximum energy of the intermediates and
        the elementary reactions.

        Returns:
            list of two floats containing the min and max value.
        """
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
        return visualize_reaction(self.reactions[idx], show_uncertainty)


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

    @property
    def stoichiometric_matrix(self) -> np.ndarray:
        """
        Return the stoichiometric matrix of the network.
        """
        v = np.zeros(
            (len(self.intermediates) + 1, len(self.reactions))
        )  # +1 for the surface
        for i, inter in enumerate(self.intermediates.values()):
            for j, reaction in enumerate(self.reactions):
                if inter.code in reaction.stoic.keys():
                    v[i, j] = reaction.stoic[inter.code]
                v[-1, j] = reaction.stoic["*"] if "*" in reaction.stoic.keys() else 0
        return v

    @property
    def df_stoic(self) -> pd.DataFrame:
        """
        Return the stoichiometric matrix of the network.
        """

        df = pd.DataFrame(
            self.stoichiometric_matrix,
            index=list(self.intermediates.keys()) + ["*"],
            columns=range(1, len(self.reactions) + 1),
        )
        return df

    def get_kinetic_constants(self, t: float = None):
        """
        Return the kinetic constants of the reactions.
        """
        if t is None:
            if self.oc["T"] is None:
                raise ValueError("temperature not specified")
            else:
                t = self.oc["T"]
        else:
            self.oc["T"] = t

        for reaction in self.reactions:
            reaction.k_eq = np.exp(-reaction.e_rxn[0] / t / K_B)
            if reaction.r_type == "adsorption":  # Hertz-Knudsen
                adsorbate_mass = list(reaction.products)[0].mass
                reaction.k_dir = 1e-18 / (2 * np.pi * adsorbate_mass * K_BU * t) ** 0.5
                reaction.k_dir *= np.exp(-reaction.e_act[0] / K_B / t)
                reaction.k_rev = reaction.k_dir / reaction.k_eq
            elif reaction.r_type == "desorption":
                reaction.k_dir = (K_B * t / H) * np.exp(-reaction.e_act[0] / t / K_B)
                reaction.k_rev = reaction.k_dir / reaction.k_eq
                # adsorbate_mass = list(reaction.reactants)[0].mass
                # reaction.k_rev = 1e-19 / (2 * np.pi * adsorbate_mass * K_BU * t) ** 0.5
                # reaction.k_rev *= np.exp(-(reaction.e_is[0] - reaction.e_fs[0]) / K_B / t)
                # reaction.k_dir = reaction.k_rev / np.exp(reaction.e_rxn[0] / t / K_B)
            else:  # Surface reaction
                reaction.k_dir = (K_B * t / H) * np.exp(-reaction.e_act[0] / t / K_B)
                reaction.k_rev = reaction.k_dir / reaction.k_eq

    def run_microkinetic(
        self,
        iv: dict[str, float],
        temperature: float = None,
        pressure: float = None,
        rtol: float = 1e-12,
        atol: float = 1e-64,
        sstol: float = 1e-12,
        model: Union[DifferentialPFR, DynamicCSTR] = DifferentialPFR,
        **kwargs,
    ):
        """
        Run microkinetic simulation on a differential reactor.

        Args:
            iv (dict): Initial values of the molar fractions of the gas phase
                intermediates. Surface is assumed to be empty. Keys are the
                chemical formulas of the intermediates and values are the
                molar fractions (e.g. {"CH4": 0.1, "H2O": 0.2, "N2": 0.7}), 
                where the sum of the values must be 1.0.
            rtol (float, optional): Relative tolerance. Defaults to 1e-12.
            atol (float, optional): Absolute tolerance. Defaults to 1e-64.
            sstol (float, optional): Steady state tolerance. Defaults to 1e-12.

        Returns:
            dict containing the results of the simulation.
        """
        # SET UP INITIAL CONDITIONS AND PARAMETERS
        if sum(iv.values()) != 1.0:
            raise ValueError("Sum of molar fractions is not 1.0")
        if temperature is None:
            if self.temperature is None:
                raise ValueError("temperature not specified")
            else:
                T = temperature
        else:
            T = temperature

        if pressure is None:
            if self.pressure is None:
                raise ValueError("pressure not specified")
            else:
                P = pressure
        else:   
            P = pressure
        inters = [inter.code for inter in self.intermediates.values()]
        inters_formula = [inter.molecule.get_chemical_formula() for inter in self.intermediates.values()]
        gas_mask = np.array(
            [inter.phase == "gas" for inter in self.intermediates.values()] + [False]
        )
        v = self.stoichiometric_matrix.copy()
        y0 = np.zeros(len(inters) + 1)
        y0[-1] = 1.0   # Initial condition: empty surface
        num_inerts = 0
        for key in iv.keys():
            if key not in inters_formula:  # inert
                v = np.vstack((v, np.zeros(len(self.reactions))))
                y0 = np.append(y0, self.pressure * iv[key])
                gas_mask = np.append(gas_mask, True)
                num_inerts += 1
                # print(f"Added inert {key}")
            elif key in inters_formula:
                for i, inter in enumerate(self.intermediates.values()):
                    if inter.molecule.get_chemical_formula() == key and inter.phase == "gas":
                        y0[i] = P * iv[key]
                        # print(f"Added gas reactant {key}")
                        break
            else:
                continue

        self.get_kinetic_constants(T)
        kd = np.array([reaction.k_dir for reaction in self.reactions])
        kr = np.array([reaction.k_rev for reaction in self.reactions])

        # # REDIRECT ADSORPTION/DESORPTION TO FACILITATE CONVERGENCE
        # for i, step in enumerate(self.reactions):
        #     if step.r_type == "adsorption":
        #         # if none of the iv keys is in the reactants, define the step in the opposite direction
        #         if not any(
        #             [inter.molecule.get_chemical_formula() in iv.keys() for inter in list(step.reactants)+list(step.products)]
        #         ):
        #             kd[i], kr[i] = kr[i], kd[i]
        #             # change sign to the stoichiometric matrix, column i
        #             v[:, i] = -v[:, i]
        

        # RUN SIMULATION
        reactor_args = (T, P, v, *kwargs)
        reactor = model(*reactor_args)
        results = reactor.integrate(y0, (kd, kr, gas_mask), rtol, atol, sstol)
        results["inters"] = inters

        # REWIRE CRN GRAPH BASED ON CRN OUTPUT
        run_graph = self.graph.copy()
        run_graph.remove_edges_from(list(run_graph.edges))
        run_graph.temperature = self.temperature
        run_graph.pressure = self.pressure
        run_graph.max_rate = np.max(np.abs(results["consumption_rate"]))
        run_graph.min_rate = np.min(np.abs(results["consumption_rate"][results["consumption_rate"] != 0]))

        for i, inter in enumerate(self.intermediates.values()):
            if inter.phase == "gas":
                run_graph.nodes[inter]["molar_fraction"] = (
                    results["y"][i, -1] / self.pressure
                )
            elif inter.phase == "ads":
                run_graph.nodes[inter]["coverage"] = results["y"][i, -1]
        run_graph.nodes["*"]["coverage"] = results["y"][
            -1, -1
        ]  # intermediates do not contain *

        # Reaction node: net rate (forward - reverse), positive or negative
        for i, reaction in enumerate(self.reactions):
            run_graph.nodes[reaction.code]["rate"] = results["rate"][i]
            
        for i, inter in enumerate(self.intermediates.values()):
            for j, reaction in enumerate(self.reactions):
                if inter.code in reaction.stoic.keys():
                    if results["consumption_rate"][i, j] < 0:
                        run_graph.add_edge(inter.code, reaction.code, rate=-results["consumption_rate"][i, j])
                    else:
                        run_graph.add_edge(reaction.code, inter.code, rate=results["consumption_rate"][i, j])
        run_graph.remove_node("*")
        results["run_graph"] = run_graph

        return results
