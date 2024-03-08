import re

from shutil import rmtree
from typing import Union
from copy import deepcopy

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from ase import Atoms
from ase.visualize import view

from care import ElementaryReaction, Intermediate, Surface
from care.constants import INTER_ELEMS, K_B, K_BU, OC_KEYS, OC_UNITS, H
from care.crn.reactors import DifferentialPFR
from care.crn.visualize import visualize_reaction
from care.crn.microkinetic import max_flux


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
        self.num_intermediates = len(self.intermediates)
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
            len(self.intermediates) - self.num_closed_shell_mols,
            self.num_closed_shell_mols,
            len(self.reactions)
        )
        string += "Surface: {}\n".format(self.surface)
        string += "Network Carbon cutoff: {}\n".format(self.ncc)
        string += "Network Oxygen cutoff: {}\n".format(self.noc)
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

    def add_reactions(self, rxn_lst: list[ElementaryReaction]) -> None:
        """
        Add elementary reactions to the network.

        Args:
            rxn_lst (list of obj:`ElementaryReaction`): List containing the
                elementary reactions.
        """
        self.reactions += rxn_lst

    def del_reactions(self, rxn_lst: list[str]) -> None:
        """
        Delete elementary reactions and consequently the intermediates 
        that are not participating in any reaction.

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
        Generate a network graph representation.

        Returns:
            graph (obj:`nx.DiGraph`): Network graph. Nodes represent species and elementary reactions.
            Species nodes can only be linked to reaction nodes and vice versa.

        Notes:
            The returned graph is correctly directed at the elementary reaction level, 
            i.e. if A + B -> C + D is a reaction, the species on the same side (A and B)
            are linked to the reaction node in the same way (both entering or exiting).
            However, at the global level, the graph is ill-defined and further processing 
            (kinetic modeling) is needed to obtain a correct representation of the network.
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

        graph.add_node(
            "*",
            category="intermediate",
            phase="surf",
            formula="*",
            nC=0,
            nH=0,
            nO=0,
        )

        for reaction in self.reactions:
            r_type = (
                reaction.r_type
                if reaction.r_type in ("adsorption", "desorption", "eley_rideal", "PCET")
                else "surface_reaction"
            )
            graph.add_node(
                reaction.code,
                category="reaction",
                r_type=r_type,
            )

            for inter in reaction.components[0]:
                graph.add_edge(inter.code, reaction.code)
            for inter in reaction.components[1]:
                graph.add_edge(reaction.code, inter.code)  #TODO: add electrochemistry

        return graph

    def gen_electro_graph(self, graph) -> nx.DiGraph:

        electro_graph = deepcopy(graph)

        # Adding electrochemical species
        electro_graph.add_node(
            "H2O(aq)",
            category="electro",
            phase="solv",
            formula="H2O(aq)",
            nC = 0,
            nH = 2,
            nO = 1,
        )
        electro_graph.add_node(
            "H+",
            category="electro",
            phase="solv",
            formula="H+",
            nC = 0,
            nH = 1,
            nO = 0,
        )
        electro_graph.add_node(
            "OH-",
            category="electro",
            phase="solv",
            formula="OH-",
            nC = 0,
            nH = 1,
            nO = 1,
        )
        electro_graph.add_node(
            "e-",
            category="electro",
            phase="electro",
            formula="e-",
            nC = 0,
            nH = 0,
            nO = 0,
        )

        graph_electro_reactions = []
        for reaction in electro_graph.nodes:
            if electro_graph.nodes[reaction]["category"] == "reaction" and electro_graph.nodes[reaction]["r_type"] == "PCET":
                graph_electro_reactions.append(reaction)

        for reaction_code in graph_electro_reactions:
           
            crn_rxn = self.search_reaction(code=reaction_code)[0]

            for inter in crn_rxn.components[0]:
                if inter.phase in ("solv", "electro"):
                    # if not electro_graph.has_node(inter.code):
                    #     electro_graph.add_node(inter.code, category="electro", phase=inter.phase, formula=inter.formula, nC=inter["C"], nH=inter["H"], nO=inter["O"])
                    electro_graph.add_edge(inter.code, crn_rxn.code)
            for inter in crn_rxn.components[1]:
                if inter.phase in ("solv", "electro"):
                    # if not electro_graph.has_node(inter.code):
                    #     electro_graph.add_node(inter.code, category="electro", phase=inter.phase, formula=inter.formula, nC=inter["C"], nH=inter["H"], nO=inter["O"])
                    electro_graph.add_edge(crn_rxn.code, inter.code)
        return electro_graph


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
            (len(self.intermediates) + 1, len(self.reactions)), dtype=np.int8
        )
        sorted_intermediates = sorted(
            self.intermediates.keys())
        for i, inter in enumerate(sorted_intermediates):
            for j, reaction in enumerate(self.reactions):
                if inter in reaction.stoic.keys():
                    v[i, j] = reaction.stoic[inter]
                v[-1, j] = reaction.stoic["*"] if "*" in reaction.stoic.keys() else 0
        return v


    def get_kinetic_constants(self, 
                              t: float = None, 
                              uq: bool = False, 
                              thermo: bool = False) -> None:
        """
        Evaluate the kinetic constants of the reactions in the network.

        Args:
            t (float, optional): Temperature in Kelvin. Defaults to None.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            thermo (bool, optional): If True, the activation barriers will be 
                neglected and only the thermodynamic path is considered. Defaults
                to False.
        """
        if t is None:
            if self.oc["T"] is None:
                raise ValueError("temperature not specified")
            else:
                t = self.oc["T"]
        else:
            self.oc["T"] = t

        for reaction in self.reactions:
            if thermo:
                if uq:
                    e_rxn = np.random.normal(reaction.e_rxn[0], reaction.e_rxn[1])
                    e_act = 0 if e_rxn < 0 else e_rxn
                else:
                    e_rxn = reaction.e_rxn[0]
                    e_act = 0 if e_rxn < 0 else e_rxn
            else:
                if uq:
                    e_act = np.random.normal(reaction.e_act[0], reaction.e_act[1])
                    e_rxn = np.random.normal(reaction.e_rxn[0], reaction.e_rxn[1])
                else:
                    e_act = reaction.e_act[0]
                    e_rxn = reaction.e_rxn[0]
            reaction.k_eq = np.exp(-e_rxn / t / K_B)
            if reaction.r_type == "adsorption":  
                reaction.k_dir = 1e-18 / (2 * np.pi * reaction.adsorbate_mass * K_BU * t) ** 0.5
                reaction.k_dir *= np.exp(-e_act / K_B / t)
            elif reaction.r_type == "desorption":
                reaction.k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)
            else: 
                reaction.k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)
            reaction.k_rev = reaction.k_dir / reaction.k_eq

    def get_hubs(self, n:int = None) -> dict[str, int]:
        """
        Get hubs of the network.

        Returns:
            hubs (dict): Dictionary containing the intermediates and the number
                of reactions in which they are involved, sorted in descending
                order.
        """
        hubs = {inter.code: 0 for inter in self.intermediates.values()}
        for inter in self.intermediates.values():
            for reaction in self.reactions:
                hubs[inter.code] += 1 if inter.code in reaction.stoic.keys() else 0
        hubs = dict(sorted(hubs.items(), key=lambda item: item[1], reverse=True))
        if n is not None:
            return dict(list(hubs.items())[:n])
        else:
            return hubs

    def run_microkinetic(
        self,
        iv: dict[str, float],
        main_reactant: str,
        temperature: float = None,
        pressure: float = None,
        uq: bool = False,
        uq_samples: int = 100,
        thermo: bool = False,
        solver: str = "Julia",
        barrier_threshold: float = None, 
        ss_tol: float = 1e-10
    ) -> dict:
        """
        Run microkinetic simulation.

        Args:
            iv (dict): Initial values of the molar fractions of the gas phase
                intermediates. Surface is assumed to be empty. Keys are the
                chemical formulas of the intermediates and values are the
                molar fractions (e.g. {"CH4": 0.1, "H2O": 0.2, "N2": 0.7}), 
                where the sum of the values must be 1.0.
            temperature (float, optional): Temperature in Kelvin. Defaults to None.
            pressure (float, optional): Pressure in bar. Defaults to None.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            uq_samples (int, optional): Number of samples for the uncertainty
                quantification. Defaults to 100.
            thermo (bool, optional): If True, the activation barriers will be
                neglected and only the thermodynamic path is considered. Defaults
                to False.
            solver (str, optional): Solver to be used. Defaults to "Julia".
            barrier_threshold (float, optional): Threshold for the activation
                energy of the reactions. If a reaction has an activation energy
                below this threshold, it will be filtered out. Defaults to None.

        Returns:
            dict containing the results of the simulation.
        """
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
        inters = sorted(
            self.intermediates.keys(), key=lambda x: self.intermediates[x].code)
        inters_formula = [self.intermediates[x].molecule.get_chemical_formula() for x in inters]
        gas_mask = np.array(
            [self.intermediates[inter].phase == "gas" for inter in inters] + [False], dtype=bool
        )

        inlet_molecules = [inter for inter in iv.keys() if inter in inters_formula]
        for reaction in self.reactions:
            if reaction.r_type in ("adsorption", "desorption"):
                formula_reactants = [inter.molecule.get_chemical_formula() for inter in reaction.components[0] if inter.code != "*"]
                formula_products = [inter.molecule.get_chemical_formula() for inter in reaction.components[1] if inter.code != "*"]
                rxn_formulas = formula_reactants + formula_products
                if any(inlet_molecule in rxn_formulas for inlet_molecule in inlet_molecules):
                    if reaction.r_type == "adsorption":
                        continue
                    else:
                        reaction.reverse()
                else:
                    if reaction.r_type == "adsorption":
                        reaction.reverse()

        v = self.stoichiometric_matrix.copy()
        y0 = np.zeros(len(inters) + 1, dtype=np.float64)
        y0[-1] = 1.0   # Initial condition: empty surface
        num_inerts = 0
        for key in iv.keys():
            if key not in inters_formula:  # inert
                v = np.vstack((v, np.zeros(len(self.reactions), dtype=np.int8)), dtype=np.int8)
                y0 = np.append(y0, P * iv[key])
                gas_mask = np.append(gas_mask, True)
                num_inerts += 1
            else:
                for i, inter in enumerate(self.intermediates.values()):
                    if inter.molecule.get_chemical_formula() == key and inter.phase == "gas":
                        y0[i] = P * iv[key]
                        break

        self.get_kinetic_constants(T, uq, thermo)
        kd = np.array([reaction.k_dir for reaction in self.reactions], dtype=np.float64)
        kr = np.array([reaction.k_rev for reaction in self.reactions], dtype=np.float64)
        # get min and max among all k_dir and k_rev
        ktot = np.concatenate((kd, kr))
        kmax, kmin = np.max(ktot), np.min(ktot)
        print("Ratio between max and min k_dir: {:.2e}".format(kmax / kmin))


        # APPLY BARRIER THRESHOLD
        if isinstance(barrier_threshold, (int, float)) and barrier_threshold > 0:
            filtered_idxs = []
            condition = lambda x, y: x >= barrier_threshold and y >= barrier_threshold
            for i, reaction in enumerate(self.reactions):
                eact_dir, eact_rev = reaction.e_act[0], reaction.e_act[0] - reaction.e_rxn[0]
                if condition(eact_dir, eact_rev):
                    filtered_idxs.append(i)
                    v[:, i] = 0
            print(f"Filtered {len(filtered_idxs)} reactions")   

        # RUN SIMULATION
        reactor = DifferentialPFR(v, kd, kr, gas_mask, ss_tol)
        print(reactor)
        if solver == "Julia":
            results = reactor.integrate(y0)
            print("Steady state reached")
        elif solver == "Python":
            rtol, atol, sstol = 1e-6, 1e-12, 1e-10
            count_atol_decrease = 0
            status = None
            while status != 1: # play with atol and rtol to get successful integration  
                try:  
                    results = reactor.integrate(y0, "Python", rtol, atol, sstol)
                    status = results["status"]                    
                except:
                    status = None
                    
                if status != 1:
                    atol *= 1e2
                    count_atol_decrease += 1
                    if count_atol_decrease % 4 == 0: 
                        rtol *= 1e2
                        print('Lowering relative tolerance to reach steady state... (rtol = {})'.format(rtol))
                    print('Lowering absolute tolerance to reach steady state... (atol = {})'.format(atol))
                    

            path = []
            while path == []:
                mkm_rxns = deepcopy(self.reactions)
                new_graph = deepcopy(self.graph)
                print("ENTERING MAX FLUX")
                new_graph.remove_edges_from(list(new_graph.edges))

                for i, inter in enumerate(inters):
                    if self.intermediates[inter].phase == "gas":
                        new_graph.nodes[inter]["molar_fraction"] = results["y"][i] / P
                    elif self.intermediates[inter].phase == "ads":
                        new_graph.nodes[inter]["coverage"] = results["y"][i]
                new_graph.nodes["*"]["coverage"] = results["y"][-1]

                for i, reaction in enumerate(mkm_rxns):
                    if results["rate"][i] < 0:
                        new_graph.remove_node(reaction.code)
                        reaction.reverse()
                        new_graph.add_node(reaction.code, category="reaction", r_type=reaction.r_type, 
                                    rate = results["rate"][i], e_act = reaction.e_act[0], e_rxn = reaction.e_rxn[0])
                    else:
                        new_graph.nodes[reaction.code]["rate"] = results["rate"][i]
                        new_graph.nodes[reaction.code]["e_act"] = reaction.e_act[0]
                        new_graph.nodes[reaction.code]["e_rxn"] = reaction.e_rxn[0]

                for i, inter in enumerate(inters):
                    for j, reaction in enumerate(mkm_rxns):
                        if inter in reaction.stoic.keys():
                            if results["consumption_rate"][i, j] < 0:  # Reaction j consumes inter i
                                if reaction.e_act[0] == 0:
                                    delta = 0
                                elif reaction.e_act[0] == reaction.e_rxn[0]:  # for energy diagram
                                    delta = reaction.e_rxn[0]
                                else:
                                    delta = reaction.e_act[0]
                                new_graph.add_edge(inter, 
                                            reaction.code, 
                                            rate=abs(results["consumption_rate"][i, j]), 
                                            delta=delta, 
                                            weight=1/abs(results["consumption_rate"][i, j]))
                            else:  # Reaction j produces inter i (v,r > 0 or v,r<0)
                                new_graph.add_edge(reaction.code, 
                                            inter, 
                                            rate=abs(results["consumption_rate"][i, j]), 
                                            delta=-(reaction.e_act[0] - reaction.e_rxn[0]), 
                                            weight=1/abs(results["consumption_rate"][i, j]))
                new_graph.remove_node("*")

                path = max_flux(new_graph, main_reactant)
                if not path:
                    atol /= 1e2  # Tight steady state precision to avoid oscillations and help convergence
                    print('Lowering relative tolerance to improve numerical stability... (atol = {})'.format(atol))
                    results = reactor.integrate(y0, "Python", rtol, atol, sstol)

            if any({reaction.r_type == "PCET" for reaction in self.reactions}):
                new_graph = self.gen_electro_graph(new_graph)
            results["run_graph"] = new_graph
            results["inters"] = inters

            print('Steady state reached (rtol = {}, atol = {}, sstol = {})'.format(rtol, atol, sstol))
        return results
