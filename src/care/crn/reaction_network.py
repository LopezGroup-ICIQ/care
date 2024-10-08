from typing import Union
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from ase import Atoms
from ase.visualize import view

from care import ElementaryReaction, Intermediate, Surface
from care.constants import K_B, K_BU, OC_KEYS, H
from care.crn.utils.electro import Electron, Proton, Water
from care.crn.reactors import DifferentialPFR
from care.crn.visualize import visualize_reaction
from care.crn.microkinetic import gen_graph, calc_eapp


class ReactionNetwork:
    """
    Base class for surface reaction networks.

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
        type: str = None,
    ):
        self.intermediates = intermediates
        self.reactions = reactions
        self.surface = surface
        self.ncc = ncc
        self.noc = noc
        self.excluded = None
        self.num_closed_shell_mols = len(
            [
                inter
                for inter in self.intermediates.values()
                if inter.closed_shell and inter.phase == "gas"
            ]
        )
        self.num_intermediates = len(self.intermediates)
        self.num_reactions = len(self.reactions)
        if type not in ("thermal", "electrochemical"):
            raise ValueError("type must be either 'thermal' or 'electrochemical'")
        self.type = type

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
            len(self.reactions),
        )
        string += "Surface: {}\n".format(self.surface)
        string += "Network Carbon cutoff: {}\n".format(self.ncc)
        string += "Network Oxygen cutoff: {}\n".format(self.noc)
        string += "Type: {}\n".format(self.type)
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

    def add_reactions(
        self, rxns: Union[list[ElementaryReaction], ElementaryReaction]
    ) -> None:
        """
        Add elementary reactions to the network.

        Args:
            rxns (list of obj:`ElementaryReaction` or obj:`ElementaryReaction`): List
                containing the elementary reactions that will be added to the network.
        """
        if isinstance(rxns, list):
            self.reactions += rxns
        elif isinstance(rxns, ElementaryReaction):
            self.reactions.append(rxns)
        else:
            raise TypeError(
                "rxns must be a list of ElementaryReaction or ElementaryReaction"
            )

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

    @property
    def graph(self) -> nx.DiGraph:
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
                closed_shell=inter.closed_shell,
            )

        graph.add_node(
            "*",
            category="intermediate",
            phase="surf",
            formula="*",
            nC=0,
            nH=0,
            nO=0,
            closed_shell=False,
        )

        for idx, reaction in enumerate(self.reactions):
            r_type = (
                reaction.r_type
                if reaction.r_type
                in ("adsorption", "desorption", "eley_rideal", "PCET")
                else "surface_reaction"
            )
            graph.add_node(
                reaction.code,
                category="reaction",
                r_type=r_type,
                rr=reaction.r_type,
                idx=idx,
            )

            for inter in reaction.components[0]:
                if inter.phase in ("solv", "electro"):
                    if not graph.has_node(inter.code):
                        graph.add_node(
                            inter.code,
                            category="electro",
                            phase=inter.phase,
                            formula=inter.formula,
                            nC=inter["C"],
                            nH=inter["H"],
                            nO=inter["O"],
                        )
                    graph.add_edge(
                        inter.code,
                        reaction.code,
                        dir="in",
                        v=reaction.stoic[inter.code],
                    )
                else:
                    graph.add_edge(
                        inter.code,
                        reaction.code,
                        dir="in",
                        v=reaction.stoic[inter.code],
                    )
            for inter in reaction.components[1]:
                if inter.phase in ("solv", "electro"):
                    if not graph.has_node(inter.code):
                        graph.add_node(
                            inter.code,
                            category="electro",
                            phase=inter.phase,
                            formula=inter.formula,
                            nC=inter["C"],
                            nH=inter["H"],
                            nO=inter["O"],
                        )
                    graph.add_edge(
                        reaction.code,
                        inter.code,
                        dir="out",
                        v=reaction.stoic[inter.code],
                    )
                else:
                    graph.add_edge(
                        reaction.code,
                        inter.code,
                        dir="out",
                        v=reaction.stoic[inter.code],
                    )

        graph.remove_node("*")
        return graph

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

    def search_inter_by_elements(self, element_dict: dict[str, int]) -> tuple:
        """Given a dictionary with the elements as keys and the number of each
        element as value, returns all the intermediates matching the provided
        stoichiometry.

        Args:
            element_dict (dict): Dictionary with the symbol of the elements as
            keys and the number of each element as value.

        Returns:
            tuple of obj:`Intermediate` containing all the matching intermediates.
        """
        for elem in ['C', 'H', 'O']:  # hardcoded as INTER_ELEMS contains also '*' and 'q' (to fix)
            if elem not in element_dict.keys():
                element_dict[elem] = 0
        matches = []
        for inter in self.intermediates.values():
            elem_tmp = {element: inter[element] for element in ['C', 'H', 'O']}
            if all(
                [
                    elem_tmp[element] == element_dict[element]
                    for element in elem_tmp.keys()
                ]
            ):
                matches.append(inter)
        return tuple(matches)

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
        if self.reactions[idx].e_rxn is None or self.reactions[idx].e_act is None:
            raise ValueError("Reaction energetic properties are not estimated")
        return visualize_reaction(self.reactions[idx], show_uncertainty)

    def get_reaction_table(self) -> pd.DataFrame:
        repr_hr_width = max(len(step.repr_hr) for step in self) + 2
        dhr_width = 10
        eact_width = 10
        class_width = 55

        # Print header (optional)
        header = "{:<{}} {:<{}} {:<{}} {}".format("Step", repr_hr_width, "DHR (eV)", dhr_width, "Eact (eV)", eact_width, "Class")
        print(header)
        print("=" * (repr_hr_width + dhr_width + eact_width + class_width))

        # Print each step with aligned columns
        for step in self:
            repr_hr_str = step.repr_hr.ljust(repr_hr_width)
            dhr_str = "{:.2f}".format(step.e_rxn[0]).ljust(dhr_width)
            eact_str = "{:.2f}".format(step.e_act[0]).ljust(eact_width)
            class_str = str(step.__class__).ljust(class_width)
            print(f"{repr_hr_str} {dhr_str} {eact_str} {class_str}")

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
        v = np.zeros((len(self.intermediates) + 1, len(self.reactions)), dtype=np.int8)
        sorted_intermediates = sorted(self.intermediates.keys())
        for i, inter in enumerate(sorted_intermediates):
            for j, reaction in enumerate(self.reactions):
                if inter in reaction.stoic.keys():
                    v[i, j] = reaction.stoic[inter]
                v[-1, j] = reaction.stoic["*"] if "*" in reaction.stoic.keys() else 0
        return v

    def get_kinetic_constants(
        self, t: float = None, uq: bool = False, thermo: bool = False
    ) -> None:
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
                reaction.k_dir = (
                    1e-18 / (2 * np.pi * reaction.adsorbate_mass * K_BU * t) ** 0.5
                )
            else:
                reaction.k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)
            reaction.k_rev = reaction.k_dir / reaction.k_eq

    def get_hubs(self, n: int = None) -> dict[str, int]:
        """
        Get hubs of the network.

        Returns:
            hubs (dict): Dictionary containing the intermediates and the number
                of reactions in which they are involved, sorted in descending
                order.
        """
        hubs = defaultdict(int)
        for reaction in self.reactions:
            for inter_code in reaction.stoic.keys():
                # if inter_code not in ["*", "e-", "H+", "OH-", "H2O(aq)"]:
                hubs[inter_code] += 1
        hubs = dict(sorted(hubs.items(), key=lambda item: item[1], reverse=True))
        if n is not None:
            return dict(list(hubs.items())[:n])
        else:
            return hubs

    def run_microkinetic(
        self,
        iv: dict[str, float],
        oc: dict[str, float],
        uq: bool = False,
        nruns: int = 100,
        thermo: bool = False,
        solver: str = "Julia",
        barrier_threshold: float = None,
        ss_tol: float = 1e-10,
        tfin: float = 1e6,
        target_products: list[str] = None,
        eapp: bool = False,
        gpu: bool = False,
    ) -> dict:
        """
        Run microkinetic simulation.

        Args:
            iv (dict): Initial values of the molar fractions of the gas phase
                intermediates. Surface is assumed to be empty. Keys are the
                chemical formulas of the intermediates and values are the
                molar fractions (e.g. {"CH4": 0.1, "H2O": 0.2, "N2": 0.7}),
                where the sum of the values must be 1.0.
            oc (dict): Operating conditions of the simulation. Keys are "T"
                (temperature in Kelvin), "P" (pressure in bar), "U" (potential
                in V) and "pH" (pH of the solution).
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            uq_samples (int, optional): Number of samples for the uncertainty
                quantification. Defaults to 100.
            thermo (bool, optional): If True, the activation barriers will be
                neglected and only the thermodynamic path is considered
                (i.e. E_{TS}=max(E_{IS},E_{FS})). Defaults to False.
            solver (str, optional): Solver to be used ("Python" or "Julia").
                Defaults to "Julia".
            barrier_threshold (float, optional): Threshold for the activation
                energy of the reactions. If a reaction has an activation energy
                higher than the threshold in both directions, it will be filtered
                out. Defaults to None.
            ss_tol (float, optional): Tolerance for the steady state condition.
                ODE Integration stops when the sum of the absolute values of the
                derivatives of the surface coverages is less than ss_tol.
                Defaults to 1e-10 [-].
            tfin (float, optional): Final simulation time. Defaults to 1e6 [s].
            target_products (list, optional): List of the target products. If
                provided, the reaction network will be pruned to only include
                the defined closed-shell products and the reactions leading to
                them. Defaults to None.
            eapp (bool, optional): If True, the apparent activation energy will
                be calculated. Defaults to False.
            gpu (bool, optional): If True, the simulation will be run on the GPU
                if available (only available for Julia). Defaults to False.

        Returns:
            dict containing the results of the simulation at steady-state.
        """
        if sum(iv.values()) != 1.0:
            raise ValueError("Sum of molar fractions is not 1.0")
        if oc["T"] is None:
            if self.temperature is None:
                raise ValueError("temperature not specified")
            else:
                T = oc["T"]
        else:
            T = oc["T"]

        if oc["P"] is None:
            if self.pressure is None:
                raise ValueError("pressure not specified")
            else:
                P = oc["P"]
        else:
            P = oc["P"]

        if self.type == "electrochemical":
            if oc["U"] is None:
                raise ValueError("potential not specified")
            else:
                U = oc["U"]
            if oc["pH"] is None:
                raise ValueError("pH not specified")
            else:
                PH = oc["pH"]

        MKM_GRAPH = deepcopy(self.graph)

        if self.type == "electrochemical":
            ELECTRO_SPECIES = [Proton().code, Electron().code, Water().code]
            ELECTRO_INTERS = [Proton(), Electron(), Water()]
        else:
            ELECTRO_SPECIES, ELECTRO_INTERS = [], []

        if target_products is not None:
            count_removed_inters = 0
            count_removed_reactions = 0
            undesired_closed_shell = []
            reactants_products = list(iv.keys()) + target_products
            for inter in self.intermediates.values():
                if inter.closed_shell and inter.formula not in reactants_products:
                    undesired_closed_shell.append(inter.code)
                    count_removed_inters += 1
            MKM_GRAPH.remove_nodes_from(undesired_closed_shell)
            rm_condition = 0
            while rm_condition == 0:
                inter_to_remove, rxns_to_remove = [], []
                for node in MKM_GRAPH.nodes:
                    if MKM_GRAPH.nodes[node]["category"] == "reaction":
                        idx = MKM_GRAPH.nodes[node]["idx"]
                        if MKM_GRAPH.degree(node) <= 1:
                            rxns_to_remove.append(node)
                            count_removed_reactions += 1
                        else:
                            in_list = [
                                MKM_GRAPH.edges[edge]["dir"]
                                for edge in MKM_GRAPH.in_edges(node)
                            ]
                            reactants = [
                                reactant
                                for reactant in self.reactions[idx].reactants
                                if reactant.phase != "surf"
                            ]
                            products = [
                                product
                                for product in self.reactions[idx].products
                                if product.phase != "surf"
                            ]
                            out_list = [
                                MKM_GRAPH.edges[edge]["dir"]
                                for edge in MKM_GRAPH.out_edges(node)
                            ]
                            if len(in_list) != len(reactants) or len(out_list) != len(
                                products
                            ):
                                rxns_to_remove.append(node)
                                count_removed_reactions += 1

                    if MKM_GRAPH.nodes[node]["category"] == "intermediate":
                        if MKM_GRAPH.degree(node) == 0:
                            inter_to_remove.append(node)
                            count_removed_inters += 1
                        if (
                            MKM_GRAPH.degree(node) == 1
                            and MKM_GRAPH.nodes[node]["phase"] == "ads"
                        ):
                            inter_to_remove.append(node)
                            count_removed_inters += 1

                if len(inter_to_remove) == 0 and len(rxns_to_remove) == 0:
                    rm_condition = 1
                else:
                    MKM_GRAPH.remove_nodes_from(list(set(inter_to_remove)))
                    MKM_GRAPH.remove_nodes_from(list(set(rxns_to_remove)))

            print(
                "Filtered {} species and {} reactions (target products filter)".format(
                    count_removed_inters, count_removed_reactions
                )
            )

        if barrier_threshold > 0.0:
            count_removed_inters, count_removed_reactions = 0, 0
            rxns_to_remove = []
            condition = lambda x, y: x >= barrier_threshold and y >= barrier_threshold
            for node in MKM_GRAPH.nodes():
                if MKM_GRAPH.nodes[node]["category"] == "reaction":
                    rxn = self.reactions[MKM_GRAPH.nodes[node]["idx"]]
                    if condition(rxn.e_act[0], rxn.e_act[0] - rxn.e_rxn[0]):
                        rxns_to_remove.append(node)
                        print(f"Removing {node} from the network")
                        count_removed_reactions += 1
            print(
                "Found {} reactions with Eact > {:.1f} eV".format(
                    count_removed_reactions, barrier_threshold
                )
            )
            MKM_GRAPH.remove_nodes_from(rxns_to_remove)
            rm_condition, n = 0, 0
            while rm_condition == 0:
                n += 1
                inter_to_remove, rxns_to_remove = [], []
                for node in MKM_GRAPH.nodes:
                    if MKM_GRAPH.nodes[node]["category"] == "reaction":
                        if MKM_GRAPH.degree(node) <= 1:
                            print(f"{n}) Removing {node} from the network (0/1 edge)")
                            rxns_to_remove.append(node)
                            count_removed_reactions += 1
                        else:
                            in_list = [
                                MKM_GRAPH.edges[edge]["dir"]
                                for edge in MKM_GRAPH.in_edges(node)
                            ]
                            out_list = [
                                MKM_GRAPH.edges[edge]["dir"]
                                for edge in MKM_GRAPH.out_edges(node)
                            ]
                            if in_list == [] or out_list == []:
                                print(
                                    f"{n}) Removing {node} from the network (edges in same direction)"
                                )
                                rxns_to_remove.append(node)
                                count_removed_reactions += 1
                    if MKM_GRAPH.nodes[node]["category"] == "intermediate":
                        if MKM_GRAPH.degree(node) == 0:
                            print(f"{n}) Removing {node} from the network (isolated)")
                            inter_to_remove.append(node)
                            count_removed_inters += 1
                        if (
                            MKM_GRAPH.degree(node) == 1
                            and MKM_GRAPH.nodes[node]["phase"] == "ads"
                        ):
                            print(
                                f"{n}) Removing {node} from the network (adsorbed participating in 1 reaction)"
                            )
                            inter_to_remove.append(node)
                            count_removed_inters += 1
                if len(inter_to_remove + rxns_to_remove) == 0:
                    rm_condition = 1
                else:
                    MKM_GRAPH.remove_nodes_from(list(set(inter_to_remove)))
                    MKM_GRAPH.remove_nodes_from(list(set(rxns_to_remove)))

            print(
                "Filtered {} reactions and {} species (Eact > {:.1f} eV)".format(
                    count_removed_reactions, count_removed_inters, barrier_threshold
                )
            )

        RXN_IDXS = [
            MKM_GRAPH.nodes[node]["idx"]
            for node in MKM_GRAPH.nodes
            if MKM_GRAPH.nodes[node]["category"] == "reaction"
        ]
        INTERS_CODE = [
            node
            for node in MKM_GRAPH.nodes
            if MKM_GRAPH.nodes[node]["category"] == "intermediate"
        ]
        MKM_RXNS = [self.reactions[i] for i in RXN_IDXS]
        MKM_INTERS = {inter: self.intermediates[inter] for inter in INTERS_CODE}

        # CHECK that no intermediate is missing in the reactions
        for step in MKM_RXNS:
            for inter in step.reactants:
                if inter.code not in MKM_INTERS.keys() and inter.code not in (
                    "*",
                    "e-",
                    "H+",
                    "OH-",
                    "H2O(aq)",
                ):
                    print(
                        f"Reaction {step.code} includes reactant {inter.code} which is not in the network"
                    )
            for inter in step.products:
                if inter.code not in MKM_INTERS.keys() and inter.code not in (
                    "*",
                    "e-",
                    "H+",
                    "OH-",
                    "H2O(aq)",
                ):
                    print(
                        f"Reaction {step.code} includes product {inter.code} which is not in the network"
                    )

        # BUILD THE MICROKINETIC MODEL
        inters = sorted(
            MKM_INTERS.keys(), key=lambda x: MKM_INTERS[x].code
        )  # order of y vector defined here!
        inters_formula = [MKM_INTERS[x].formula for x in inters]
        gas_mask = np.array(
            [MKM_INTERS[inter].phase == "gas" for inter in inters], dtype=bool
        )
        MKM_INTERS["*"] = Intermediate(
            "*", molecule=Atoms(), is_surface=True, phase="surf"
        )
        inters.append("*")
        inters_formula.append("*")
        gas_mask = np.append(gas_mask, False)

        inlet_molecules = [inter for inter in iv.keys() if inter in inters_formula]
        for reaction in MKM_RXNS:
            if reaction.r_type in ("adsorption", "desorption"):
                formula_reactants = [
                    inter.formula
                    for inter in reaction.components[0]
                    if inter.code != "*"
                ]
                formula_products = [
                    inter.formula
                    for inter in reaction.components[1]
                    if inter.code != "*"
                ]
                rxn_formulas = formula_reactants + formula_products
                if any(
                    inlet_molecule in rxn_formulas for inlet_molecule in inlet_molecules
                ):
                    if reaction.r_type == "adsorption":
                        continue
                    else:
                        reaction.reverse()
                else:
                    if reaction.r_type == "adsorption":
                        reaction.reverse()
            elif reaction.r_type == "PCET":
                if U < 0:
                    if Electron() not in reaction.reactants:
                        reaction.reverse()
                else:
                    if Electron() in reaction.reactants:
                        reaction.reverse()
            else:
                if any(
                    [
                        inter.closed_shell and inter.formula not in inlet_molecules
                        for inter in reaction.components[0]
                    ]
                ):
                    reaction.reverse()

                if any(
                    [
                        inter.closed_shell and inter.formula in inlet_molecules
                        for inter in reaction.components[1]
                    ]
                ):
                    reaction.reverse()

        # Generating new graph with reversed reactions
        MKM_GRAPH = gen_graph(MKM_INTERS, MKM_RXNS)

        v = np.zeros((len(MKM_INTERS), len(MKM_RXNS)), dtype=np.int8)

        for i, reaction in enumerate(MKM_RXNS):
            for inter, stoic in reaction.stoic.items():
                if inter not in ELECTRO_SPECIES:
                    v[inters.index(inter), i] = stoic

        y0 = np.zeros(len(inters), dtype=np.float64)
        y0[-1] = 1.0  # Initial condition: empty surface
        num_inerts = 0
        inerts = []
        sorted_dict_keys = sorted(iv.keys())
        for key in sorted_dict_keys:
            if key not in inters_formula:  # inert
                v = np.vstack(
                    (v, np.zeros(len(MKM_RXNS), dtype=np.int8)), dtype=np.int8
                )
                y0 = np.append(y0, P * iv[key])
                gas_mask = np.append(gas_mask, True)
                num_inerts += 1
                inerts.append(key)
                # inters.append(key)
                # inters_formula.append(key)
        for key in sorted_dict_keys:
            for i, inter in enumerate(inters_formula):
                if inter == key and gas_mask[i] == True:
                    y0[i] = P * iv[key]
                    break

        for i, inter in enumerate(inters):
            if gas_mask[i]:
                if not any(v[i, :]):
                    raise ValueError(
                        f"Gas species {inter} does not participate in any reaction"
                    )
            else:
                if np.sum(np.abs(v[i, :])) < 2:
                    raise ValueError(
                        f"Surface species {inter} does not participate in at least two reactions"
                    )

        dfv = pd.DataFrame(
            v,
            index=inters_formula + inerts,
            columns=["R{}".format(i + 1) for i, _ in enumerate(MKM_RXNS)],
        )
        dfv.columns = pd.MultiIndex.from_tuples(
            [(col, MKM_RXNS[i].r_type) for i, col in enumerate(dfv.columns)]
        )
        dfv.to_csv("stoichiometric_matrix.csv")

        if not uq:
            kf, kr = np.zeros(len(MKM_RXNS), dtype=np.float64), np.zeros(
                len(MKM_RXNS), dtype=np.float64
            )
            for i, reaction in enumerate(MKM_RXNS):
                kf[i], kr[i] = reaction.get_kinetic_constants(T, uq, thermo)
            dfk = pd.DataFrame(
                {"k_dir": kf, "k_rev": kr},
                index=["R{}".format(i + 1) for i, _ in enumerate(MKM_RXNS)],
            )
            dfk.to_csv("kinetic_constants.csv")
        else:
            NRUNS = nruns
            kf, kr = np.zeros((len(MKM_RXNS), NRUNS), dtype=np.float64), np.zeros(
                (len(MKM_RXNS), NRUNS), dtype=np.float64
            )
            for run in range(NRUNS):
                for i, reaction in enumerate(MKM_RXNS):
                    kf[i, run], kr[i, run] = reaction.get_kinetic_constants(
                        T, uq, thermo
                    )
            # dataframe with all constants
            dfk = pd.DataFrame(
                {"k_dir": kf.flatten(), "k_rev": kr.flatten()},
                index=pd.MultiIndex.from_product(
                    [
                        ["R{}".format(i + 1) for i, _ in enumerate(MKM_RXNS)],
                        range(NRUNS),
                    ]
                ),
            )
            dfk.to_csv("kinetic_constants.csv")

        if eapp:
            DELTA = 1  # temperature delta to get apparent activation energy
            kf_plus, kr_plus = np.zeros(len(MKM_RXNS), dtype=np.float64), np.zeros(
                len(MKM_RXNS), dtype=np.float64
            )
            kf_minus, kr_minus = np.zeros(len(MKM_RXNS), dtype=np.float64), np.zeros(
                len(MKM_RXNS), dtype=np.float64
            )
            for i, reaction in enumerate(MKM_RXNS):
                kf_plus[i], kr_plus[i] = reaction.get_kinetic_constants(
                    T + DELTA, uq, thermo
                )
                kf_minus[i], kr_minus[i] = reaction.get_kinetic_constants(
                    T - DELTA, uq, thermo
                )

        dfgm = pd.DataFrame(
            gas_mask, index=inters_formula + inerts, columns=["gas_mask"]
        )
        dfgm.to_csv("gas_mask.csv")

        ktot = np.concatenate((kf, kr))
        kmax, kmin = np.max(ktot), np.min(ktot)
        print("Ratio between max and min k_dir: {:.2e}".format(kmax / kmin))
        print("Shape of stoichiometric matrix: {}".format(v.shape))
        print("Number of values in stoichiometry matrix: {}".format(v.size))

        reactants_mask = gas_mask.copy()
        products_mask = gas_mask.copy()
        for i, inter in enumerate(inters):
            if MKM_INTERS[inter].formula in iv.keys():
                products_mask[i] = False
            else:
                reactants_mask[i] = False

        reactor = DifferentialPFR(v, kf, kr, gas_mask, inters, P, T)
        print(reactor)

        if eapp:
            reactor_plus = DifferentialPFR(
                v, kf_plus, kr_plus, gas_mask, inters, P, T + DELTA
            )
            reactor_minus = DifferentialPFR(
                v, kf_minus, kr_minus, gas_mask, inters, P, T - DELTA
            )

        RTOL, ATOL, SSTOL = 1e-8, 1e-20, ss_tol
        RTOL_MIN, ATOL_MIN = 1e-16, 1e-40
        count_atol_increase = 0
        status = None

        if uq:
            results_uq = []
            for run in range(NRUNS):
                print(f"Run {run+1}/{NRUNS}")
                reactor.kd = kf[:, run]
                reactor.kr = kr[:, run]
                results_uq.append(
                    reactor.integrate(y0, solver, RTOL, ATOL, SSTOL, tfin, gpu)
                )
            results = {}
            results["runs"] = results_uq
            # define all as mean values and std as std of the runs
            results["y"] = np.mean(
                [results["runs"][i]["y"] for i in range(NRUNS)], axis=0
            )
            results["y_std"] = np.std(
                [results["runs"][i]["y"] for i in range(NRUNS)], axis=0
            )
            results["forward_rate"] = np.mean(
                [results["runs"][i]["forward_rate"] for i in range(NRUNS)], axis=0
            )
            results["forward_rate_std"] = np.std(
                [results["runs"][i]["forward_rate"] for i in range(NRUNS)], axis=0
            )
            results["backward_rate"] = np.mean(
                [results["runs"][i]["backward_rate"] for i in range(NRUNS)], axis=0
            )
            results["backward_rate_std"] = np.std(
                [results["runs"][i]["backward_rate"] for i in range(NRUNS)], axis=0
            )
            results["net_rate"] = np.mean(
                [results["runs"][i]["net_rate"] for i in range(NRUNS)], axis=0
            )
            results["net_rate_std"] = np.std(
                [results["runs"][i]["net_rate"] for i in range(NRUNS)], axis=0
            )
            results["consumption_rate"] = np.mean(
                [results["runs"][i]["consumption_rate"] for i in range(NRUNS)], axis=0
            )
            results["consumption_rate_std"] = np.std(
                [results["runs"][i]["consumption_rate"] for i in range(NRUNS)], axis=0
            )
            results["total_consumption_rate"] = np.mean(
                [results["runs"][i]["total_consumption_rate"] for i in range(NRUNS)],
                axis=0,
            )
            results["total_consumption_rate_std"] = np.std(
                [results["runs"][i]["total_consumption_rate"] for i in range(NRUNS)],
                axis=0,
            )
            results["inters"] = inters
            results["gas_mask"] = gas_mask
            results["y0"] = y0
        else:
            while status not in (0, 1):
                results = reactor.integrate(y0, solver, RTOL, ATOL, SSTOL, tfin, gpu)
                if eapp:
                    results_plus = reactor_plus.integrate(
                        y0, solver, RTOL, ATOL, SSTOL, tfin, gpu
                    )
                    results_minus = reactor_minus.integrate(
                        y0, solver, RTOL, ATOL, SSTOL, tfin, gpu
                    )
                status = results["status"]

                if status not in (0, 1):
                    ATOL /= 10
                    count_atol_increase += 1
                    if count_atol_increase % 2 == 0:
                        RTOL /= 10
                        print(
                            "Decreasing rtol to {} to reach steady state...".format(
                                RTOL
                            )
                        )
                    print("Decreasing atol to {} to reach steady state...".format(ATOL))

                if ATOL < ATOL_MIN or RTOL < RTOL_MIN:
                    print("Failed to reach steady state")
                    return None

            df_rates = pd.DataFrame(
                {
                    "type": [reaction.r_type for reaction in MKM_RXNS],
                    "k_dir": kf,
                    "k_rev": kr,
                    "forward_rate": results["forward_rate"],
                    "reverse_rate": results["backward_rate"],
                    "net_rate": results["net_rate"],
                },
                index=["R{}".format(i + 1) for i, _ in enumerate(MKM_RXNS)],
            )
            df_rates.to_csv("rates.csv")

            g = deepcopy(MKM_GRAPH)
            MKM_RXNS_COPY = deepcopy(MKM_RXNS)
            g.remove_edges_from(list(g.edges))

            for i, inter in enumerate(inters):
                if MKM_INTERS[inter].phase == "gas":
                    g.nodes[inter]["molar_fraction"] = results["y"][i] / P
                elif MKM_INTERS[inter].phase == "ads":
                    g.nodes[inter]["coverage"] = results["y"][i]
                else:
                    g.nodes[inter]["coverage"] = results["y"][i]

            for i, reaction in enumerate(MKM_RXNS_COPY):
                if results["net_rate"][i] < 0:
                    g.remove_node(reaction.code)
                    reaction.reverse()  # IN-PLACE OPERATION
                    r_type = (
                        reaction.r_type
                        if reaction.r_type
                        in ("adsorption", "desorption", "eley_rideal", "PCET")
                        else "surface_reaction"
                    )
                    g.add_node(
                        reaction.code,
                        category="reaction",
                        r_type=r_type,
                        rr=reaction.r_type,
                        idx=i,
                        rate=abs(results["net_rate"][i]),
                        e_act=reaction.e_act[0],
                        e_rxn=reaction.e_rxn[0],
                    )
                else:
                    g.nodes[reaction.code]["rate"] = results["net_rate"][i]
                    g.nodes[reaction.code]["e_act"] = reaction.e_act[0]
                    g.nodes[reaction.code]["e_rxn"] = reaction.e_rxn[0]

            for i, inter in enumerate(inters):
                for j, reaction in enumerate(MKM_RXNS_COPY):
                    if inter in reaction.stoic.keys():
                        if (
                            results["consumption_rate"][i, j] < 0
                        ):  # Reaction j consumes inter i (sign(v) != sign(r))
                            g.add_edge(
                                inter,
                                reaction.code,
                                rate=abs(results["consumption_rate"][i, j]),
                                delta=max(0, reaction.e_act[0], reaction.e_rxn[0]),
                                v=reaction.stoic[inter],
                            )
                        else:  # Reaction j produces inter i (sign(v) == sign(r))
                            g.add_edge(
                                reaction.code,
                                inter,
                                rate=abs(results["consumption_rate"][i, j]),
                                delta=-(reaction.e_act[0] - reaction.e_rxn[0]),
                                v=reaction.stoic[inter],
                            )

            # Add electrochemical species (e-, H+, OH-, H2O(aq))
            for inter in ELECTRO_INTERS:
                g.add_node(
                    inter.code,
                    category="intermediate",
                    phase="electro",
                    nC=inter["C"],
                    nH=inter["H"],
                    nO=inter["O"],
                )
            for j, reaction in enumerate(MKM_RXNS_COPY):
                if reaction.r_type == "PCET":
                    for i, inter in enumerate(ELECTRO_SPECIES):
                        if inter in reaction.stoic.keys():
                            if reaction.stoic[inter] < 0:
                                g.add_edge(
                                    inter,
                                    reaction.code,
                                    rate=0,
                                    delta=0,
                                    v=reaction.stoic[inter],
                                )
                            else:
                                g.add_edge(
                                    reaction.code,
                                    inter,
                                    rate=0,
                                    delta=0,
                                    v=reaction.stoic[inter],
                                )
            g.remove_node("*")

            results["run_graph"] = g
            results["inters"] = inters
            results["gas_mask"] = gas_mask
            results["y0"] = y0
            # Saving the tolerance values used
            results["rtol"] = RTOL
            results["atol"] = ATOL
            results["sstol"] = SSTOL
            results["k_ratio"] = kmax / kmin
            if eapp:
                r_minus = results_minus["total_consumption_rate"]
                r_plus = results_plus["total_consumption_rate"]
                r = results["total_consumption_rate"]
                t_vec = np.array([T - DELTA, T, T + DELTA])
                r_vec = np.array([r_minus, r, r_plus])
                results["eapp"] = calc_eapp(t_vec, r_vec, gas_mask)
                for i, inter in enumerate(gas_mask[:-1]):
                    if inter:
                        print(f"{inters_formula[i]}: {results['eapp'][i]}")
            print(
                "Steady state reached (rtol = {}, atol = {}, sstol = {})".format(
                    RTOL, ATOL, SSTOL
                )
            )
        return results
