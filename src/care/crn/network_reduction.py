"""
ReactionNetworkSimplifier class implementation.
Given a network, it provides a toolbox of methods to simplify, removing and modifying intermediates and reactions.
"""

import networkx as nx

from care import Intermediate, ReactionNetwork


class ReactionNetworkSimplifier:
    def __init__(self):
        pass

    def del_by_elements(
        self, network: ReactionNetwork, nc: int = None, nh: int = None, no: int = None
    ) -> None:
        """
        Deletes all the intermediates containing the specified number of atoms.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        elements : list
            List of elements to be deleted.
        """
        pass

    def del_by_chemical_family(
        self, network: ReactionNetwork, family: str, closed_shell_only: bool = True
    ) -> None:
        """
        Delete from the reaction network all the intermediates belonging to the specified chemical family.
        """
        return None

    def del_by_formula(self, network: ReactionNetwork, formula: str) -> None:
        """
        Deletes all the intermediates containing the formula.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        formula : str
            Chemical formula.
        """
        for inter in list(network.intermediates.values()):
            if formula in inter.molecule.get_chemical_formula():
                network.del_inter(inter.code)
        return None

    def del_by_code(self, network: ReactionNetwork, code: str) -> None:
        """
        Deletes the intermediate with the given code.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        code : str
            Code of the intermediate to be deleted.
        """
        network.del_inter(code)
        return None

    def del_by_energy(
        self, network: ReactionNetwork, energy: float, tol: float = 0.0
    ) -> None:
        """
        Deletes all the elementary reactions whose energy is greater than the given energy.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        energy : float
            Energy threshold.
        tol : float, optional
            Tolerance, by default 0.0
        """
        for rxn in list(network.reactions.values()):
            if rxn.energy > energy - tol:
                network.del_rxn(rxn.code)
        return None

    def del_by_barrier(
        self, network: ReactionNetwork, barrier: float, tol: float = 0.0
    ) -> None:
        """
        Deletes all the elementary reactions whose barrier is greater than the given barrier.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        barrier : float
            Barrier threshold.
        tol : float, optional
            Tolerance, by default 0.0
        """
        for rxn in list(network.reactions.values()):
            if rxn.barrier > barrier - tol:
                network.del_rxn(rxn.code)
        return None


def is_alkane(inter: Intermediate) -> bool:
    """
    Checks if the given intermediate is an alkane.

    Parameters
    ----------
    inter : Intermediate
        Intermediate.

    Returns
    -------
    bool
        True if the intermediate is an alkane, False otherwise.
    """
    return set(inter.molecule.get_chemical_symbols()) == set(["C", "H"])


def find_flux(graph: nx.DiGraph, source: str) -> nx.Digraph:
    """
    Finds the flux of the given source node.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph.
    source : str
        Source node.

    Returns
    -------
    nx.DiGraph
        Flux graph.
    """

    #
