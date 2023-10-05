"""
NetworkReduction class implementation.
Given a network, it reduces it to a smaller network based on the rules provided by the user.
"""

from GAMERNet.rnet.networks.reaction_network import ReactionNetwork

class NetworkReduction:
    def __init__(self, 
                 gas_reactants: list[str]):
        
        
        return None
    
    def del_by_elements(self, network: ReactionNetwork, elements: list) -> None:
        """
        Deletes all the intermediates containing the elements in the list.

        Parameters
        ----------
        network : ReactionNetwork
            Reaction network.
        elements : list
            List of elements to be deleted.
        """
        for inter in list(network.intermediates.values()):
            if set(elements).issubset(set(inter.molecule.get_chemical_symbols())):
                network.del_inter(inter.code)
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
    
    def del_by_energy(self, network: ReactionNetwork, energy: float, tol: float = 0.0) -> None:
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
    
    def del_by_barrier(self, network: ReactionNetwork, barrier: float, tol: float = 0.0) -> None:
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