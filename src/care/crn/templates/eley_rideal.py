from care import Intermediate
from care.crn.intermediate import GasSpecies, AdsorbedSpecies
from care.crn.templates import Adsorption, Desorption

class AssociativeAdsorption(Adsorption):
    """
    Eley-Rideal (associative adsorption) elementary reaction:
    A* + B(g) -> C*
    """
    def __init__(self, components, stoic=None):
        super().__init__(components, stoic)

    def reverse(self):
        super().reverse()
        self.__class__ = DissociativeDesorption


class DissociativeDesorption(Desorption):
    """
    Dissociative desorption elementary reaction of the type:
    A* -> B(g) * C*
    """
    def __init__(self, components, stoic=None):
        super().__init__(components, stoic)

    def reverse(self):
        super().reverse()
        self.__class__ = AssociativeAdsorption

def gen_eleyrideal(gas_inters: list[GasSpecies],
                   ads_inters: list[AdsorbedSpecies]) -> list[AssociativeAdsorption]:
    """
    Generate Eley-Rideal reactions from a list of gas-phase intermediates and
    adsorbed intermediates.

    Args:
        gas_inters (list[Intermediate]): List of gas-phase intermediates.
        ads_inters (list[Intermediate]): List of adsorbed intermediates.

    Returns:
        list[AssociativeAdsorption]: List of Eley-Rideal reactions.
    """
    eleyrideal_reactions = []
    for gas in gas_inters:
        for ads in ads_inters:
            products = check_eleyrideal(gas, ads)
            if products:
                for product in products:
                    eleyrideal_reactions.append(AssociativeAdsorption(
                        components=[gas, ads, product]))
    return eleyrideal_reactions

def check_eleyrideal(gas: Intermediate,
                     ads: Intermediate) -> list[Intermediate]:
    """
    Check if a Eley-Rideal reaction is possible between a gas-phase intermediate
    and an adsorbed intermediate.
    """
    # TODO: Implement this function
    pass
